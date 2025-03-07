import random
from collections import OrderedDict
from json import dumps

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_utils
import train_utils
from args import get_classifier_args
from models.models import GCNEncoder, GCNPredictor, DataExpander, SimpleDataset, simple_collate_fn
import os


def get_model(args, log):
    # Convert embed_sizes to a list if it's an integer
    embed_sizes = [args.embed_sizes, args.hidden_size] if isinstance(args.embed_sizes, int) else args.embed_sizes
    
    expander = DataExpander(
        input_size=args.input_size,
        output_size=args.expander_output_size,
        hidden_size=args.hidden_size,
        embed_size=embed_sizes
    )

    encoder = GCNEncoder(
        input_size=args.expander_output_size,  
        hidden_size=args.hidden_size,
        num_layers=args.num_layer,
        drop_prob=args.drop_prob,
        JK=args.JK
    )

    predictor = GCNPredictor(
        in_channels=args.hidden_size,  
        pool_ratio=args.pool_ratio,
        pred_mlp_hidden=args.pred_mlp_hidden,
        pred_output_size=args.output_size,
        gene_mlp_hidden=args.gene_mlp_hidden,
        drop_prob=args.drop_prob
    )

    expander = nn.DataParallel(expander, args.gpu_ids)
    encoder = nn.DataParallel(encoder, args.gpu_ids)
    predictor = nn.DataParallel(predictor, args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        expander, epoch = train_utils.load_model(
            expander, args.load_path + "_expander.pth.tar", args.gpu_ids
        )
        encoder, epoch = train_utils.load_model(
            encoder, args.load_path + "_encoder.pth.tar", args.gpu_ids
        )
        predictor, epoch = train_utils.load_model(
            predictor, args.load_path + "_predictor.pth.tar", args.gpu_ids
        )
    else:
        epoch = 0  

    return expander, encoder, predictor, epoch

def create_batch_subgraph(original_edge_index, batch_indices, num_nodes_total, device):
    """
    Create a subgraph from the original edge_index that only includes edges between nodes in the batch.
    
    Args:
        original_edge_index (torch.Tensor): The original edge_index tensor (2, num_edges)
        batch_indices (torch.Tensor): Indices of nodes in the current batch
        num_nodes_total (int): Total number of nodes in the original graph
        device (torch.device): Device to place tensors on
        
    Returns:
        torch.Tensor: A new edge_index tensor for the batch subgraph
    """
    # Create a mask for each node indicating if it's in the batch
    in_batch = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
    in_batch[batch_indices] = True
    
    # Create a mapping from original node indices to batch-local indices
    node_mapper = torch.full((num_nodes_total,), -1, dtype=torch.long, device=device)
    node_mapper[batch_indices] = torch.arange(len(batch_indices), device=device)
    
    # Filter edges where both source and target nodes are in the batch
    edge_mask = in_batch[original_edge_index[0]] & in_batch[original_edge_index[1]]
    filtered_edges = original_edge_index[:, edge_mask]
    
    # If no edges remain, create self-loops as a fallback
    if filtered_edges.shape[1] == 0 or edge_mask.sum() == 0:
        batch_size = len(batch_indices)
        batch_edge_index = torch.zeros((2, batch_size), dtype=torch.long, device=device)
        batch_edge_index[0] = torch.arange(batch_size, device=device)
        batch_edge_index[1] = torch.arange(batch_size, device=device)
        return batch_edge_index
    
    # Remap node indices to be consecutive in the batch
    remapped_edges = node_mapper[filtered_edges]
    
    # Add self-loops to ensure all nodes have at least one edge
    # This is important for GCN to work properly with isolated nodes
    has_edge = torch.zeros(len(batch_indices), dtype=torch.bool, device=device)
    has_edge[remapped_edges[0]] = True
    has_edge[remapped_edges[1]] = True
    
    # For nodes without any edges, add self-loops
    isolated_nodes = torch.nonzero(~has_edge, as_tuple=True)[0]
    if len(isolated_nodes) > 0:
        self_loops = torch.stack([isolated_nodes, isolated_nodes], dim=0)
        remapped_edges = torch.cat([remapped_edges, self_loops], dim=1)
    
    return remapped_edges


def train(epoch, expander, encoder, predictor, optimizer, train_loader, llm_embedding_X, log, device, edge_index, num_nodes_total):
    """
    Train the model for one epoch using GCNPredictor with batch-specific subgraphs from the original edge_index.
    """
    sum_train_loss = 0
    
    with torch.enable_grad(), tqdm(total=len(train_loader.dataset)) as progress_bar:
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Get the indices of the nodes in this batch within the full dataset
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_nodes_total)
            batch_indices = torch.arange(start_idx, end_idx, device=device)
            
            # Handle last batch which might be smaller
            if end_idx - start_idx < batch_size:
                batch_size = end_idx - start_idx
                batch_x = batch_x[:batch_size]
                batch_y = batch_y[:batch_size]
                batch_indices = batch_indices[:batch_size]
            
            # Create a subgraph for this batch from the original edge_index
            batch_edge_index = create_batch_subgraph(edge_index, batch_indices, num_nodes_total, device)
            
            # Create a batch vector for GCNPredictor
            batch_vector = torch.arange(batch_size, device=device)
            
            optimizer.zero_grad()
            
            # Step 1: Combine expression data with gene descriptions
            batch_x = expander(batch_x, llm_embedding_X)
            
            # Step 2: Process through GCN with the batch-specific edge_index
            x_enc = encoder(batch_x, batch_edge_index)
            
            # Step 3: Make prediction and identify important genes
            pred, gene_scores, perm = predictor(x_enc, batch_edge_index, batch_vector)
            prediction = torch.log_softmax(pred, dim=-1)
            
            # Calculate loss with class weighting
            weight_vector = torch.zeros([args.output_size], device=device)
            for i in range(args.output_size):
                n_samplei = torch.sum(batch_y == i).item()
                if n_samplei > 0:
                    weight_vector[i] = batch_size / (n_samplei * args.output_size)
            nll_loss = F.nll_loss(prediction, batch_y, weight=weight_vector)
            loss = nll_loss
            
            sum_train_loss += loss.item()
            
            # Backpropagation and optimization
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            nn.utils.clip_grad_norm_(predictor.parameters(), args.max_grad_norm)
            optimizer.step()
            
            # Update progress bar
            progress_bar.update(batch_size)
            progress_bar.set_postfix(epoch=epoch, NLL=loss.item())
            
        log.info(f"Train Loss: {sum_train_loss / len(train_loader.dataset) * args.batch_size}")
    
    return


def evaluate(encoder, expander, predictor, data_loader, llm_embedding_X, device, save_dir, epoch, log, args, edge_index, num_nodes_total, test=False):
    """
    Evaluate the model using batch-specific subgraphs from the original edge_index.
    """
    nll_meter = train_utils.AverageMeter()
    metrics_meter = train_utils.MetricsMeter()

    encoder.eval()
    predictor.eval()
    expander.eval()

    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        sum_nll_loss = 0

        for i, (batch_x, batch_y) in enumerate(data_loader):
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Get the indices of the nodes in this batch within the full dataset
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_nodes_total)
            batch_indices = torch.arange(start_idx, end_idx, device=device)
            
            # Handle last batch which might be smaller
            if end_idx - start_idx < batch_size:
                batch_size = end_idx - start_idx
                batch_x = batch_x[:batch_size]
                batch_y = batch_y[:batch_size]
                batch_indices = batch_indices[:batch_size]
            
            # Create a subgraph for this batch from the original edge_index
            batch_edge_index = create_batch_subgraph(edge_index, batch_indices, num_nodes_total, device)
            
            # Create batch vector for GCNPredictor
            batch_vector = torch.arange(batch_size, device=device)

            # Process through the model using the batch-specific edge_index
            batch_x = expander(batch_x, llm_embedding_X)
            node_emb = encoder(batch_x, batch_edge_index)
            pred, gene_scores, perm = predictor(node_emb, batch_edge_index, batch_vector)
            prediction = torch.log_softmax(pred, dim=-1)

            # Calculate metrics
            nll_loss = F.nll_loss(prediction, batch_y)
            sum_nll_loss += nll_loss.item()
            nll_meter.update(nll_loss.item(), batch_size)
            metrics_meter.update(prediction.exp()[:, 1].cpu(), batch_y.cpu())

            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

        # Return to training mode
        expander.train()
        encoder.train()
        predictor.train()

        # Log evaluation results
        log.info(f"NLL Loss: {sum_nll_loss / len(data_loader.dataset) * args.batch_size}")

        # Compute and return metrics
        metrics_result = metrics_meter.return_metrics()
        results_list = [
            ('Loss', nll_meter.avg),
            ('Accuracy', metrics_result["Accuracy"]),
            ('Recall', metrics_result["Recall"]),
            ('Precision', metrics_result["Precision"]),
            ('Specificity', metrics_result["Specificity"]),
            ('F1', metrics_result["F1"]),
            ('AUC', metrics_result["AUC"])
        ]
        results = OrderedDict(results_list)

        return results


# def cal_emb(model, expander, path_list, path_index, path_edge_type, path_positions, prior_path_weight,data_loader,llm_embedding_X,pathway_embeddings, device, save_dir, epoch,log,):
#     model.eval()
#     expander.eval()
#     all_path_embs = []
#     with torch.no_grad(), \
#             tqdm(total=len(data_loader.dataset)) as progress_bar:
#         times = 0
#         for batch_x, batch_y, batch_in_deg, batch_out_deg, batch_edge_types, batch_node_index, batch_mask in data_loader:

#             batch_size = batch_x.size(0)
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             batch_in_deg = batch_in_deg.to(device)
#             batch_out_deg = batch_out_deg.to(device)
#             batch_edge_types = batch_edge_types.to(device)
#             batch_node_index = batch_node_index.to(device)
#             batch_mask = batch_mask.to(device)
#             batch_x = expander(batch_x,llm_embedding_X)
#             _, path_emb = model(batch_x, batch_mask, batch_in_deg, batch_out_deg, batch_edge_types,
#                                 batch_node_index, path_list, path_index, path_edge_type, path_positions,pathway_embeddings)
#             # Log info
#             # print(path_emb.shape)
#             if path_emb.dim() == 1:
#                 path_emb = path_emb.unsqueeze(0)
#             progress_bar.update(batch_size)
#             all_path_embs.append(path_emb.cpu())
#         all_path_embs = torch.cat(all_path_embs, dim=0).cpu().numpy()
#         output_file = f"{save_dir}/output_cell_embeddings.csv"
#         df = pd.DataFrame(all_path_embs)
#         df.to_csv(output_file, index=False)
#         log.info(f"Cell embedding saved!")
#         return


def main(args):
    # Set up directory
    args.save_dir = train_utils.get_save_dir(args.save_dir, args.name, "train_classifier")

    # Get and process input data - simplified version
    args, expression, label, llm_embedding_X, edge_index, gene_list = data_utils.process_input_data(args)

    # Set up logging and devices
    log = train_utils.get_logger(args.save_dir, args.name)
    device, args.gpu_ids = train_utils.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    best_train_results = []
    best_val_results = []
    best_test_results = []
    
    # Multiple runs
    for run in range(args.runs):
        log.info(f"Start run {run + 1}")

        # Set random seed
        seed = train_utils.get_seed(args.seed)
        log.info(f'Using random seed {seed}...')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Get model
        log.info('Building model...')
        expander, encoder, predictor, epoch = get_model(args, log)
        expander = expander.to(device)
        encoder = encoder.to(device)
        predictor = predictor.to(device)
        
        # Set models to training mode
        expander.train()
        encoder.train()
        predictor.train()
        
        # Move data to device
        llm_embedding_X = llm_embedding_X.to(device)
        edge_index = edge_index.to(device)

        # Get saver
        saver = train_utils.CheckpointSaver(args.save_dir,
                                          max_checkpoints=args.max_checkpoints,
                                          metric_name=args.metric_name,
                                          maximize_metric=args.maximize_metric,
                                          log=log)

        # Set optimizer and scheduler
        if args.maximize_metric:
            mode = "max"
        else:
            mode = "min"
            
        optimizer = optim.Adam(
            list(expander.parameters()) + list(encoder.parameters()) + list(predictor.parameters()), 
            lr=args.lr, 
            weight_decay=args.l2_wd
        )
        
        scheduler = sched.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=1, 
            eta_min=1e-7
        )

        # Prepare dataset splits
        index = np.arange(expression.shape[0])
        train_index, val_index, test_index = train_utils.split_train_val_test(
            index, seed, args.val_ratio, args.test_ratio, label
        )
        
        # Create simplified datasets
        train_dataset = SimpleDataset(expression[train_index], label[train_index])
        val_dataset = SimpleDataset(expression[val_index], label[val_index])
        test_dataset = SimpleDataset(expression[test_index], label[test_index])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=simple_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.num_workers,
            collate_fn=simple_collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=simple_collate_fn
        )

                    # Train loop
        log.info('Training the model on classification...')
        num_nodes_total = expression.shape[0]  # Total number of cells
        
        while epoch != args.num_epochs:
            epoch += 1
            log.info(f'Starting epoch {epoch}...')

            train(epoch, expander, encoder, predictor, optimizer, 
                 train_loader, llm_embedding_X, log, device, edge_index, num_nodes_total)

            # Evaluate and save checkpoint
            log.info(f'Evaluating after epoch {epoch}...')
            val_results = evaluate(encoder, expander, predictor, 
                                 val_loader, llm_embedding_X, device, 
                                 args.save_dir, epoch, log, args, edge_index, num_nodes_total, test=False)
            
            model_dict = {
                f"_expander_{run + 1}": expander,
                f"_encoder_{run + 1}": encoder, 
                f"_predictor_{run + 1}": predictor
            }
            
            scheduler.step(val_results[args.metric_name])
            
            for param_group in optimizer.param_groups:
                log.info(f"Current LR: {param_group['lr']}")

            # Log to console
            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in val_results.items())
            log.info(f'Val {results_str}')

            if saver.is_best(val_results[args.metric_name]):
                test_results = evaluate(encoder, expander, predictor, 
                                      test_loader, llm_embedding_X, device, 
                                      args.save_dir, epoch, log, args, edge_index, num_nodes_total, test=True)
                results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in test_results.items())
                log.info(f'Test {results_str}')

            saver.save(epoch, model_dict, val_results[args.metric_name], device)
            torch.cuda.empty_cache()

        # End of training - load best model and get final results
        best_val_results.append(saver.best_val)
        expander, _ = train_utils.load_model(expander, f"{args.save_dir}/best" + f"_expander_{run + 1}.pth.tar", args.gpu_ids)
        encoder, _ = train_utils.load_model(encoder, f"{args.save_dir}/best" + f"_encoder_{run + 1}.pth.tar", args.gpu_ids)
        predictor, _ = train_utils.load_model(predictor, f"{args.save_dir}/best" + f"_predictor_{run + 1}.pth.tar", args.gpu_ids)
        
        # Get final results
        train_results = evaluate(encoder, expander, predictor, train_loader, llm_embedding_X, device, args.save_dir, epoch, log, args, edge_index, num_nodes_total, test=False)
        best_train_results.append(train_results[args.metric_name])
        val_results = evaluate(encoder, expander, predictor, val_loader, llm_embedding_X, device, args.save_dir, epoch, log, args, edge_index, num_nodes_total, test=False)
        test_results = evaluate(encoder, expander, predictor, test_loader, llm_embedding_X, device, args.save_dir, epoch, log, args, edge_index, num_nodes_total, test=True)
        best_test_results.append(test_results[args.metric_name])
        
        # Log final results
        train_results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in train_results.items())
        val_results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in val_results.items())
        test_results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in test_results.items())

        log.info(f"Best train {train_results_str}")
        log.info(f"Best val {val_results_str}")
        log.info(f'Best test {test_results_str}')
        
        # Save top genes information if available
        if hasattr(predictor.module, 'get_top_genes'):
            top_genes = predictor.module.get_top_genes(gene_list)
            with open(f'{args.save_dir}/top_genes_run_{run+1}.txt', 'w') as f:
                for gene, score in top_genes:
                    f.write(f"{gene}\t{score:.4f}\n")

    # Final results across all runs
    log.info(f"Finish training, compute average results.")
    mean_train = np.mean(best_train_results)
    std_train = np.std(best_train_results)
    mean_val = np.mean(best_val_results)
    std_val = np.std(best_val_results)
    mean_test = np.mean(best_test_results)
    std_test = np.std(best_test_results)

    train_desc = '{:.3f} ± {:.3f}'.format(mean_train, std_train)
    val_desc = '{:.3f} ± {:.3f}'.format(mean_val, std_val)
    test_desc = '{:.3f} ± {:.3f}'.format(mean_test, std_test)

    log.info(f"Train result: {train_desc}")
    log.info(f"Validation result: {val_desc}")
    log.info(f"Test result: {test_desc}")
    
    # Save gene list for reference
    np.savez(f"{args.save_dir}/save_result.npz", gene_list=gene_list)
    
    return



if __name__ == "__main__":
    # Load args.
    args = get_classifier_args()
    main(args)
    # generate_intra_network(args.save_dir, 300)