"""
Utils files
"""
import logging
import os
import queue
import random
import shutil
import time
import networkx as nx
from networkx import Graph
import numpy as np
import torch
import torch.utils.data as data
import tqdm
import ujson as json
from sklearn.metrics import roc_curve, auc
from torch_geometric.data import Data
from torch_scatter import scatter
from typing import Optional, Tuple
from torch import Tensor, LongTensor
import torch.nn as nn


def reindex_nx_graph(G: Graph, ordered_node_list: list) -> Graph:
    r"""reindex the nodes in nx graph according to given ordering.
    Args:
        G (Graph): Networkx graph object.
        ordered_node_list (list): A list served as node ordering.
    """

    ordered_node_dict = dict(zip(ordered_node_list, range(len(ordered_node_list))))
    return nx.relabel_nodes(G, ordered_node_dict)


def save_json(filename: str,
              obj: dict,
              message: Optional[str] = None,
              ascii: Optional[bool] = True):
    r"""Save data in JSON format.
    Args:
        filename (str) : Name of save directory (including file name).
        obj (object): Data to be saved.
        message (Optional, str): Anything to print.
        ascii (Optional, bool): If ture, ensure the encoding is ascii.
    """
    if message is not None:
        print(f"Saving {message}...")
    with open(filename, "w") as fh:
        json.dump(obj, fh, ensure_ascii=ascii)


def nx_to_graph_data(G: Graph, num_nodes: int ) -> Data:
    r"""convert nx graph to torch geometric Data object
    Args:
        G（nx.Graph): Networkx graph.
        num_nodes(Tensor): a sclar tensor to save the number of node in the graph
    """
    edge_list = G.edges
    edge_index = torch.from_numpy(np.array(edge_list).T).to(torch.long)
    in_deg = torch.tensor([node for (node, val) in sorted(G.in_degree, key=lambda pair: pair[0])]).long().unsqueeze(-1)
    out_deg = torch.tensor([node for (node, val) in sorted(G.out_degree, key=lambda pair: pair[0])]).long().unsqueeze(
        -1)
    return Data(edge_index=edge_index, num_nodes=num_nodes, in_deg=in_deg, out_deg=out_deg)


def nx_compute_in_and_out_degree(G: Graph) -> Tuple[Tensor, Tensor]:
    r"""Compute in and out degree of each node in the input graph.
    Args:
        G (nx.Graph): Networkx graph.
    """
    in_deg = torch.tensor([val for (node, val) in sorted(G.in_degree, key=lambda pair: pair[0])]).int().unsqueeze(-1)
    out_deg = torch.tensor([val for (node, val) in sorted(G.out_degree, key=lambda pair: pair[0])]).int().unsqueeze(-1)
    return in_deg, out_deg


def nx_compute_shortest_path(G: Graph,
                             min_length: int,
                             max_length: int,
                             num_edges: int,
                             keep_gene_list: Optional[list] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    r"""Compute all pair shortest path in the graph.
    Args:
        G (nx.Graph): Networkx graph.
        min_length (int): Minimum length of the path.
        max_length (int): Maximum length considered when computing the shortest path.
        keep_gene_list (Optional, list): If specified, only keep path that start from a gene in the list.
    """
    shortest_path_pair = nx.all_pairs_shortest_path(G, max_length)
    all_path_list = []
    path_index = []
    path_edge_type = []
    path_position = []
    path_count = 0
    for shortest_path in shortest_path_pair:
        index, paths = shortest_path
        if keep_gene_list is not None:
            if index not in keep_gene_list:
                continue

        for end_node, path in paths.items():
            if end_node == index:
                continue
            elif len(path) < min_length and G.get_edge_data(path[-2], path[-1])["edge_type"] != 5:
                continue
            else:
                path_edges = []
                for i in range(len(path) - 1):
                    path_edges.append(G.get_edge_data(path[i], path[i + 1])["edge_type"])
                # add padding edge type
                path_edges.append(num_edges)

                path_edge_type.extend(path_edges)
                path_position.extend([i for i in range(len(path))])
                all_path_list.extend(path)
                path_index.extend([path_count for _ in range(len(path))])
                path_count += 1

    return torch.tensor(all_path_list).long(), \
           torch.tensor(path_index).long(), \
           torch.tensor(path_edge_type).long(), \
           torch.tensor(path_position).long(),\
           path_count


def nx_compute_all_simple_paths(G: Graph,
                                source_node_list: list,
                                target_node_list: list,
                                min_length: int,
                                max_length: int,
                                num_edges: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    r"""Compute all possible paths between the source node and target node in the list. The path must follow the rule
        receptor -> target / receptor -> tf -> target / receptor -> sig -> tf -> target.
    Args:
        G (nx.Graph): Networkx graph.
        source_node_list (list): List of the source node.
        target_node_list (list): List of the target node.
        min_length (int): Minimum length of the path.
        max_length (int): Maximum length of the path.
        num_edges (int): Number of edge types in the graph.
    """
    all_path_list = []
    path_index = []
    path_edge_type = []
    path_position = []
    path_count = 0
    count = 0
    for source in source_node_list:
        for target in target_node_list:
            path_list = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
            for path in path_list:
                if len(path) >= min_length:
                    if G.get_edge_data(path[-2], path[-1])["edge_type"] in [4, 5]:
                        path_edges = []
                        for i in range(len(path) - 1):
                            path_edges.append(G.get_edge_data(path[i], path[i + 1])["edge_type"])
                        # add padding edge type
                        path_edges.append(num_edges)
                        all_path_list.extend(path)
                        path_index.extend([path_count for _ in range(len(path))])
                        path_edge_type.extend(path_edges)
                        path_position.extend([i for i in range(len(path))])
                        path_count += 1
            count += 1

    return torch.tensor(all_path_list).long(), \
           torch.tensor(path_index).long(), \
           torch.tensor(path_edge_type).long(), \
           torch.tensor(path_position).long(),\
           path_count


def nx_combine_shortest_path_and_simple_path(G: Graph,
                                             source_node_list: list,
                                             target_node_list: list,
                                             min_length: int,
                                             max_length: int,
                                             num_edges: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    r"""Compute and combin both the shortest path and
        all possible simple paths between the source node and target node in the list.
    Args:
        G (nx.Graph): Networkx graph.
        source_node_list (list): List of the source node.
        target_node_list (list): List of the target node.
        min_length (int): Minimum length of the path.
        max_length (int): Maximum length of the path.
    """
    shortest_path_list, shortest_path_index, shortest_path_type, shortest_path_positions, shortest_path_count\
        = nx_compute_shortest_path(G, min_length, max_length, num_edges, source_node_list)
    simple_path_list, simple_path_index, simple_path_type, simple_path_positions, simple_path_count = \
        nx_compute_all_simple_paths(G, source_node_list, target_node_list, min_length, max_length, num_edges)

    total_path_list = torch.cat([shortest_path_list, simple_path_list], dim=0)
    total_path_index = torch.cat([shortest_path_index, simple_path_index + shortest_path_count], dim=0)
    total_path_type = torch.cat([shortest_path_type, simple_path_type], dim=0)
    total_path_positions = torch.cat([shortest_path_positions, simple_path_positions], dim=0)
    total_path_count = simple_path_count + shortest_path_count
    # total_path_list = []
    # total_path_index = []
    # total_path_type = []
    # total_path_positions = []
    # total_path_count = shortest_path_count
    # for i in range(shortest_path_count):
    #     path = shortest_path_list[shortest_path_index == i].numpy().tolist()
    #     total_path_list.append(path)
    #     total_path_index.append([i for _ in range(len(path))])
    #
    # for i in range(simple_path_count):
    #     path = simple_path_list[simple_path_index == i].numpy().tolist()
    #     if path not in total_path_list:
    #         total_path_list.append(path)
    #         total_path_index.append([total_path_count for _ in range(len(path))])
    #         total_path_count += 1

    print(total_path_count)
    return total_path_list, \
           total_path_index, \
           total_path_type, \
           total_path_positions,\
           total_path_count




def nx_compute_all_random_path(G: Graph,
                               min_length: int,
                               max_length: int) -> Tensor:
    r"""Random select one path for each pair of the node if there exist.
    Args:
        G (nx.Graph): Networkx graph.
        min_length (int): Minimum length when computing path.
        max_length (int): Maximum length when computing path.
    """
    all_path_list = []
    path_index = []
    path_count = 0
    graph_nodes = list(G.nodes)
    for source in graph_nodes:
        for target in graph_nodes:
            path_list = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
            if len(path_list) > 0:
                pair_path_index = [i for i in range(len(path_list))]
                while len(pair_path_index) > 0:
                    random.shuffle(pair_path_index)
                    index = pair_path_index.pop()
                    path = path_list[index]
                    if len(path) < min_length:
                        continue
                    else:
                        all_path_list.extend(path)
                        path_index.extend([path_count for _ in range(len(path))])
                        path_count += 1
                        break
    return torch.tensor(all_path_list, dtype=torch.long), torch.tensor(path_index, dtype=torch.long), path_count


def get_path_prior_weight(fold_change: Tensor,
                          path_list: LongTensor,
                          path_index: LongTensor,
                          mode: str = "up") -> Tensor:
    r"""Compute the prior weight for each path based on the fold-change value of all genes in the path.

    Args:
        fold_change (Tensor): Fold-change value for each gene.
        path_list (LongTensor): Path gene index.
        path_index (LongTensor): Path index.
        mode (str, optional): prior weight mode, choose from (up, down, deg).


    """
    if mode == "down":
        fold_change = -fold_change
    elif mode == "deg":
        fold_change = torch.abs(fold_change)

    weight = fold_change[path_list].view(-1, 1)
    total_weight = scatter(weight, path_index, dim=-2, reduce="mean")
    normalized_weight = (total_weight - total_weight.min()) / (total_weight.max() - total_weight.min())
    return normalized_weight


def nx_compute_shortest_path_length(G: Graph,
                                    max_length: int) -> Tensor:
    r"""Compute all pair the shortest path length in the graph.
    Args:
        G (nx.Graph): Networkx graph.
        max_length (int): Maximum length when computing the shortest path.
    """
    num_node = G.number_of_nodes()
    shortest_path_length_matrix = torch.zeros([num_node, num_node]).int()
    all_shortest_path_lengths = nx.all_pairs_shortest_path_length(G, max_length)
    for shortest_path_lengths in all_shortest_path_lengths:
        index, path_lengths = shortest_path_lengths
        for end_node, path_length in path_lengths.items():
            if end_node == index:
                continue
            else:
                shortest_path_length_matrix[index, end_node] = path_length
    return shortest_path_length_matrix


def nx_return_edge_feature_index(G: Graph) -> Tensor:
    r"""Return edge type for each edge in the graph.
    Args:
        G (nx.Graph): Networkx graph.
    """
    return torch.from_numpy(nx.adjacency_matrix(G, weight="edge_type").toarray()).long()


class LoadClassifierDataset(data.Dataset):
    r"""construct classification dataset with scRNA-seq expression data and pre-defined signaling network
    Args:
        expression (np.array): scRNA-seq expression data.
        gs_path (str): Path of hallmark gene set database.
        gene_list (list): Gene symbol list of input dataset.
        in_degree (Tensor): In degree for each genes.
        out_degree (Tensor): Out degree fpr each genes.
        shortest_path_length (LongTensor): The length of the shortest path between each gene pair.
        edge_types (LongTensor): Edge type for each edge in the graph.
        label (np.array): Label of each sample.
    """

    def __init__(self,
                 expression: np.array,
                 gs_path: str,
                 gene_list: list,
                 in_degree: Tensor,
                 out_degree: Tensor,
                 shortest_path_length: LongTensor,
                 edge_types: LongTensor,
                 label: np.array):
        super(LoadClassifierDataset, self).__init__()

        self.gene_list = gene_list
        self.expression = torch.from_numpy(expression).float()
        self.label = torch.from_numpy(label).long()
        self.num_nodes = len(gene_list)
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.shortest_path_length = shortest_path_length
        self.edge_types = edge_types
        self.node_index = torch.tensor([i for i in range(len(self.gene_list))])

        # hallmark gene set feature
        with open(gs_path) as f:
            gene_feature_dict = json.load(f)
        gene_feature = np.zeros([len(gene_list), 50], dtype=np.int32)
        for i in range(len(gene_list)):
            gene = gene_list[i]
            gs_list = gene_feature_dict.get(gene.split("_")[0])
            if gs_list is not None and len(gs_list) > 0:
                for j in gs_list:
                    gene_feature[i, j] = 1
        self.gene_feature = torch.from_numpy(gene_feature).float()

    def __len__(self):
        return self.expression.size(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor, LongTensor]:
        cell_expression = self.expression[idx]
        x = cell_expression.unsqueeze(-1)
        y = self.label[idx]
        return (x, y, self.in_degree, self.out_degree, self.node_index, self.edge_types)


def classifier_collate_fn(examples: list) -> Tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor, LongTensor, Tensor]:
    r"""Create batch tensors from a list of individual examples returned Merge examples of different length by padding
    all examples to the maximum length in the batch.
    Args:
        examples (list): List of tuples

    """

    def merge_features(tensor_list):
        tensor_list = [tensor.unsqueeze(0) for tensor in tensor_list]
        return torch.cat(tensor_list, dim=0)

    # Group by tensor type
    x_list, y_list, in_deg_list, out_deg_list, node_index_list, edge_type_list = zip(*examples)
    batch_x = merge_features(x_list)
    batch_y = torch.tensor(y_list)
    batch_in_deg = merge_features(in_deg_list)
    batch_out_deg = merge_features(out_deg_list)
    edge_types = merge_features(edge_type_list)
    batch_node_index = merge_features(node_index_list)
    batch_mask = torch.ones([len(x_list), x_list[0].size(0)])
    return (batch_x, batch_y, batch_in_deg, batch_out_deg, edge_types, batch_node_index, batch_mask)


class AverageMeter:
    r"""Keep track of average values over time.
    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        r"""Reset meter."""
        self.__init__()

    def update(self, val: float, num_samples: Optional[int] = 1):
        r"""Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class EMA:
    r"""Exponential moving average of model parameters.
    Args:
        model (nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model: nn.Module, num_updates: float):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model: nn.Module):
        r"""Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model: nn.Module):
        r"""Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class CheckpointSaver:
    r"""Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self,
                 save_dir: str,
                 max_checkpoints: int,
                 metric_name: str,
                 maximize_metric: bool = False,
                 log: logging.Logger = None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val: float) -> bool:
        r"""Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message: str):
        r"""Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self,
             step: int,
             model_dict: nn.Module,
             metric_val: float,
             device: str):
        r"""Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (nn.Module): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (str): Device where model resides.
        """

        checkpoint_path = os.path.join(self.save_dir, f'step_{step}')
        for name, model in model_dict.items():
            ckpt_dict = {
                'model_name': model.__class__.__name__,
                'model_state': model.cpu().state_dict(),
                'step': step
            }

            model.to(device)
            torch.save(ckpt_dict, f"{checkpoint_path}{name}.pth.tar")
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best')
            for name in model_dict.keys():
                shutil.copy(f"{checkpoint_path}{name}.pth.tar", f"{best_path}{name}.pth.tar")

            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                for name in model_dict.keys():
                    os.remove(f"{worst_ckpt}{name}.pth.tar")
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model: nn.Module,
               checkpoint_path: str,
               gpu_ids: list,
               return_step: bool = True) -> nn.Module:
    r"""Load model parameters from disk.

    Args:
        model (nn.Module): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (nn.Module): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_available_devices() -> Tuple[str, list]:
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def get_save_dir(base_dir: str, name: str, type: str, id_max: int = 100) -> str:
    r"""Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        type (str): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = type
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir: str, name: str) -> logging.Logger:
    r"""Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def torch_from_json(path: str, dtype: Optional[torch.dtype]=torch.float32) -> Tensor:
    r"""Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


class MetricsMeter:
    r"""Keep track of model performance.
    """

    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.threshold = 0.5
        self.prediction = np.array([1])
        self.label = np.array([1])

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, input: Tensor, target: Tensor):
        """Update meter with new result

        Args:
            input (torch.tensor, Batch_size*1): predicted probability tensor.
            target (torch.tensor, Batch_size*1): ground true, 1 represent positive

        """
        predict = (input > self.threshold).int()
        self.TP += (target[torch.where(predict == 1)] == 1).sum().item()
        self.FP += (target[torch.where(predict == 1)] == 0).sum().item()
        self.TN += (target[torch.where(predict == 0)] == 0).sum().item()
        self.FN += (target[torch.where(predict == 0)] == 1).sum().item()
        input = input.view(-1).numpy()
        target = target.view(-1).numpy()
        self.prediction = np.concatenate([self.prediction, input], axis=-1)
        self.label = np.concatenate([self.label, target], axis=-1)

    def return_metrics(self) -> dict:
        recall = self.TP / (self.TP + self.FN + 1e-30)
        precision = self.TP / (self.TP + self.FP + 1e-30)
        specificity = self.TN / (self.TN + self.FP + 1e-30)
        accuracy = (self.TP + self.TN) / (self.TP + self.FP + self.TN + self.FN + 1e-30)
        F1 = self.TP / (self.TP + 0.5 * (self.FP + self.FN) + 1e-30)
        fpr, tpr, thresholds = roc_curve(self.label[1:], self.prediction[1:])
        AUC = auc(fpr, tpr)
        metrics_result = {'Accuracy': accuracy,
                          "Recall": recall,
                          "Precision": precision,
                          "Specificity": specificity,
                          "F1": F1,
                          "AUC": AUC,
                          "fpr": fpr,
                          "tpr": tpr,
                          "thresholds": thresholds
                          }
        return metrics_result

def get_seed(seed=234) -> int:
    r"""Return random seed based on current time.
    Args:
        seed (int): base seed.
    """
    t = int(time.time() * 1000.0)
    seed = seed + ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24)
    return seed