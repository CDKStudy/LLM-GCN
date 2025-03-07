"""
Data utils files
"""


import random
from typing import Optional, Tuple
import networkx as nx
import numpy as np
import torch
import ujson as json
from networkx import Graph
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_scatter import scatter
import pandas as pd
from sklearn.preprocessing import scale
from constants import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
# from process_pathway_text import *




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
    # llm_weight =
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

def generate_edge_index(expression, k=10, method="knn"):
    """
    生成细胞网络的 edge_index。
    
    Args:
        expression (np.ndarray or torch.Tensor): scRNA 数据 (num_cells, num_genes)。
        k (int): 最近邻数目。
        method (str): 构建方法 ("knn" 或 "cosine_threshold")。
        
    Returns:
        edge_index (torch.Tensor): 边索引 (2, num_edges)。
    """
    # 将输入数据转换为 numpy 数组
    if isinstance(expression, torch.Tensor):
        expression = expression.numpy()
    
    num_cells = expression.shape[0]
    
    if method == "knn":
        # 使用 KNN 构建邻接矩阵
        adjacency_matrix = kneighbors_graph(expression, n_neighbors=k, mode='connectivity', include_self=False)
        edge_index = np.array(adjacency_matrix.nonzero())  # 转换为边索引 (2, num_edges)
    
    elif method == "cosine_threshold":
        # 使用余弦相似度构建邻接矩阵
        similarity_matrix = cosine_similarity(expression)
        threshold = 0.5  # 相似度阈值
        adjacency_matrix = (similarity_matrix >= threshold).astype(int)  # 转换为邻接矩阵
        np.fill_diagonal(adjacency_matrix, 0)  # 移除自环
        edge_index = np.array(adjacency_matrix.nonzero())  # 转换为边索引 (2, num_edges)
    
    else:
        raise ValueError("Unsupported method. Use 'knn' or 'cosine_threshold'.")
    
    # 转换为 PyTorch Tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def process_input_data(args):
    """
    Process input data focusing only on gene expression and descriptions.
    """
    # Load expression data
    control_X = pd.read_csv(args.control_file_path, header=0, sep=',').values.T
    test_X = pd.read_csv(args.test_file_path, header=0, sep=',').values.T
    input_gene_list = np.load(args.gene_symbol_file_path, allow_pickle=True)["gene_list"]
    
    # Load LLM embedding data
    llm_embedding_X = np.load(args.llm_text_embedding_file_path)
    
    # Only keep top genes in the data
    control_X = control_X[:, :args.top_gene]
    test_X = test_X[:, :args.top_gene]
    input_gene_list = input_gene_list[:args.top_gene]

    # Create dataset and labels
    expression = np.concatenate([control_X, test_X], axis=0)
    num_control = control_X.shape[0]
    num_test = test_X.shape[0]
    label = np.array([0 for _ in range(expression.shape[0])])
    label[num_control:] = 1

    # Remove duplicated genes from dataset
    unique, counts = np.unique(input_gene_list, return_counts=True)
    duplicated_gene_list = unique[np.where(counts > 1)[0]].tolist()
    unique_gene_index = np.array([i for i, gene in enumerate(input_gene_list) if gene not in duplicated_gene_list])
    keep_expression = np.zeros([expression.shape[0], len(duplicated_gene_list)])
    for i, gene in enumerate(duplicated_gene_list):
        index = np.where(input_gene_list == gene)[0]
        keep_index = index[np.argmax(np.var(expression[:, index], axis=0))]
        keep_expression[:, i] = expression[:, keep_index]
    expression = expression[:, unique_gene_index]
    expression = np.concatenate([expression, keep_expression], axis=-1)
    input_gene_list = input_gene_list[unique_gene_index]
    input_gene_list = np.concatenate([input_gene_list, duplicated_gene_list])
    
    # Split back to control and test
    control_X = expression[:num_control]
    test_X = expression[num_control:]
    
    # Filter genes with low expression
    control_expr_pre = np.sum(control_X > 0, axis=0) / control_X.shape[0]
    control_expr_mean = np.mean(control_X, axis=0)
    control_expr_filter = np.logical_and(control_expr_pre > 0.05, control_expr_mean > 0.05)

    test_expr_pre = np.sum(test_X > 0, axis=0) / test_X.shape[0]
    test_expr_mean = np.mean(test_X, axis=0)
    test_expr_filter = np.logical_and(test_expr_pre > 0.05, test_expr_mean > 0.05)
    
    # Keep genes that pass the filter
    expr_filter = np.logical_and(control_expr_filter, test_expr_filter)
    expression = expression[:, expr_filter]
    input_gene_list = input_gene_list[expr_filter]
    
    # Generate cell-cell graph structure for GCN
    edge_index = generate_edge_index(expression, k=10, method="knn")
    
    # Match gene descriptions to filtered genes
    with open('data/text/gene_text_description.json', 'r') as f:
        gene_data = json.load(f)
    gene_to_index = {gene.upper(): idx for idx, gene in enumerate(gene_data.keys())}
    
    # Find indices of genes in the description data
    valid_genes = [gene.upper() for gene in input_gene_list if gene.upper() in gene_to_index]
    indices = [gene_to_index[gene] for gene in valid_genes]
    
    # Get LLM embeddings for valid genes
    llm_embedding_X = llm_embedding_X[indices, :]
    llm_embedding_X = torch.from_numpy(llm_embedding_X).float()
    
    # Convert to tensors
    expression = torch.from_numpy(expression).float()
    label = torch.from_numpy(label).long()
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Set model dimensions in args
    args.input_size = expression.shape[1]  # Number of genes after filtering
    args.num_nodes = expression.shape[0]   # Number of cells
    args.embed_sizes = llm_embedding_X.shape[1]  # Size of LLM embeddings
    
    return args, expression, label, llm_embedding_X, edge_index, input_gene_list