import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import List, Optional
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from typing import Union

class GCNEncoder(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 drop_prob: float = 0.1, 
                 JK: str = "last"):

        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        self.JK = JK

        self.gcn_layers = nn.ModuleList([
            GCNConv(input_size if i == 0 else hidden_size, hidden_size) 
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_list = []
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            h_list.append(x)
            
        if self.JK == "concat":
            node_emb = torch.cat(h_list, dim=-1)
        elif self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "max":
            node_emb = torch.stack(h_list, dim=-1).max(dim=-1)[0]
        elif self.JK == "sum":
            node_emb = torch.stack(h_list, dim=-1).sum(dim=-1)
        else:
            raise ValueError("Invalid JK method: choose from 'concat', 'last', 'max', 'sum'.")
            
        return node_emb


class SimpleDataset(torch.utils.data.Dataset):
    """Simplified dataset class for gene expression data."""
    def __init__(self, expression, labels):
        self.expression = expression
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.expression[idx], self.labels[idx]

def simple_collate_fn(batch):
    """Simplified collate function for DataLoader."""
    x = torch.stack([item[0] for item in batch])
    y = torch.tensor([item[1] for item in batch])
    return x, y


class SimpleClassifier(nn.Module):
    def __init__(self,
                 hidden_sizes: List[int],
                 output_size: int,
                 drop_prob: Optional[float] = 0.1):

        super(SimpleClassifier, self).__init__()
        layers = []
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_prob))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GCNPredictor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 pool_ratio: float,
                 pred_mlp_hidden: List[int],
                 pred_output_size: int,
                 gene_mlp_hidden: List[int],
                 drop_prob: float = 0.1):

        super(GCNPredictor, self).__init__()
        self.topk_pool = TopKPooling(in_channels, ratio=pool_ratio)
        self.prediction_head = SimpleClassifier([in_channels] + pred_mlp_hidden, pred_output_size, drop_prob)
        self.gene_selection_head = SimpleClassifier([in_channels] + gene_mlp_hidden, 1, drop_prob)
        
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch: torch.Tensor):
        x_pooled, edge_index, _, batch, perm, _ = self.topk_pool(x, edge_index, None, batch)
        graph_repr = global_mean_pool(x_pooled, batch)
        pred = self.prediction_head(graph_repr)
        gene_scores = self.gene_selection_head(x_pooled)
        
        return pred, gene_scores, perm


class DataExpander(nn.Module):
    def __init__(self, input_size: int, 
                 output_size: int,
                 hidden_size: int,
                 embed_size: Union[int, List[int]],
                 drop_prob: Optional[float] = 0.1):
        super(DataExpander, self).__init__()

        # Process gene expression
        self.gene_expander = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # Handle both single integer and list of integers for embed_size
        if isinstance(embed_size, int):
            # If embed_size is an integer, create a simple projection
            self.embed_mlp = nn.Sequential(
                nn.Linear(embed_size, hidden_size),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(drop_prob)
            )
        else:
            # If embed_size is a list, create multiple layers
            layers = []
            for i in range(1, len(embed_size)):
                layers.append(nn.Linear(embed_size[i-1], embed_size[i]))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                layers.append(nn.Dropout(drop_prob))
            self.embed_mlp = nn.Sequential(*layers)
        
        # Final layer to combine features
        self.final_layer = nn.Linear(hidden_size * 2, output_size)
        self.final_activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, embedding_x):
        # Process gene expression data
        # x shape: [batch_size, num_genes]
        gene_features = self.gene_expander(x)  # [batch_size, hidden_size]
        
        # Process LLM embeddings
        # embedding_x shape: [num_genes, embed_dim]
        embed_features = self.embed_mlp(embedding_x)  # [num_genes, hidden_size]
        
        # For each sample, get the average embedding
        embed_features = embed_features.mean(0, keepdim=True)  # [1, hidden_size]
        embed_features = embed_features.expand(gene_features.size(0), -1)  # [batch_size, hidden_size]
        
        # Combine gene expression features with LLM embedding features
        combined = torch.cat([gene_features, embed_features], dim=1)  # [batch_size, hidden_size*2]
        
        # Final processing
        output = self.final_layer(combined)
        output = self.final_activation(output)
        
        return output