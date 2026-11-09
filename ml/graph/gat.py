"""Graph Attention Network for brain connectivity graphs (training, PyTorch Geometric).

    GATConv x layers (multi-head, edge_dim=2 for [FC, SC], ELU)
    -> readout: [global mean pool, attention pool]
    -> Linear -> ReLU -> Dropout -> Linear -> softmax over 3 classes

The deployed forward pass is ml/graph/gat_numpy.py; export_weights() writes the
parameters it needs, under the same names.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

CLASSES = ["Normal (CDR 0)", "MCI (CDR 0.5)", "Dementia (CDR >= 1)"]


def to_data(graph: dict, label: int | None = None) -> Data:
    d = Data(
        x=torch.from_numpy(graph["x"]),
        edge_index=torch.from_numpy(graph["edge_index"]),
        edge_attr=torch.from_numpy(graph["edge_attr"]),
    )
    if label is not None:
        d.y = torch.tensor([label])
    return d


class GraphGAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, heads: int = 4, layers: int = 2,
                 dropout: float = 0.3, n_classes: int = 3) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            # Self-loops come from graph_build (attr [1, SC_ii]); see its docstring.
            self.convs.append(GATConv(d, hidden, heads=heads, edge_dim=2, concat=True,
                                      add_self_loops=False, dropout=dropout))
            d = hidden * heads
        self.gate = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

    def forward(self, x, edge_index, edge_attr, batch, return_attention: bool = False):
        attentions = []
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if return_attention:
                x, (_, alpha) = conv(x, edge_index, edge_attr, return_attention_weights=True)
                attentions.append(alpha)
            else:
                x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
        weights = pyg_softmax(self.gate(x), batch)
        att_pool = torch.zeros(int(batch.max()) + 1, x.shape[1], device=x.device).index_add_(0, batch, weights * x)
        logits = self.head(torch.cat([global_mean_pool(x, batch), att_pool], dim=1))
        return (logits, attentions) if return_attention else logits


def export_weights(model: GraphGAT) -> dict[str, np.ndarray]:
    """Parameters for gat_numpy.forward, keyed by their torch names."""
    out = {}
    for name, t in model.state_dict().items():
        key = name.replace(".lin_src.", ".lin.")  # older PyG names the shared projection lin_src
        out[key] = t.detach().cpu().numpy().astype(np.float32)
    return out
