"""Numpy forward pass of the trained Graph Attention Network (inference only).

Re-implements exactly what ml/graph/gat.py computes in eval mode (dropout
off), using weights exported by gat.export_weights() into graph_model.npz.
train_graph.py checks the two agree to 1e-4 before it writes any artifact.

Why numpy and not torch at inference: the backend runs on a 512 MB instance
already carrying Whisper; torch + torch_geometric are several hundred MB and
would not fit, and the backend tests must run without them. A GAT forward
pass on one ~50-node graph is a few small matrix products.

GATConv (torch_geometric 2.x, add_self_loops=False, concat heads), per layer:
    h        = x W^T                          -> (N, H, C)
    a_ij     = LeakyReLU_0.2(att_src.h_j + att_dst.h_i + att_edge.(W_e e_ij))
    alpha_ij = softmax over the incoming edges j -> i, per head
    out_i    = sum_j alpha_ij h_j  (heads concatenated) + bias
followed by ELU. Readout = [mean of node embeddings, attention pooling
(softmax over a learned per-node gate)], then Linear-ReLU-Linear -> softmax.
"""

from __future__ import annotations

import numpy as np


def _leaky_relu(z: np.ndarray, slope: float = 0.2) -> np.ndarray:
    return np.where(z > 0, z, slope * z)


def _elu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, np.expm1(np.minimum(z, 0)))


def _segment_softmax(scores: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    """Softmax of scores (E, H) grouped by index (E,) into n groups."""
    peak = np.full((n, scores.shape[1]), -np.inf)
    np.maximum.at(peak, index, scores)
    e = np.exp(scores - peak[index])
    total = np.zeros((n, scores.shape[1]))
    np.add.at(total, index, e)
    return e / total[index]


def gat_layer(x, edge_index, edge_attr, p: dict, heads: int):
    """One GATConv. Returns (out (N, H*C), attention (E, H))."""
    n = x.shape[0]
    w = p["lin.weight"]                                   # (H*C, in)
    c = w.shape[0] // heads
    h = (x @ w.T).reshape(n, heads, c)
    src, dst = edge_index
    a_src = (h * p["att_src"].reshape(1, heads, c)).sum(-1)
    a_dst = (h * p["att_dst"].reshape(1, heads, c)).sum(-1)
    e = (edge_attr @ p["lin_edge.weight"].T).reshape(-1, heads, c)
    a_edge = (e * p["att_edge"].reshape(1, heads, c)).sum(-1)
    alpha = _segment_softmax(_leaky_relu(a_src[src] + a_dst[dst] + a_edge), dst, n)
    out = np.zeros((n, heads, c))
    np.add.at(out, dst, alpha[:, :, None] * h[src])
    return out.reshape(n, heads * c) + p["bias"], alpha


def forward(graph: dict, weights: dict, config: dict) -> dict:
    """Class probabilities + per-layer attention for one graph."""
    x = graph["x"].astype(np.float64)
    ei, ea = graph["edge_index"], graph["edge_attr"].astype(np.float64)
    heads = int(config["heads"])
    attentions = []
    for i in range(int(config["layers"])):
        p = {k.split(".", 2)[2]: v for k, v in weights.items() if k.startswith(f"convs.{i}.")}
        x, alpha = gat_layer(x, ei, ea, p, heads)
        x = _elu(x)
        attentions.append(alpha)

    gate = x @ weights["gate.weight"].T + weights["gate.bias"]      # (N, 1)
    g = np.exp(gate - gate.max())
    g /= g.sum()
    z = np.concatenate([x.mean(axis=0), (g * x).sum(axis=0)])
    hidden = np.maximum(z @ weights["head.0.weight"].T + weights["head.0.bias"], 0.0)
    logits = hidden @ weights["head.3.weight"].T + weights["head.3.bias"]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return {"probabilities": probs, "logits": logits, "attention": attentions}


def top_edges(graph: dict, attentions: list[np.ndarray], region_names: list[str], k: int = 10) -> list[dict]:
    """Region pairs the model attended to most.

    Attention is averaged over heads and layers, self-loops dropped, and the
    two directions of each undirected edge averaged. It says where the model
    looked, not which connection is abnormal.
    """
    src, dst = graph["edge_index"]
    att = np.mean([a.mean(axis=1) for a in attentions], axis=0)
    pairs: dict[tuple[int, int], list[float]] = {}
    for s, d, a in zip(src, dst, att):
        if s != d:
            pairs.setdefault((min(s, d), max(s, d)), []).append(float(a))
    ranked = sorted(((np.mean(v), key) for key, v in pairs.items()), reverse=True)[:k]
    fc_col = graph["edge_attr"][:, 0]
    fc_of = {(int(s), int(d)): float(f) for s, d, f in zip(src, dst, fc_col)}
    return [
        {"region_a": region_names[a], "region_b": region_names[b],
         "attention": round(float(score), 4), "importance": round(float(score), 4),
         "fc": round(fc_of.get((a, b), 0.0), 4)}
        for score, (a, b) in ranked
    ]


def load_weights(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].astype(np.float64) for k in z.files}
