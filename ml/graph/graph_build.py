"""Connectivity matrices -> one brain graph (numpy only).

Shared by training (ml/graph/train_graph.py wraps the output in a PyG Data
object) and the backend's numpy forward pass, so a graph is built exactly one
way.

    nodes      = atlas regions (N)
    edges      = strongest functional connections: per node the `top_k`
                 neighbours by |FC|, and/or every pair with |FC| > `threshold`;
                 made symmetric (an edge kept for either end is kept for both)
                 and stored in both directions, plus one self-loop per node
    edge_attr  = [FC_ij, SC_ij]                       (edge_dim = 2)
    x (node i) = [FC row i (Fisher z), SC row i, degree_i, strength_i]
                 (N + 2 features without DTI, 2N + 2 with)

SC (structural connectivity from DTI tractography) arrives as streamline
counts spanning orders of magnitude, so it is log1p-scaled and divided by its
maximum to [0, 1]. Without DTI, SC is all zeros and the model sees FC only;
graph_config.json records which was used in training.

Self-loops are added here (attr [1, SC_ii]) rather than inside GATConv, so the
PyG layer runs with add_self_loops=False and the numpy re-implementation does
not have to copy PyG's "mean of incoming edge attributes" fill rule.
"""

from __future__ import annotations

import io

import numpy as np

DEFAULT_TOP_K = 10
MAX_MATRIX_BYTES = 20 * 1024 * 1024
SYMMETRY_TOL = 1e-4


class MatrixError(ValueError):
    """An uploaded connectivity matrix is unusable (the client's fault)."""


def load_matrix_bytes(raw: bytes, suffix: str) -> np.ndarray:
    """Parse an uploaded .csv or .npy SC matrix into a float64 array."""
    if not raw:
        raise MatrixError("the connectivity matrix file is empty")
    if len(raw) > MAX_MATRIX_BYTES:
        raise MatrixError("the connectivity matrix file is too large")
    try:
        if suffix == ".npy":
            m = np.load(io.BytesIO(raw), allow_pickle=False)  # never unpickle uploads
        elif suffix == ".csv":
            text = raw.decode("utf-8-sig")
            m = np.loadtxt(io.StringIO(text), delimiter="," if "," in text else None, ndmin=2)
        else:
            raise MatrixError("structural connectivity must be a .csv or .npy file")
        return np.asarray(m, dtype=np.float64)
    except MatrixError:
        raise
    except Exception as exc:
        raise MatrixError(f"could not read the connectivity matrix: {exc}") from exc


def validate_matrix(m: np.ndarray, n: int, name: str = "structural connectivity") -> np.ndarray:
    """Square n x n, finite, symmetric (within tolerance). Returns it symmetrised."""
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2 or m.shape != (n, n):
        raise MatrixError(f"{name} matrix must be {n}x{n} (one row per atlas region), got {tuple(m.shape)}")
    if not np.all(np.isfinite(m)):
        raise MatrixError(f"{name} matrix contains NaN or infinite values")
    scale = max(1.0, float(np.abs(m).max()))
    if np.abs(m - m.T).max() > SYMMETRY_TOL * scale:
        raise MatrixError(f"{name} matrix must be symmetric (connections are undirected)")
    return (m + m.T) / 2


def normalise_sc(sc: np.ndarray) -> np.ndarray:
    sc = np.clip(sc, 0.0, None)
    sc = np.log1p(sc)
    np.fill_diagonal(sc, 0.0)
    peak = sc.max()
    return sc / peak if peak > 0 else sc


def select_edges(fc: np.ndarray, top_k: int | None = DEFAULT_TOP_K,
                 threshold: float | None = None) -> np.ndarray:
    """Boolean N x N symmetric adjacency (no diagonal).

    top_k and threshold combine with AND when both are given; with neither,
    every pair is connected.
    """
    n = fc.shape[0]
    strength = np.abs(fc).astype(np.float64)
    np.fill_diagonal(strength, -np.inf)
    keep = np.ones((n, n), dtype=bool)
    if threshold is not None:
        keep &= strength > threshold
    if top_k is not None:
        k = max(1, min(int(top_k), n - 1))
        idx = np.argsort(-strength, axis=1, kind="stable")[:, :k]
        topk = np.zeros((n, n), dtype=bool)
        np.put_along_axis(topk, idx, True, axis=1)
        keep &= topk | topk.T            # symmetric: kept if either end ranks it
    np.fill_diagonal(keep, False)
    return keep | keep.T


def build_graph(fc: np.ndarray, sc: np.ndarray | None = None, *,
                top_k: int | None = DEFAULT_TOP_K, threshold: float | None = None,
                fisher_z: bool = False) -> dict:
    """One subject's graph as numpy arrays (PyG layout: edge_index[0]=source)."""
    fc = validate_matrix(fc, fc.shape[0], "functional connectivity")
    n = fc.shape[0]
    fc = np.clip(fc, -1.0, 1.0)
    np.fill_diagonal(fc, 1.0)
    sc_n = np.zeros((n, n)) if sc is None else normalise_sc(validate_matrix(sc, n))

    adj = select_edges(fc, top_k, threshold)
    src, dst = np.nonzero(adj)                      # both directions already present
    loops = np.arange(n)
    edge_index = np.vstack([np.concatenate([src, loops]), np.concatenate([dst, loops])]).astype(np.int64)
    edge_attr = np.column_stack([
        np.concatenate([fc[src, dst], np.ones(n)]),
        np.concatenate([sc_n[src, dst], np.diag(sc_n)]),
    ]).astype(np.float32)

    denom = max(1, n - 1)
    degree = adj.sum(axis=1) / denom
    strength = (np.abs(fc) * adj).sum(axis=1) / denom
    rows = np.arctanh(np.clip(fc, -0.999, 0.999)) if fisher_z else fc
    # Without DTI the SC block would be N all-zero columns per node: noise for
    # a model trained on ~650 graphs, so it is left out entirely.
    blocks = [rows] + ([sc_n] if sc is not None else []) + [degree[:, None], strength[:, None]]
    x = np.hstack(blocks).astype(np.float32)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": n,
        "num_edges": int(adj.sum()),         # directed, self-loops excluded
        "has_sc": sc is not None,
    }


def node_feature_dim(n_regions: int, with_sc: bool = True) -> int:
    return (2 if with_sc else 1) * n_regions + 2
