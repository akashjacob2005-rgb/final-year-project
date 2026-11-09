"""Graph learning pathway: brain connectivity graphs -> GAT -> Normal / MCI / Dementia.

Shared by training (ml/graph/train_graph.py) and the backend
(backend/app/ml/graph_model.py). Everything the backend imports from here is
numpy / nibabel / nilearn only — torch and torch_geometric are training-time
dependencies (see ml/README.md, "Graph pathway: why inference is numpy").
"""
