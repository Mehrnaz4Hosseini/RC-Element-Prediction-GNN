"""
preprocessing.py

This module provides utility functions for preprocessing graph data,
including feature normalization and Laplacian positional encoding.

Functions:
- normalize_features: Applies standard normalization to features.
- compute_laplacian_pe: Computes Laplacian positional encoding using eigenvectors.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

def normalize_features(features, scaler):
    """
    Normalizes the input features using StandardScaler.

    Args:
        features (np.ndarray): Raw input features (n_nodes x n_features).

    Returns:
        np.ndarray: Normalized features.
    """
    return scaler.fit_transform(features)

def to_tensor(edge_df, x_np, y_np):

    edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)
    x = torch.tensor(x_np, dtype=torch.float)
    y = torch.tensor(y_np, dtype=torch.float)
    return edge_index, x, y

def compute_laplacian_pe(edge_index, num_nodes, pe_dim):
    """
    Computes Laplacian positional encoding from the graph adjacency matrix.

    Args:
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        num_nodes (int): Number of nodes in the graph.
        pe_dim (int): Number of positional encoding dimensions.

    Returns:
        torch.Tensor: Positional encoding matrix of shape [num_nodes, pe_dim]
    """
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    lap = csgraph.laplacian(adj, normed=True)
    eigvals, eigvecs = eigsh(lap, k=pe_dim + 1, which='SM')
    return torch.tensor(eigvecs[:, 1:], dtype=torch.float)