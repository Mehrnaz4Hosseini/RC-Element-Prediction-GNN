"""
This module provides utility functions for preprocessing heterogeneous graph data,
including feature normalization and global Laplacian positional encoding.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def normalize_features(features, scaler=None):
    """
    Normalizes input features. If a scaler is provided, it fits/transforms.
    """
    print(f"[NORMALIZE DEBUG] Input features shape: {features.shape}")

    if scaler is None:
        scaler = StandardScaler()
        print(f"[NORMALIZE DEBUG] Creating new scaler")
    else:
        print(f"[NORMALIZE DEBUG] Using existing scaler")

    # Handle cases with empty data
    if features.shape[0] == 0:
        print(f"[NORMALIZE DEBUG] Empty features, returning as-is")
        return features, scaler

    normalized = scaler.fit_transform(features)
    print(f"[NORMALIZE DEBUG] Normalized shape: {normalized.shape}")

    return normalized, scaler


def compute_laplacian_pe(edge_index, num_nodes, pe_dim):
    """
    Computes Laplacian positional encoding with robust error handling.
    """
    # Check if we have any edges
    if edge_index.numel() == 0 or num_nodes == 0:
        print(f"[WARNING] Empty graph for Laplacian PE, returning zeros")
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    # 1. Convert to scipy sparse adjacency matrix
    try:
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    except Exception as e:
        print(f"[WARNING] Failed to create adjacency matrix: {e}. Returning zeros.")
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    # 2. Compute the normalized Laplacian
    try:
        lap = csgraph.laplacian(adj, normed=True)
    except Exception as e:
        print(f"[WARNING] Failed to compute Laplacian: {e}. Returning zeros.")
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    # 3. Compute eigenvectors
    # Ensure k is less than num_nodes
    k = min(pe_dim + 1, num_nodes - 1)
    if k <= 0:
        print(
            f"[WARNING] Not enough nodes ({num_nodes}) for PE dimension {pe_dim}. Returning zeros."
        )
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    try:
        eigvals, eigvecs = eigsh(
            lap,
            k=k,
            which="SM",
            ncv=max(20, k * 2),  # Increase ncv for better convergence
            tol=1e-4,  # Slightly higher tolerance
            maxiter=1000,  # Limit iterations
        )
    except Exception as e:
        print(f"[WARNING] Eigenvalue decomposition failed: {e}. Returning zeros.")
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    # 4. Exclude the first eigenvector and return
    if eigvecs.shape[1] > 1:
        # Sort by eigenvalues
        idx = eigvals.argsort()
        eigvecs = eigvecs[:, idx]

        # Extract eigenvectors (skip first trivial one)
        actual_pe_dim = min(pe_dim, eigvecs.shape[1] - 1)
        if actual_pe_dim > 0:
            pe = torch.tensor(eigvecs[:, 1 : actual_pe_dim + 1], dtype=torch.float)
            return pe
        else:
            print(
                f"[WARNING] Could not extract non-trivial eigenvectors. Returning zeros."
            )
            return torch.zeros((num_nodes, pe_dim), dtype=torch.float)
    else:
        print(
            f"[WARNING] Only found {eigvecs.shape[1]} eigenvector(s). Returning zeros."
        )
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)
