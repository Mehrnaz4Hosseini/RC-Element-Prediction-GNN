"""
Utility functions for preprocessing heterogeneous graph data.
Includes feature normalization and robust positional encoding.
"""

import numpy as np
import torch
import scipy as sp
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
import traceback

from src.utils.logger import get_system_logger

logger = get_system_logger()

# ==================================================
# HETERODATA -> HOMOGENEOUS (PyG 2.7 compatible)
# ==================================================


def hetero_to_homo(data):
    """
    Converts a HeteroData graph into:
      - homo_edge_index (global)
      - num_nodes
      - typewise node starting offsets (for mapping PE back)
    """
    node_offsets = {}
    current = 0

    # Compute global offsets for each node type
    for node_type in data.node_types:
        num = data[node_type].num_nodes
        node_offsets[node_type] = current
        current += num

    homo_edges = []

    for (src_type, rel, dst_type), edge_index in data.edge_index_dict.items():
        src_offset = node_offsets[src_type]
        dst_offset = node_offsets[dst_type]

        # Shift node indices into global space
        e = edge_index.clone()
        e[0] += src_offset
        e[1] += dst_offset
        homo_edges.append(e)

    if len(homo_edges) == 0:
        homo_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        homo_edge_index = torch.cat(homo_edges, dim=1)

    num_nodes = current
    return homo_edge_index, num_nodes, node_offsets


# ==================================================
# FEATURE NORMALIZATION
# ==================================================


def normalize_features(features: np.ndarray, scaler: Optional[StandardScaler] = None):
    if features.shape[0] == 0:
        logger.debug("Empty feature array provided for normalization")
        return features, scaler or StandardScaler()

    try:
        if scaler is None:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(features)
        else:
            normalized = scaler.transform(features)
        return normalized, scaler

    except Exception as e:
        logger.error(f"Feature normalization failed: {e}")
        raise


# ==================================================
# RANDOM WALK POSITIONAL ENCODING
# ==================================================


def compute_random_walk_pe(edge_index, num_nodes, pe_dim):
    logger.debug(f"Computing Random Walk PE (nodes={num_nodes}, dim={pe_dim})")

    if num_nodes == 0:
        return torch.zeros((0, pe_dim))

    row, col = edge_index
    adj = torch.zeros((num_nodes, num_nodes))
    adj[row, col] = 1

    deg = adj.sum(dim=1)
    deg[deg == 0] = 1
    P = adj / deg.unsqueeze(1)

    pe_list = []
    Pk = P.clone()

    for _ in range(pe_dim):
        diag = torch.diag(Pk)
        pe_list.append(diag)
        Pk = torch.matmul(Pk, P)

    return torch.stack(pe_list, dim=1)


# ==================================================
# LAPLACIAN POSITIONAL ENCODING
# ==================================================


def compute_laplacian_pe(edge_index, num_nodes, pe_dim):
    logger.debug(f"Computing Laplacian PE (nodes={num_nodes}, dim={pe_dim})")

    if num_nodes <= 1:
        return torch.zeros((num_nodes, pe_dim))

    row, col = edge_index.cpu().numpy()
    adj = sp.sparse.csr_matrix(
        (np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes)
    )
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)

    degree = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1))
    D_inv_sqrt = sp.sparse.diags(deg_inv_sqrt)

    L = sp.sparse.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
    k = min(pe_dim + 1, num_nodes - 1)

    eigvals, eigvecs = sp.sparse.linalg.eigsh(L, k=k, which="SM", maxiter=20000)

    idx = eigvals.argsort()
    eigvecs = eigvecs[:, idx]

    pe = torch.tensor(eigvecs[:, 1 : pe_dim + 1], dtype=torch.float)

    if pe.shape[1] < pe_dim:
        padding = torch.zeros((num_nodes, pe_dim - pe.shape[1]))
        pe = torch.cat([pe, padding], dim=1)

    return pe


# ==================================================
# DEGREE FALLBACK PE
# ==================================================


def compute_degree_pe(edge_index, num_nodes, pe_dim):
    logger.warning("Using Degree Positional Encoding fallback")
    deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    return deg.unsqueeze(1).repeat(1, pe_dim)


# ==================================================
# ROBUST PE WRAPPER
# ==================================================


def compute_positional_encoding(data, pe_dim):
    logger.info(f"Computing positional encoding (dim={pe_dim})")

    try:
        edge_index, num_nodes, _ = hetero_to_homo(data)

        # 1. Try random walk PE
        try:
            return compute_random_walk_pe(edge_index, num_nodes, pe_dim)
        except:
            pass

        # 2. Try Laplacian PE
        try:
            return compute_laplacian_pe(edge_index, num_nodes, pe_dim)
        except:
            pass

        # 3. Fallback
        return compute_degree_pe(edge_index, num_nodes, pe_dim)

    except Exception as e:
        logger.error(f"PE failed: {e}")
        return torch.zeros((data.num_nodes, pe_dim))


# ==================================================
# ADD PE TO HETERO GRAPH
# ==================================================


def add_positional_encoding(data, pe_dim=8, verbose=False):
    if pe_dim == 0:
        return data

    try:
        pe = compute_positional_encoding(data, pe_dim)

        # rebuild offsets to re‑split PE
        _, _, offsets = hetero_to_homo(data)

        for node_type, start in offsets.items():
            end = start + data[node_type].num_nodes
            node_pe = pe[start:end]
            data[node_type].x = torch.cat([data[node_type].x, node_pe], dim=1)

        return data

    except Exception as e:
        logger.error(f"Failed to add PE: {e}")
        return data
