"""
Utility functions for preprocessing heterogeneous graph data.
Includes feature normalization and positional encoding.
"""

import numpy as np
import torch
import scipy as sp
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
import traceback

# Import logger
try:
    from src.utils.logger import get_system_logger
except ImportError:
    import logging

    get_system_logger = lambda: logging.getLogger("SYSTEM")


logger = get_system_logger()


def normalize_features(
    features: np.ndarray, scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize input features using StandardScaler.

    Args:
        features: Input feature array
        scaler: Optional pre-fitted StandardScaler

    Returns:
        normalized_features, updated_scaler
    """
    if features.shape[0] == 0:
        logger.debug("Empty feature array provided for normalization")
        return features, scaler or StandardScaler()

    try:
        if scaler is None:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(features)
            logger.debug(f"Fitted new scaler on {features.shape[0]} samples")
        else:
            normalized = scaler.transform(features)
            logger.debug(
                f"Transformed {features.shape[0]} samples with existing scaler"
            )

        return normalized, scaler

    except Exception as e:
        logger.error(f"Feature normalization failed: {e}")
        raise


# ==================================================
# LAPLACIAN POSITIONAL ENCODING
# ==================================================


def _safe_eigsh_laplacian(
    L: sp.sparse.csr_matrix, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safely compute eigenvectors of Laplacian matrix with fallback strategies.

    Args:
        L: Normalized Laplacian matrix
        k: Number of eigenvectors to compute

    Returns:
        eigenvalues, eigenvectors
    """
    try:
        # Try with standard parameters first
        eigvals, eigvecs = sp.sparse.linalg.eigsh(
            L, k=k, which="SM", sigma=0, v0=np.ones(L.shape[0]), maxiter=10000, tol=1e-6
        )
        return eigvals, eigvecs

    except Exception as e:  # CATCH ALL EXCEPTIONS, not just ArpackNoConvergence
        logger.warning(f"Eigsh did not converge: {e}")

        # Try with increased maxiter
        try:
            eigvals, eigvecs = sp.sparse.linalg.eigsh(
                L,
                k=k,
                which="SM",
                sigma=0,
                v0=np.ones(L.shape[0]),
                maxiter=50000,
                tol=1e-5,
            )
            logger.info("Eigsh converged with increased maxiter")
            return eigvals, eigvecs
        except:
            logger.error("Eigsh failed even with increased maxiter")

            # Fallback: return zeros
            eigvals = np.zeros(k)
            eigvecs = np.zeros((L.shape[0], k))
            return eigvals, eigvecs


def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    pe_dim: int,
    normalize: bool = True,
    k_extra: int = 2,
) -> torch.Tensor:
    """
    Compute Laplacian positional encoding for a graph.

    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes in graph
        pe_dim: Dimension of positional encoding
        normalize: Whether to normalize eigenvectors
        k_extra: Extra eigenvectors to compute for stability

    Returns:
        Positional encoding tensor of shape [num_nodes, pe_dim]
    """
    # Handle edge cases
    if edge_index.numel() == 0:
        logger.warning(f"No edges provided for Laplacian PE, returning zeros")
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    if num_nodes == 0 or pe_dim == 0:
        logger.warning(
            f"Zero nodes or PE dimension: num_nodes={num_nodes}, pe_dim={pe_dim}"
        )
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    # Ensure k is valid
    k = min(pe_dim + k_extra, num_nodes - 1)
    if k <= 0:
        logger.warning(f"Invalid k value: {k} (pe_dim={pe_dim}, num_nodes={num_nodes})")
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    logger.debug(
        f"Computing Laplacian PE: {num_nodes} nodes, {edge_index.shape[1]} edges, k={k}"
    )

    try:
        # Create adjacency matrix
        row, col = edge_index.cpu().numpy()

        # Check for valid indices
        if row.max() >= num_nodes or col.max() >= num_nodes:
            logger.error(
                f"Edge indices out of bounds: max index={max(row.max(), col.max())}, num_nodes={num_nodes}"
            )
            return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

        adj = sp.sparse.csr_matrix(
            (np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes)
        )

        # Make symmetric for undirected graphs
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)  # Binary adjacency

        # Compute normalized Laplacian
        degree = np.array(adj.sum(axis=1)).flatten()

        # Check for isolated nodes (degree = 0)
        isolated_nodes = np.sum(degree == 0)
        if isolated_nodes > 0:
            logger.warning(
                f"Graph has {isolated_nodes} isolated nodes ({isolated_nodes/num_nodes*100:.1f}%)"
            )

        degree_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
        L = sp.sparse.eye(num_nodes) - adj.multiply(degree_inv_sqrt).multiply(
            degree_inv_sqrt[:, np.newaxis]
        )

        # Compute eigenvectors
        eigvals, eigvecs = _safe_eigsh_laplacian(L, k)

        # Sort by eigenvalues
        idx = eigvals.argsort()
        eigvecs = eigvecs[:, idx]

        # Skip first eigenvector (all ones) and extract pe_dim
        if eigvecs.shape[1] > 1:
            actual_dim = min(pe_dim, eigvecs.shape[1] - 1)

            if actual_dim < pe_dim:
                logger.warning(f"Could only compute {actual_dim}/{pe_dim} eigenvectors")

            pe = torch.tensor(eigvecs[:, 1 : actual_dim + 1], dtype=torch.float)

            # Normalize if requested
            if normalize and actual_dim > 0:
                pe = (pe - pe.mean(dim=0)) / (pe.std(dim=0) + 1e-8)

            # Pad if needed
            if actual_dim < pe_dim:
                padding = torch.zeros((num_nodes, pe_dim - actual_dim))
                pe = torch.cat([pe, padding], dim=1)

            logger.debug(f"Computed Laplacian PE: {pe.shape}")
            return pe

        else:
            logger.warning("Could not compute sufficient eigenvectors, returning zeros")
            return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    except Exception as e:
        logger.error(f"Laplacian computation failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)


def compute_hetero_laplacian_pe(
    data: "HeteroData",
    pe_dim: int,
    include_cross_type: bool = False,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute heterogeneous-aware Laplacian positional encoding.

    Args:
        data: Heterogeneous graph data
        pe_dim: Dimension of positional encoding
        include_cross_type: Whether to include cross-type edges in PE computation
        normalize: Whether to normalize eigenvectors

    Returns:
        Dictionary mapping node_type -> positional encoding tensor
    """
    pe_dict = {}
    node_types = data.node_types if hasattr(data, "node_types") else []

    logger.debug(
        f"Computing heterogeneous Laplacian PE for {len(node_types)} node types"
    )

    for node_type in node_types:
        if node_type not in data.x_dict:
            logger.warning(f"Node type {node_type} not found in data")
            continue

        num_nodes = data[node_type].x.size(0)
        logger.debug(f"Processing {node_type}: {num_nodes} nodes")

        if include_cross_type:
            # Collect all edges where this node type appears
            all_edges = []

            # Same-type edges
            edge_key = (node_type, "to", node_type)
            if edge_key in data.edge_index_dict:
                edges = data[edge_key].edge_index
                if edges.shape[1] > 0:
                    all_edges.append(edges)
                    logger.debug(f"  Added {edges.shape[1]} same-type edges")

            # Incoming edges from other types
            for edge_type in data.edge_types:
                src_type, _, dst_type = edge_type
                if dst_type == node_type and src_type != node_type:
                    edges = data[edge_type].edge_index
                    if edges.shape[1] > 0:
                        reversed_edges = torch.stack([edges[1], edges[0]])
                        all_edges.append(reversed_edges)
                        logger.debug(
                            f"  Added {edges.shape[1]} incoming edges from {src_type}"
                        )

            # Outgoing edges to other types
            for edge_type in data.edge_types:
                src_type, _, dst_type = edge_type
                if src_type == node_type and dst_type != node_type:
                    edges = data[edge_type].edge_index
                    if edges.shape[1] > 0:
                        all_edges.append(edges)
                        logger.debug(
                            f"  Added {edges.shape[1]} outgoing edges to {dst_type}"
                        )

            if all_edges:
                combined_edges = torch.cat(all_edges, dim=1)
                logger.debug(
                    f"  Total edges for {node_type}: {combined_edges.shape[1]}"
                )
                pe = compute_laplacian_pe(combined_edges, num_nodes, pe_dim, normalize)
            else:
                logger.warning(f"No edges found for node type {node_type}")
                pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float)

        else:
            # Only same-type edges
            edge_key = (node_type, "to", node_type)
            if edge_key in data.edge_index_dict:
                edges = data[edge_key].edge_index
                if edges.shape[1] > 0:
                    pe = compute_laplacian_pe(edges, num_nodes, pe_dim, normalize)
                else:
                    logger.warning(f"No same-type edges for {node_type}")
                    pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float)
            else:
                logger.warning(f"Edge type {edge_key} not found for {node_type}")
                pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float)

        pe_dict[node_type] = pe

    logger.info(f"Computed PE for {len(pe_dict)} node types")
    return pe_dict


def compute_random_walk_pe(
    data: "HeteroData", pe_dim: int, num_walks: int = 10, walk_length: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Compute Random Walk Positional Encoding for heterogeneous graphs.

    Args:
        data: Heterogeneous graph data
        pe_dim: Dimension of positional encoding
        num_walks: Number of random walks per node
        walk_length: Length of each random walk

    Returns:
        Dictionary mapping node_type -> positional encoding tensor
    """
    logger.info(
        f"Computing Random Walk PE (dim={pe_dim}, walks={num_walks}, length={walk_length})"
    )

    pe_dict = {}

    for node_type in data.node_types:
        if node_type not in data.x_dict:
            continue

        num_nodes = data[node_type].x.size(0)

        # TODO: Implement actual random walk PE
        # For now, using random initialization as placeholder
        rwpe = torch.randn((num_nodes, pe_dim)) * 0.1

        pe_dict[node_type] = rwpe
        logger.debug(f"  {node_type}: Random PE shape {rwpe.shape}")

    return pe_dict


def compute_sin_cos_pe(
    num_nodes: int, pe_dim: int, base: float = 10000.0
) -> torch.Tensor:
    """
    Compute sinusoidal positional encoding (like in Transformers).

    Args:
        num_nodes: Number of nodes
        pe_dim: Dimension of positional encoding
        base: Base for exponential calculation

    Returns:
        Positional encoding tensor of shape [num_nodes, pe_dim]
    """
    logger.debug(f"Computing sinusoidal PE for {num_nodes} nodes (dim={pe_dim})")

    if num_nodes == 0 or pe_dim == 0:
        return torch.zeros((num_nodes, pe_dim), dtype=torch.float)

    position = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, pe_dim, 2, dtype=torch.float) * -(np.log(base) / pe_dim)
    )

    pe = torch.zeros((num_nodes, pe_dim), dtype=torch.float)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def add_positional_encoding(
    data: "HeteroData",
    pe_type: str = "hetero_laplacian",
    pe_dim: int = 8,
    include_cross_type: bool = False,
    normalize: bool = True,
    verbose: bool = False,
) -> "HeteroData":
    """
    Add positional encoding to heterogeneous graph.

    Args:
        data: Heterogeneous graph data
        pe_type: Type of positional encoding
        pe_dim: Dimension of positional encoding
        include_cross_type: Whether to include cross-type edges in PE computation
        normalize: Whether to normalize eigenvectors
        verbose: Whether to print detailed information

    Returns:
        Updated heterogeneous graph data with positional encoding
    """
    if pe_type == "none" or pe_dim == 0:
        if verbose:
            logger.info("Positional encoding disabled")
        return data

    logger.info(f"Adding {pe_type} positional encoding (dim={pe_dim})")

    try:
        if pe_type == "hetero_laplacian":
            pe_dict = compute_hetero_laplacian_pe(
                data, pe_dim, include_cross_type, normalize
            )

        elif pe_type == "random_walk":
            pe_dict = compute_random_walk_pe(data, pe_dim)

        elif pe_type == "sin_cos":
            pe_dict = {}
            for node_type in data.node_types:
                if node_type in data.x_dict:
                    num_nodes = data[node_type].x.size(0)
                    pe_dict[node_type] = compute_sin_cos_pe(num_nodes, pe_dim)

        elif pe_type == "learnable":
            # Initialize learnable PE
            pe_dict = {}
            for node_type in data.node_types:
                if node_type in data.x_dict:
                    num_nodes = data[node_type].x.size(0)
                    pe_dict[node_type] = torch.zeros(
                        (num_nodes, pe_dim), dtype=torch.float
                    )

            # Store indices for learnable embedding lookup
            data.learnable_pe_indices = {}
            for node_type in data.node_types:
                if node_type in data.x_dict:
                    num_nodes = data[node_type].x.size(0)
                    data.learnable_pe_indices[node_type] = torch.arange(num_nodes)

            logger.info("Initialized learnable positional encoding")

        else:
            logger.error(f"Unknown PE type: {pe_type}")
            raise ValueError(f"Unknown PE type: {pe_type}")

        # Concatenate PE to node features
        nodes_with_pe = 0
        for node_type, pe in pe_dict.items():
            if node_type in data.x_dict:
                data.x_dict[node_type] = torch.cat([data.x_dict[node_type], pe], dim=1)
                nodes_with_pe += 1

                if verbose:
                    orig_features = data[node_type].x.shape[1] - pe_dim
                    logger.debug(
                        f"  {node_type}: {data[node_type].x.shape[1]} features "
                        f"({orig_features} orig + {pe_dim} PE)"
                    )

        logger.info(f"Added PE to {nodes_with_pe} node types")
        return data

    except Exception as e:
        logger.error(f"Failed to add positional encoding: {e}")
        if verbose:
            logger.debug(f"Traceback: {traceback.format_exc()}")
        # Return data unchanged if PE fails
        return data
