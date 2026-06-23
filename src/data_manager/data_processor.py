"""
Utility functions for preprocessing heterogeneous graph data.

Includes:
  - Positional encoding (random-walk / Laplacian / degree fallback)
    that NEVER silently changes the feature width (Bug #2 fix).
  - FeatureNormalizer: a leakage-safe scaler that is fit on TRAINING
    graphs only and applied to all splits (Bug #3 fix).
"""

import numpy as np
import torch
import scipy as sp
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_system_logger

logger = get_system_logger()


# ==================================================
# HETERODATA -> HOMOGENEOUS (PyG 2.7 compatible)
# ==================================================
def hetero_to_homo(data):
    """
    Converts a HeteroData graph into:
      - homo_edge_index (global indices)
      - num_nodes
      - typewise node starting offsets (to map PE back per type)
    """
    node_offsets = {}
    current = 0

    for node_type in data.node_types:
        num = data[node_type].num_nodes
        node_offsets[node_type] = current
        current += num

    homo_edges = []
    for (src_type, rel, dst_type), edge_index in data.edge_index_dict.items():
        e = edge_index.clone()
        e[0] += node_offsets[src_type]
        e[1] += node_offsets[dst_type]
        homo_edges.append(e)

    if len(homo_edges) == 0:
        homo_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        homo_edge_index = torch.cat(homo_edges, dim=1)

    return homo_edge_index, current, node_offsets


# ==================================================
# (legacy) FEATURE NORMALIZATION HELPER
#   Kept for backwards compatibility; the builder no
#   longer calls this. Use FeatureNormalizer instead.
# ==================================================
def normalize_features(features: np.ndarray, scaler: Optional[StandardScaler] = None):
    if features.shape[0] == 0:
        return features, scaler or StandardScaler()
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(features), scaler
    return scaler.transform(features), scaler


# ==================================================
# POSITIONAL ENCODINGS
# ==================================================
def compute_random_walk_pe(edge_index, num_nodes, pe_dim):
    if num_nodes == 0:
        return torch.zeros((0, pe_dim))

    row, col = edge_index
    adj = torch.zeros((num_nodes, num_nodes))
    adj[row, col] = 1.0

    deg = adj.sum(dim=1)
    deg[deg == 0] = 1.0
    P = adj / deg.unsqueeze(1)

    pe_list, Pk = [], P.clone()
    for _ in range(pe_dim):
        pe_list.append(torch.diag(Pk))
        Pk = Pk @ P
    return torch.stack(pe_list, dim=1)


def compute_laplacian_pe(edge_index, num_nodes, pe_dim):
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
    eigvecs = eigvecs[:, eigvals.argsort()]

    pe = torch.tensor(eigvecs[:, 1 : pe_dim + 1], dtype=torch.float)
    if pe.shape[1] < pe_dim:  # pad if not enough eigenvectors
        pad = torch.zeros((num_nodes, pe_dim - pe.shape[1]))
        pe = torch.cat([pe, pad], dim=1)
    return pe


def compute_degree_pe(edge_index, num_nodes, pe_dim):
    deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    return deg.unsqueeze(1).repeat(1, pe_dim)


def compute_positional_encoding(data, pe_dim):
    """Try random-walk -> Laplacian -> degree. Always returns (num_nodes, pe_dim)."""
    edge_index, num_nodes, _ = hetero_to_homo(data)

    for fn in (compute_random_walk_pe, compute_laplacian_pe, compute_degree_pe):
        try:
            pe = fn(edge_index, num_nodes, pe_dim)
            if pe.shape == (num_nodes, pe_dim):
                return pe
            logger.warning(f"{fn.__name__} returned shape {tuple(pe.shape)}, trying next")
        except Exception as e:
            logger.warning(f"{fn.__name__} failed ({e}), trying next method")

    logger.error("All PE methods failed; using zeros")
    return torch.zeros((num_nodes, pe_dim))


# ==================================================
# ADD PE TO HETERO GRAPH  (Bug #2 fix)
# ==================================================
def add_positional_encoding(data, pe_dim=8, verbose=False):
    """
    Append exactly `pe_dim` PE columns to every node type.

    GUARANTEE: every node always receives exactly `pe_dim` extra columns
    (zeros if the math fails). This keeps feature width identical across
    all graphs, which the model relies on.

    Also records `data[nt].num_raw_features` = feature count BEFORE PE,
    so the normalizer knows which columns are real features vs PE.
    """
    if pe_dim == 0:
        for nt in data.node_types:
            data[nt].num_raw_features = data[nt].x.shape[1]
        return data

    try:
        pe = compute_positional_encoding(data, pe_dim)
    except Exception as e:
        logger.error(f"PE computation crashed ({e}); falling back to zeros")
        pe = None

    _, _, offsets = hetero_to_homo(data)

    for nt, start in offsets.items():
        n = data[nt].num_nodes
        # record raw width BEFORE appending PE
        data[nt].num_raw_features = data[nt].x.shape[1]

        if pe is not None:
            node_pe = pe[start : start + n]
        else:
            node_pe = torch.zeros((n, pe_dim))

        # final safety net: force exact width
        if node_pe.shape != (n, pe_dim):
            node_pe = torch.zeros((n, pe_dim))

        data[nt].x = torch.cat([data[nt].x, node_pe], dim=1)

    return data


# ==================================================
# LEAKAGE-SAFE FEATURE NORMALIZER  (Bug #3 fix)
# ==================================================
class FeatureNormalizer:
    """
    Per-node-type StandardScaler fit on TRAINING graphs only.

    Correct usage (in the training notebook, AFTER the split):

        normalizer = FeatureNormalizer()
        normalizer.fit(train_graphs)          # learn mean/std from train only
        normalizer.transform(train_graphs)    # apply to every split
        normalizer.transform(val_graphs)
        normalizer.transform(test_graphs)
        normalizer.save("../checkpoints/hgt/normalizer.pkl")

    Only the RAW feature columns (the first `num_raw_features`) are scaled;
    positional-encoding columns are left untouched.
    """

    def __init__(self, node_types: Optional[List[str]] = None):
        self.node_types = node_types or ["beam", "column"]
        self.scalers: Dict[str, StandardScaler] = {}
        self.num_raw: Dict[str, int] = {}
        self.fitted = False

    def _raw_count(self, graph, nt: int) -> int:
        # prefer the value recorded by add_positional_encoding
        if hasattr(graph[nt], "num_raw_features"):
            return int(graph[nt].num_raw_features)
        return graph[nt].x.shape[1]  # fallback: treat all columns as raw

    def fit(self, train_graphs: List) -> "FeatureNormalizer":
        for nt in self.node_types:
            chunks, raw_c = [], None
            for g in train_graphs:
                if nt not in g.node_types or g[nt].x.shape[0] == 0:
                    continue
                raw_c = self._raw_count(g, nt) if raw_c is None else raw_c
                chunks.append(g[nt].x[:, :raw_c].cpu().numpy())

            if not chunks:
                logger.warning(f"No '{nt}' nodes in training set; skipping scaler")
                continue

            stacked = np.vstack(chunks).astype(np.float64)
            scaler = StandardScaler().fit(stacked)
            self.scalers[nt] = scaler
            self.num_raw[nt] = raw_c
            logger.info(
                f"Fitted '{nt}' scaler on {stacked.shape[0]} nodes, "
                f"{raw_c} raw features"
            )

        self.fitted = True
        return self

    def transform(self, graphs: List) -> List:
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer.transform called before fit")

        for g in graphs:
            for nt in self.node_types:
                if nt not in self.scalers or nt not in g.node_types:
                    continue
                if g[nt].x.shape[0] == 0:
                    continue

                raw_c = self.num_raw[nt]
                x = g[nt].x.cpu().numpy()
                x[:, :raw_c] = self.scalers[nt].transform(x[:, :raw_c])
                g[nt].x = torch.tensor(x, dtype=torch.float)
        return graphs

    def fit_transform(self, train_graphs: List) -> List:
        return self.fit(train_graphs).transform(train_graphs)

    def save(self, path: str):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {"scalers": self.scalers, "num_raw": self.num_raw,
                 "node_types": self.node_types},
                f,
            )
        logger.info(f"Normalizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(node_types=state["node_types"])
        obj.scalers = state["scalers"]
        obj.num_raw = state["num_raw"]
        obj.fitted = True
        return obj


# ==================================================
# POSITIONAL ENCODER  (geometric / topological / hybrid)
# ==================================================
class PositionalEncoder:
    """
    Appends a positional-encoding block to every node's feature vector.

    Apply ONCE to all graphs after loading (deterministic -> no leakage),
    BEFORE the train/test split. Switch `mode` to compare approaches without
    rebuilding graphs:

        PositionalEncoder(mode="geometric").transform(graphs)
        PositionalEncoder(mode="topological").transform(graphs)
        PositionalEncoder(mode="hybrid").transform(graphs)

    Modes
    -----
    geometric   : from real 3D coordinates (X1,Y1,Z1)->(X2,Y2,Z2).
                  8 dims: 3 direction cosines, 3 normalized midpoint coords
                  (incl. story fraction z), normalized length, plan-radial dist.
    topological : from graph connectivity. per-relation (log) degree
                  (= load-path proxy, robust to isolated nodes) + random-walk
                  return-probability PE (degenerate step dropped).
    hybrid      : geometric  ++  topological  (best for isolated nodes, which
                  have zero topological signal but valid coordinates).
    none        : no-op.

    The encoder appends AFTER the raw features, so `num_raw_features` (set at
    build time) still marks the boundary the FeatureNormalizer scales up to.
    """

    GEO_KEYS = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"]

    def __init__(self, mode: str = "geometric", dim: int = 8,
                 node_types: Optional[List[str]] = None):
        assert mode in {"geometric", "topological", "hybrid", "none"}
        self.mode = mode
        self.dim = dim
        self.node_types = node_types or ["beam", "column"]

    # ------------------------------------------------------------------ #
    def transform(self, graphs: List) -> List:
        for g in graphs:
            if getattr(g, "_pe_added", False) or self.mode == "none":
                g._pe_added = True
                continue

            geo = self._geometric(g) if self.mode in ("geometric", "hybrid") else None
            topo = self._topological(g) if self.mode in ("topological", "hybrid") else None

            for nt in self.node_types:
                if nt not in g.node_types:
                    continue
                parts = []
                if geo is not None:
                    parts.append(geo[nt])
                if topo is not None:
                    parts.append(topo[nt])
                pe = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                g[nt].x = torch.cat([g[nt].x, pe.float()], dim=1)
            g._pe_added = True
        return graphs

    # ------------------------------------------------------------------ #
    def _coords(self, g, nt):
        names = list(g[nt].feature_names)
        idx = {n: i for i, n in enumerate(names)}
        if not all(k in idx for k in self.GEO_KEYS):
            raise KeyError(
                f"geometric PE needs columns {self.GEO_KEYS}; "
                f"'{nt}' has {names[:8]}..."
            )
        x = g[nt].x
        p1 = x[:, [idx["X1"], idx["Y1"], idx["Z1"]]]
        p2 = x[:, [idx["X2"], idx["Y2"], idx["Z2"]]]
        return p1, p2

    def _geometric(self, g) -> Dict[str, torch.Tensor]:
        coords, mids = {}, []
        for nt in self.node_types:
            if nt not in g.node_types:
                continue
            p1, p2 = self._coords(g, nt)
            coords[nt] = (p1, p2)
            mids.append((p1 + p2) / 2)

        M = torch.cat(mids, dim=0)              # building-level normalization
        mn, mx = M.min(0).values, M.max(0).values
        rng = (mx - mn).clamp(min=1e-6)
        centroid = M.mean(0)
        plan_diag = torch.sqrt(((mx[:2] - mn[:2]) ** 2).sum()).clamp(min=1e-6)
        max_len = max(
            (torch.norm(p2 - p1, dim=1).max().item() for p1, p2 in coords.values()),
            default=1.0,
        ) or 1.0

        out = {}
        for nt, (p1, p2) in coords.items():
            d = p2 - p1
            L = torch.norm(d, dim=1).clamp(min=1e-6)
            mid = (p1 + p2) / 2
            direction = d / L.unsqueeze(1)               # 3: orientation
            mid_norm = (mid - mn) / rng                  # 3: position (z = story)
            len_norm = (L / max_len).unsqueeze(1)        # 1: span/height
            radial = (torch.norm(mid[:, :2] - centroid[:2], dim=1)
                      / plan_diag).unsqueeze(1)          # 1: torsion proxy
            out[nt] = torch.cat([direction, mid_norm, len_norm, radial], dim=1)
        return out

    def _topological(self, g) -> Dict[str, torch.Tensor]:
        # neighbor types per node type -> per-relation degree
        neighbors = {"beam": ["beam", "column"], "column": ["column", "beam"]}
        degs = {}
        for nt in self.node_types:
            if nt not in g.node_types:
                continue
            N = g[nt].x.shape[0]
            cols = []
            for other in neighbors.get(nt, []):
                et = (nt, "to", other)
                d = torch.zeros(N)
                if et in g.edge_types and g[et].edge_index.shape[1] > 0:
                    d = torch.bincount(g[et].edge_index[0], minlength=N).float()
                cols.append(torch.log1p(d).unsqueeze(1))   # log -> robust scale
            degs[nt] = torch.cat(cols, dim=1) if cols else torch.zeros(N, 1)

        # random-walk PE on the homogeneous collapse (drop degenerate step 1)
        rw_dim = max(1, self.dim - 2)
        edge_index, num_nodes, offsets = hetero_to_homo(g)
        rw_full = compute_random_walk_pe(edge_index, num_nodes, rw_dim + 1)
        rw = rw_full[:, 1: rw_dim + 1]

        out = {}
        for nt in self.node_types:
            if nt not in g.node_types:
                continue
            N = g[nt].x.shape[0]
            start = offsets[nt]
            node_rw = (
                rw[start: start + N]
                if rw.shape[0] >= start + N
                else torch.zeros(N, rw_dim)
            )
            out[nt] = torch.cat([degs[nt], node_rw], dim=1)
        return out


# ==================================================
# ISOLATED-NODE HANDLER  (none / self_loop / knn)
# ==================================================
class IsolatedNodeHandler:
    """
    Three switchable strategies for the ~11% beams that have no edges, so you
    can report results for each. Apply BEFORE the PositionalEncoder (it changes
    edges, which the topological PE reads).

        IsolatedNodeHandler(strategy="none")                 # keep as-is
        IsolatedNodeHandler(strategy="self_loop")            # add i->i edges
        IsolatedNodeHandler(strategy="knn", k=4).transform(graphs)

    All strategies only touch SAME-TYPE relations (beam->beam, column->column),
    so the model architecture is identical across options -> a fair comparison.

    knn: each isolated node is connected (bidirectionally) to its k physically
    nearest same-type nodes using the stored 3D position. `k` is a single
    hyper-parameter; try a few values per run since the ideal k differs by
    building.
    """

    def __init__(self, strategy: str = "none", k: int = 4,
                 node_types: Optional[List[str]] = None):
        assert strategy in {"none", "self_loop", "knn"}
        self.strategy = strategy
        self.k = k
        self.node_types = node_types or ["beam", "column"]

    def transform(self, graphs: List) -> List:
        for g in graphs:
            if getattr(g, "_iso_done", False) or self.strategy == "none":
                g._iso_done = True
                continue
            if self.strategy == "self_loop":
                self._self_loops(g)
            else:
                self._knn(g)
            g._iso_done = True
        return graphs

    def _isolated(self, g, nt) -> List[int]:
        N = g[nt].x.shape[0]
        conn = set()
        for et in g.edge_types:
            s, _, d = et
            ei = g[et].edge_index
            if ei.shape[1] == 0:
                continue
            if s == nt:
                conn.update(ei[0].tolist())
            if d == nt:
                conn.update(ei[1].tolist())
        return [i for i in range(N) if i not in conn]

    def _append(self, g, nt, src, dst):
        et = (nt, "to", nt)
        new = torch.tensor([src, dst], dtype=torch.long)
        if et in g.edge_types and g[et].edge_index.shape[1] > 0:
            g[et].edge_index = torch.cat([g[et].edge_index, new], dim=1)
        else:
            g[et].edge_index = new

    def _self_loops(self, g):
        for nt in self.node_types:
            if nt not in g.node_types:
                continue
            N = g[nt].x.shape[0]
            idx = list(range(N))
            self._append(g, nt, idx, idx)   # self-loop for every node

    def _knn(self, g):
        for nt in self.node_types:
            if nt not in g.node_types:
                continue
            iso = self._isolated(g, nt)
            if not iso:
                continue
            pos = getattr(g[nt], "pos", None)
            if pos is None:
                logger.warning(f"knn: '{nt}' has no .pos; skipping")
                continue
            N = pos.shape[0]
            k = min(self.k, N - 1)
            if k < 1:
                continue
            src, dst = [], []
            for i in iso:
                d = torch.norm(pos - pos[i], dim=1)
                d[i] = float("inf")
                for j in torch.topk(d, k, largest=False).indices.tolist():
                    src += [i, j]      # bidirectional
                    dst += [j, i]
            self._append(g, nt, src, dst)


# ==================================================
# TARGET NORMALIZER  (scales y = [width, height])
# ==================================================
class TargetNormalizer:
    """
    Per-node-type StandardScaler for the regression TARGETS (y).

    Train in scaled target space (stable, fast), then call `inverse(...)`
    on predictions so all reported metrics (MAE/RMSE/R2) are in REAL units.

    Fit on TRAINING graphs only -- exactly like FeatureNormalizer.
    """

    def __init__(self, node_types: Optional[List[str]] = None):
        self.node_types = node_types or ["beam", "column"]
        self.mean: Dict[str, np.ndarray] = {}
        self.scale: Dict[str, np.ndarray] = {}
        self.fitted = False

    def fit(self, train_graphs: List) -> "TargetNormalizer":
        for nt in self.node_types:
            chunks = [
                g[nt].y.cpu().numpy()
                for g in train_graphs
                if nt in g.node_types
                and hasattr(g[nt], "y")
                and g[nt].y.numel() > 0
            ]
            if not chunks:
                continue
            stacked = np.vstack(chunks).astype(np.float64)
            scaler = StandardScaler().fit(stacked)
            # guard against zero-variance targets
            scale = scaler.scale_.copy()
            scale[scale == 0] = 1.0
            self.mean[nt] = scaler.mean_.astype(np.float32)
            self.scale[nt] = scale.astype(np.float32)
            logger.info(f"Fitted '{nt}' target scaler on {stacked.shape[0]} nodes")
        self.fitted = True
        return self

    def transform(self, graphs: List) -> List:
        if not self.fitted:
            raise RuntimeError("TargetNormalizer.transform called before fit")
        for g in graphs:
            for nt in self.node_types:
                if nt not in self.mean or nt not in g.node_types:
                    continue
                if not hasattr(g[nt], "y") or g[nt].y.numel() == 0:
                    continue
                y = g[nt].y.cpu().numpy()
                y = (y - self.mean[nt]) / self.scale[nt]
                g[nt].y = torch.tensor(y, dtype=torch.float)
        return graphs

    def inverse(self, node_type: str, t: "torch.Tensor") -> "torch.Tensor":
        """Map a scaled tensor [N, 2] back to real units (on its own device)."""
        if node_type not in self.mean:
            return t
        mean = torch.as_tensor(self.mean[node_type], dtype=t.dtype, device=t.device)
        scale = torch.as_tensor(self.scale[node_type], dtype=t.dtype, device=t.device)
        return t * scale + mean

    def fit_transform(self, train_graphs: List) -> List:
        return self.fit(train_graphs).transform(train_graphs)

    def save(self, path: str):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {"mean": self.mean, "scale": self.scale, "node_types": self.node_types},
                f,
            )
        logger.info(f"Target normalizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TargetNormalizer":
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(node_types=state["node_types"])
        obj.mean = state["mean"]
        obj.scale = state["scale"]
        obj.fitted = True
        return obj
