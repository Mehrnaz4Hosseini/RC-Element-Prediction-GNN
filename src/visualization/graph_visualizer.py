import plotly.graph_objects as go
import networkx as nx
import colorsys
import logging
from collections import defaultdict
import numpy as np


class GraphVisualizer:

    def __init__(self):

        # Node colors
        self.colors = {
            "beam": "#1f77b4",
            "column": "#ff7f0e",
            "edge_default": "#888",
        }

        # Fixed palette for merged edge types
        self.edge_palette = {
            "beam—beam": "#0000FF",
            "column—column": "#FF0000",
            "beam—column": "#229451",
        }

        self._edge_color_cache = {}

        self.logger = logging.getLogger("SimpleGraphVisualizer")
        self.logger.setLevel(logging.INFO)

        # remove any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s", "%H:%M:%S"
        )

        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

        self.logger.propagate = False

    def _edge_color(self, label):

        if label in self.edge_palette:
            return self.edge_palette[label]

        if label not in self._edge_color_cache:
            idx = len(self._edge_color_cache)
            hue = (idx * 0.21) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)

            self._edge_color_cache[label] = (
                f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            )

        return self._edge_color_cache[label]

    def _merged_edge_label(self, src, dst):

        types = tuple(sorted([src, dst]))

        if types == ("beam", "beam"):
            return "beam—beam"

        elif types == ("column", "column"):
            return "column—column"

        else:
            return "beam—column"

    def visualize(self, graph, title="3D Building Graph"):

        G = nx.Graph()

        # ============================
        # Add Nodes
        # ============================
        for ntype in graph.node_types:

            count = graph[ntype].num_nodes

            for i in range(count):

                G.add_node((ntype, i), node_type=ntype)

        # ============================
        # Add Edges
        # ============================
        edge_types_seen = defaultdict(list)

        for src, rel, dst in graph.edge_types:

            merged_label = self._merged_edge_label(src, dst)

            edge_index = graph[(src, rel, dst)].edge_index

            for s, t in edge_index.t().tolist():

                u, v = (src, s), (dst, t)

                key = tuple(sorted([u, v]))

                G.add_edge(*key, type=merged_label)

                edge_types_seen[merged_label].append(key)

        # ============================
        # Graph Statistics
        # ============================

        node_counts = {ntype: graph[ntype].num_nodes for ntype in graph.node_types}
        total_nodes = sum(node_counts.values())

        edge_counts = {label: len(edges) for label, edges in edge_types_seen.items()}
        total_edges = sum(edge_counts.values())

        isolated_nodes = list(nx.isolates(G))
        num_isolated = len(isolated_nodes)

        # ============================
        # Structured Logging
        # ============================

        summary = []

        summary.append("Graph Summary")
        summary.append("-------------")

        summary.append("Nodes")
        for k, v in node_counts.items():
            summary.append(f"  {k:<7}: {v}")
        summary.append(f"  {'total':<7}: {total_nodes}")

        summary.append("")
        summary.append("Edges")
        for k, v in edge_counts.items():
            summary.append(f"  {k:<13}: {v}")
        summary.append(f"  {'total':<13}: {total_edges}")

        summary.append("")
        summary.append("Isolated Nodes")
        summary.append(f"  count : {num_isolated}")

        self.logger.info("\n" + "\n".join(summary))

        # ============================
        # Layout
        # ============================

        pos = nx.spring_layout(G, dim=3, seed=42, k=0.2)

        fig = go.Figure()

        # ============================
        # Draw Edges
        # ============================

        for label, edges in edge_types_seen.items():

            edge_x, edge_y, edge_z = [], [], []

            for u, v in edges:

                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]

                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_z += [z0, z1, None]

            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    name=label,
                    line=dict(color=self._edge_color(label), width=2),  # original width
                    opacity=0.5,
                    hoverinfo="none",
                )
            )

        # ============================
        # Draw Nodes
        # ============================

        for ntype in graph.node_types:

            nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == ntype]

            node_x = [pos[n][0] for n in nodes]
            node_y = [pos[n][1] for n in nodes]
            node_z = [pos[n][2] for n in nodes]

            color = self.colors.get(ntype, self.colors["edge_default"])

            fig.add_trace(
                go.Scatter3d(
                    x=node_x,
                    y=node_y,
                    z=node_z,
                    mode="markers",
                    name=f"{ntype.capitalize()}s",
                    marker=dict(
                        size=4,  # original size
                        color=color,
                        line=dict(color="white", width=0.5),
                    ),
                    text=[f"{ntype} {n[1]}" for n in nodes],
                    hoverinfo="text",
                )
            )

        # ============================
        # Layout
        # ============================

        fig.update_layout(
            title=dict(text=title, x=0.5),
            template="plotly_dark",
            showlegend=True,
            width=1000,
            height=800,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
        )

        fig.show()


# ==========================================================
# GeometricGraphVisualizer
# ==========================================================
class GeometricGraphVisualizer:

    def __init__(self):

        # -----------------------------------------
        # Colors
        # -----------------------------------------
        self.colors = {
            "beam": "#4C9AFF",
            "column": "#FF9F43",
            "edge_default": "#AAAAAA",
        }

        self.edge_palette = {
            "beam—beam": "#0000FF",
            "column—column": "#FF0000",
            "beam—column": "#00FF00",
        }

        self.overlap_color = "#FF00FF"
        self.isolated_color = "#FFFF00"

        self._edge_color_cache = {}

        # -----------------------------------------
        # Logger (clean, no duplication)
        # -----------------------------------------
        self.logger = logging.getLogger("GeometricGraphVisualizer")
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s", "%H:%M:%S"
        )
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.propagate = False

    # ------------------------------------------------------
    # Edge helpers
    # ------------------------------------------------------
    def _edge_color(self, label):
        return self.edge_palette.get(label, self.colors["edge_default"])

    def _merged_edge_label(self, src, dst):
        types = tuple(sorted([src, dst]))

        if types == ("beam", "beam"):
            return "beam—beam"
        elif types == ("column", "column"):
            return "column—column"
        else:
            return "beam—column"

    # ======================================================
    # MAIN VISUALIZE
    # ======================================================
    def visualize(self, graph, title="3D Building Graph"):

        G = nx.Graph()

        # ----------------------------------------------------
        # Add Nodes
        # ----------------------------------------------------
        for ntype in graph.node_types:
            for i in range(graph[ntype].num_nodes):
                G.add_node((ntype, i), node_type=ntype)

        # ----------------------------------------------------
        # Add Edges (merged)
        # ----------------------------------------------------
        edge_types_seen = defaultdict(list)

        for src, rel, dst in graph.edge_types:

            label = self._merged_edge_label(src, dst)
            edge_index = graph[(src, rel, dst)].edge_index

            for s, t in edge_index.t().tolist():

                u = (src, s)
                v = (dst, t)
                key = tuple(sorted([u, v]))

                G.add_edge(*key, type=label)
                edge_types_seen[label].append(key)

        for label in edge_types_seen:
            edge_types_seen[label] = list(set(edge_types_seen[label]))

        # ----------------------------------------------------
        # Positions
        # ----------------------------------------------------
        pos = self._get_node_positions(graph, G)

        # ----------------------------------------------------
        # Detect overlapping nodes
        # ----------------------------------------------------
        coord_map = defaultdict(list)

        for node, coord in pos.items():
            key = tuple(np.round(coord, 6))
            coord_map[key].append(node)

        overlapping_nodes = set()
        overlap_hover = {}

        for group in coord_map.values():
            if len(group) > 1:
                label = "<br>".join([f"{n[0]} {n[1]}" for n in group])
                for n in group:
                    overlapping_nodes.add(n)
                    overlap_hover[n] = label

        # ----------------------------------------------------
        # Detect isolated nodes
        # ----------------------------------------------------
        isolated_nodes = set(nx.isolates(G))

        # ----------------------------------------------------
        # Logging Summary
        # ----------------------------------------------------
        node_counts = {nt: graph[nt].num_nodes for nt in graph.node_types}
        total_nodes = sum(node_counts.values())

        edge_counts = {lbl: len(edges) for lbl, edges in edge_types_seen.items()}
        total_edges = sum(edge_counts.values())

        summary = []
        summary.append("Graph Summary")
        summary.append("-------------")

        summary.append("Nodes")
        for k, v in node_counts.items():
            summary.append(f"  {k:<7}: {v}")
        summary.append(f"  {'total':<7}: {total_nodes}")

        summary.append("")
        summary.append("Edges")
        for k, v in edge_counts.items():
            summary.append(f"  {k:<13}: {v}")
        summary.append(f"  {'total':<13}: {total_edges}")

        summary.append("")
        summary.append("Isolated Nodes")
        summary.append(f"  count : {len(isolated_nodes)}")

        summary.append("")
        summary.append("Overlapping Nodes")
        summary.append(f"  count : {len(overlapping_nodes)}")

        self.logger.info("\n" + "\n".join(summary))

        # ----------------------------------------------------
        # Create Figure
        # ----------------------------------------------------
        fig = go.Figure()

        # ---------------- EDGES ----------------
        for label, edges in edge_types_seen.items():

            edge_x, edge_y, edge_z = [], [], []

            for u, v in edges:
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]

                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_z += [z0, z1, None]

            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    name=label,
                    line=dict(color=self._edge_color(label), width=3),
                    opacity=0.5,
                    hoverinfo="none",
                )
            )

        # ---------------- NORMAL NODES ----------------
        for ntype in graph.node_types:

            nodes = [
                n
                for n, d in G.nodes(data=True)
                if d["node_type"] == ntype
                and n not in overlapping_nodes
                and n not in isolated_nodes
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=[pos[n][0] for n in nodes],
                    y=[pos[n][1] for n in nodes],
                    z=[pos[n][2] for n in nodes],
                    mode="markers",
                    name=ntype.capitalize(),
                    marker=dict(
                        size=7,
                        color=self.colors.get(ntype),
                        line=dict(color="white", width=0.5),
                    ),
                    text=[f"{ntype} {n[1]}" for n in nodes],
                    hoverinfo="text",
                )
            )

        # ---------------- OVERLAPPING ----------------
        if overlapping_nodes:
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[n][0] for n in overlapping_nodes],
                    y=[pos[n][1] for n in overlapping_nodes],
                    z=[pos[n][2] for n in overlapping_nodes],
                    mode="markers",
                    name="Overlapping Nodes",
                    marker=dict(
                        size=8,
                        color=self.overlap_color,
                        symbol="diamond",
                    ),
                    text=[overlap_hover[n] for n in overlapping_nodes],
                    hoverinfo="text",
                )
            )

        # ---------------- ISOLATED ----------------
        if isolated_nodes:
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[n][0] for n in isolated_nodes],
                    y=[pos[n][1] for n in isolated_nodes],
                    z=[pos[n][2] for n in isolated_nodes],
                    mode="markers",
                    name="Isolated Nodes",
                    marker=dict(
                        size=8,
                        color=self.isolated_color,
                        symbol="square",
                    ),
                    text=[f"{n[0]} {n[1]}" for n in isolated_nodes],
                    hoverinfo="text",
                )
            )

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template="plotly_dark",
            width=1000,
            height=800,
            showlegend=True,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        fig.show()

    # ------------------------------------------------------
    # Position Extraction
    # ------------------------------------------------------
    def _get_node_positions(self, graph, G):

        pos = {}
        self.logger.info("Extracting node positions...")

        for ntype in graph.node_types:

            node_data = graph[ntype]

            if "pos" not in node_data:
                self.logger.warning(f"{ntype}: missing 'pos'")
                continue

            coords = node_data.pos.cpu().numpy()

            for i in range(coords.shape[0]):
                pos[(ntype, i)] = coords[i].tolist()

        if len(pos) != G.number_of_nodes():
            self.logger.warning("Missing coordinates → fallback spring layout")
            return nx.spring_layout(G, dim=3, seed=42)

        self.logger.info(f"Positions collected: {len(pos)}")

        return pos
