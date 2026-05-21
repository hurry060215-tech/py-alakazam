"""Lightweight directed-graph container for lineage trees.

A minimal, dependency-free stand-in for the igraph objects produced by
alakazam. A :class:`LineageTree` stores vertices (with arbitrary
attributes) and weighted directed edges, and provides the traversal
primitives required by the topology-analysis functions.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

__all__ = ["LineageTree"]


class LineageTree:
    """Directed, vertex- and edge-annotated lineage tree.

    Attributes
    ----------
    vertices : list of str
        Ordered vertex names.
    vertex_attr : dict
        ``{attr_name: {vertex: value}}``.
    edges : list of tuple
        ``(from, to)`` pairs in insertion order.
    edge_attr : dict
        ``{attr_name: {(from, to): value}}``.
    graph_attr : dict
        Graph-level attributes (clone, v_gene, j_gene, junc_len, ...).
    """

    def __init__(self):
        self.vertices: list[str] = []
        self.vertex_attr: dict[str, dict] = {}
        self.edges: list[tuple[str, str]] = []
        self.edge_attr: dict[str, dict] = {}
        self.graph_attr: dict = {}

    # -- construction ----------------------------------------------------
    def add_vertex(self, name: str, **attrs):
        if name not in self.vertices:
            self.vertices.append(name)
        for k, v in attrs.items():
            self.vertex_attr.setdefault(k, {})[name] = v

    def add_edge(self, frm: str, to: str, **attrs):
        self.add_vertex(frm)
        self.add_vertex(to)
        self.edges.append((frm, to))
        for k, v in attrs.items():
            self.edge_attr.setdefault(k, {})[(frm, to)] = v

    def set_vertex_attr(self, name: str, vertex: str, value):
        self.vertex_attr.setdefault(name, {})[vertex] = value

    # -- accessors -------------------------------------------------------
    def get_vertex_attr(self, name: str, vertex: str | None = None):
        d = self.vertex_attr.get(name, {})
        if vertex is None:
            return [d.get(v) for v in self.vertices]
        return d.get(vertex)

    def get_edge_attr(self, name: str, edge=None):
        d = self.edge_attr.get(name, {})
        if edge is None:
            return [d.get(e) for e in self.edges]
        return d.get(edge)

    @property
    def vertex_attr_names(self):
        return list(self.vertex_attr.keys())

    @property
    def edge_attr_names(self):
        return list(self.edge_attr.keys())

    # -- topology --------------------------------------------------------
    def children(self, v: str) -> list[str]:
        return [b for (a, b) in self.edges if a == v]

    def parents(self, v: str) -> list[str]:
        return [a for (a, b) in self.edges if b == v]

    def outdegree(self, v: str) -> int:
        return sum(1 for (a, _) in self.edges if a == v)

    def shortest_paths_out(self, root: str, weighted: bool = False):
        """Return ``{vertex: (steps, distance, vpath)}`` from ``root``.

        Traverses the directed tree breadth-first; ``vpath`` is the list
        of vertices from ``root`` to the target (inclusive).
        """
        adj = defaultdict(list)
        weights = self.edge_attr.get("weight", {})
        for (a, b) in self.edges:
            adj[a].append(b)
        result = {root: (0, 0.0, [root])}
        queue = deque([root])
        while queue:
            u = queue.popleft()
            su, du, pu = result[u]
            for w in adj[u]:
                if w not in result:
                    wt = weights.get((u, w), 1)
                    result[w] = (su + 1, du + (wt if wt is not None else 0),
                                 pu + [w])
                    queue.append(w)
        return result

    # -- export ----------------------------------------------------------
    def vertices_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"name": self.vertices})
        for attr, d in self.vertex_attr.items():
            df[attr] = [d.get(v) for v in self.vertices]
        return df

    def edges_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.edges, columns=["from", "to"])
        for attr, d in self.edge_attr.items():
            df[attr] = [d.get(e) for e in self.edges]
        return df

    def __repr__(self):
        return (f"LineageTree(clone={self.graph_attr.get('clone')!r}, "
                f"{len(self.vertices)} vertices, {len(self.edges)} edges)")
