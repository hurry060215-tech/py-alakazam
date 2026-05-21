"""Lineage-tree topology analysis (port of alakazam ``R/Topology.R``).

Path lengths, MRCA detection, edge/subtree summaries and the
permutation tests for annotation enrichment.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .graph import LineageTree

__all__ = [
    "getPathLengths", "getMRCA", "tableEdges", "summarizeSubtrees",
    "permuteLabels", "testEdges", "testMRCA", "EdgeTest", "MRCATest",
]


# ==========================================================================
# Result containers
# ==========================================================================
@dataclass
class EdgeTest:
    """Result of :func:`testEdges`."""
    tests: pd.DataFrame
    permutations: pd.DataFrame
    nperm: int

    def __repr__(self):
        return f"EdgeTest(nperm={self.nperm}, {len(self.tests)} edges)"


@dataclass
class MRCATest:
    """Result of :func:`testMRCA`."""
    tests: pd.DataFrame
    permutations: pd.DataFrame
    nperm: int

    def __repr__(self):
        return f"MRCATest(nperm={self.nperm}, {len(self.tests)} annotations)"


# ==========================================================================
# Path lengths
# ==========================================================================
def getPathLengths(graph: LineageTree, root: str = "Germline",
                   field: str | None = None, exclude=None) -> pd.DataFrame:
    """Unweighted (steps) and weighted (distance) path lengths from ``root``.

    Faithful port of alakazam ``getPathLengths``.
    """
    exclude = [] if exclude is None else (
        exclude if isinstance(exclude, (list, tuple, set)) else [exclude])
    paths = graph.shortest_paths_out(root, weighted=True)

    skip = {root}
    if field is not None:
        fvals = graph.vertex_attr.get(field, {})
        for v in graph.vertices:
            val = fvals.get(v)
            if val in exclude or (val is None and any(
                    e is None or (isinstance(e, float) and np.isnan(e))
                    for e in exclude)):
                skip.add(v)

    rows = []
    for v in graph.vertices:
        if v in paths:
            steps_path, dist, vpath = paths[v]
            steps = sum(1 for x in vpath if x not in skip)
            rows.append({"name": v, "steps": steps, "distance": dist})
        else:
            rows.append({"name": v, "steps": np.nan, "distance": np.nan})
    return pd.DataFrame(rows)


def getMRCA(graph: LineageTree, path: str = "distance",
            root: str = "Germline", field: str | None = None,
            exclude=None) -> pd.DataFrame:
    """Return the most-recent-common-ancestor node(s) of a lineage tree.

    Faithful port of alakazam ``getMRCA``.
    """
    if path not in ("distance", "steps"):
        raise ValueError("path must be 'distance' or 'steps'")
    exclude = [] if exclude is None else (
        exclude if isinstance(exclude, (list, tuple, set)) else [exclude])

    path_df = getPathLengths(graph, root=root, field=field, exclude=exclude)

    skip = {root}
    if field is not None:
        fvals = graph.vertex_attr.get(field, {})
        for v in graph.vertices:
            val = fvals.get(v)
            if val in exclude or (val is None and any(
                    e is None or (isinstance(e, float) and np.isnan(e))
                    for e in exclude)):
                skip.add(v)

    cand = path_df[~path_df["name"].isin(skip)].copy()
    col = path
    cand = cand[cand[col].notna()]
    if len(cand) == 0:
        return pd.DataFrame()
    mn = cand[col].min()
    mrca_names = cand[cand[col] == mn]["name"].tolist()

    vdf = graph.vertices_dataframe()
    out = vdf[vdf["name"].isin(mrca_names)].copy()
    out = out.merge(path_df[["name", "steps", "distance"]], on="name",
                    how="left")
    return out.reset_index(drop=True)


# ==========================================================================
# Subtree summaries
# ==========================================================================
def summarizeSubtrees(graph: LineageTree, fields=None,
                      root: str = "Germline") -> pd.DataFrame:
    """Per-node subtree summary statistics.

    Faithful port of alakazam ``summarizeSubtrees``.
    """
    fields = list(fields or [])
    rows = []
    parent_of = {b: a for (a, b) in graph.edges}

    for v in graph.vertices:
        paths = graph.shortest_paths_out(v, weighted=True)
        size = len(paths)
        depth = max((s for (s, _, _) in paths.values()), default=0) + 1
        pathlen = max((d for (_, d, _) in paths.values()), default=0)
        rec = {"name": v}
        for f in fields:
            rec[f] = graph.vertex_attr.get(f, {}).get(v)
        rec["parent"] = parent_of.get(v)
        rec["outdegree"] = graph.outdegree(v)
        rec["size"] = size
        rec["depth"] = depth
        rec["pathlength"] = pathlen
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["outdegree_norm"] = df["outdegree"] / df["outdegree"].sum()
    df["size_norm"] = df["size"] / df["size"].max()
    df["depth_norm"] = df["depth"] / df["depth"].max()
    pl_max = df["pathlength"].max()
    df["pathlength_norm"] = (df["pathlength"] / pl_max if pl_max else
                             np.nan)
    return df


# ==========================================================================
# Edge tabulation
# ==========================================================================
def tableEdges(graph: LineageTree, field: str, indirect: bool = False,
               exclude=None) -> pd.DataFrame:
    """Count parent-child annotation pairs over a lineage tree.

    Faithful port of alakazam ``tableEdges``.
    """
    exclude = [] if exclude is None else list(exclude)

    def _excluded(val):
        if val in exclude:
            return True
        if (val is None or (isinstance(val, float) and np.isnan(val))):
            return any(e is None or (isinstance(e, float) and np.isnan(e))
                       for e in exclude)
        return False

    fvals = graph.vertex_attr.get(field, {})

    if indirect:
        skip = {v for v in graph.vertices if _excluded(fvals.get(v))}
        rows = []
        for v in graph.vertices:
            if v in skip:
                continue
            parent = fvals.get(v)
            paths = graph.shortest_paths_out(v, weighted=False)
            for w, (_, _, vpath) in paths.items():
                if w == v:
                    continue
                # first non-skipped node along the path after v
                trimmed = [x for x in vpath if x not in skip and x != v]
                if len(trimmed) == 1:
                    rows.append({"parent": parent,
                                 "child": fvals.get(trimmed[0])})
        edge_df = pd.DataFrame(rows)
    else:
        rows = []
        for (a, b) in graph.edges:
            pa, ch = fvals.get(a), fvals.get(b)
            if _excluded(pa) or _excluded(ch):
                continue
            rows.append({"parent": pa, "child": ch})
        edge_df = pd.DataFrame(rows)

    if len(edge_df) == 0:
        return pd.DataFrame(columns=["parent", "child", "count"])
    return (edge_df.groupby(["parent", "child"], dropna=False)
            .size().reset_index(name="count"))


# ==========================================================================
# Permutation
# ==========================================================================
def permuteLabels(graph: LineageTree, field: str,
                  exclude=("Germline", np.nan),
                  rng: np.random.Generator | None = None) -> LineageTree:
    """Permute vertex annotations of a lineage tree.

    Faithful port of alakazam ``permuteLabels``.
    """
    rng = rng or np.random.default_rng()
    exclude = list(exclude)

    def _excluded(val):
        if val in exclude:
            return True
        if (val is None or (isinstance(val, float) and np.isnan(val))):
            return any(e is None or (isinstance(e, float) and np.isnan(e))
                       for e in exclude)
        return False

    fvals = graph.vertex_attr.get(field, {})
    labels = {v: fvals.get(v) for v in graph.vertices}
    movable = [v for v in graph.vertices if not _excluded(labels[v])]
    if len(movable) < 2:
        return graph

    perm = copy.deepcopy(graph)
    vals = [labels[v] for v in movable]
    shuffled = list(np.array(vals, dtype=object)[rng.permutation(len(vals))])
    for v, nv in zip(movable, shuffled):
        perm.set_vertex_attr(field, v, nv)
    return perm


def testEdges(graphs, field: str, indirect: bool = False,
              exclude=("Germline", np.nan), nperm: int = 200,
              seed: int | None = None) -> EdgeTest:
    """Permutation test for parent-child annotation enrichment.

    Faithful port of alakazam ``testEdges``.

    Note
    ----
    The label permutation uses NumPy's RNG; with a fixed ``seed`` results
    are reproducible but differ from R's RNG.
    """
    rng = np.random.default_rng(seed)
    graphs = list(graphs)

    def _count(gs):
        parts = [tableEdges(g, field, indirect=indirect, exclude=exclude)
                 for g in gs]
        allc = pd.concat(parts, ignore_index=True)
        if len(allc) == 0:
            return pd.DataFrame(columns=["parent", "child", "count"])
        return (allc.groupby(["parent", "child"], dropna=False)["count"]
                .sum().reset_index())

    obs = _count(graphs)
    if len(obs) == 0:
        raise ValueError("No valid edges found in graphs")

    perm_list = []
    for i in range(nperm):
        permed = [permuteLabels(g, field, exclude=exclude, rng=rng)
                  for g in graphs]
        tmp = _count(permed)
        tmp["iter"] = i + 1
        perm_list.append(tmp)
    perm = pd.concat(perm_list, ignore_index=True)

    obs = obs.copy()
    for i in obs.index:
        p, c = obs.at[i, "parent"], obs.at[i, "child"]
        d = perm[(perm["parent"] == p) & (perm["child"] == c)]["count"]
        d = d.to_numpy(dtype=float)
        obs.at[i, "expected"] = d.mean() if len(d) else np.nan
        if len(d):
            obs.at[i, "pvalue"] = 1 - np.mean(d <= obs.at[i, "count"])
        else:
            obs.at[i, "pvalue"] = np.nan
    return EdgeTest(tests=obs, permutations=perm, nperm=nperm)


def testMRCA(graphs, field: str, root: str = "Germline",
             exclude=("Germline", np.nan), nperm: int = 200,
             seed: int | None = None) -> MRCATest:
    """Permutation test for MRCA annotation enrichment.

    Faithful port of alakazam ``testMRCA``.
    """
    rng = np.random.default_rng(seed)
    graphs = list(graphs)

    def _count(gs):
        rows = []
        for g in gs:
            mrca = getMRCA(g, path="distance", root=root, field=field,
                           exclude=exclude)
            if len(mrca) == 0:
                continue
            # resolve ambiguous founders: unique annotations only
            uniq = mrca.drop_duplicates(subset=[field])
            if len(uniq) == 1:
                rows.append(uniq.iloc[0][field])
        if not rows:
            return pd.DataFrame(columns=["annotation", "count"])
        s = pd.Series(rows, name="annotation")
        return s.value_counts().rename_axis("annotation").reset_index(
            name="count")

    obs = _count(graphs)

    perm_list = []
    for i in range(nperm):
        permed = [permuteLabels(g, field, exclude=exclude, rng=rng)
                  for g in graphs]
        tmp = _count(permed)
        tmp["iter"] = i + 1
        perm_list.append(tmp)
    perm = pd.concat(perm_list, ignore_index=True)

    obs = obs.copy()
    for i in obs.index:
        x = obs.at[i, "annotation"]
        d = perm[perm["annotation"] == x]["count"].to_numpy(dtype=float)
        obs.at[i, "expected"] = d.mean() if len(d) else np.nan
        if len(d):
            obs.at[i, "pvalue"] = 1 - np.mean(d <= obs.at[i, "count"])
        else:
            obs.at[i, "pvalue"] = np.nan
    return MRCATest(tests=obs, permutations=perm, nperm=nperm)
