"""Matplotlib plotting (port of the alakazam ``plot*`` family).

Gene-usage bars, the Hill diversity curve with CI ribbons, the clonal
rank-abundance curve, and a lineage-tree drawing.
"""
from __future__ import annotations

import numpy as np

from .diversity import AbundanceCurve, DiversityCurve
from .graph import LineageTree

__all__ = [
    "plotGeneUsage", "plotDiversityCurve", "plotAbundanceCurve",
    "plotLineageTree",
]

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _import_mpl():
    import matplotlib.pyplot as plt  # noqa
    return plt


def plotGeneUsage(gene_tab, value: str = "seq_freq", group: str | None = None,
                  top: int = 20, ax=None, title: str = "Gene Usage"):
    """Bar plot of V(D)J gene-usage frequencies (output of ``countGenes``)."""
    plt = _import_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    df = gene_tab.copy()
    if group is None:
        df = df.sort_values(value, ascending=False).head(top)
        ax.bar(df["gene"].astype(str), df[value], color=_PALETTE[0])
    else:
        groups = list(df[group].dropna().unique())
        genes = (df.groupby("gene")[value].sum()
                 .sort_values(ascending=False).head(top).index.tolist())
        width = 0.8 / max(len(groups), 1)
        x = np.arange(len(genes))
        for gi, g in enumerate(groups):
            sub = df[df[group] == g].set_index("gene")
            vals = [sub[value].get(gn, 0) for gn in genes]
            ax.bar(x + gi * width, vals, width=width,
                   label=str(g), color=_PALETTE[gi % len(_PALETTE)])
        ax.set_xticks(x + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(genes)
        ax.legend(title=group)

    ax.set_ylabel(value)
    ax.set_title(title)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
    ax.figure.tight_layout()
    return ax


def plotDiversityCurve(data: DiversityCurve, colors: dict | None = None,
                       main_title: str = "Diversity",
                       legend_title: str = "Group", ax=None):
    """Plot the Hill diversity curve with confidence-interval ribbons."""
    plt = _import_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    div = data.diversity
    group = data.group_by
    for gi, g in enumerate(data.groups):
        sub = div[div[group].astype(str) == str(g)].sort_values("q")
        col = (colors.get(g) if colors else None) or _PALETTE[gi % len(_PALETTE)]
        ax.plot(sub["q"], sub["d"], "-o", color=col, label=str(g), ms=3)
        ax.fill_between(sub["q"], sub["d_lower"], sub["d_upper"],
                        color=col, alpha=0.2)
    ax.set_xlabel("q")
    ax.set_ylabel("Diversity (D)")
    ax.set_title(main_title)
    ax.legend(title=legend_title)
    ax.figure.tight_layout()
    return ax


def plotAbundanceCurve(data: AbundanceCurve, colors: dict | None = None,
                       main_title: str = "Rank Abundance",
                       legend_title: str = "Group", ax=None):
    """Plot the clonal rank-abundance distribution with CI ribbons."""
    plt = _import_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    abund = data.abundance
    group = data.group_by
    for gi, g in enumerate(data.groups):
        sub = abund[abund[group].astype(str) == str(g)].sort_values("rank")
        col = (colors.get(g) if colors else None) or _PALETTE[gi % len(_PALETTE)]
        ax.plot(sub["rank"], sub["p"], "-", color=col, label=str(g))
        ax.fill_between(sub["rank"], sub["lower"], sub["upper"],
                        color=col, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Abundance")
    ax.set_title(main_title)
    ax.legend(title=legend_title)
    ax.figure.tight_layout()
    return ax


def _tree_layout(graph: LineageTree, root: str = "Germline"):
    """Simple top-down tree layout: ``{vertex: (x, y)}``."""
    if root not in graph.vertices:
        root = graph.vertices[0]
    children = {}
    for (a, b) in graph.edges:
        children.setdefault(a, []).append(b)

    depth = {root: 0}
    order = [root]
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        for c in children.get(u, []):
            if c not in seen:
                seen.add(c)
                depth[c] = depth[u] + 1
                order.append(c)
                stack.append(c)

    leaves = [v for v in order if not children.get(v)]
    xpos = {v: i for i, v in enumerate(leaves)}
    for v in reversed(order):
        kids = children.get(v, [])
        if kids:
            xpos[v] = np.mean([xpos[k] for k in kids])
    return {v: (xpos.get(v, 0), -depth.get(v, 0)) for v in graph.vertices}


def plotLineageTree(graph: LineageTree, label_field: str | None = None,
                    root: str = "Germline", ax=None,
                    title: str = "Lineage Tree"):
    """Draw an Ig lineage tree (nodes, weighted edges, optional labels)."""
    plt = _import_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    pos = _tree_layout(graph, root=root)
    weights = graph.edge_attr.get("weight", {})
    for (a, b) in graph.edges:
        xa, ya = pos[a]
        xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], "-", color="grey", zorder=1)
        w = weights.get((a, b))
        if w is not None:
            ax.text((xa + xb) / 2, (ya + yb) / 2, str(w), fontsize=7,
                    color="darkred")

    for v in graph.vertices:
        x, y = pos[v]
        is_inf = str(v).startswith("Inferred")
        is_germ = v == root
        color = "white" if is_inf else ("grey" if is_germ else "steelblue")
        ax.scatter([x], [y], s=400, c=color, edgecolors="black", zorder=2)
        if label_field:
            lbl = graph.vertex_attr.get(label_field, {}).get(v)
        else:
            lbl = v
        ax.text(x, y, "" if lbl is None else str(lbl), ha="center",
                va="center", fontsize=7, zorder=3)

    ax.set_title(title)
    ax.axis("off")
    ax.figure.tight_layout()
    return ax
