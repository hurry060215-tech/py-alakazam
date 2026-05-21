"""Matplotlib plotting (port of the alakazam ``plot*`` family).

Gene-usage bars, the Hill diversity curve with CI ribbons, the clonal
rank-abundance curve, and a lineage-tree drawing.
"""
from __future__ import annotations

import numpy as np

from .diversity import AbundanceCurve, DiversityCurve
from .graph import LineageTree
from .topology import EdgeTest, MRCATest, summarizeSubtrees

__all__ = [
    "plotGeneUsage", "plotDiversityCurve", "plotAbundanceCurve",
    "plotLineageTree", "plotDiversityTest", "plotEdgeTest", "plotMRCATest",
    "plotSubtrees", "gridPlot",
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


# ==========================================================================
# Diversity / topology test plots
# ==========================================================================
def plotDiversityTest(data: DiversityCurve, q, colors: dict | None = None,
                      main_title: str = "Diversity",
                      legend_title: str = "Group", log_d: bool = False,
                      annotate: str = "none", ax=None):
    """Plot the mean +/- SD diversity at a single order ``q``.

    Faithful matplotlib port of alakazam ``plotDiversityTest``.
    Draws the mean diversity value (with an SD range bar) per group at
    Hill order ``q``, using the significance-test data attached to a
    :class:`~pyalakazam.DiversityCurve`.

    Parameters
    ----------
    data : DiversityCurve
        Output of :func:`~pyalakazam.alphaDiversity` (must carry tests).
    q : float
        Hill diversity order to display; must be one of ``data.q``.
    colors : dict, optional
        ``{group: color}`` mapping.
    main_title, legend_title : str
        Plot and legend titles.
    log_d : bool
        If ``True``, plot diversity on a log2 axis.
    annotate : {'none', 'depth'}
        If ``'depth'``, append the sampling depth ``N`` to each label.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    See Also
    --------
    plotDiversityCurve
    """
    plt = _import_mpl()
    if data.tests is None:
        raise ValueError("Test data missing from input object.")
    if annotate not in ("none", "depth"):
        raise ValueError("annotate must be 'none' or 'depth'")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    div = data.diversity
    group = data.group_by
    sub = div[np.isclose(div["q"].astype(float), float(q))]
    if len(sub) == 0:
        raise ValueError(f"Test for order q={q} not found in input object.")

    n_map = getattr(data, "n", None) or {}
    labels = {}
    for g in data.groups:
        n = n_map.get(g)
        labels[g] = (f"{g} (N={n})" if annotate == "depth" and n is not None
                     else str(g))

    for gi, g in enumerate(data.groups):
        row = sub[sub[group].astype(str) == str(g)]
        if len(row) == 0:
            continue
        d = float(row["d"].iloc[0])
        sd = float(row["d_sd"].iloc[0])
        col = (colors.get(g) if colors else None) or _PALETTE[gi % len(_PALETTE)]
        ax.plot([gi, gi], [d - sd, d + sd], "-", color=col, alpha=0.8, lw=2)
        ax.plot([gi], [d], "o", color=col, label=labels[g], ms=6)

    ax.set_xticks(range(len(data.groups)))
    ax.set_xticklabels([labels[g] for g in data.groups])
    ax.set_ylabel(f"Mean {q}D +/- SD")
    ax.set_title(main_title)
    if log_d:
        ax.set_yscale("log", base=2)
    ax.legend(title=legend_title)
    ax.figure.tight_layout()
    return ax


def _plot_test(perm_count, obs_rows, label_fn, color, main_title,
               style, xlabel, ax):
    """Shared histogram/CDF renderer for the permutation-test plots."""
    plt = _import_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if style not in ("histogram", "cdf"):
        raise ValueError("style must be 'histogram' or 'cdf'")

    counts = np.asarray(perm_count, dtype=float)
    if style == "histogram":
        ax.hist(counts, bins=50, color=color)
        ax.set_ylabel("Number of realizations")
    else:
        order = np.sort(counts)
        y = np.arange(1, len(order) + 1) / len(order)
        ax.step(order, y, where="post", color=color, lw=1.5)
        ax.set_ylabel("P-value")
    for obs, pval in obs_rows:
        ax.axvline(obs, color="firebrick", ls=":", lw=1.5)
        if style == "cdf" and pval is not None:
            ax.axhline(pval, color="steelblue", ls=":", lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_title(main_title)
    ax.figure.tight_layout()
    return ax


def plotEdgeTest(data: EdgeTest, color: str = "steelblue",
                 main_title: str = "Edge Test", style: str = "histogram",
                 ax=None):
    """Plot the permutation distribution of the lineage edge test.

    Faithful matplotlib port of alakazam ``plotEdgeTest``. Shows the
    null distribution of parent-child edge counts (aggregated over all
    parent/child annotation pairs) with observed values marked.

    Parameters
    ----------
    data : EdgeTest
        Output of :func:`~pyalakazam.testEdges`.
    color : str
        Fill/line color for the null distribution.
    main_title : str
        Plot title.
    style : {'histogram', 'cdf'}
        Render the null distribution as a histogram or empirical CDF.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    See Also
    --------
    plotMRCATest, plotSubtrees
    """
    obs_rows = [(float(r["count"]), float(r["pvalue"]))
                for _, r in data.tests.iterrows()]
    return _plot_test(data.permutations["count"], obs_rows, None, color,
                      main_title, style, "Number of edges", ax)


def plotMRCATest(data: MRCATest, color: str = "steelblue",
                 main_title: str = "MRCA Test", style: str = "histogram",
                 ax=None):
    """Plot the permutation distribution of the MRCA test.

    Faithful matplotlib port of alakazam ``plotMRCATest``. Shows the
    null distribution of the number of MRCA nodes by annotation, with
    observed values marked.

    Parameters
    ----------
    data : MRCATest
        Output of :func:`~pyalakazam.testMRCA`.
    color : str
        Fill/line color for the null distribution.
    main_title : str
        Plot title.
    style : {'histogram', 'cdf'}
        Render the null distribution as a histogram or empirical CDF.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    See Also
    --------
    plotEdgeTest, plotSubtrees
    """
    obs_rows = [(float(r["count"]), float(r["pvalue"]))
                for _, r in data.tests.iterrows()]
    return _plot_test(data.permutations["count"], obs_rows, None, color,
                      main_title, style, "Number of MRCAs", ax)


def plotSubtrees(graphs, field: str, stat: str, root: str = "Germline",
                 exclude=("Germline", np.nan), colors: dict | None = None,
                 main_title: str = "Subtrees",
                 legend_title: str = "Annotation", style: str = "box",
                 ax=None):
    """Plot per-annotation subtree summary statistics.

    Faithful matplotlib port of alakazam ``plotSubtrees``. Pools the
    :func:`~pyalakazam.summarizeSubtrees` output over a list of lineage
    trees and draws the normalized statistic grouped by node annotation.

    Parameters
    ----------
    graphs : list of LineageTree
        Lineage trees to summarize.
    field : str
        Vertex-annotation field to group by.
    stat : {'outdegree', 'size', 'depth', 'pathlength'}
        Which normalized subtree statistic to plot.
    root : str
        Root vertex name.
    exclude : iterable
        Annotation values to drop before plotting.
    colors : dict, optional
        ``{annotation: color}`` mapping.
    main_title, legend_title : str
        Plot and legend titles.
    style : {'box', 'violin'}
        Box plot or violin plot.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes

    See Also
    --------
    plotEdgeTest, plotMRCATest
    """
    plt = _import_mpl()
    stat_map = {"outdegree": ("outdegree_norm", "Node outdegree"),
                "size": ("size_norm", "Subtree size"),
                "depth": ("depth_norm", "Depth under node"),
                "pathlength": ("pathlength_norm",
                               "Path length under node")}
    if stat not in stat_map:
        raise ValueError("stat must be one of "
                         + ", ".join(sorted(stat_map)))
    if style not in ("box", "violin"):
        raise ValueError("style must be 'box' or 'violin'")
    stat_col, y_lab = stat_map[stat]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    exclude = list(exclude)

    def _excluded(v):
        if v in exclude:
            return True
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return any(e is None or (isinstance(e, float) and np.isnan(e))
                       for e in exclude)
        return False

    import pandas as pd
    parts = [summarizeSubtrees(g, fields=[field], root=root)
             for g in graphs]
    sum_df = pd.concat(parts, ignore_index=True)
    sum_df = sum_df[[not _excluded(v) for v in sum_df[field]]]
    sum_df = sum_df[np.isfinite(sum_df[stat_col].astype(float))]

    levels = (list(colors) if colors else
              sorted(str(x) for x in sum_df[field].dropna().unique()))
    data = [sum_df[sum_df[field].astype(str) == str(lvl)]
            [stat_col].to_numpy(dtype=float) for lvl in levels]
    pos = np.arange(1, len(levels) + 1)

    if style == "box":
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True)
        for gi, patch in enumerate(bp["boxes"]):
            col = (colors.get(levels[gi]) if colors else None) \
                or _PALETTE[gi % len(_PALETTE)]
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
    else:
        vp = ax.violinplot([d for d in data if len(d)], positions=pos,
                           widths=0.7, showmeans=True)
        for gi, body in enumerate(vp["bodies"]):
            col = (colors.get(levels[gi]) if colors else None) \
                or _PALETTE[gi % len(_PALETTE)]
            body.set_facecolor(col)
            body.set_alpha(0.8)

    ax.set_xticks(pos)
    ax.set_xticklabels(levels)
    ax.set_ylabel(y_lab)
    ax.set_title(main_title)
    ax.figure.tight_layout()
    return ax


def gridPlot(*axes, ncol: int = 1):
    """Arrange several plots on a grid of subplots.

    Matplotlib equivalent of alakazam ``gridPlot``. Each argument
    should be a callable taking an ``ax`` keyword (e.g. a
    ``functools.partial`` of one of the ``plot*`` functions), which is
    rendered into its own panel of a shared figure.

    Parameters
    ----------
    *axes : callables
        Plot callables, each accepting an ``ax`` keyword argument.
    ncol : int
        Number of columns in the grid.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from functools import partial                        # doctest: +SKIP
    >>> gridPlot(partial(plotEdgeTest, et),
    ...          partial(plotMRCATest, mt), ncol=2)          # doctest: +SKIP
    """
    plt = _import_mpl()
    n = len(axes)
    nrow = int(np.ceil(n / ncol))
    fig, grid = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow),
                             squeeze=False)
    flat = grid.flatten()
    for i, fn in enumerate(axes):
        fn(ax=flat[i])
    for j in range(n, len(flat)):
        flat[j].axis("off")
    fig.tight_layout()
    return fig
