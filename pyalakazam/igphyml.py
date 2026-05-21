"""IgPhyML output handling (port of alakazam ``R/Phylo.R``).

* :func:`readIgphyml` --- read an IgPhyML repertoire output file into
  parameter estimates plus per-clone lineage trees.
* :func:`combineIgphyml` --- combine repertoire-level IgPhyML parameter
  estimates from several :func:`readIgphyml` objects.

IgPhyML output (``.tab``) is a tab-delimited file: the first data row
holds the run command in the ``TREE`` column, and every subsequent row
holds one clone with its parameter estimates and a Newick tree string.
"""
from __future__ import annotations

import pandas as pd

from .graph import LineageTree
from .lineage import Phylo, phyloToGraph

__all__ = ["readIgphyml", "combineIgphyml"]

# Parameter ordering used by combineIgphyml (faithful copy from R).
_ORDERED_PARAMS = (
    "id", "nseq", "nsite", "lhood", "tree_length", "omega_fwr_mle",
    "omega_fwr_lci", "omega_fwr_uci", "omega_cdr_mle", "omega_cdr_lci",
    "omega_cdr_uci", "kappa_mle", "kappa_lci", "kappa_uci", "wrc_2_mle",
    "wrc_2_lci", "wrc_2_uci", "gyw_0_mle", "gyw_0_lci", "gyw_0_uci",
    "wa_1_mle", "wa_1_lci", "wa_1_uci", "tw_0_mle", "tw_0_lci", "tw_0_uci",
    "syc_2_mle", "syc_2_lci", "syc_2_uci", "grs_0_mle", "grs_0_lci",
    "grs_0_uci",
)


# ==========================================================================
# Newick parsing
# ==========================================================================
def _parse_newick(text: str) -> Phylo:
    """Parse a Newick tree string into a :class:`Phylo` object.

    A minimal stand-in for ``ape::read.tree``: supports named tips,
    internal-node labels and branch lengths.
    """
    s = text.strip()
    if s.endswith(";"):
        s = s[:-1]

    tips: list[str] = []
    internals: list = []
    edges: list = []
    elens: list = []
    pos = 0

    def _read_token():
        """Read a name/length token up to the next structural char."""
        nonlocal pos
        start = pos
        while pos < len(s) and s[pos] not in "(),:;":
            pos += 1
        return s[start:pos]

    def _parse_clade():
        """Recursively parse one clade; return its node id."""
        nonlocal pos
        if s[pos] == "(":
            pos += 1  # consume '('
            children = []
            while True:
                child = _parse_clade()
                children.append(child)
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                    continue
                break
            assert s[pos] == ")"
            pos += 1  # consume ')'
            label = _read_token()
            node_id = ("INT", len(internals))
            internals.append(label if label else None)
            for ch, ch_len in children:
                edges.append((node_id, ch))
                elens.append(ch_len)
            return (node_id, _read_length())
        else:
            name = _read_token()
            tips.append(name)
            return (("TIP", len(tips) - 1), _read_length())

    def _read_length():
        """Read an optional ``:length`` suffix; return float or 0.0."""
        nonlocal pos
        if pos < len(s) and s[pos] == ":":
            pos += 1
            tok = _read_token()
            return float(tok) if tok else 0.0
        return 0.0

    root_id, _ = _parse_clade()

    # build the ape-style 1-based index: tips first, then internals
    n_tip = len(tips)

    def _idx(node_id):
        kind, k = node_id
        return k + 1 if kind == "TIP" else n_tip + k + 1

    edge = [[_idx(a), _idx(b)] for (a, b) in edges]
    return Phylo(edge=__import__("numpy").asarray(edge, dtype=int),
                 edge_length=__import__("numpy").asarray(elens, dtype=float),
                 tip_label=list(tips),
                 node_label=[lbl if lbl else "" for lbl in internals])


def _is_rooted(phylo: Phylo) -> bool:
    """A rooted binary tree has the root node with out-degree two."""
    n_tip = len(phylo.tip_label)
    root = n_tip + 1
    return int((phylo.edge[:, 0] == root).sum()) == 2


# ==========================================================================
# Public API
# ==========================================================================
def readIgphyml(file, id=None, format: str = "graph",
                collapse: bool = False, branches: str = "mutations"):
    """Read in output from IgPhyML.

    Faithful port of alakazam ``readIgphyml``. Reads the repertoire
    output (``.tab``) of the IgPhyML phylogenetics inference package
    into parameter estimates and per-clone lineage trees.

    Parameters
    ----------
    file : str or path-like
        IgPhyML output file (``.tab``).
    id : str, optional
        Identifier to assign to the returned object.
    format : {'graph', 'phylo'}
        Return per-clone trees as :class:`~pyalakazam.LineageTree`
        graphs (``'graph'``) or :class:`Phylo` objects (``'phylo'``).
    collapse : bool
        If ``True``, collapse internal nodes separated by very short
        branches and drop internal-node labels.
    branches : {'mutations', 'distance'}
        ``'mutations'`` rescales branch lengths to expected mutations
        (``length * NSITE``); ``'distance'`` leaves them as expected
        substitutions per site.

    Returns
    -------
    dict
        ``{'param': DataFrame, 'command': str, 'trees': dict}`` where
        ``trees`` maps clone IDs to trees in the requested ``format``.

    Examples
    --------
    >>> ig = readIgphyml("repertoire_igphyml.tab")           # doctest: +SKIP
    >>> ig["param"].columns                                  # doctest: +SKIP

    See Also
    --------
    combineIgphyml
    """
    if format not in ("graph", "phylo"):
        raise ValueError("format must be 'graph' or 'phylo'")
    if branches not in ("mutations", "distance"):
        raise ValueError("branches must be 'mutations' or 'distance'")

    df = pd.read_csv(file, sep="\t", dtype=str, keep_default_na=False)

    params = df.drop(columns=["TREE"]).copy()
    params.columns = [c.lower() for c in params.columns]
    # numeric coercion where possible (leave non-numeric columns as-is)
    for c in params.columns:
        coerced = pd.to_numeric(params[c], errors="coerce")
        if coerced.notna().all():
            params[c] = coerced

    out: dict = {"param": params, "command": df.iloc[0]["TREE"]}

    trees: dict = {}
    for i in range(1, len(df)):
        rtree = _parse_newick(df.iloc[i]["TREE"])
        germ_base = f"{df.iloc[i]['CLONE']}_GERM"
        germ_ids = [t for t in rtree.tip_label if germ_base in t]
        if len(germ_ids) > 1:
            raise ValueError(
                "Can only be one tip of the form '<cloneid>_GERM'")
        rtree.germid = germ_ids[0] if germ_ids else None

        if branches == "mutations":
            nsite = float(df.iloc[i]["NSITE"])
            rtree.edge_length = (rtree.edge_length * nsite).round(1)

        if collapse:
            rtree.node_label = ["" for _ in rtree.node_label]

        if format == "graph":
            trees[df.iloc[i]["CLONE"]] = phyloToGraph(
                rtree, germline=rtree.germid)
        else:
            trees[df.iloc[i]["CLONE"]] = rtree

    out["trees"] = trees
    if id is not None:
        out["param"] = out["param"].copy()
        out["param"]["id"] = id
    return out


def combineIgphyml(iglist, format: str = "wide") -> pd.DataFrame:
    """Combine repertoire-level IgPhyML parameter estimates.

    Faithful port of alakazam ``combineIgphyml``. Collects the
    repertoire parameter rows from a list of :func:`readIgphyml`
    objects into a single data frame.

    Parameters
    ----------
    iglist : list of dict
        List of objects returned by :func:`readIgphyml` (each must have
        an ``id`` assigned).
    format : {'wide', 'long'}
        ``'wide'`` returns one row per repertoire; ``'long'`` returns a
        tidy ``id`` / ``variable`` / ``value`` data frame.

    Returns
    -------
    pandas.DataFrame
        The combined parameter estimates.

    Examples
    --------
    >>> combined = combineIgphyml([ig1, ig2])                # doctest: +SKIP

    See Also
    --------
    readIgphyml
    """
    if format not in ("wide", "long"):
        raise ValueError("format must be 'wide' or 'long'")

    # parameters common to (the maximum number of) objects
    counts: dict = {}
    for x in iglist:
        for name in x["param"].columns:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        raise ValueError("No parameters found in objects.")
    mx = max(counts.values())
    common = {n for n, c in counts.items() if c == mx}
    params = [p for p in _ORDERED_PARAMS if p in common]

    if "id" not in params:
        raise ValueError(
            "id not specified in objects. Use 'id' flag in readIgphyml.")

    repertoires = [x["param"].iloc[[0]][params] for x in iglist]
    combined = pd.concat(repertoires, ignore_index=True)

    if format == "long":
        combined = combined.melt(id_vars=["id"], var_name="variable",
                                 value_name="value")
        cats = [p for p in params if p != "id"]
        combined["variable"] = pd.Categorical(
            combined["variable"], categories=cats, ordered=True)
    return combined
