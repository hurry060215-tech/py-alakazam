"""V(D)J gene-annotation parsing and usage tabulation.

Port of alakazam ``R/Gene.R`` --- :func:`getSegment`, :func:`getAllele`,
:func:`getGene`, :func:`getFamily`, :func:`getLocus`, :func:`getChain`,
:func:`countGenes`, :func:`sortGenes` and :func:`groupGenes`.
"""
from __future__ import annotations

import re
import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from .core import checkColumns

__all__ = [
    "getSegment", "getAllele", "getGene", "getFamily", "getLocus",
    "getChain", "countGenes", "sortGenes", "groupGenes",
]

_ALLELE_REGEX = r"((IG[HKL]|TR[ABDG])[VDJADEGMC][A-R0-9\(\)]*[-/\w]*[-\*]*[\.\w]+)"
_GENE_REGEX = r"((IG[HKL]|TR[ABDG])[VDJADEGMC][A-R0-9\(\)]*[-/\w]*)"
_FAMILY_REGEX = r"((IG[HKL]|TR[ABDG])[VDJADEGMC][A-R0-9\(\)]*)"
_LOCUS_REGEX = r"((IG[HLK]|TR[ABDG]))"


def _as_list(x):
    if isinstance(x, str):
        return [x], True
    if isinstance(x, pd.Series):
        return list(x), False
    return list(x), False


def getSegment(segment_call, segment_regex: str, first: bool = True,
               collapse: bool = True, strip_d: bool = True,
               omit_nl: bool = False, sep: str = ","):
    """Generic Ig/TCR segment-call parser.

    Extracts the substring matching ``segment_regex`` from each
    delimited call. Faithful port of alakazam ``getSegment``.
    """
    calls, scalar = _as_list(segment_call)
    edge = f"[^{re.escape(sep)}]*"
    out = []
    for call in calls:
        if call is None or (isinstance(call, float) and np.isnan(call)):
            out.append(np.nan)
            continue
        s = str(call)

        if omit_nl:
            s = re.sub(edge + "(" + _ALLELE_REGEX + ")" + edge, r"\1", s)
            nl_regex = (r"(IG[HKL]|TR[ABDG])[VDJADEGMC][0-9]+-NL[0-9]"
                        r"([-/\w]*[-\*][\.\w]+)*(" + re.escape(sep) + r"|$)")
            s = re.sub(nl_regex, "", s)

        r = re.sub(edge + "(" + segment_regex + ")" + edge, r"\1", s)

        if strip_d:
            strip_regex = (r"(?<=[A-Z0-9][0-9])D(?=\*|-|"
                           + re.escape(sep) + r"|$)")
            r = re.sub(strip_regex, "", r)

        if first:
            r = re.sub(re.escape(sep) + r".*$", "", r)
        elif collapse:
            parts = r.split(sep)
            seen = []
            for p in parts:
                if p not in seen:
                    seen.append(p)
            r = sep.join(seen)
        out.append(r)
    return out[0] if scalar else out


def getAllele(segment_call, first: bool = True, collapse: bool = True,
              strip_d: bool = True, omit_nl: bool = False, sep: str = ","):
    """Extract allele name(s) from segment call(s)."""
    return getSegment(segment_call, _ALLELE_REGEX, first=first,
                      collapse=collapse, strip_d=strip_d, omit_nl=omit_nl,
                      sep=sep)


def getGene(segment_call, first: bool = True, collapse: bool = True,
            strip_d: bool = True, omit_nl: bool = False, sep: str = ","):
    """Extract gene name(s) from segment call(s)."""
    return getSegment(segment_call, _GENE_REGEX, first=first,
                      collapse=collapse, strip_d=strip_d, omit_nl=omit_nl,
                      sep=sep)


def getFamily(segment_call, first: bool = True, collapse: bool = True,
              strip_d: bool = True, omit_nl: bool = False, sep: str = ","):
    """Extract family name(s) from segment call(s)."""
    return getSegment(segment_call, _FAMILY_REGEX, first=first,
                      collapse=collapse, strip_d=strip_d, omit_nl=omit_nl,
                      sep=sep)


def getLocus(segment_call, first: bool = True, collapse: bool = True,
             strip_d: bool = True, omit_nl: bool = False, sep: str = ","):
    """Extract locus name(s) (IGH/IGK/IGL/TRA/...) from segment call(s)."""
    return getSegment(segment_call, _LOCUS_REGEX, first=first,
                      collapse=collapse, strip_d=strip_d, omit_nl=omit_nl,
                      sep=sep)


def getChain(segment_call, first: bool = True, collapse: bool = True,
             strip_d: bool = True, omit_nl: bool = False, sep: str = ","):
    """Map segment call(s) to chain type ``VH`` (heavy) or ``VL`` (light)."""
    r = getLocus(segment_call, first=first, collapse=collapse,
                 strip_d=strip_d, omit_nl=omit_nl, sep=sep)
    scalar = isinstance(r, str)
    vals = [r] if scalar else list(r)
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.nan)
            continue
        v = re.sub(r"(IGH)|(TR[BD])", "VH", str(v))
        v = re.sub(r"(IG[KL])|(TR[AG])", "VL", v)
        out.append(v)
    return out[0] if scalar else out


def countGenes(data: pd.DataFrame, gene: str, groups=None, copy: str | None = None,
               clone: str | None = None, fill: bool = False, first: bool = True,
               collapse: bool = True, mode: str = "gene",
               remove_na: bool = True) -> pd.DataFrame:
    """Tabulate V(D)J allele/gene/family usage frequencies within each locus.

    Faithful port of alakazam ``countGenes``. Frequencies are computed
    within each locus; optional ``copy`` (copy-number weighted) and
    ``clone`` (one gene per clone) accounting are supported.
    """
    if mode not in ("gene", "allele", "family", "asis"):
        raise ValueError("mode must be one of gene/allele/family/asis")
    if groups is None:
        groups = []
    elif isinstance(groups, str):
        groups = [groups]
    else:
        groups = list(groups)
    data = data.copy()

    if remove_na:
        na = data[gene].isna()
        if na.any():
            data = data[~na].copy()

    if mode != "asis":
        fn = {"allele": getAllele, "gene": getGene, "family": getFamily}[mode]
        data[gene] = fn(data[gene], first=first, collapse=collapse)

    if "locus" not in data.columns:
        data["locus"] = data[gene].astype(str).str[:3]

    if clone is None and copy is None:
        locus_tab = (data.groupby(groups + ["locus"], dropna=False)
                     .size().reset_index(name="locus_count"))
        gene_tab = (data.groupby(groups + ["locus", gene], dropna=False)
                    .size().reset_index(name="seq_count"))
        gene_tab = gene_tab.merge(locus_tab, on=groups + ["locus"], how="left")
        gene_tab["seq_freq"] = gene_tab["seq_count"] / gene_tab["locus_count"]
        gene_tab = gene_tab.sort_values("seq_count", ascending=False)
    elif clone is not None and copy is None:
        if data[clone].isna().all():
            raise ValueError("No clone IDs are present in the data.")
        data = data[~data[clone].isna()].copy()
        # one gene per clone (most common)
        cg = (data.groupby(groups + ["locus", clone, gene], dropna=False)
              .size().reset_index(name="clone_gene_count"))
        # keep the max-count gene per clone
        cg = cg.sort_values("clone_gene_count", ascending=False)
        keep = cg.drop_duplicates(subset=groups + ["locus", clone])
        locus_tab = (keep.groupby(groups + ["locus"], dropna=False)
                     .size().reset_index(name="locus_clone_count"))
        gene_tab = (keep.groupby(groups + ["locus", gene], dropna=False)
                    .size().reset_index(name="clone_count"))
        gene_tab = gene_tab.merge(locus_tab, on=groups + ["locus"], how="left")
        gene_tab["clone_freq"] = (gene_tab["clone_count"]
                                  / gene_tab["locus_clone_count"])
        gene_tab = gene_tab.sort_values("clone_count", ascending=False)
    else:
        locus_tab = (data.groupby(groups + ["locus"], dropna=False)
                     .agg(locus_count=(gene, "size"),
                          locus_copy_count=(copy, "sum")).reset_index())
        gene_tab = (data.groupby(groups + ["locus", gene], dropna=False)
                    .agg(seq_count=(gene, "size"),
                         copy_count=(copy, "sum")).reset_index())
        gene_tab = gene_tab.merge(locus_tab, on=groups + ["locus"], how="left")
        gene_tab["seq_freq"] = gene_tab["seq_count"] / gene_tab["locus_count"]
        gene_tab["copy_freq"] = (gene_tab["copy_count"]
                                 / gene_tab["locus_copy_count"])
        gene_tab = gene_tab.sort_values("copy_count", ascending=False)

    if fill and groups:
        idx_cols = groups + [gene]
        full = pd.MultiIndex.from_product(
            [gene_tab[c].unique() for c in idx_cols], names=idx_cols)
        gene_tab = (gene_tab.set_index(idx_cols).reindex(full)
                    .reset_index())
        for c in ("seq_count", "seq_freq", "copy_count", "copy_freq",
                  "clone_count", "clone_freq"):
            if c in gene_tab.columns:
                gene_tab[c] = gene_tab[c].fillna(0)
        gene_tab["locus"] = gene_tab[gene].astype(str).str[:3]

    gene_tab = gene_tab.rename(columns={gene: "gene"}).reset_index(drop=True)
    return gene_tab


def sortGenes(genes, method: str = "name"):
    """Sort gene/allele names by name or by chromosomal position.

    Faithful port of alakazam ``sortGenes``.
    """
    if method not in ("name", "position"):
        raise ValueError("method must be 'name' or 'position'")

    calls = sorted(getAllele(list(genes), first=False, strip_d=False))
    flat = []
    for c in calls:
        for part in str(c).split(","):
            flat.append(part)
    flat = sorted(set(flat))

    rows = []
    for call in flat:
        fam = getFamily(call, first=True, strip_d=False)
        gen = getGene(call, first=True, strip_d=False)
        alle = getAllele(call, first=True, strip_d=False)
        g1 = re.sub(r"[^-]+-([^-\*D]+).*", r"\1", str(gen))
        g1 = re.sub(r"[^0-9]+", "99", g1)
        g2 = re.sub(r"[^-]+-[^-]+-?", "", str(gen))
        g2 = re.sub(r"[^0-9]+", "99", g2)
        a1 = re.sub(r"[^\*]+\*|[^\*]+$", "", str(alle))
        rows.append({
            "CALL": call, "FAMILY": str(fam),
            "G1": float(g1) if g1 else 0.0,
            "G2": float(g2) if g2 else 0.0,
            "A1": float(a1) if a1 else 0.0,
        })
    df = pd.DataFrame(rows).fillna(0)
    if method == "name":
        df = df.sort_values(["FAMILY", "G1", "G2", "A1"])
    else:
        df = df.sort_values(["G1", "G2", "FAMILY", "A1"],
                            ascending=[False, False, True, True])
    return df["CALL"].tolist()


# ----------------------------------------------------------------------
# groupGenes --- group sequences by shared V/J(/junction-length) genes
# ----------------------------------------------------------------------
_VALID_LOCI = ("IGH", "IGI", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG")
_HEAVY_LOCI = ("IGH", "TRB", "TRD")
_LIGHT_LOCI = ("IGK", "IGL", "TRA", "TRG")


def _getAllVJL(v, j, l, first, sep_anno=",", sep_chain=";"):
    """Port of alakazam internal ``getAllVJL``.

    Expands a single (possibly multi-chain, possibly ambiguous) ``v``/
    ``j``/``l`` annotation into the set of ``V@J`` (or ``V@J@L``) gene
    tokens used for grouping.
    """
    l_null = l is None
    multi_chain = sep_chain in v
    multi_anno_v = sep_anno in v
    multi_anno_j = sep_anno in j

    if multi_chain:
        v = v.split(sep_chain)
        j = j.split(sep_chain)
        if not l_null:
            l = l.split(sep_chain)
    else:
        v = [v]
        j = [j]
        if not l_null:
            l = [l]

    # split ambiguous (comma-separated) annotations within each chain
    if multi_anno_v:
        v = [chain.split(sep_anno) for chain in v]
        if first:
            v = [[anno[0]] for anno in v]
    else:
        v = [[chain] for chain in v]

    if multi_anno_j:
        j = [chain.split(sep_anno) for chain in j]
        if first:
            j = [[anno[0]] for anno in j]
    else:
        j = [[chain] for chain in j]

    # reduce allele calls to gene level (union of ambiguous genes)
    v = [getGene(annos, first=False, collapse=True) for annos in v]
    j = [getGene(annos, first=False, collapse=True) for annos in j]

    if multi_chain or (multi_anno_v and first) or (multi_anno_j and first):
        exp = []
        for i in range(len(v)):
            for vi in v[i]:
                for ji in j[i]:
                    if not l_null:
                        exp.append("@".join([vi, ji, str(l[i])]))
                    else:
                        exp.append("@".join([vi, ji]))
        seen = []
        for e in exp:
            if e not in seen:
                seen.append(e)
        return seen
    else:
        exp = []
        for vi in v[0]:
            for ji in j[0]:
                if not l_null:
                    exp.append("@".join([vi, ji, str(l[0])]))
                else:
                    exp.append("@".join([vi, ji]))
        return exp


def groupGenes(data: pd.DataFrame, v_call: str = "v_call",
               j_call: str = "j_call", junc_len: str | None = None,
               sequence_alignment: str | None = None,
               cell_id: str | None = None, split_light: bool = False,
               locus: str | None = "locus", only_heavy: bool = True,
               first: bool = False) -> pd.DataFrame:
    """Group sequences by shared V and J gene assignments.

    Faithful port of alakazam ``groupGenes``. Rows are grouped by shared
    V and J gene calls, and optionally also by junction length. In the
    case of ambiguous (multiple, comma-separated) gene assignments, the
    grouping is the union across all ambiguous V and J gene pairs ---
    analogous to single-linkage clustering, i.e. allowing for chaining.
    Both unpaired bulk sequencing and paired single-cell BCR/TCR data
    (IGH:IGK/IGL, TRB:TRA, TRD:TRG) are supported.

    Parameters
    ----------
    data
        :class:`~pandas.DataFrame` containing sequence data.
    v_call, j_call
        Names of the columns with the heavy/long chain V- and J-segment
        allele calls.
    junc_len
        Name of the column containing the junction length. If ``None``
        (default) only the first stage of a 2-stage partitioning is
        performed --- grouping by V and J gene only. If specified, a
        1-stage partitioning by V gene, J gene and junction length is
        performed.
    sequence_alignment
        Name of the column containing the sequence alignment (validated
        only).
    cell_id
        Name of the column with cell identifiers. If specified, grouping
        is performed in single-cell mode using only the heavy/long chain
        (IGH, TRB, TRD). If ``None``, bulk sequencing is assumed.
    split_light
        Deprecated; ignored.
    locus
        Name of the column containing locus information. Only used in
        single-cell mode.
    only_heavy
        Deprecated; grouping always uses the heavy/long chain only.
    first
        If ``True`` only the first call of an ambiguous gene assignment
        is used. If ``False`` (default) the union of ambiguous gene
        assignments is used to group all sequences with any overlapping
        gene calls.

    Returns
    -------
    pandas.DataFrame
        A copy of ``data`` with an added ``vj_group`` column holding the
        disjoint group index (``"G1"``, ``"G2"``, ...).

    Examples
    --------
    >>> import pyalakazam as ak
    >>> db = ak.load_example_db()
    >>> grouped = ak.groupGenes(db)
    >>> grouped["vj_group"].head()  # doctest: +SKIP
    0    G28
    1    G50
    2    G36
    3    G36
    4    G86
    Name: vj_group, dtype: object
    """
    if not only_heavy:
        warnings.warn("only_heavy = False is deprecated. Running as if "
                      "only_heavy = True")
        only_heavy = True
    if split_light:
        warnings.warn("split_light = True is deprecated. Please use "
                      "split_light = False. After clonal identification, "
                      "light chain groups can be found with "
                      "dowser::resolveLightChains")
        split_light = False

    if (cell_id is None and "cell_id" in data.columns
            and locus is not None):
        nmissing = int(data["cell_id"].isna().sum())
        if 0 < nmissing < len(data):
            raise ValueError(
                "A cell_id column was found in the data, but was not "
                "specified. Additionally, the data appears to have paired "
                "and unpaired cell data. This data type requires the "
                "single cell workflow, please specify the cell_id and "
                "rerun.")
        elif nmissing == 0:
            raise ValueError(
                "A cell_id column was found in the data, but was not "
                "specified. This data type requires the single cell "
                "workflow, please specify the cell_id and rerun.")
        else:
            warnings.warn("A cell_id column was found in the data, but was "
                          "not specified. All values are NA.")

    check = checkColumns(
        data, [c for c in (v_call, j_call, junc_len, sequence_alignment,
                           locus) if c is not None])
    if check is not True:
        raise ValueError(
            "A column or some combination of columns v_call, j_call, "
            "junc_len, sequence_alignment, and locus were not found in "
            "the data")

    data = data.copy()
    data[v_call] = data[v_call].astype("object")
    data[j_call] = data[j_call].astype("object")

    sep_anno = ","
    sep_chain = ";"
    mixed = False
    single_cell = False

    if cell_id is not None and locus is not None:
        n_na_cell = int(data[cell_id].isna().sum())
        if n_na_cell == 0:
            single_cell = True
        elif (~data[cell_id].isna()).any():
            single_cell = True
            mixed = True
        if single_cell:
            data[cell_id] = data[cell_id].astype("object")
            data[locus] = data[locus].astype("object")
            loci = data[locus].dropna().unique()
            if not all(x in _VALID_LOCI for x in loci):
                raise ValueError(
                    "The locus column contains invalid loci annotations.")

    if single_cell:
        data_orig = data.reset_index(drop=True)
        cell_col = data_orig[cell_id].to_numpy(dtype=object)
        locus_col = data_orig[locus].to_numpy(dtype=object)
        # unique cell IDs, in order of first appearance (NA last)
        cell_id_uniq = list(pd.unique(data_orig[cell_id]))
        cell_seq_idx = []
        for x in cell_id_uniq:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                if mixed:
                    idx_h = np.where(
                        pd.isna(cell_col)
                        & np.isin(locus_col, _HEAVY_LOCI))[0]
                else:
                    idx_h = np.where(
                        pd.isna(cell_col)
                        & np.isin(locus_col, _HEAVY_LOCI))[0]
                idx_l = np.array([], dtype=int)
            else:
                idx_h = np.where(
                    (cell_col == x) & np.isin(locus_col, _HEAVY_LOCI))[0]
                idx_l = np.where(
                    (cell_col == x) & np.isin(locus_col, _LIGHT_LOCI))[0]
            cell_seq_idx.append({"heavy": idx_h, "light": idx_l})

        if not mixed:
            heavy_mask = np.isin(locus_col, _HEAVY_LOCI)
            heavy = data_orig[heavy_mask]
            heavy_counts = heavy[cell_id].value_counts(dropna=False)
            if (heavy_counts > 1).any():
                raise ValueError(
                    "Only one heavy chain is allowed per cell. Remove "
                    "cells with multiple heavy chains.")
            light = data_orig[np.isin(locus_col, _LIGHT_LOCI)]
            if not light[cell_id].isin(heavy[cell_id]).all():
                raise ValueError("Unpaired light chains were found in the "
                                 "data. Please remove them.")

        # collapse per-cell heavy-chain annotations
        cols = [c for c in (cell_id, v_call, j_call, junc_len)
                if c is not None]
        rows = []
        v_orig = data_orig[v_call].to_numpy(dtype=object)
        j_orig = data_orig[j_call].to_numpy(dtype=object)
        l_orig = (data_orig[junc_len].to_numpy(dtype=object)
                  if junc_len is not None else None)
        for i_cell, x in enumerate(cell_id_uniq):
            idx_h = cell_seq_idx[i_cell]["heavy"]
            row = {cell_id: x}
            row[v_call] = sep_chain.join(str(s) for s in v_orig[idx_h])
            row[j_call] = sep_chain.join(str(s) for s in j_orig[idx_h])
            if junc_len is not None:
                row[junc_len] = sep_chain.join(
                    str(s) for s in l_orig[idx_h])
            rows.append(row)
        work = pd.DataFrame(rows, columns=cols)
    else:
        work = data.reset_index(drop=True)

    # check one-to-one V/J chain correspondence
    n_sep_v = work[v_call].astype(str).str.count(re.escape(sep_chain))
    n_sep_j = work[j_call].astype(str).str.count(re.escape(sep_chain))
    if (n_sep_v != n_sep_j).any():
        raise ValueError("Requirement not met: one-to-one annotation-to-"
                         "chain correspondence for both V and J (heavy)")

    cols_grp = [c for c in (v_call, j_call, junc_len) if c is not None]
    sub = work[cols_grp]
    bool_na = sub.isna().any(axis=1)
    bool_empty = (sub.astype(str) == "").any(axis=1)
    bool_bad = (bool_na | bool_empty).to_numpy()
    if bool_bad.any():
        entity = " cell(s)" if single_cell else " sequence(s)"
        warnings.warn(
            "NA(s) found in one or more of { "
            + ", ".join(cols_grp) + " } columns. "
            + f"{int(bool_bad.sum())}{entity} removed.")
        keep = ~bool_bad
        work = work[keep].reset_index(drop=True)
        if single_cell:
            cell_id_uniq = [c for c, k in zip(cell_id_uniq, keep) if k]
            cell_seq_idx = [c for c, k in zip(cell_seq_idx, keep) if k]

    n_entities = len(work)
    if n_entities == 0:
        raise ValueError("No sequences/cells remain after NA removal.")

    # expand each row into its set of V@J(@L) tokens
    v_vals = work[v_call].astype(str).tolist()
    j_vals = work[j_call].astype(str).tolist()
    l_vals = (work[junc_len].astype(str).tolist()
              if junc_len is not None else None)
    row_tokens = []
    for i in range(n_entities):
        li = None if l_vals is None else l_vals[i]
        toks = _getAllVJL(v_vals[i], j_vals[i], li, first=first,
                          sep_anno=sep_anno, sep_chain=sep_chain)
        row_tokens.append(toks)

    # single-linkage union-find: rows sharing any token are connected
    token_to_root = {}
    parent = list(range(n_entities))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, toks in enumerate(row_tokens):
        for tok in toks:
            if tok in token_to_root:
                _union(i, token_to_root[tok])
            else:
                token_to_root[tok] = i

    # assign group labels in order of first row appearance
    label_map = {}
    groups = []
    for i in range(n_entities):
        r = _find(i)
        if r not in label_map:
            label_map[r] = f"G{len(label_map) + 1}"
        groups.append(label_map[r])

    if not single_cell:
        work = work.copy()
        work["vj_group"] = groups
        return work

    # single-cell: propagate per-cell group back to all chains
    out = data_orig.copy()
    out["vj_group"] = np.nan
    out["vj_group"] = out["vj_group"].astype("object")
    for i_cell in range(len(work)):
        idx_h = cell_seq_idx[i_cell]["heavy"]
        idx_l = cell_seq_idx[i_cell]["light"]
        if mixed and len(idx_h) == 0 and len(idx_l) == 1:
            out.loc[out.index[idx_l], "vj_group"] = np.nan
            continue
        idx_all = np.concatenate([idx_h, idx_l])
        if len(idx_all):
            out.loc[out.index[idx_all], "vj_group"] = groups[i_cell]
    return out
