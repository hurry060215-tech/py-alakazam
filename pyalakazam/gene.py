"""V(D)J gene-annotation parsing and usage tabulation.

Port of alakazam ``R/Gene.R`` --- :func:`getSegment`, :func:`getAllele`,
:func:`getGene`, :func:`getFamily`, :func:`getLocus`, :func:`getChain`,
:func:`countGenes` and :func:`sortGenes`.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "getSegment", "getAllele", "getGene", "getFamily", "getLocus",
    "getChain", "countGenes", "sortGenes",
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
