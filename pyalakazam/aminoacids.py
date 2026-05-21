"""Amino-acid physicochemical property calculations.

Port of alakazam ``R/AminoAcids.R`` --- :func:`gravy`, :func:`bulk`,
:func:`polar`, :func:`aliphatic`, :func:`charge`, :func:`isValidAASeq`,
:func:`countPatterns` and :func:`aminoAcidProperties`.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

from .constants import (ABBREV_AA, BULKINESS_ZIMJ68, HYDROPATHY_KYTJ82,
                        PK_EMBOSS, POLARITY_GRAR74)
from .sequence import translateDNA

__all__ = [
    "gravy", "bulk", "polar", "aliphatic", "charge", "isValidAASeq",
    "countPatterns", "aminoAcidProperties",
]

_NONINFO_RE = re.compile(r"[X.\*-]")
_VALID_AA = set(ABBREV_AA.keys()) | {"X", ".", "*", "-"}


def _as_str_list(seq):
    if isinstance(seq, str):
        return [seq], True
    return [("" if (s is None or (isinstance(s, float) and np.isnan(s)))
             else str(s)) for s in seq], False


def _score_avg(seq, table):
    """Average score over informative positions (gravy/bulk/polar)."""
    seqs, scalar = _as_str_list(seq)
    out = []
    for s in seqs:
        clean = _NONINFO_RE.sub("", s)
        if len(clean) == 0:
            out.append(np.nan)
            continue
        total = sum(table.get(c, np.nan) for c in clean)
        out.append(total / len(clean))
    return out[0] if scalar else np.array(out)


def gravy(seq, hydropathy=None):
    """Grand average of hydrophobicity (Kyte & Doolittle)."""
    return _score_avg(seq, hydropathy or HYDROPATHY_KYTJ82)


def bulk(seq, bulkiness=None):
    """Average bulkiness (Zimmerman et al, 1968)."""
    return _score_avg(seq, bulkiness or BULKINESS_ZIMJ68)


def polar(seq, polarity=None):
    """Average polarity (Grantham, 1974)."""
    return _score_avg(seq, polarity or POLARITY_GRAR74)


def _count_occurrences(seq, pattern):
    """Count non-overlapping regex matches per sequence."""
    seqs, scalar = _as_str_list(seq)
    rx = re.compile(pattern)
    out = [len(rx.findall(s)) for s in seqs]
    return out[0] if scalar else np.array(out)


def aliphatic(seq, normalize: bool = True):
    """Aliphatic index (Ikai, 1980)."""
    seqs, scalar = _as_str_list(seq)
    ala = np.asarray(_count_occurrences(seqs, "[A]"), dtype=float)
    val = np.asarray(_count_occurrences(seqs, "[V]"), dtype=float)
    leu_ile = np.asarray(_count_occurrences(seqs, "[LI]"), dtype=float)
    out = ala + 2.9 * val + 3.9 * leu_ile
    if normalize:
        lens = np.array([len(_NONINFO_RE.sub("", s)) for s in seqs],
                        dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = out / lens
    return float(out[0]) if scalar else out


def charge(seq, pH: float = 7.4, pK=None, normalize: bool = False):
    """Net charge (Moore, 1985), excluding N/C terminus charges."""
    pK = pK or PK_EMBOSS
    seqs, scalar = _as_str_list(seq)

    def cnt(letter):
        return np.asarray(_count_occurrences(seqs, letter), dtype=float)

    arg = cnt("R") * (1 / (1 + 10 ** (pH - pK["R"])))
    his = cnt("H") * (1 / (1 + 10 ** (pH - pK["H"])))
    lys = cnt("K") * (1 / (1 + 10 ** (pH - pK["K"])))
    asp = cnt("D") * (-1 / (1 + 10 ** (-(pH - pK["D"]))))
    glu = cnt("E") * (-1 / (1 + 10 ** (-(pH - pK["E"]))))
    cys = cnt("C") * (-1 / (1 + 10 ** (-(pH - pK["C"]))))
    tyr = cnt("Y") * (-1 / (1 + 10 ** (-(pH - pK["Y"]))))
    out = arg + lys + his + asp + glu + tyr + cys
    if normalize:
        lens = np.array([len(_NONINFO_RE.sub("", s)) for s in seqs],
                        dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = out / lens
    return float(out[0]) if scalar else out


def isValidAASeq(seq):
    """Return True for each valid (non-ambiguous) amino-acid sequence."""
    scalar = isinstance(seq, str)
    seqs = [seq] if scalar else list(seq)
    out = []
    for s in seqs:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            out.append(False)
            continue
        out.append(all(c in _VALID_AA for c in str(s)))
    return out[0] if scalar else np.array(out)


def countPatterns(seq, patterns, nt: bool = True, trim: bool = False,
                  label: str = "region") -> pd.DataFrame:
    """Count the fraction of sequence positions matching each pattern."""
    region_aa = translateDNA(list(seq), trim=trim) if nt else list(seq)
    region_aa = [("" if (s is None or (isinstance(s, float) and np.isnan(s)))
                  else str(s)) for s in region_aa]
    aa_len = np.array([len(s) for s in region_aa], dtype=float)

    if isinstance(patterns, dict):
        names = list(patterns.keys())
        pats = list(patterns.values())
    else:
        pats = list(patterns)
        names = [f"X{i + 1}" for i in range(len(pats))]

    cols = {}
    for name, pat in zip(names, pats):
        col = f"{label}_{name}" if label else name
        counts = np.asarray(_count_occurrences(region_aa, pat), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            cols[col] = counts / aa_len
    return pd.DataFrame(cols)


_PROP_COLS = {
    "length": "aa_length", "gravy": "aa_gravy", "bulk": "aa_bulk",
    "aliphatic": "aa_aliphatic", "polarity": "aa_polarity",
    "charge": "aa_charge", "basic": "aa_basic", "acidic": "aa_acidic",
    "aromatic": "aa_aromatic",
}
_DEFAULT_PROPS = ["length", "gravy", "bulk", "aliphatic", "polarity",
                  "charge", "basic", "acidic", "aromatic"]


def aminoAcidProperties(data: pd.DataFrame, property: Sequence[str] | None = None,
                        seq: str = "junction", nt: bool = True,
                        trim: bool = False, label: str | None = None,
                        **kwargs) -> pd.DataFrame:
    """Compute per-sequence amino-acid physicochemical descriptors.

    Faithful port of alakazam ``aminoAcidProperties``. Adds ``*_aa_*``
    columns for length, gravy, bulk, aliphatic, polarity, charge, and
    basic/acidic/aromatic residue fractions.
    """
    props = list(property) if property else list(_DEFAULT_PROPS)
    if label is None:
        label = seq
    out_cols = {p: f"{label}_{_PROP_COLS[p]}" for p in props}

    region = data[seq].astype(object)
    if nt:
        region_aa = translateDNA(list(region), trim=trim)
    else:
        region_aa = list(region)
        if trim:
            region_aa = [(s[1:len(s) - 1]
                          if isinstance(s, str) else s) for s in region_aa]
    region_aa = ["" if (s is None or (isinstance(s, float) and np.isnan(s)))
                 else str(s) for s in region_aa]

    valid = np.asarray(isValidAASeq(region_aa))
    out = pd.DataFrame(np.nan, index=data.index,
                       columns=[out_cols[p] for p in props])

    valid_idx = np.where(valid)[0]
    valid_seqs = [region_aa[i] for i in valid_idx]
    if len(valid_seqs):
        labels = data.index[valid_idx]
        gravy_kw = {k: v for k, v in kwargs.items() if k == "hydropathy"}
        bulk_kw = {k: v for k, v in kwargs.items() if k == "bulkiness"}
        ali_kw = {k: v for k, v in kwargs.items() if k == "normalize"}
        pol_kw = {k: v for k, v in kwargs.items() if k == "polarity"}
        chg_kw = {k: v for k, v in kwargs.items()
                  if k in ("pH", "pK", "normalize")}

        if "length" in props:
            out.loc[labels, out_cols["length"]] = [len(s) for s in valid_seqs]
        if "gravy" in props:
            out.loc[labels, out_cols["gravy"]] = gravy(valid_seqs, **gravy_kw)
        if "bulk" in props:
            out.loc[labels, out_cols["bulk"]] = bulk(valid_seqs, **bulk_kw)
        if "aliphatic" in props:
            out.loc[labels, out_cols["aliphatic"]] = aliphatic(
                valid_seqs, **ali_kw)
        if "polarity" in props:
            out.loc[labels, out_cols["polarity"]] = polar(valid_seqs, **pol_kw)
        if "charge" in props:
            out.loc[labels, out_cols["charge"]] = charge(valid_seqs, **chg_kw)

        info = np.array([len(_NONINFO_RE.sub("", s)) for s in valid_seqs],
                        dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            if "basic" in props:
                cnt = np.asarray(_count_occurrences(valid_seqs, "[RHK]"),
                                 dtype=float)
                out.loc[labels, out_cols["basic"]] = cnt / info
            if "acidic" in props:
                cnt = np.asarray(_count_occurrences(valid_seqs, "[DE]"),
                                 dtype=float)
                out.loc[labels, out_cols["acidic"]] = cnt / info
            if "aromatic" in props:
                cnt = np.asarray(_count_occurrences(valid_seqs, "[FWHY]"),
                                 dtype=float)
                out.loc[labels, out_cols["aromatic"]] = cnt / info

    keep = [c for c in data.columns if c not in out.columns]
    return pd.concat([data[keep], out], axis=1)
