"""DNA / amino-acid sequence operations (port of alakazam ``R/Sequence.R``).

Covers translation, gap/end masking, padding, V-region extraction,
sequence-distance models (Hamming with IUPAC ambiguity and indel
handling) and duplicate collapsing.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .constants import CODON_TABLE, IMGT_REGIONS, IUPAC_AA, IUPAC_DNA

__all__ = [
    "translateDNA", "maskSeqGaps", "maskSeqEnds", "padSeqEnds",
    "extractVRegion", "getDNAMatrix", "getAAMatrix", "seqDist",
    "seqEqual", "pairwiseDist", "pairwiseEqual", "nonsquareDist",
    "collapseDuplicates",
]


# ==========================================================================
# Translation
# ==========================================================================
def _translate_one(seq: str) -> str | float:
    """Translate a single DNA string to amino acids (seqinr-compatible)."""
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return np.nan
    seq = str(seq)
    if len(seq) < 3:
        return np.nan
    out = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        codon = seq[i:i + 3].upper()
        out.append(CODON_TABLE.get(codon, "X"))
    return "".join(out)


def translateDNA(seq, trim: bool = False):
    """Translate nucleotide sequence(s) to amino acids.

    Parameters
    ----------
    seq : str or iterable of str
        DNA sequence(s).
    trim : bool
        If True, drop the first and last codon (junction -> CDR3).

    Returns
    -------
    str or list of str
    """
    scalar = isinstance(seq, str)
    seqs = [seq] if scalar else list(seq)
    out = []
    for s in seqs:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            out.append(np.nan)
            continue
        s = str(s)
        if trim:
            s = s[3:len(s) - 3]
        s = re.sub(r"[-.]", "N", s)
        out.append(_translate_one(s))
    return out[0] if scalar else out


# ==========================================================================
# Masking / padding
# ==========================================================================
def maskSeqGaps(seq, mask_char: str = "N", outer_only: bool = False):
    """Replace gap characters ``-`` and ``.`` with ``mask_char``."""
    scalar = isinstance(seq, str)
    seqs = [seq] if scalar else list(seq)
    out = []
    for s in seqs:
        s = str(s)
        if outer_only:
            m_head = re.match(r"^[-.]+", s)
            m_tail = re.search(r"[-.]+$", s)
            if m_head:
                s = mask_char * len(m_head.group(0)) + s[m_head.end():]
            if m_tail:
                s = s[:m_tail.start()] + mask_char * len(m_tail.group(0))
        else:
            s = re.sub(r"[-.]", mask_char, s)
        out.append(s)
    return out[0] if scalar else out


def maskSeqEnds(seq, mask_char: str = "N", max_mask: int | None = None,
                trim: bool = False):
    """Uniformly mask (or trim) ragged leading/trailing ``mask_char`` runs."""
    scalar = isinstance(seq, str)
    seqs = [str(s) for s in ([seq] if scalar else list(seq))]
    if not seqs:
        return seq

    def _head(s):
        m = re.match(re.escape(mask_char) + "*", s)
        return len(m.group(0)) if m else 0

    def _tail(s):
        m = re.search(re.escape(mask_char) + "*$", s)
        return len(m.group(0)) if m else 0

    left = max(_head(s) for s in seqs)
    right = max(_tail(s) for s in seqs)
    if max_mask is not None:
        left = min(left, max_mask)
        right = min(right, max_mask)

    out = []
    for s in seqs:
        n = len(s)
        if trim:
            out.append(s[left:n - right])
        else:
            s2 = mask_char * left + s[left:n - right] + mask_char * right
            out.append(s2)
    return out[0] if scalar else out


def padSeqEnds(seq, length: int | None = None, start: bool = False,
               pad_char: str = "N", mod3: bool = True):
    """Pad ragged sequence ends to a uniform length."""
    scalar = isinstance(seq, str)
    seqs = [str(s) for s in ([seq] if scalar else list(seq))]
    width = max([len(s) for s in seqs] + ([length] if length else [0]))
    if mod3 and width % 3 != 0:
        width += 3 - width % 3
    out = []
    for s in seqs:
        pad = pad_char * max(0, width - len(s))
        out.append(pad + s if start else s + pad)
    return out[0] if scalar else out


# ==========================================================================
# V-region extraction
# ==========================================================================
def extractVRegion(sequences,
                   region: str | Sequence[str] = ("fwr1", "cdr1", "fwr2",
                                                  "cdr2", "fwr3")):
    """Extract FWR/CDR sub-sequences from IMGT-gapped sequences."""
    scalar_seq = isinstance(sequences, str)
    seqs = [sequences] if scalar_seq else list(sequences)
    if isinstance(region, str):
        s, e = IMGT_REGIONS[region]
        out = [str(x)[s - 1:e] for x in seqs]
        return out[0] if scalar_seq else out
    cols = {}
    for r in region:
        s, e = IMGT_REGIONS[r]
        cols[r] = [str(x)[s - 1:e] for x in seqs]
    return pd.DataFrame(cols)


# ==========================================================================
# Distance matrices
# ==========================================================================
def getDNAMatrix(gap: int = -1) -> pd.DataFrame:
    """Hamming distance matrix for IUPAC DNA characters.

    Returns a square DataFrame indexed by characters. Two characters have
    distance 0 if their IUPAC expansions intersect, else 1. Gap characters
    ``-``/``.`` carry distance ``gap`` against concrete bases.
    """
    chars = list(IUPAC_DNA.keys()) + ["-", ".", "?"]
    n = len(chars)
    sub = np.eye(n)
    keys = list(IUPAC_DNA.keys())
    for i in range(len(keys)):
        for j in range(i, len(keys)):
            same = bool(set(IUPAC_DNA[keys[i]]) & set(IUPAC_DNA[keys[j]]))
            sub[i, j] = sub[j, i] = same
    idx = {c: k for k, c in enumerate(chars)}
    gi, di = idx["-"], idx["."]
    for a in (gi, di):
        for b in (gi, di):
            sub[a, b] = 1
        for k in range(15):
            sub[a, k] = 1 - gap
            sub[k, a] = 1 - gap
    mat = 1 - sub
    return pd.DataFrame(mat, index=chars, columns=chars)


def getAAMatrix(gap: int = 0) -> pd.DataFrame:
    """Hamming distance matrix for IUPAC amino-acid characters."""
    chars = list(IUPAC_AA.keys()) + ["-", "."]
    n = len(chars)
    sub = np.eye(n)
    keys = list(IUPAC_AA.keys())
    for i in range(len(keys)):
        for j in range(i, len(keys)):
            same = bool(set(IUPAC_AA[keys[i]]) & set(IUPAC_AA[keys[j]]))
            sub[i, j] = sub[j, i] = same
    idx = {c: k for k, c in enumerate(chars)}
    gi, di = idx["-"], idx["."]
    for a in (gi, di):
        for b in (gi, di):
            sub[a, b] = 1
        for k in range(n):
            sub[a, k] = 1 - gap
            sub[k, a] = 1 - gap
    mat = 1 - sub
    return pd.DataFrame(mat, index=chars, columns=chars)


# ==========================================================================
# Sequence distance / equality
# ==========================================================================
def seqEqual(seq1: str, seq2: str,
             ignore: Iterable[str] = ("N", "-", ".", "?")) -> bool:
    """Test two DNA sequences for equality, ignoring ambiguous positions."""
    seq1, seq2 = str(seq1), str(seq2)
    if len(seq1) != len(seq2):
        return False
    ign = set(ignore)
    for c1, c2 in zip(seq1, seq2):
        if c1 != c2 and c1 not in ign and c2 not in ign:
            return False
    return True


def seqDist(seq1: str, seq2: str, dist_mat: pd.DataFrame | None = None
            ) -> float:
    """Distance between two equal-length DNA sequences.

    Faithful port of ``seqDistRcpp``: positive per-position distances are
    summed; runs of gap characters (distance ``-1``) that differ between
    the two sequences each count as a single mismatch.
    """
    if dist_mat is None:
        dist_mat = getDNAMatrix()
    seq1, seq2 = str(seq1), str(seq2)
    if len(seq1) != len(seq2):
        raise ValueError("Sequences of different length.")
    mat = dist_mat.to_numpy()
    ridx = {c: i for i, c in enumerate(dist_mat.index)}
    cidx = {c: i for i, c in enumerate(dist_mat.columns)}
    d_seen = 0
    indels = 0
    d_sum = 0.0
    for c1, c2 in zip(seq1, seq2):
        if c1 not in ridx:
            raise ValueError(f"Character {c1!r} not found in dist_mat.")
        if c2 not in cidx:
            raise ValueError(f"Character {c2!r} not found in dist_mat.")
        d_i = mat[ridx[c1], cidx[c2]]
        if d_i > 0:
            d_sum += d_i
        elif d_i == -1 and d_seen != -1:
            indels += 1
        d_seen = d_i
    return d_sum + indels


def pairwiseDist(seq, dist_mat: pd.DataFrame | None = None) -> pd.DataFrame:
    """All-pairs distance matrix for a set of DNA sequences."""
    if dist_mat is None:
        dist_mat = getDNAMatrix()
    if isinstance(seq, dict):
        names, seqs = list(seq.keys()), list(seq.values())
    elif isinstance(seq, pd.Series):
        names, seqs = list(seq.index), list(seq.values)
    else:
        seqs = list(seq)
        names = list(range(len(seqs)))
    n = len(seqs)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            d = seqDist(seqs[i], seqs[j], dist_mat)
            out[i, j] = out[j, i] = d
    return pd.DataFrame(out, index=names, columns=names)


def pairwiseEqual(seq) -> pd.DataFrame:
    """All-pairs equivalence matrix (ignores ambiguous positions)."""
    if isinstance(seq, dict):
        names, seqs = list(seq.keys()), list(seq.values())
    elif isinstance(seq, pd.Series):
        names, seqs = list(seq.index), list(seq.values)
    else:
        seqs = list(seq)
        names = list(range(len(seqs)))
    n = len(seqs)
    out = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1):
            eq = seqEqual(seqs[i], seqs[j])
            out[i, j] = out[j, i] = eq
    return pd.DataFrame(out, index=names, columns=names)


def nonsquareDist(seq, indx, dist_mat: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    """Pairwise distances between all sequences and a subset ``indx``.

    ``indx`` uses 1-based indices to mirror the R API.
    """
    if dist_mat is None:
        dist_mat = getDNAMatrix()
    if isinstance(seq, dict):
        names, seqs = list(seq.keys()), list(seq.values())
    elif isinstance(seq, pd.Series):
        names, seqs = list(seq.index), list(seq.values)
    else:
        seqs = list(seq)
        names = list(range(len(seqs)))
    idx0 = sorted(int(i) - 1 for i in indx)
    m, ncol = len(idx0), len(seqs)
    out = np.zeros((m, ncol))
    for r, ii in enumerate(idx0):
        for j in range(ncol):
            if ii == j:
                out[r, j] = 0
            else:
                out[r, j] = seqDist(seqs[ii], seqs[j], dist_mat)
    return pd.DataFrame(out, index=[names[i] for i in idx0], columns=names)


# ==========================================================================
# Duplicate collapsing
# ==========================================================================
def _informative_length(s: str) -> int:
    return len(re.sub(r"[N\-.?]", "", str(s)))


def collapseDuplicates(data: pd.DataFrame, id: str = "sequence_id",
                       seq: str = "sequence_alignment",
                       text_fields: Sequence[str] | None = None,
                       num_fields: Sequence[str] | None = None,
                       seq_fields: Sequence[str] | None = None,
                       add_count: bool = False,
                       ignore=("N", "-", ".", "?"),
                       sep: str = ",", verbose: bool = False) -> pd.DataFrame:
    """Remove duplicate DNA sequences and merge their annotations.

    Faithful port of alakazam ``collapseDuplicates`` (non-dry path).
    Duplicate clusters are sets of mutually-equivalent sequences;
    sequences that match multiple non-equivalent clusters (ambiguous)
    are discarded.
    """
    data = data.reset_index(drop=True).copy()
    text_fields = list(text_fields or [])
    num_fields = list(num_fields or [])
    seq_fields = list(seq_fields or [])

    if data[id].duplicated().any():
        raise ValueError("All values in the id column are not unique")

    if add_count:
        data["collapse_count"] = 1
        num_fields = num_fields + ["collapse_count"]

    nseq = len(data)
    if nseq <= 1:
        return data

    uniq_seqs = list(dict.fromkeys(data[seq].astype(str)))
    exact_dups = data[seq].astype(str).duplicated().any()
    d_mat = pairwiseEqual(uniq_seqs).to_numpy()
    n_uniq = d_mat.shape[0]

    lower = d_mat[np.tril_indices(n_uniq, -1)]
    if not lower.any() and not exact_dups:
        return data

    # identify ambiguous unique sequences
    ambig_rows = []
    for i in range(n_uniq):
        idx = np.where(d_mat[i])[0]
        sub = d_mat[np.ix_(idx, idx)]
        if not sub.all():
            ambig_rows.append(i)
    discard_count = len(ambig_rows)

    seq_str = data[seq].astype(str)
    ambig_seqs = {uniq_seqs[i] for i in ambig_rows}
    data_ambig = seq_str.isin(ambig_seqs).to_numpy()

    # exclude ambiguous sequences from clustering
    if discard_count > 0:
        keep_u = [i for i in range(n_uniq) if i not in ambig_rows]
        d_mat = d_mat[np.ix_(keep_u, keep_u)]
        uniq_seqs = [uniq_seqs[i] for i in keep_u]
        data = data[~data_ambig].reset_index(drop=True)
        seq_str = data[seq].astype(str)

    if len(data) == 0:
        return data

    # cluster remaining unique sequences
    n_uniq2 = d_mat.shape[0]
    done = set()
    clusters = []
    for i in range(n_uniq2):
        if i in done:
            continue
        idx = list(np.where(d_mat[i])[0])
        done.update(idx)
        clusters.append([uniq_seqs[k] for k in idx])

    out_rows = []
    for clust in clusters:
        rows = data.index[seq_str.isin(clust)].tolist()
        if len(rows) == 1:
            out_rows.append(data.loc[rows[0]].copy())
            continue
        sub = data.loc[rows]
        inform = sub[seq].astype(str).map(_informative_length)
        best = inform.idxmax()
        rec = data.loc[best].copy()
        for f in text_fields:
            vals = sub[f].dropna()
            if len(vals):
                parts = sorted(set(
                    p for v in vals for p in str(v).split(sep)))
                rec[f] = sep.join(parts)
            else:
                rec[f] = np.nan
        for f in num_fields:
            vals = sub[f].dropna()
            rec[f] = vals.sum() if len(vals) else np.nan
        for f in seq_fields:
            vals = sub[f].dropna()
            if len(vals):
                lens = vals.map(_informative_length)
                rec[f] = vals.loc[lens.idxmax()]
            else:
                rec[f] = np.nan
        out_rows.append(rec)

    result = pd.DataFrame(out_rows).reset_index(drop=True)
    if verbose:
        print(f" FUNCTION> collapseDuplicates")
        print(f" FIRST_ID> {data[id].iloc[0]}")
        print(f"    TOTAL> {nseq}")
        print(f"   UNIQUE> {len(result)}")
        print(f"COLLAPSED> {nseq - len(result) - discard_count}")
        print(f"DISCARDED> {discard_count}\n")
    return result
