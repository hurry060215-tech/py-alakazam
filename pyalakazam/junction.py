"""Junction-region alignment annotation (port of alakazam ``R/Junction.R``).

:func:`junctionAlignment` determines the number of deleted germline
nucleotides in the junction region and the number of V- and J-gene
nucleotides contributed to the CDR3.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .core import checkColumns
from .gene import getAllele

__all__ = ["junctionAlignment"]

_GAP_CHARS = (".", "-")
_JUNCTION_START = 310  # 1-based IMGT position where the junction begins


def _count_deleted(row, allele_call, germline_start, germline_end,
                   germline_db, junction, junction_length,
                   sequence_alignment):
    """Per-row deletion / CDR3-contribution counts for one segment.

    Faithful port of the internal alakazam ``countDeleted``. Returns
    ``[deleted_head, deleted_tail, germ_cdr3_length]``.
    """
    allele = getAllele(row[allele_call], first=True)
    deleted = [np.nan, np.nan, np.nan]
    if allele is None or (isinstance(allele, float) and np.isnan(allele)):
        return deleted

    if allele not in germline_db:
        raise KeyError(f"{allele} not found in germline_db.")
    germline = germline_db[allele]

    g_start = float(row[germline_start])
    g_end = float(row[germline_end])

    # nucleotides deleted from the head/tail of the germline alignment
    germline_head = germline[:int(g_start) - 1]
    deleted_head = len(germline_head.replace(".", ""))
    germline_tail = germline[int(g_end):]
    deleted_tail = len(germline_tail.replace(".", ""))
    deleted[0] = deleted_head
    deleted[1] = deleted_tail

    junc = row[junction]
    if junc is None or (isinstance(junc, float) and np.isnan(junc)):
        return deleted

    junction_len = row[junction_length]
    if not (junction_len is not None and float(junction_len) > 6):
        return deleted
    junction_len = int(junction_len)

    # locate the (ungapped) end of the junction within the alignment
    seq_aln = str(row[sequence_alignment])
    non_gap = np.array([c not in _GAP_CHARS for c in seq_aln], dtype=int)
    non_gap[: _JUNCTION_START - 1] = 0
    cum = np.cumsum(non_gap)
    over = np.where(cum > junction_len)[0]
    junction_end = (int(over[0]) + 1) - 1 if len(over) else len(seq_aln)

    germ_cdr3_length = np.nan
    if re.search(r"[Vv]", allele):
        last_cdr3_pre_np = int(g_end) - int(g_start) + 1
        first_cdr3_pre_np = _JUNCTION_START + 3
        germ_seq = seq_aln[first_cdr3_pre_np - 1:last_cdr3_pre_np]
        germ_cdr3_length = len(re.sub(r"[.\-]", "", germ_seq))
    elif re.search(r"[Jj]", allele):
        j_aln_len = int(g_end) - int(g_start) + 1
        start = len(seq_aln) - j_aln_len + 1
        germ_seq = seq_aln[start - 1:junction_end - 3]
        germ_cdr3_length = len(germ_seq.replace("-", ""))

    return [deleted_head, deleted_tail, germ_cdr3_length]


def junctionAlignment(data: pd.DataFrame, germline_db: dict,
                      v_call: str = "v_call", d_call: str = "d_call",
                      j_call: str = "j_call",
                      v_germline_start: str = "v_germline_start",
                      v_germline_end: str = "v_germline_end",
                      d_germline_start: str = "d_germline_start",
                      d_germline_end: str = "d_germline_end",
                      j_germline_start: str = "j_germline_start",
                      j_germline_end: str = "j_germline_end",
                      np1_length: str = "np1_length",
                      np2_length: str = "np2_length",
                      junction: str = "junction",
                      junction_length: str = "junction_length",
                      sequence_alignment: str = "sequence_alignment"
                      ) -> pd.DataFrame:
    """Calculate junction-region alignment properties.

    Faithful port of alakazam ``junctionAlignment``. Determines the
    number of deleted germline nucleotides in the junction region and
    the number of V- and J-gene nucleotides contributed to the CDR3.

    Parameters
    ----------
    data : pandas.DataFrame
        AIRR-format data frame with V/D/J calls, germline start/end
        coordinates, N/P-region lengths, junction and alignment columns.
    germline_db : dict
        ``{allele: germline_sequence}`` reference database for the
        V, D and J genes referenced in ``data``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``data`` with six additional columns:

        * ``e3v_length`` --- 3' V germline nucleotides deleted.
        * ``e5d_length`` --- 5' D germline nucleotides deleted.
        * ``e3d_length`` --- 3' D germline nucleotides deleted.
        * ``e5j_length`` --- 5' J germline nucleotides deleted.
        * ``v_cdr3_length`` --- V nucleotides in the CDR3.
        * ``j_cdr3_length`` --- J nucleotides in the CDR3.

    Examples
    --------
    >>> germline_db = {"IGHV3-11*05": "...", "IGHD3-10*01": "...",
    ...                "IGHJ5*02": "..."}                    # doctest: +SKIP
    >>> db = junctionAlignment(SingleDb, germline_db)        # doctest: +SKIP
    """
    chk = checkColumns(data, [
        v_call, d_call, j_call, v_germline_start, v_germline_end,
        d_germline_start, d_germline_end, j_germline_start, j_germline_end,
        np1_length, np2_length, junction, junction_length,
        sequence_alignment])
    if chk is not True:
        raise ValueError(chk)

    data = data.reset_index(drop=True).copy()
    for col in ("e3v_length", "e5d_length", "e3d_length", "e5j_length",
                "v_cdr3_length", "j_cdr3_length"):
        if col not in data.columns:
            data[col] = np.nan

    for i in range(len(data)):
        row = data.iloc[i]
        v_dels = _count_deleted(
            row, allele_call=v_call, germline_start=v_germline_start,
            germline_end=v_germline_end, germline_db=germline_db,
            junction=junction, junction_length=junction_length,
            sequence_alignment=sequence_alignment)
        d_dels = _count_deleted(
            row, allele_call=d_call, germline_start=d_germline_start,
            germline_end=d_germline_end, germline_db=germline_db,
            junction=junction, junction_length=junction_length,
            sequence_alignment=sequence_alignment)
        j_dels = _count_deleted(
            row, allele_call=j_call, germline_start=j_germline_start,
            germline_end=j_germline_end, germline_db=germline_db,
            junction=junction, junction_length=junction_length,
            sequence_alignment=sequence_alignment)
        data.at[i, "e3v_length"] = v_dels[1]
        data.at[i, "e5d_length"] = d_dels[0]
        data.at[i, "e3d_length"] = d_dels[1]
        data.at[i, "e5j_length"] = j_dels[0]
        data.at[i, "v_cdr3_length"] = v_dels[2]
        data.at[i, "j_cdr3_length"] = j_dels[2]

    return data
