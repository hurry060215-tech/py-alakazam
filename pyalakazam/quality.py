"""Per-position sequencing-quality handling (port of alakazam ``R/Fastq.R``).

Reads FASTQ Phred qualities into an AIRR-format data frame, maps them
onto the IMGT-numbered ``sequence_alignment`` positions using the
V/D/J CIGAR strings, and provides utilities for extracting and masking
low-quality positions.

* :func:`readFastqDb` --- merge FASTQ qualities into a Db.
* :func:`getPositionQuality` --- per-position long-format quality table.
* :func:`maskPositionsByQuality` --- mask low-quality positions with ``N``.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from .core import checkColumns

__all__ = ["readFastqDb", "getPositionQuality", "maskPositionsByQuality"]

_GAP_CHARS = (".", "-")


# ==========================================================================
# CIGAR parsing
# ==========================================================================
def _explode_cigar(cigar: str):
    """Split a CIGAR string into ``(ops, lengths)`` lists.

    Equivalent to ``GenomicAlignments::explodeCigarOps`` /
    ``explodeCigarOpLengths``.
    """
    ops, lengths = [], []
    for m in re.finditer(r"(\d+)([MIDNSHP=X])", str(cigar)):
        lengths.append(int(m.group(1)))
        ops.append(m.group(2))
    return ops, lengths


# ==========================================================================
# Internal: per-sequence alignment-quality calculation
# ==========================================================================
def _calc_sequence_alignment_quality(
        row, sequence, sequence_id, sequence_alignment, quality,
        quality_num, v_cigar, d_cigar, j_cigar, np1_length, np2_length,
        v_sequence_end, d_sequence_end, raw=False) -> pd.DataFrame:
    """Map raw read qualities onto IMGT alignment positions for one row.

    Faithful port of the internal alakazam ``calcSequenceAlignmentQuality``.
    """
    seq = row[sequence]
    quality_phred = list(str(row[quality]))
    quality_num_values = str(row[quality_num]).split(",")

    # build the pseudo-CIGARs for the N/P insertions
    vd_pseudo = None
    np1 = row.get(np1_length)
    if pd.notna(np1) and float(np1) > 0:
        vd_pseudo = f"{int(row[v_sequence_end])}S{int(np1)}X"
    dj_pseudo = None
    np2 = row.get(np2_length)
    if pd.notna(np2) and float(np2) > 0:
        dj_pseudo = f"{int(row[d_sequence_end])}S{int(np2)}X"

    cigars = [row.get(v_cigar), vd_pseudo, row.get(d_cigar), dj_pseudo,
              row.get(j_cigar)]
    cigars = [c for c in cigars if c is not None and pd.notna(c)]

    # accumulate read-coordinate ranges
    range_rows = []
    for cigar in cigars:
        ops, lengths = _explode_cigar(cigar)
        keep = [op not in ("N", "I") for op in ops]
        ops = [o for o, k in zip(ops, keep) if k]
        lengths = [v for v, k in zip(lengths, keep) if k]
        sub = []
        start = 1
        for i, (op, w) in enumerate(zip(ops, lengths)):
            end = None
            if op in ("S", "=", "X", "D"):
                end = start + w - 1
            sub.append({"start": start, "end": end, "width": w,
                        "operator": op})
            if end is not None:
                start = end + 1
        sub = [r for r in sub if r["operator"] not in ("S", "D")]
        range_rows.extend(sub)

    ranges = pd.DataFrame(range_rows)

    positions = []
    for _, r in ranges.iterrows():
        positions.extend(range(int(r["start"]), int(r["end"]) + 1))
    positions = np.asarray(positions, dtype=int)

    # reconstruct sequence (read coordinates -> spliced sequence)
    reconstructed = "".join(seq[p - 1] for p in positions)
    expected = re.sub(r"[.\-]", "", str(row[sequence_alignment]))
    if reconstructed != expected:
        raise ValueError("Reconstructed sequence_alignment from cigar "
                         "doesn't match db sequence_alignment.")

    qnum = np.asarray(
        [np.nan if x in ("", "NA") else float(x)
         for x in quality_num_values], dtype=float)

    quality_df = pd.DataFrame({
        "sequence_position": positions,
        "sequence_alignment_position": np.nan,
        quality: [quality_phred[p - 1] for p in positions],
        quality_num: qnum[positions - 1],
        sequence_id: row[sequence_id],
    })
    quality_df["sequence_alignment_nt"] = None

    # map sequenced positions onto the gapped alignment coordinates
    nt_aln = list(str(row[sequence_alignment]))
    pos = 0
    for aln_position, nt in enumerate(nt_aln, start=1):
        if nt not in _GAP_CHARS:
            quality_df.at[pos, "sequence_alignment_position"] = aln_position
            quality_df.at[pos, "sequence_alignment_nt"] = nt
            pos += 1

    if raw:
        return quality_df

    n = len(nt_aln)
    qual_num = np.full(n, np.nan)
    aln_idx = quality_df["sequence_alignment_position"].to_numpy(dtype=float)
    valid = ~np.isnan(aln_idx)
    qual_num[(aln_idx[valid] - 1).astype(int)] = \
        quality_df.loc[valid, quality_num].to_numpy(dtype=float)
    qual_num_str = ",".join("NA" if np.isnan(x) else _fmt_num(x)
                            for x in qual_num)

    qual_phred = [" "] * n
    for idx, ph in zip(aln_idx[valid].astype(int),
                       quality_df.loc[valid, quality].tolist()):
        qual_phred[idx - 1] = ph

    return pd.DataFrame({
        "quality_alignment_num": [qual_num_str],
        "quality_alignment": ["".join(qual_phred)],
        sequence_id: [row[sequence_id]],
    })


def _fmt_num(x: float) -> str:
    """Render a numeric quality like R (integers without a decimal point)."""
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))


def _sequence_alignment_quality(
        data, sequence_id="sequence_id", sequence="sequence",
        sequence_alignment="sequence_alignment", quality="quality",
        quality_num="quality_num", v_cigar="v_cigar", d_cigar="d_cigar",
        j_cigar="j_cigar", np1_length="np1_length", np2_length="np2_length",
        v_sequence_end="v_sequence_end", d_sequence_end="d_sequence_end",
        raw=False) -> pd.DataFrame:
    """Vectorised wrapper of :func:`_calc_sequence_alignment_quality`."""
    parts = []
    for _, row in data.iterrows():
        parts.append(_calc_sequence_alignment_quality(
            row, sequence=sequence, sequence_id=sequence_id,
            sequence_alignment=sequence_alignment, quality=quality,
            quality_num=quality_num, v_cigar=v_cigar, d_cigar=d_cigar,
            j_cigar=j_cigar, np1_length=np1_length, np2_length=np2_length,
            v_sequence_end=v_sequence_end, d_sequence_end=d_sequence_end,
            raw=raw))
    qual = pd.concat(parts, ignore_index=True)
    if raw:
        return qual
    return data.merge(qual, on=sequence_id, how="left")


# ==========================================================================
# FASTQ reading
# ==========================================================================
def _read_fastq(path, quality_offset: int = -33):
    """Read a FASTQ file -> ``(names, sequences, qualities)`` lists.

    ``qualities`` are integer Phred score lists (``ord(char) + offset``).
    Equivalent to ``ape::read.fastq`` with the given ``offset``.
    """
    names, seqs, quals = [], [], []
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    i = 0
    while i < len(lines):
        # skip blank / whitespace-only separator lines between records
        if not lines[i].strip():
            i += 1
            continue
        if not lines[i].startswith("@"):
            raise ValueError("Malformed FASTQ record.")
        names.append(lines[i][1:])
        seqs.append(lines[i + 1])
        quals.append([ord(c) + quality_offset for c in lines[i + 3]])
        i += 4
    return names, seqs, quals


def readFastqDb(data: pd.DataFrame, fastq_file, quality_offset: int = -33,
                header: str = "presto", sequence_id: str = "sequence_id",
                sequence: str = "sequence",
                sequence_alignment: str = "sequence_alignment",
                v_cigar: str = "v_cigar", d_cigar: str = "d_cigar",
                j_cigar: str = "j_cigar", np1_length: str = "np1_length",
                np2_length: str = "np2_length",
                v_sequence_end: str = "v_sequence_end",
                d_sequence_end: str = "d_sequence_end",
                style: str = "num",
                quality_sequence: bool = False) -> pd.DataFrame:
    """Add sequencing-quality columns from a FASTQ file to a Db.

    Faithful port of alakazam ``readFastqDb``. Reads per-base Phred
    qualities from ``fastq_file``, joins them to ``data`` by sequence
    ID, and maps them onto the IMGT-numbered ``sequence_alignment``
    positions via the V/D/J CIGAR strings.

    Parameters
    ----------
    data : pandas.DataFrame
        AIRR-format data frame.
    fastq_file : str or path-like
        FASTQ file containing the raw reads.
    quality_offset : int
        ASCII offset applied when decoding Phred quality scores
        (``-33`` for Sanger/Illumina 1.8+).
    header : {'presto', 'asis'}
        ``'presto'`` strips a pRESTO-style ``|...`` annotation suffix
        from the FASTQ read names; ``'asis'`` uses them unchanged.
    style : {'num', 'ascii', 'both'}
        Whether to keep the numeric quality column
        (``quality_alignment_num``), the ASCII column
        (``quality_alignment``), or both.
    quality_sequence : bool
        If ``True``, also retain the raw-read ``quality`` /
        ``quality_num`` columns.

    Returns
    -------
    pandas.DataFrame
        ``data`` with the requested quality columns appended.

    Examples
    --------
    >>> db = readFastqDb(db, "reads.fastq", style="both")    # doctest: +SKIP

    See Also
    --------
    getPositionQuality, maskPositionsByQuality
    """
    if header not in ("presto", "asis"):
        raise ValueError("header must be 'presto' or 'asis'")
    if style not in ("num", "ascii", "both"):
        raise ValueError("style must be 'num', 'ascii' or 'both'")

    check_cols = [sequence_id, sequence, sequence_alignment, v_cigar,
                  d_cigar, j_cigar, np1_length, np2_length, v_sequence_end,
                  d_sequence_end]
    chk = checkColumns(data, check_cols)
    if chk is not True:
        raise ValueError(chk)

    names, _, quals = _read_fastq(fastq_file, quality_offset=quality_offset)
    quality_num = [",".join(str(x) for x in q) for q in quals]
    quality = ["".join(chr(int(x) - quality_offset) for x in q)
               for q in quals]
    ids = [re.sub(r"\|.+", "", nm) if header == "presto" else nm
           for nm in names]

    fastq_db = pd.DataFrame({sequence_id: ids, "quality_num": quality_num,
                             "quality": quality})

    data = data.merge(fastq_db, on=sequence_id, how="left")
    data = _sequence_alignment_quality(
        data, sequence=sequence, sequence_id=sequence_id,
        sequence_alignment=sequence_alignment, quality="quality",
        quality_num="quality_num", v_cigar=v_cigar, d_cigar=d_cigar,
        j_cigar=j_cigar, np1_length=np1_length, np2_length=np2_length,
        v_sequence_end=v_sequence_end, d_sequence_end=d_sequence_end,
        raw=False)

    if not quality_sequence:
        data = data.drop(columns=[c for c in ("quality", "quality_num")
                                  if c in data.columns])
    if style != "both":
        drop = ("quality_alignment_num" if style == "ascii"
                else "quality_alignment")
        if drop in data.columns:
            data = data.drop(columns=[drop])
    return data


# ==========================================================================
# Per-position quality
# ==========================================================================
def getPositionQuality(data: pd.DataFrame, sequence_id: str = "sequence_id",
                       sequence: str = "sequence_alignment",
                       quality_num: str = "quality_alignment_num"
                       ) -> pd.DataFrame:
    """Build a long-format per-position quality table.

    Faithful port of alakazam ``getPositionQuality``. For each
    sequence, returns one row per alignment position with the
    position index, the numeric quality, the sequence ID and the
    nucleotide at that position.

    Parameters
    ----------
    data : pandas.DataFrame
        Data frame with a sequence column and a comma-separated numeric
        quality column (as produced by :func:`readFastqDb`).
    sequence_id : str
        Sequence-identifier column.
    sequence : str
        Aligned-sequence column.
    quality_num : str
        Comma-separated numeric quality column.

    Returns
    -------
    pandas.DataFrame
        Columns ``position``, ``<quality_num>``, ``<sequence_id>``, ``nt``.

    Examples
    --------
    >>> pq = getPositionQuality(db)                          # doctest: +SKIP

    See Also
    --------
    readFastqDb, maskPositionsByQuality
    """
    chk = checkColumns(data, [sequence, quality_num])
    if chk is not True:
        raise ValueError(chk)

    out = []
    for _, row in data.reset_index(drop=True).iterrows():
        seq_id = row[sequence_id]
        seq = str(row[sequence])
        seq_len = len(seq)
        qual_parts = str(row[quality_num]).split(",")
        qual_values = np.asarray(
            [np.nan if x == "NA" else float(x) for x in qual_parts],
            dtype=float)
        if seq_len != len(qual_values):
            raise ValueError(
                f"Different length, for sequence: {seq_id}. "
                f"seq: {sequence}. qual: {quality_num}")
        out.append(pd.DataFrame({
            "position": np.arange(1, seq_len + 1),
            quality_num: qual_values,
            sequence_id: seq_id,
            "nt": list(seq),
        }))
    return pd.concat(out, ignore_index=True)


def maskPositionsByQuality(data: pd.DataFrame, min_quality: float = 70,
                           sequence: str = "sequence_alignment",
                           quality_num: str = "quality_alignment_num"
                           ) -> pd.DataFrame:
    """Mask low-quality positions of a sequence with ``N``.

    Faithful port of alakazam ``maskPositionsByQuality``. Positions
    whose numeric quality is below ``min_quality`` are replaced with
    ``N``. The masked sequence is written to a new column named
    ``<sequence>_masked``.

    Parameters
    ----------
    data : pandas.DataFrame
        Data frame with a sequence column and a comma-separated numeric
        quality column.
    min_quality : float
        Minimum acceptable quality; positions below it are masked.
    sequence : str
        Aligned-sequence column.
    quality_num : str
        Comma-separated numeric quality column.

    Returns
    -------
    pandas.DataFrame
        ``data`` with an added ``<sequence>_masked`` column.

    Examples
    --------
    >>> masked = maskPositionsByQuality(db, min_quality=90)  # doctest: +SKIP

    See Also
    --------
    getPositionQuality, readFastqDb
    """
    chk = checkColumns(data, [sequence, quality_num])
    if chk is not True:
        raise ValueError(chk)

    sequence_masked = f"{sequence}_masked"
    data = data.reset_index(drop=True).copy()
    if sequence_masked not in data.columns:
        data[sequence_masked] = data[sequence]

    num_masked = 0
    for i in range(len(data)):
        seq_qual = str(data.at[i, quality_num]).split(",")
        low = [j for j, x in enumerate(seq_qual)
               if x != "NA" and float(x) < min_quality]
        if low:
            num_masked += 1
            seq = list(str(data.at[i, sequence]))
            for j in low:
                seq[j] = "N"
            data.at[i, sequence_masked] = "".join(seq)

    print(f"Number of masked sequences: {num_masked}", file=sys.stderr)
    return data
