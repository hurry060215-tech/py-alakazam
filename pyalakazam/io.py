"""Change-O database I/O (port of alakazam ``R/Changeo.R``).

The Change-O tab-delimited database format is the AIRR-precursor TSV
written by older Immcantation tools (Change-O, IgBLAST/MakeDb pipelines).
This module provides :func:`readChangeoDb` / :func:`writeChangeoDb` for
reading and writing that format.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["readChangeoDb", "writeChangeoDb"]

# --------------------------------------------------------------------------
# Sequence columns whose values are upper-cased when ``seq_upper=True``.
# Faithful copy of the ``seq_columns`` vector in alakazam ``readChangeoDb``.
# --------------------------------------------------------------------------
_SEQ_COLUMNS = (
    "SEQUENCE_INPUT", "SEQUENCE_VDJ", "SEQUENCE_IMGT", "JUNCTION",
    "JUNCTION_AA", "CDR3_IGBLAST_NT", "CDR3_IGBLAST_AA", "GERMLINE_VDJ",
    "GERMLINE_VDJ_V_REGION", "GERMLINE_VDJ_D_MASK", "GERMLINE_IMGT",
    "GERMLINE_IMGT_V_REGION", "GERMLINE_IMGT_D_MASK", "FWR1_IMGT",
    "FWR2_IMGT", "FWR3_IMGT", "FWR4_IMGT", "CDR1_IMGT", "CDR2_IMGT",
    "CDR3_IMGT",
)

# Strings interpreted as missing values (matches alakazam ``na`` argument).
_NA_VALUES = ("", "NA", "None")


def readChangeoDb(file, select=None, drop=None, seq_upper: bool = True
                  ) -> pd.DataFrame:
    """Read a Change-O tab-delimited database file.

    Faithful port of alakazam ``readChangeoDb``. Reads the legacy
    Change-O ``.tab`` database (the AIRR-precursor TSV) into a
    :class:`pandas.DataFrame`.

    Parameters
    ----------
    file : str or path-like
        Path to a tab-delimited database file.
    select : list of str, optional
        Columns to keep; if ``None`` all columns are kept.
    drop : list of str, optional
        Columns to remove; applied after ``select``.
    seq_upper : bool
        If ``True`` (default), sequence columns are converted to
        upper case.

    Returns
    -------
    pandas.DataFrame
        The parsed database.

    Examples
    --------
    >>> db = readChangeoDb("example_changeo.tab")            # doctest: +SKIP
    >>> db = readChangeoDb("example.tab", select=["SEQUENCE_ID", "V_CALL"])
    ... # doctest: +SKIP

    See Also
    --------
    writeChangeoDb : write a database back to disk.
    """
    db = pd.read_csv(file, sep="\t", dtype=str, na_values=list(_NA_VALUES),
                     keep_default_na=False)

    select_columns = list(db.columns)
    if select is not None:
        select = [select] if isinstance(select, str) else list(select)
        select_columns = [c for c in select_columns if c in select]
    if drop is not None:
        drop = [drop] if isinstance(drop, str) else list(drop)
        select_columns = [c for c in select_columns if c not in drop]
    db = db[select_columns]

    if seq_upper:
        upper_cols = [c for c in _SEQ_COLUMNS if c in db.columns]
        for c in upper_cols:
            db[c] = db[c].map(lambda x: x.upper() if isinstance(x, str)
                              else x)
    return db


def writeChangeoDb(data: pd.DataFrame, file) -> None:
    """Write a Change-O tab-delimited database file.

    Faithful port of alakazam ``writeChangeoDb``. Writes a
    :class:`pandas.DataFrame` to a tab-delimited file, with missing
    values rendered as the string ``NA``.

    Parameters
    ----------
    data : pandas.DataFrame
        Database to write.
    file : str or path-like
        Output path.

    Examples
    --------
    >>> writeChangeoDb(db, "out.tab")                        # doctest: +SKIP

    See Also
    --------
    readChangeoDb : read a database from disk.
    """
    data.to_csv(file, sep="\t", na_rep="NA", index=False)
