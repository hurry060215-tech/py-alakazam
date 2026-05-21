"""Common utilities (port of alakazam ``R/Core.R``).

String translation, column validation and the Stouffer meta-analysis.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["translateStrings", "checkColumns", "stoufferMeta"]


def translateStrings(strings, translation: dict):
    """Substitute whole-string values according to a translation dict.

    ``translation`` maps replacement -> value(s) to replace; a value is
    only substituted when it matches a complete string entry. Faithful
    port of alakazam ``translateStrings``.
    """
    scalar = isinstance(strings, str)
    vals = [strings] if scalar else list(strings)
    out = list(vals)
    for new, old in translation.items():
        olds = old if isinstance(old, (list, tuple)) else [old]
        pat = re.compile("^(" + "|".join(re.escape(o) for o in olds) + ")$")
        out = [new if (isinstance(s, str) and pat.match(s)) else s
               for s in out]
    return out[0] if scalar else out


def checkColumns(data: pd.DataFrame, columns, logic: str = "all"):
    """Validate that ``columns`` exist and contain non-NA data.

    Returns ``True`` on success, otherwise a descriptive message string
    (matching alakazam ``checkColumns``).
    """
    if logic not in ("all", "any"):
        raise ValueError("logic must be 'all' or 'any'")
    columns = [c for c in (columns if isinstance(columns, (list, tuple))
                           else [columns]) if c is not None]
    names = list(data.columns)

    if logic == "all":
        for f in columns:
            if f not in names:
                return f"The column {f} was not found"
        for f in columns:
            if data[f].isna().all():
                return f"The column {f} contains no data"
    else:
        if not any(c in names for c in columns):
            return ("Input must contain at least one of the columns: "
                    + ", ".join(columns))
        found = [c for c in columns if c in names]
        if all(data[c].isna().all() for c in found):
            return ("None of the columns " + ", ".join(found)
                    + " contain data")
    return True


def stoufferMeta(p, w=None) -> dict:
    """Weighted meta-analysis of p-values via Stouffer's Z-score method.

    Faithful port of alakazam ``stoufferMeta``.
    """
    p = np.asarray(p, dtype=float)
    if w is None:
        w = np.repeat(1.0, len(p)) / len(p)
    else:
        w = np.asarray(w, dtype=float)
        if len(w) != len(p):
            raise ValueError("Length of p and w must be equal.")
        w = w / w.sum()
    x = norm.ppf(1 - p)
    z = np.sum(w * x) / np.sqrt(np.sum(w ** 2))
    pvalue = 1 - norm.cdf(z)
    return {"Z": z, "pvalue": pvalue}
