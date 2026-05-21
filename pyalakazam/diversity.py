"""Clonal diversity and abundance analysis.

Port of alakazam ``R/Diversity.R`` --- Hill-number diversity curves with
rarefaction-style resampling and bootstrap confidence intervals, clonal
abundance estimation, sample-coverage estimators and clone tabulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

__all__ = [
    "calcCoverage", "calcDiversity", "countClones", "estimateAbundance",
    "alphaDiversity", "rarefyDiversity", "testDiversity",
    "AbundanceCurve", "DiversityCurve",
]


# ==========================================================================
# Result containers
# ==========================================================================
@dataclass
class AbundanceCurve:
    """Clonal relative abundance distribution with bootstrap CIs."""
    abundance: pd.DataFrame
    bootstrap: pd.DataFrame
    clone_by: str
    group_by: str
    groups: list
    n: dict
    nboot: int
    ci: float

    def __repr__(self):
        return (f"AbundanceCurve(groups={self.groups}, nboot={self.nboot}, "
                f"ci={self.ci})")


@dataclass
class DiversityCurve:
    """Hill diversity curve over diversity orders ``q``."""
    diversity: pd.DataFrame
    tests: pd.DataFrame | None
    method: str
    group_by: str
    groups: list
    q: np.ndarray
    n: dict
    ci: float

    def __repr__(self):
        return (f"DiversityCurve(method={self.method!r}, "
                f"groups={self.groups}, ci={self.ci})")


# ==========================================================================
# Coverage estimators
# ==========================================================================
def _chao1_coverage(x: np.ndarray) -> float:
    x = x[x >= 1]
    n = x.sum()
    f1 = np.sum(x == 1)
    f2 = np.sum(x == 2)
    if f2 > 0:
        return 1 - (f1 / n) * (((n - 1) * f1) / ((n - 1) * f1 + 2 * f2))
    return 1 - (f1 / n) * (((n - 1) * (f1 - 1)) / ((n - 1) * (f1 - 1) + 2))


def calcCoverage(x, r: int = 1) -> float:
    """Sample coverage of order ``r`` (Chao et al, 2015; Chao1 at r=1)."""
    x = np.asarray(x, dtype=float)
    if r == 1:
        return _chao1_coverage(x)
    x = x[x >= 1]
    n = x.sum()
    fr = np.sum(x == r)
    fs = np.sum(x == r + 1)
    if fr == 0:
        raise ValueError(f"No abundance data with count={r}.")
    if fs == 0:
        raise ValueError(f"No abundance data with count={r + 1}.")
    a = np.math.factorial(r) * fr / np.sum(x[x >= r] ** r)
    b = ((n - r) * fr / ((n - r) * fr + (r + 1) * fs)) ** r
    return 1 - a * b


def _infer_unseen_count(x: np.ndarray) -> int:
    x = x[x >= 1]
    n = x.sum()
    f1 = np.sum(x == 1)
    f2 = np.sum(x == 2)
    if f2 > 0:
        return int(np.ceil(((n - 1) * f1 ** 2) / (n * 2 * f2)))
    return int(np.ceil(((n - 1) * f1 * (f1 - 1)) / (n * 2)))


def _infer_unseen_abundance(x: np.ndarray) -> np.ndarray:
    x = x[x >= 1]
    rc1 = calcCoverage(x, r=1)
    f0 = _infer_unseen_count(x)
    if f0 <= 0:
        return np.array([])
    return np.repeat((1 - rc1) / f0, f0)


def _adjust_observed_abundance(x: np.ndarray) -> np.ndarray:
    x = x[x >= 1]
    n = x.sum()
    rc1 = calcCoverage(x, r=1)
    lam = (1 - rc1) / np.sum(x / n * np.exp(-x))
    return x / n * (1 - lam * np.exp(-x))


def _infer_complete_abundance(x: np.ndarray, names):
    """Return (names, probabilities) of the complete abundance distribution."""
    x = np.asarray(x, dtype=float)
    keep = x >= 1
    x_obs = x[keep]
    obs_names = list(np.asarray(names, dtype=object)[keep])
    p1 = _adjust_observed_abundance(x_obs)
    p2 = _infer_unseen_abundance(x_obs)
    u_names = [f"U{i + 1}" for i in range(len(p2))]
    return obs_names + u_names, np.concatenate([p1, p2])


# ==========================================================================
# Clone tabulation
# ==========================================================================
def countClones(data: pd.DataFrame, groups=None, copy: str | None = None,
                 clone: str = "clone_id", remove_na: bool = True
                 ) -> pd.DataFrame:
    """Tabulate clone sizes (sequence and copy counts) per group.

    Faithful port of alakazam ``countClones`` (bulk path).
    """
    if groups is None:
        groups = []
    elif isinstance(groups, str):
        groups = [groups]
    else:
        groups = list(groups)
    data = data.copy()
    if remove_na:
        na = data[clone].isna()
        if na.any():
            data = data[~na].copy()

    if copy is None:
        tab = (data.groupby(groups + [clone], dropna=False)
               .size().reset_index(name="seq_count"))
        if groups:
            tab["seq_freq"] = tab.groupby(groups)["seq_count"].transform(
                lambda s: s / s.sum())
        else:
            tab["seq_freq"] = tab["seq_count"] / tab["seq_count"].sum()
        tab = tab.sort_values("seq_count", ascending=False)
    else:
        tab = (data.groupby(groups + [clone], dropna=False)
               .agg(seq_count=(clone, "size"),
                    copy_count=(copy, "sum")).reset_index())
        if groups:
            tab["seq_freq"] = tab.groupby(groups)["seq_count"].transform(
                lambda s: s / s.sum())
            tab["copy_freq"] = tab.groupby(groups)["copy_count"].transform(
                lambda s: s / s.sum())
        else:
            tab["seq_freq"] = tab["seq_count"] / tab["seq_count"].sum()
            tab["copy_freq"] = tab["copy_count"] / tab["copy_count"].sum()
        tab = tab.sort_values("copy_count", ascending=False)
    return tab.reset_index(drop=True)


# ==========================================================================
# Abundance estimation
# ==========================================================================
def estimateAbundance(data: pd.DataFrame, clone: str = "clone_id",
                      copy: str | None = None, group: str | None = None,
                      min_n: int = 30, max_n: int | None = None,
                      uniform: bool = True, ci: float = 0.95,
                      nboot: int = 200, seed: int | None = None
                      ) -> AbundanceCurve:
    """Estimate the complete clonal relative abundance distribution.

    Faithful port of alakazam ``estimateAbundance``: clones are tabulated,
    the complete abundance distribution is inferred via the Chao1
    correction, and a multinomial bootstrap yields confidence intervals.

    Note
    ----
    The bootstrap draws (multinomial resampling) use NumPy's RNG, which
    differs from R's RNG; bootstrap point estimates converge to the same
    values but realisation-level numbers are not bit-identical.
    """
    rng = np.random.default_rng(seed)
    ci_z = ci + (1 - ci) / 2
    ci_x = norm.ppf(ci_z)

    count_col = "copy_count" if copy is not None else "seq_count"
    clone_tab = countClones(data, copy=copy, clone=clone,
                            groups=[group] if group else None)
    clone_tab = clone_tab.copy()
    clone_tab["clone_count"] = clone_tab[count_col]

    if group is not None:
        group_tab = (clone_tab.groupby(group, dropna=False)["clone_count"]
                     .sum().reset_index(name="count")
                     .rename(columns={group: "group"}))
    else:
        group_tab = pd.DataFrame({"group": ["All"],
                                  "count": [clone_tab["clone_count"].sum()]})
    group_all = group_tab["group"].astype(str).tolist()
    group_tab = group_tab[group_tab["count"] >= min_n]
    group_keep = group_tab["group"].astype(str).tolist()

    if uniform:
        cand = list(group_tab["count"])
        if max_n is not None:
            cand.append(max_n)
        nval = min(cand) if cand else 0
        nsam = {g: int(nval) for g in group_keep}
    else:
        nsam = {}
        for g, c in zip(group_keep, group_tab["count"]):
            nsam[g] = int(min(c, max_n) if max_n is not None else c)

    abund_list = []
    boot_list = []
    for g in group_keep:
        n = nsam[g]
        if group is not None:
            mask = clone_tab[group].astype(str) == g
        else:
            mask = pd.Series(True, index=clone_tab.index)
        abund_obs = clone_tab.loc[mask, "clone_count"].to_numpy(dtype=float)
        obs_names = clone_tab.loc[mask, clone].tolist()

        names, p = _infer_complete_abundance(abund_obs, obs_names)
        p = p / p.sum()
        boot_mat = rng.multinomial(n, p, size=nboot).T / n  # rows=clones

        p_mean = boot_mat.mean(axis=1)
        p_sd = boot_mat.std(axis=1, ddof=1)
        p_err = ci_x * p_sd
        p_lower = np.maximum(p_mean - p_err, 0)
        p_upper = p_mean + p_err

        abund_df = pd.DataFrame({clone: names, "p": p_mean, "p_sd": p_sd,
                                 "lower": p_lower, "upper": p_upper})
        abund_df = abund_df.sort_values("p", ascending=False).reset_index(
            drop=True)
        abund_df["rank"] = np.arange(1, len(abund_df) + 1)
        abund_df.insert(0, "group", g)
        abund_list.append(abund_df)

        boot_df = pd.DataFrame(boot_mat,
                               columns=[f"V{i + 1}" for i in range(nboot)])
        boot_df.insert(0, clone, names)
        boot_df.insert(0, "group", g)
        boot_list.append(boot_df)

    id_col = group if group is not None else "group"
    abundance_df = pd.concat(abund_list, ignore_index=True)
    bootstrap_df = pd.concat(boot_list, ignore_index=True)
    abundance_df = abundance_df.rename(columns={"group": id_col})
    bootstrap_df = bootstrap_df.rename(columns={"group": id_col})

    return AbundanceCurve(abundance=abundance_df, bootstrap=bootstrap_df,
                          clone_by=clone, group_by=id_col, groups=group_keep,
                          n=nsam, nboot=nboot, ci=ci)


# ==========================================================================
# Diversity
# ==========================================================================
def calcDiversity(p, q):
    """Hill diversity index for clone counts/proportions ``p`` and orders ``q``.

    Faithful port of alakazam ``calcDiversity``.
    """
    q = np.asarray(q, dtype=float).copy()
    q[q == 1] = 0.9999
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    p = p / p.sum()
    out = np.array([np.sum(p ** x) ** (1 / (1 - x)) for x in q])
    return out


def _infer_rarefied_diversity(x, q, m):
    """Hill numbers under rarefaction to ``m`` sequences."""
    x = np.asarray(x, dtype=float)
    x = x[x >= 1].astype(int)
    n = int(x.sum())
    if m > n:
        raise ValueError("m must be <= total observed sequences.")
    q = np.asarray(q, dtype=float).copy()
    q[q == 1] = 0.9999
    fk_n = np.bincount(x, minlength=n + 1)[1:n + 1]

    def lchoose(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)

    fk_m = np.zeros(m)
    for k in range(1, m + 1):
        ks = np.arange(k, m + 1)
        terms = np.exp(lchoose(ks, k) + lchoose(n - ks, m - k)
                       - lchoose(n, m)) * fk_n[ks - 1]
        fk_m[k - 1] = terms.sum()
    rel = np.arange(1, m + 1) / m
    return np.array([np.sum(rel ** r * fk_m) ** (1 / (1 - r)) for r in q])


def _helper_alpha(boot_mat: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Diversity of each bootstrap realisation. boot_mat rows=clones."""
    return np.column_stack([calcDiversity(boot_mat[:, j], q)
                            for j in range(boot_mat.shape[1])])


def alphaDiversity(data, min_q: float = 0, max_q: float = 4,
                   step_q: float = 0.1, ci: float = 0.95, **kwargs
                   ) -> DiversityCurve:
    """Compute the clonal alpha-diversity (Hill) curve with bootstrap CIs.

    ``data`` may be a DataFrame (passed to :func:`estimateAbundance`) or
    a pre-computed :class:`AbundanceCurve`. Faithful port of alakazam
    ``alphaDiversity``.
    """
    if isinstance(data, AbundanceCurve):
        abundance = data
    elif isinstance(data, pd.DataFrame):
        abundance = estimateAbundance(data, ci=0.95, **kwargs)
    else:
        raise TypeError("data must be a DataFrame or AbundanceCurve.")

    ci_z = ci + (1 - ci) / 2
    ci_x = norm.ppf(ci_z)
    q = np.arange(min_q, max_q + step_q / 2, step_q)
    if 0 not in q:
        q = np.concatenate([[0], q])

    clone = abundance.clone_by
    group = abundance.group_by
    boot = abundance.bootstrap
    boot_cols = [c for c in boot.columns if c not in (clone, group)]

    div_rows = []
    boot_div = {}  # group -> array (n_q, nboot)
    for g in abundance.groups:
        sub = boot[boot[group].astype(str) == g]
        bm = sub[boot_cols].to_numpy(dtype=float)
        dvals = _helper_alpha(bm, q)  # (n_q, nboot)
        boot_div[g] = dvals
        d = dvals.mean(axis=1)
        d_sd = dvals.std(axis=1, ddof=1)
        d_lower = np.maximum(d - d_sd * ci_x, 0)
        d_upper = d + d_sd * ci_x
        for i, qi in enumerate(q):
            div_rows.append({group: g, "q": qi, "d": d[i], "d_sd": d_sd[i],
                             "d_lower": d_lower[i], "d_upper": d_upper[i]})
    div_df = pd.DataFrame(div_rows)

    # evenness = D / D(q=0)
    d0 = div_df[div_df["q"] == 0].set_index(group)["d"]
    div_df["e"] = div_df.apply(
        lambda r: r["d"] / d0[r[group]], axis=1)
    div_df["e_lower"] = div_df.apply(
        lambda r: r["d_lower"] / d0[r[group]], axis=1)
    div_df["e_upper"] = div_df.apply(
        lambda r: r["d_upper"] / d0[r[group]], axis=1)

    test_df = None
    if len(abundance.groups) > 1:
        test_df = _helper_test(boot_div, q, abundance.groups)

    return DiversityCurve(diversity=div_df, tests=test_df, method="alpha",
                          group_by=group, groups=list(abundance.groups),
                          q=q, n=abundance.n, ci=ci)


def _helper_test(boot_div: dict, q: np.ndarray, groups) -> pd.DataFrame:
    """Permutation-free bootstrap delta test between groups, per ``q``."""
    rows = []
    for ga, gb in combinations(groups, 2):
        for i, qi in enumerate(q):
            m1 = boot_div[ga][i]
            m2 = boot_div[gb][i]
            if m1.mean() >= m2.mean():
                delta = m1 - m2
            else:
                delta = m2 - m1
            # ECDF at 0
            p = np.mean(delta <= 0)
            p = p * 2 if p <= 0.5 else (1 - p) * 2
            rows.append({"test": f"{ga} != {gb}", "q": qi,
                         "delta_mean": delta.mean(),
                         "delta_sd": delta.std(ddof=1), "pvalue": p})
    return pd.DataFrame(rows)


def testDiversity(data, q, group: str, **kwargs) -> DiversityCurve:
    """Compute diversity and test group differences at a fixed order ``q``.

    Convenience wrapper around :func:`alphaDiversity` that mirrors the
    behaviour of alakazam's (deprecated) ``testDiversity``.
    """
    div = alphaDiversity(data, min_q=q, max_q=q, step_q=1, group=group,
                         **kwargs)
    return div


def rarefyDiversity(data, min_q: float = 0, max_q: float = 4,
                    step_q: float = 0.1, ci: float = 0.95, **kwargs
                    ) -> DiversityCurve:
    """Rarefaction-based diversity curve (alias of :func:`alphaDiversity`).

    In modern alakazam ``rarefyDiversity`` is deprecated in favour of
    ``alphaDiversity``; the uniform-resampling estimator is identical.
    """
    return alphaDiversity(data, min_q=min_q, max_q=max_q, step_q=step_q,
                          ci=ci, **kwargs)
