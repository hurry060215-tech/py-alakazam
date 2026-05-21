"""R-parity tests --- pyalakazam vs R/CRAN alakazam 1.4.3.

The R driver (:file:`r_reference_driver.R`) runs alakazam on its bundled
``ExampleDb`` / ``ExampleTrees`` datasets --- the exact same data that
pyalakazam ships --- and writes numeric results to TSVs.

Deterministic functions (gene parsing, ``countGenes``, ``countClones``,
``aminoAcidProperties``, ``translateDNA``, sequence distances, topology
analysis) are checked to bit-exact / rel-diff < 1e-6 agreement.

Bootstrap functions (``estimateAbundance``, ``alphaDiversity``) use
multinomial resampling: R and NumPy RNGs differ, so the point estimates
are required to agree to within a few percent, not bit-exact.

Tests skip gracefully when the CMAP R env or alakazam is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pyalakazam as ak

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'suppressMessages(library(alakazam)); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or alakazam not installed.",
)


@pytest.fixture(scope="module")
def r_ref(tmp_path_factory):
    """Run the alakazam R reference once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("alakazam_R")
    cmd = (f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
           f"&& Rscript {R_DRIVER} {out_dir}")
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                         text=True, timeout=1200)
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


@pytest.fixture(scope="module")
def db():
    return ak.load_example_db()


@pytest.fixture(scope="module")
def trees():
    return ak.load_example_trees()


# ======================================================================
# Gene annotation --- bit-exact
# ======================================================================
def test_get_gene_calls_exact(r_ref, db):
    r = pd.read_csv(r_ref / "gene_calls.tsv", sep="\t", dtype=str,
                    keep_default_na=False)
    for col, fn in (("gene", ak.getGene), ("family", ak.getFamily),
                    ("allele", ak.getAllele), ("locus", ak.getLocus),
                    ("chain", ak.getChain)):
        py = [str(x) for x in fn(db["v_call"].tolist())]
        agree = np.mean([a == b for a, b in zip(py, r[col].tolist())])
        assert agree == 1.0, f"{col}: agreement {agree:.4f}"


def test_count_genes_exact(r_ref, db):
    r = pd.read_csv(r_ref / "count_genes.tsv", sep="\t")
    py = ak.countGenes(db, gene="v_call", groups="sample_id", mode="family")
    rk = r.set_index(["sample_id", "gene"]).sort_index()
    pk = py.set_index(["sample_id", "gene"]).sort_index()
    common = rk.index.intersection(pk.index)
    assert len(common) == len(rk)
    assert np.allclose(rk.loc[common, "seq_count"],
                       pk.loc[common, "seq_count"])
    assert np.allclose(rk.loc[common, "seq_freq"],
                       pk.loc[common, "seq_freq"], rtol=1e-9)


def test_count_genes_gene_mode_exact(r_ref, db):
    r = pd.read_csv(r_ref / "count_genes_gene.tsv", sep="\t")
    py = ak.countGenes(db, gene="v_call", mode="gene")
    rk = r.set_index("gene")["seq_count"].sort_index()
    pk = py.set_index("gene")["seq_count"].sort_index()
    common = rk.index.intersection(pk.index)
    assert len(common) == len(rk)
    assert np.allclose(rk.loc[common], pk.loc[common])


def _canonical_partition(labels):
    """Map a label Series to its partition (set of frozensets of members).

    Group label *names* differ between R and Python (component
    enumeration order), so equality is tested on the partition itself.
    """
    parts = {}
    for member, lab in labels.items():
        parts.setdefault(lab, set()).add(member)
    return frozenset(frozenset(m) for m in parts.values())


def test_group_genes_partition_exact(r_ref, db):
    """groupGenes must partition sequences identically to R."""
    r = pd.read_csv(r_ref / "group_genes.tsv", sep="\t", dtype=str)
    py = ak.groupGenes(db)
    r_lab = r.set_index("sequence_id")["vj_group"].astype(str)
    py_lab = py.set_index("sequence_id")["vj_group"].astype(str)
    assert set(py_lab.index) == set(r_lab.index)
    assert _canonical_partition(py_lab) == \
        _canonical_partition(r_lab.loc[py_lab.index])


def test_group_genes_junclen_partition_exact(r_ref, db):
    """groupGenes with junc_len must partition identically to R."""
    r = pd.read_csv(r_ref / "group_genes_junclen.tsv", sep="\t", dtype=str)
    py = ak.groupGenes(db, junc_len="junction_length")
    r_lab = r.set_index("sequence_id")["vj_group"].astype(str)
    py_lab = py.set_index("sequence_id")["vj_group"].astype(str)
    assert set(py_lab.index) == set(r_lab.index)
    assert _canonical_partition(py_lab) == \
        _canonical_partition(r_lab.loc[py_lab.index])


def test_count_clones_exact(r_ref, db):
    r = pd.read_csv(r_ref / "count_clones.tsv", sep="\t")
    py = ak.countClones(db, groups="sample_id")
    rk = r.set_index(["sample_id", "clone_id"]).sort_index()
    pk = py.set_index(["sample_id", "clone_id"]).sort_index()
    common = rk.index.intersection(pk.index)
    assert len(common) == len(rk)
    assert np.allclose(rk.loc[common, "seq_count"],
                       pk.loc[common, "seq_count"])
    assert np.allclose(rk.loc[common, "seq_freq"],
                       pk.loc[common, "seq_freq"], rtol=1e-9)


# ======================================================================
# Amino-acid properties --- rel-diff < 1e-6
# ======================================================================
def test_amino_acid_properties_exact(r_ref, db):
    r = pd.read_csv(r_ref / "aa_properties.tsv", sep="\t")
    py = ak.aminoAcidProperties(db[["sequence_id", "junction"]],
                                seq="junction")
    r = r.set_index("sequence_id")
    py = py.set_index("sequence_id")
    common = r.index.intersection(py.index)
    assert len(common) == len(r)
    for col in ("junction_aa_length", "junction_aa_gravy", "junction_aa_bulk",
                "junction_aa_aliphatic", "junction_aa_polarity",
                "junction_aa_charge", "junction_aa_basic",
                "junction_aa_acidic", "junction_aa_aromatic"):
        rv = r.loc[common, col].to_numpy(dtype=float)
        pv = py.loc[common, col].to_numpy(dtype=float)
        mask = np.isfinite(rv) & np.isfinite(pv)
        assert mask.sum() > 100
        assert np.allclose(rv[mask], pv[mask], rtol=1e-6, atol=1e-9), \
            f"{col} mismatch"
        # NaN positions must agree
        assert np.array_equal(np.isnan(rv), np.isnan(pv)), f"{col} NaN"


def test_translate_dna_exact(r_ref, db):
    r = pd.read_csv(r_ref / "translate.tsv", sep="\t", dtype=str,
                    keep_default_na=False)
    py_aa = [str(x) for x in ak.translateDNA(db["junction"].tolist())]
    py_cdr3 = [str(x) for x in
               ak.translateDNA(db["junction"].tolist(), trim=True)]
    assert py_aa == r["junction_aa"].tolist()
    assert py_cdr3 == r["cdr3_aa"].tolist()


# ======================================================================
# Sequence distances --- bit-exact
# ======================================================================
def test_pairwise_dist_exact(r_ref, db):
    r = pd.read_csv(r_ref / "pairwise_dist.tsv", sep="\t", index_col=0)
    r.index = r.index.astype(str)
    r.columns = r.columns.astype(str)
    seqs = ak.padSeqEnds(db["junction"].iloc[:25].tolist())
    seq_map = dict(zip(db["sequence_id"].iloc[:25].astype(str), seqs))
    pd_mat = ak.pairwiseDist(seq_map)
    common = [c for c in r.columns if c in pd_mat.columns]
    assert len(common) == 25
    rv = r.loc[common, common].to_numpy()
    pv = pd_mat.loc[common, common].to_numpy()
    assert np.allclose(rv, pv, atol=1e-9)


def test_coverage_exact(r_ref, db):
    r = pd.read_csv(r_ref / "coverage.tsv", sep="\t")
    clones = ak.countClones(db, groups="sample_id")
    for _, row in r.iterrows():
        sid = row["sample_id"]
        counts = clones[clones["sample_id"] == sid]["seq_count"].to_numpy()
        cov = ak.calcCoverage(counts)
        assert abs(cov - row["coverage"]) < 1e-9, f"{sid}: {cov}"


# ======================================================================
# Topology --- bit-exact
# ======================================================================
def test_path_lengths_exact(r_ref, trees):
    r = pd.read_csv(r_ref / "path_lengths.tsv", sep="\t")
    py = ak.getPathLengths(trees[22], root="Germline")
    rk = r.set_index("name").sort_index()
    pk = py.set_index("name").sort_index()
    common = rk.index.intersection(pk.index)
    assert len(common) == len(rk)
    assert np.allclose(rk.loc[common, "steps"].fillna(-1),
                       pk.loc[common, "steps"].fillna(-1))
    assert np.allclose(rk.loc[common, "distance"].fillna(-1),
                       pk.loc[common, "distance"].fillna(-1), atol=1e-9)


def test_subtrees_exact(r_ref, trees):
    r = pd.read_csv(r_ref / "subtrees.tsv", sep="\t")
    py = ak.summarizeSubtrees(trees[22], fields=["c_call"], root="Germline")
    rk = r.set_index("name").sort_index()
    pk = py.set_index("name").sort_index()
    common = rk.index.intersection(pk.index)
    assert len(common) == len(rk)
    for col in ("outdegree", "size", "depth", "pathlength"):
        assert np.allclose(rk.loc[common, col], pk.loc[common, col],
                           atol=1e-9), f"{col} mismatch"


def test_table_edges_exact(r_ref, trees):
    r = pd.read_csv(r_ref / "table_edges.tsv", sep="\t")
    py = ak.tableEdges(trees[22], "c_call", exclude=["Germline", np.nan])
    rk = r.set_index(["parent", "child"])["count"].sort_index()
    pk = py.set_index(["parent", "child"])["count"].sort_index()
    common = rk.index.intersection(pk.index)
    assert len(common) == len(rk)
    assert np.array_equal(rk.loc[common].to_numpy(),
                          pk.loc[common].to_numpy())


# ======================================================================
# Diversity / abundance --- bootstrap, point estimates within a few %
# ======================================================================
def test_estimate_abundance_point_estimates(r_ref, db):
    r = pd.read_csv(r_ref / "abundance.tsv", sep="\t")
    abund = ak.estimateAbundance(db, group="sample_id", nboot=200, seed=42)
    py = abund.abundance
    # the inferred "U*" unseen clones are unidentifiable across RNGs;
    # compare only the observed clones (matched by clone_id).
    r = r.copy()
    py = py.copy()
    r["clone_id"] = r["clone_id"].astype(str)
    py["clone_id"] = py["clone_id"].astype(str)
    r_obs = r[~r["clone_id"].str.startswith("U")]
    py_obs = py[~py["clone_id"].str.startswith("U")]
    rk = r_obs.set_index(["sample_id", "clone_id"])
    pk = py_obs.set_index(["sample_id", "clone_id"])
    common = rk.index.intersection(pk.index)
    assert len(common) > 100
    rho, _ = pearsonr(rk.loc[common, "p"], pk.loc[common, "p"])
    assert rho > 0.999, f"abundance p Pearson r = {rho:.5f}"
    # mean absolute relative error on sizeable clones
    big = rk.loc[common, "p"] > 0.005
    rel = np.abs(rk.loc[common, "p"][big] - pk.loc[common, "p"][big]) / \
        rk.loc[common, "p"][big]
    assert rel.mean() < 0.05, f"mean rel err {rel.mean():.4f}"


def test_alpha_diversity_point_estimates(r_ref, db):
    r = pd.read_csv(r_ref / "diversity.tsv", sep="\t")
    div = ak.alphaDiversity(db, group="sample_id", min_q=0, max_q=4,
                            step_q=1, nboot=200, seed=42)
    py = div.diversity
    rk = r.set_index(["sample_id", "q"]).sort_index()
    pk = py.set_index(["sample_id", "q"]).sort_index()
    common = rk.index.intersection(pk.index)
    assert len(common) == len(rk)
    rv = rk.loc[common, "d"].to_numpy()
    pv = pk.loc[common, "d"].to_numpy()
    rel = np.abs(rv - pv) / rv
    assert rel.max() < 0.05, f"diversity D max rel err {rel.max():.4f}"
    # confidence intervals within a few percent
    rv_lo = rk.loc[common, "d_lower"].to_numpy()
    pv_lo = pk.loc[common, "d_lower"].to_numpy()
    rel_lo = np.abs(rv_lo - pv_lo) / np.maximum(rv_lo, 1.0)
    assert rel_lo.max() < 0.10, f"d_lower max rel err {rel_lo.max():.4f}"


def test_calc_diversity_exact():
    """calcDiversity is deterministic --- must match R formula exactly."""
    p = np.array([1, 1, 3, 10], dtype=float)
    q = np.array([0, 1, 2], dtype=float)
    d = ak.calcDiversity(p, q)
    # richness, exp(Shannon)@0.9999, inverse-Simpson
    assert abs(d[0] - 4.0) < 1e-9
    assert abs(d[2] - (1 / np.sum((p / p.sum()) ** 2))) < 1e-6


# ======================================================================
# Change-O database I/O --- bit-exact + round-trip
# ======================================================================
_EXTDATA = Path(CONDA_ENV) / "lib/R/library/alakazam/extdata"


def test_read_changeo_db_exact(r_ref):
    """readChangeoDb must reproduce R's parsed table bit-exactly."""
    r = pd.read_csv(r_ref / "changeo_db.tsv", sep="\t", dtype=str,
                    keep_default_na=False)
    py = ak.readChangeoDb(_EXTDATA / "example_changeo.tab.gz")
    assert list(py.columns) == list(r.columns)
    assert py.shape == r.shape
    py_s = py.fillna("NA").astype(str)
    r_s = r.replace({"": "NA"}).astype(str)
    assert py_s.equals(r_s.reset_index(drop=True))


def test_write_changeo_db_roundtrip(r_ref, tmp_path):
    """writeChangeoDb -> readChangeoDb round-trip is lossless and
    matches R's own round-tripped output."""
    py = ak.readChangeoDb(_EXTDATA / "example_changeo.tab.gz")
    out = tmp_path / "rt.tab"
    ak.writeChangeoDb(py, out)
    py_rt = ak.readChangeoDb(out)
    assert py_rt.fillna("NA").astype(str).equals(
        py.fillna("NA").astype(str))
    # versus R's round-trip
    r = pd.read_csv(r_ref / "changeo_db_roundtrip.tsv", sep="\t", dtype=str,
                    keep_default_na=False)
    assert py_rt.shape == r.shape
    assert py_rt.fillna("NA").astype(str).reset_index(drop=True).equals(
        r.replace({"": "NA"}).astype(str))


# ======================================================================
# Sequencing quality --- bit-exact
# ======================================================================
def test_get_position_quality_exact(r_ref):
    """getPositionQuality must reproduce R's per-position table."""
    r = pd.read_csv(r_ref / "position_quality.tsv", sep="\t")
    qdb = ak.readChangeoDb(_EXTDATA / "example_quality.tsv")
    fdb = ak.readFastqDb(qdb, _EXTDATA / "example_quality.fastq",
                         style="both", quality_sequence=True)
    py = ak.getPositionQuality(fdb)
    assert len(py) == len(r)
    assert np.array_equal(py["position"].to_numpy(),
                          r["position"].to_numpy())
    rv = r["quality_alignment_num"].to_numpy(dtype=float)
    pv = py["quality_alignment_num"].to_numpy(dtype=float)
    mask = np.isfinite(rv) & np.isfinite(pv)
    assert np.allclose(rv[mask], pv[mask], rtol=1e-6, atol=1e-9)
    assert np.array_equal(np.isnan(rv), np.isnan(pv))
    assert py["nt"].tolist() == r["nt"].tolist()
    assert py["sequence_id"].astype(str).tolist() == \
        r["sequence_id"].astype(str).tolist()


def test_read_fastq_db_quality_exact(r_ref):
    """readFastqDb quality_alignment_num must match R bit-exactly."""
    r = pd.read_csv(r_ref / "fastq_quality.tsv", sep="\t", dtype=str)
    qdb = ak.readChangeoDb(_EXTDATA / "example_quality.tsv")
    fdb = ak.readFastqDb(qdb, _EXTDATA / "example_quality.fastq",
                         style="both", quality_sequence=True)
    r_vals = [np.nan if x in ("NA", "") else float(x)
              for x in str(r["quality_alignment_num"].iloc[0]).split(",")]
    py_vals = [np.nan if x in ("NA", "") else float(x)
               for x in str(fdb["quality_alignment_num"].iloc[0]).split(",")]
    assert len(r_vals) == len(py_vals)
    for a, b in zip(r_vals, py_vals):
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert abs(a - b) < 1e-9


# ======================================================================
# Junction alignment --- bit-exact
# ======================================================================
def test_junction_alignment_exact(r_ref):
    """junctionAlignment must reproduce R's deletion/CDR3 counts."""
    r = pd.read_csv(r_ref / "junction_alignment.tsv", sep="\t")
    sdb = ak.load_single_db()
    # gapped IMGT germline references (as in the alakazam Rd example)
    germline_db = {
        "IGHV3-11*05": (
            "CAGGTGCAGCTGGTGGAGTCTGGGGGA...GGCTTGGTCAAGCCTGGAGGG"
            "TCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTC............"
            "AGTGACTACTACATGAGCTGGATCCGCCAGGCTCCAGGGAAGGGGCTGGAGT"
            "GGGTTTCATACATTAGTAGTAGT......AGTAGTTACACAAACTACGCAGAC"
            "TCTGTGAAG...GGCCGATTCACCATCTCCAGAGACAACGCCAAGAACTCACT"
            "GTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTGTATTACTGTG"
            "CGAGAGA"),
        "IGHD3-10*01": "GTATTACTATGGTTCGGGGAGTTATTATAAC",
        "IGHJ5*02": "ACAACTGGTTCGACCCCTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG",
    }
    py = ak.junctionAlignment(sdb, germline_db)
    for col in ("e3v_length", "e5d_length", "e3d_length", "e5j_length",
                "v_cdr3_length", "j_cdr3_length"):
        rv = float(r[col].iloc[0])
        pv = float(py[col].iloc[0])
        assert abs(rv - pv) < 1e-9, f"{col}: R={rv} py={pv}"
