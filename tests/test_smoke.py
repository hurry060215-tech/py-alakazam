"""Smoke tests for pyalakazam --- exercise every public function."""
from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyalakazam as ak

warnings.filterwarnings("ignore")

# Change-O / quality example files shipped with the R alakazam package.
_EXTDATA = Path("/scratch/users/steorra/env/CMAP/lib/R/library/"
                "alakazam/extdata")


# ----------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------
def test_load_example_db():
    db = ak.load_example_db()
    assert isinstance(db, pd.DataFrame)
    assert db.shape == (1999, 19)
    for col in ("sequence_id", "v_call", "j_call", "junction", "clone_id"):
        assert col in db.columns


def test_load_single_db():
    sdb = ak.load_single_db()
    assert isinstance(sdb, pd.DataFrame)
    assert len(sdb) >= 1


def test_load_example_trees():
    trees = ak.load_example_trees()
    assert len(trees) == 49
    g = trees[22]
    assert isinstance(g, ak.LineageTree)
    assert "Germline" in g.vertices
    assert len(g.edges) > 0


# ----------------------------------------------------------------------
# gene annotation
# ----------------------------------------------------------------------
def test_get_segment_family():
    kappa = ["Homsap IGKV1D-39*01 F,Homsap IGKV1-39*02 F", "Homsap IGKJ5*01 F"]
    assert ak.getAllele(kappa) == ["IGKV1-39*01", "IGKJ5*01"]
    assert ak.getGene(kappa) == ["IGKV1-39", "IGKJ5"]
    assert ak.getFamily(kappa) == ["IGKV1", "IGKJ5"]
    assert ak.getLocus(kappa) == ["IGK", "IGK"]
    assert ak.getChain(kappa) == ["VL", "VL"]


def test_get_gene_strip_d():
    heavy = "Homsap IGHV1-69*01 F,Homsap IGHV1-69D*01 F"
    assert ak.getGene(heavy, first=False) == "IGHV1-69"
    assert ak.getGene(heavy, first=False, strip_d=False) == \
        "IGHV1-69,IGHV1-69D"


def test_get_gene_scalar():
    assert ak.getGene("Homsap IGHV3-11*05 F") == "IGHV3-11"


def test_count_genes():
    db = ak.load_example_db()
    cg = ak.countGenes(db, gene="v_call", groups="sample_id", mode="family")
    assert "gene" in cg.columns
    assert "seq_freq" in cg.columns
    grp = cg.groupby("sample_id")["seq_freq"].sum()
    assert np.allclose(grp.to_numpy(), 1.0)


def test_count_genes_copy():
    db = ak.load_example_db()
    cg = ak.countGenes(db, gene="v_call", groups="sample_id",
                       copy="duplicate_count", mode="gene")
    assert "copy_count" in cg.columns
    assert "copy_freq" in cg.columns


def test_count_genes_clone():
    db = ak.load_example_db()
    cg = ak.countGenes(db, gene="v_call", clone="clone_id", mode="family")
    assert "clone_count" in cg.columns


def test_sort_genes():
    genes = ["IGHV1-2", "IGHV3-11", "IGHV1-69", "IGHV3-7"]
    s = ak.sortGenes(genes, method="name")
    assert s[0].startswith("IGHV1")
    s2 = ak.sortGenes(genes, method="position")
    assert isinstance(s2, list)


def test_group_genes():
    db = ak.load_example_db()
    g = ak.groupGenes(db)
    assert "vj_group" in g.columns
    assert len(g) == len(db)
    assert g["vj_group"].notna().all()
    assert g["vj_group"].nunique() > 1
    assert g["vj_group"].str.match(r"G\d+").all()


def test_group_genes_junc_len():
    db = ak.load_example_db()
    g0 = ak.groupGenes(db)
    g1 = ak.groupGenes(db, junc_len="junction_length")
    # adding junction length can only split groups, never merge them
    assert g1["vj_group"].nunique() >= g0["vj_group"].nunique()


def test_group_genes_first_mode():
    # ambiguous V calls collapse to a single group when first=False,
    # since the union of gene calls chains them together
    data = pd.DataFrame({
        "sequence_id": ["s1", "s2", "s3"],
        "v_call": ["IGHV1-2*01,IGHV3-11*01", "IGHV3-11*01", "IGHV1-2*01"],
        "j_call": ["IGHJ4*02", "IGHJ4*02", "IGHJ4*02"],
        "locus": ["IGH", "IGH", "IGH"],
    })
    g_union = ak.groupGenes(data, first=False)
    assert g_union["vj_group"].nunique() == 1
    g_first = ak.groupGenes(data, first=True)
    # with first=True s1 keeps only IGHV1-2 -> groups with s3, not s2
    assert g_first.set_index("sequence_id").loc["s1", "vj_group"] == \
        g_first.set_index("sequence_id").loc["s3", "vj_group"]
    assert g_first.set_index("sequence_id").loc["s1", "vj_group"] != \
        g_first.set_index("sequence_id").loc["s2", "vj_group"]


# ----------------------------------------------------------------------
# amino-acid properties
# ----------------------------------------------------------------------
def test_translate_dna():
    assert ak.translateDNA("TGTGCGAGA") == "CAR"
    out = ak.translateDNA(["ACTGACTCGA", "ATG"])
    assert len(out) == 2


def test_translate_trim():
    seq = "TGTGCGAGAGAATGGTTC"
    full = ak.translateDNA(seq)
    trimmed = ak.translateDNA(seq, trim=True)
    assert len(trimmed) == len(full) - 2


def test_gravy_bulk_polar():
    seq = "CARDRSTPWRRGIASTTVRTSW"
    assert isinstance(ak.gravy(seq), float)
    assert isinstance(ak.bulk(seq), float)
    assert isinstance(ak.polar(seq), float)
    assert isinstance(ak.aliphatic(seq), float)
    assert isinstance(ak.charge(seq), float)


def test_is_valid_aa():
    # alakazam's valid set excludes ambiguous codes (J, B, Z); "10" invalid.
    seqs = ["CARDRSTPWRRGIASTTVRTSW", "XXTQMYVR--XX", "CARJ", "10"]
    valid = ak.isValidAASeq(seqs)
    assert valid[0] and valid[1]
    assert not valid[2] and not valid[3]


def test_amino_acid_properties():
    db = ak.load_example_db()
    aa = ak.aminoAcidProperties(db.iloc[:20][["sequence_id", "junction"]],
                                seq="junction")
    assert "junction_aa_length" in aa.columns
    assert "junction_aa_gravy" in aa.columns
    assert "junction_aa_charge" in aa.columns
    assert aa["junction_aa_length"].notna().all()


def test_count_patterns():
    seq = ["TGTCAACAGGCTAACAGTTTCCGGACGTTC",
           "TGTCAGCAATATTATATTGCTCCCTTCACTTTC"]
    cp = ak.countPatterns(seq, {"arg": "A", "val": "V"}, nt=True, trim=True,
                          label="cdr3")
    assert "cdr3_arg" in cp.columns
    assert len(cp) == 2


# ----------------------------------------------------------------------
# sequence utilities & distances
# ----------------------------------------------------------------------
def test_mask_seq_gaps():
    assert ak.maskSeqGaps("ATG-C") == "ATGNC"
    assert ak.maskSeqGaps("--ATG-C-", outer_only=True) == "NNATG-CN"


def test_mask_seq_ends():
    seq = ["CCCCTGGG", "NAACTGGN", "NNNCTGNN"]
    masked = ak.maskSeqEnds(seq)
    assert all(len(s) == 8 for s in masked)
    assert masked[0].startswith("NNN")


def test_pad_seq_ends():
    seq = ["CCCCTGGG", "ACCCTG", "CCCC"]
    padded = ak.padSeqEnds(seq)
    assert len({len(s) for s in padded}) == 1


def test_extract_v_region():
    db = ak.load_example_db()
    fwr1 = ak.extractVRegion(db["sequence_alignment"].iloc[:3].tolist(),
                             "fwr1")
    assert all(len(s) == 78 for s in fwr1)
    multi = ak.extractVRegion(db["sequence_alignment"].iloc[:3].tolist(),
                              ["cdr1", "cdr2"])
    assert isinstance(multi, pd.DataFrame)


def test_seq_dist_equal():
    assert ak.seqDist("ATGGC", "ATGGG") == 1
    # ? counts as a mismatch against a concrete base (matches alakazam)
    assert ak.seqDist("ATGGC", "ATG??") == 2
    # gaps with a gap=-1 matrix collapse contiguous indels to a single hit
    assert ak.seqDist("ATGGC", "AT--C", dist_mat=ak.getDNAMatrix(gap=-1)) == 1
    assert ak.seqEqual("ATG-C", "AT--C")
    assert not ak.seqEqual("ATGGC", "ATGGA")


def test_pairwise_dist_equal():
    seq = {"A": "ATGGC", "B": "ATGGG", "C": "ATGGG", "D": "AT--C"}
    pd_mat = ak.pairwiseDist(seq, dist_mat=ak.getDNAMatrix(gap=0))
    assert pd_mat.shape == (4, 4)
    pe = ak.pairwiseEqual(seq)
    assert pe.loc["B", "C"]


def test_nonsquare_dist():
    seq = {"A": "ATGGC", "B": "ATGGG", "C": "ATGGG", "D": "AT--C"}
    nd = ak.nonsquareDist(seq, indx=[1, 3], dist_mat=ak.getDNAMatrix(gap=0))
    assert nd.shape == (2, 4)


def test_dna_aa_matrix():
    dm = ak.getDNAMatrix()
    assert dm.loc["A", "A"] == 0
    am = ak.getAAMatrix()
    assert am.loc["A", "A"] == 0


def test_collapse_duplicates():
    db = pd.DataFrame({
        "sequence_id": list("ABCD"),
        "sequence_alignment": ["CCCCTGGG", "CCCCTGGN", "NAACTGGN", "NNNCTGNN"],
        "c_call": ["IGHM", "IGHG", "IGHG", "IGHA"],
        "duplicate_count": [1, 2, 3, 4],
    })
    out = ak.collapseDuplicates(db, text_fields=["c_call"],
                                num_fields=["duplicate_count"])
    assert len(out) < len(db)


# ----------------------------------------------------------------------
# diversity & abundance
# ----------------------------------------------------------------------
def test_count_clones():
    db = ak.load_example_db()
    cc = ak.countClones(db, groups="sample_id")
    assert "seq_count" in cc.columns
    assert "seq_freq" in cc.columns


def test_calc_coverage():
    cov = ak.calcCoverage([1, 1, 3, 10])
    assert 0 < cov < 1


def test_calc_diversity():
    d = ak.calcDiversity([1, 1, 3, 10], [0, 1, 2])
    assert len(d) == 3
    assert d[0] == 4.0  # richness


def test_estimate_abundance():
    db = ak.load_example_db()
    abund = ak.estimateAbundance(db, group="sample_id", nboot=50, seed=1)
    assert isinstance(abund, ak.AbundanceCurve)
    assert set(abund.groups) == {"+7d", "-1h"}
    assert "p" in abund.abundance.columns


def test_alpha_diversity():
    db = ak.load_example_db()
    div = ak.alphaDiversity(db, group="sample_id", min_q=0, max_q=4,
                            step_q=1, nboot=50, seed=1)
    assert isinstance(div, ak.DiversityCurve)
    assert "d" in div.diversity.columns
    assert div.tests is not None


def test_alpha_diversity_from_abundance():
    db = ak.load_example_db()
    abund = ak.estimateAbundance(db, group="sample_id", nboot=50, seed=1)
    div = ak.alphaDiversity(abund, step_q=1, max_q=4)
    assert "e" in div.diversity.columns


# ----------------------------------------------------------------------
# lineage & topology
# ----------------------------------------------------------------------
def test_make_changeo_clone():
    db = ak.load_example_db()
    sub = db[db["clone_id"] == 3138].copy()
    clone = ak.makeChangeoClone(sub, text_fields=["sample_id", "c_call"],
                                num_fields=["duplicate_count"])
    assert isinstance(clone, ak.ChangeoClone)
    assert clone.clone == "3138"
    assert len(clone.data) >= 2


def test_build_phylip_lineage():
    db = ak.load_example_db()
    sub = db[db["clone_id"] == 3138].copy()
    clone = ak.makeChangeoClone(sub, text_fields=["sample_id", "c_call"],
                                num_fields=["duplicate_count"])
    g = ak.buildPhylipLineage(clone)
    assert isinstance(g, ak.LineageTree)
    assert "Germline" in g.vertices
    # germline must be the root (no parents)
    assert len(g.parents("Germline")) == 0


def test_path_lengths():
    trees = ak.load_example_trees()
    pl = ak.getPathLengths(trees[22], root="Germline")
    assert "steps" in pl.columns
    assert "distance" in pl.columns
    germ = pl[pl["name"] == "Germline"].iloc[0]
    assert germ["steps"] == 0


def test_get_mrca():
    trees = ak.load_example_trees()
    mrca = ak.getMRCA(trees[22], path="steps", root="Germline")
    assert len(mrca) >= 1
    assert "distance" in mrca.columns


def test_summarize_subtrees():
    trees = ak.load_example_trees()
    st = ak.summarizeSubtrees(trees[22], fields=["c_call"], root="Germline")
    assert "size" in st.columns
    assert "depth" in st.columns
    germ = st[st["name"] == "Germline"].iloc[0]
    assert germ["size"] == len(trees[22].vertices)


def test_table_edges():
    trees = ak.load_example_trees()
    te = ak.tableEdges(trees[22], "c_call", exclude=["Germline", np.nan])
    assert "count" in te.columns


def test_permute_labels():
    trees = ak.load_example_trees()
    rng = np.random.default_rng(0)
    perm = ak.permuteLabels(trees[22], "c_call", rng=rng)
    assert isinstance(perm, ak.LineageTree)


def test_test_edges():
    trees = ak.load_example_trees()
    et = ak.testEdges(trees[:8], "c_call", nperm=10, seed=1)
    assert isinstance(et, ak.EdgeTest)
    assert "pvalue" in et.tests.columns


def test_test_mrca():
    trees = ak.load_example_trees()
    mt = ak.testMRCA(trees[:8], "c_call", nperm=10, seed=1)
    assert isinstance(mt, ak.MRCATest)


def test_graph_phylo_roundtrip():
    trees = ak.load_example_trees()
    g = trees[22]
    phylo = ak.graphToPhylo(g)
    g2 = ak.phyloToGraph(phylo)
    assert len(g2.edges) == len(g.edges)


# ----------------------------------------------------------------------
# core utilities
# ----------------------------------------------------------------------
def test_translate_strings():
    assert ak.translateStrings(["A", "B", "C"],
                               {"POSITION1": "A"}) == ["POSITION1", "B", "C"]


def test_check_columns():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert ak.checkColumns(df, ["A", "B"]) is True
    assert isinstance(ak.checkColumns(df, ["A", "D"]), str)


def test_stouffer_meta():
    res = ak.stoufferMeta([0.1, 0.05, 0.3], [5, 10, 1])
    assert "Z" in res and "pvalue" in res


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def test_plotting_smoke():
    import matplotlib
    matplotlib.use("Agg")
    db = ak.load_example_db()
    cg = ak.countGenes(db, gene="v_call", groups="sample_id", mode="family")
    ax = ak.plotGeneUsage(cg, group="sample_id")
    assert ax is not None

    abund = ak.estimateAbundance(db, group="sample_id", nboot=30, seed=1)
    ax2 = ak.plotAbundanceCurve(abund)
    assert ax2 is not None

    div = ak.alphaDiversity(abund, step_q=1, max_q=4)
    ax3 = ak.plotDiversityCurve(div)
    assert ax3 is not None

    ax4 = ak.plotLineageTree(ak.load_example_trees()[22], label_field="c_call")
    assert ax4 is not None


# ----------------------------------------------------------------------
# Change-O database I/O
# ----------------------------------------------------------------------
def test_write_read_changeo_db_roundtrip(tmp_path):
    df = pd.DataFrame({
        "SEQUENCE_ID": ["s1", "s2", "s3"],
        "V_CALL": ["IGHV1-2*01", "IGHV3-11*05", None],
        "JUNCTION": ["tgtgcgaga", "tgtcaacag", "tgtgcgcgc"],
        "DUPCOUNT": ["3", "1", "5"],
    })
    out = tmp_path / "db.tab"
    ak.writeChangeoDb(df, out)
    db = ak.readChangeoDb(out)
    assert list(db.columns) == list(df.columns)
    assert db.shape == df.shape
    # sequence columns upper-cased by default
    assert db["JUNCTION"].iloc[0] == "TGTGCGAGA"
    db2 = ak.readChangeoDb(out, select=["SEQUENCE_ID", "V_CALL"])
    assert list(db2.columns) == ["SEQUENCE_ID", "V_CALL"]
    db3 = ak.readChangeoDb(out, drop=["DUPCOUNT"])
    assert "DUPCOUNT" not in db3.columns
    db4 = ak.readChangeoDb(out, seq_upper=False)
    assert db4["JUNCTION"].iloc[0] == "tgtgcgaga"


# ----------------------------------------------------------------------
# Sequencing quality
# ----------------------------------------------------------------------
@pytest.mark.skipif(not (_EXTDATA / "example_quality.fastq").exists(),
                    reason="alakazam extdata not available")
def test_read_fastq_db():
    qdb = ak.readChangeoDb(_EXTDATA / "example_quality.tsv")
    fdb = ak.readFastqDb(qdb, _EXTDATA / "example_quality.fastq",
                         style="both", quality_sequence=True)
    for col in ("quality_alignment_num", "quality_alignment",
                "quality", "quality_num"):
        assert col in fdb.columns
    fdb2 = ak.readFastqDb(qdb, _EXTDATA / "example_quality.fastq",
                          style="num")
    assert "quality_alignment_num" in fdb2.columns
    assert "quality_alignment" not in fdb2.columns


@pytest.mark.skipif(not (_EXTDATA / "example_quality.fastq").exists(),
                    reason="alakazam extdata not available")
def test_get_position_quality():
    qdb = ak.readChangeoDb(_EXTDATA / "example_quality.tsv")
    fdb = ak.readFastqDb(qdb, _EXTDATA / "example_quality.fastq",
                         style="both", quality_sequence=True)
    pq = ak.getPositionQuality(fdb)
    assert set(pq.columns) == {"position", "quality_alignment_num",
                               "sequence_id", "nt"}
    assert pq["position"].iloc[0] == 1
    assert len(pq) == len(fdb["sequence_alignment"].iloc[0])


@pytest.mark.skipif(not (_EXTDATA / "example_quality.fastq").exists(),
                    reason="alakazam extdata not available")
def test_mask_positions_by_quality():
    qdb = ak.readChangeoDb(_EXTDATA / "example_quality.tsv")
    fdb = ak.readFastqDb(qdb, _EXTDATA / "example_quality.fastq",
                         style="both", quality_sequence=True)
    masked = ak.maskPositionsByQuality(fdb, min_quality=90)
    assert "sequence_alignment_masked" in masked.columns
    # high threshold -> at least one position masked to 'N'
    assert "N" in masked["sequence_alignment_masked"].iloc[0]


# ----------------------------------------------------------------------
# Junction alignment
# ----------------------------------------------------------------------
def test_junction_alignment():
    sdb = ak.load_single_db()
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
    db = ak.junctionAlignment(sdb, germline_db)
    for col in ("e3v_length", "e5d_length", "e3d_length", "e5j_length",
                "v_cdr3_length", "j_cdr3_length"):
        assert col in db.columns
    assert db["e5d_length"].iloc[0] == 5
    assert db["v_cdr3_length"].iloc[0] == 7


# ----------------------------------------------------------------------
# IgPhyML
# ----------------------------------------------------------------------
def _write_igphyml(path):
    with open(path, "w") as fh:
        fh.write("CLONE\tNSEQ\tNSITE\tLHOOD\tTREE\n")
        fh.write("REPERTOIRE\t0\t0\t-100.5\tigphyml --run\n")
        fh.write("1\t3\t100\t-50.2\t(seq1:0.1,seq2:0.2,1_GERM:0.0);\n")
        fh.write("2\t4\t100\t-60.1\t"
                 "((seqA:0.3,seqB:0.1):0.05,2_GERM:0.0);\n")


def test_read_igphyml(tmp_path):
    f = tmp_path / "ig.tab"
    _write_igphyml(f)
    ig = ak.readIgphyml(f, id="donor1", format="graph")
    assert "param" in ig and "command" in ig and "trees" in ig
    assert ig["command"] == "igphyml --run"
    assert set(ig["trees"]) == {"1", "2"}
    assert all(isinstance(t, ak.LineageTree) for t in ig["trees"].values())
    assert "id" in ig["param"].columns

    ig_p = ak.readIgphyml(f, id="donor1", format="phylo")
    assert all(hasattr(t, "tip_label") for t in ig_p["trees"].values())


def test_combine_igphyml(tmp_path):
    f = tmp_path / "ig.tab"
    _write_igphyml(f)
    ig1 = ak.readIgphyml(f, id="donor1", format="graph")
    ig2 = ak.readIgphyml(f, id="donor2", format="graph")
    wide = ak.combineIgphyml([ig1, ig2], format="wide")
    assert len(wide) == 2
    assert "id" in wide.columns
    assert set(wide["id"]) == {"donor1", "donor2"}
    long = ak.combineIgphyml([ig1, ig2], format="long")
    assert {"id", "variable", "value"}.issubset(long.columns)


# ----------------------------------------------------------------------
# Test / subtree plots
# ----------------------------------------------------------------------
def test_test_plots_smoke():
    import matplotlib
    matplotlib.use("Agg")
    db = ak.load_example_db()
    div = ak.alphaDiversity(db, group="sample_id", min_q=0, max_q=4,
                            step_q=1, nboot=20, seed=1)
    assert ak.plotDiversityTest(div, q=2) is not None
    assert ak.plotDiversityTest(div, q=2, annotate="depth",
                                log_d=True) is not None

    trees = ak.load_example_trees()
    et = ak.testEdges(trees[:8], "c_call", nperm=10, seed=1)
    assert ak.plotEdgeTest(et) is not None
    assert ak.plotEdgeTest(et, style="cdf") is not None

    mt = ak.testMRCA(trees[:8], "c_call", nperm=10, seed=1)
    assert ak.plotMRCATest(mt) is not None
    assert ak.plotMRCATest(mt, style="cdf") is not None

    assert ak.plotSubtrees(trees[:8], "c_call", "size") is not None
    assert ak.plotSubtrees(trees[:8], "c_call", "depth",
                           style="violin") is not None

    fig = ak.gridPlot(partial(ak.plotEdgeTest, et),
                      partial(ak.plotMRCATest, mt), ncol=2)
    assert fig is not None


# ----------------------------------------------------------------------
# Color constants
# ----------------------------------------------------------------------
def test_color_constants():
    assert ak.DNA_COLORS["A"] == "#64F73F"
    assert set(ak.DNA_COLORS) == {"A", "C", "G", "T"}
    assert ak.IG_COLORS["IGHM"] == "#984EA3"
    assert "IGHG" in ak.IG_COLORS
    assert ak.TR_COLORS["TRB"] == "#F4CAE4"
    assert set(ak.TR_COLORS) == {"TRA", "TRB", "TRD", "TRG"}
