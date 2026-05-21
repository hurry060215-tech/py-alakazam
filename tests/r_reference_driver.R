#!/usr/bin/env Rscript
# R reference driver for pyalakazam parity tests.
#
# Runs alakazam 1.4.3 on its bundled ExampleDb / ExampleTrees datasets and
# writes numeric results to TSVs that the Python test-suite compares
# against. Usage:  Rscript r_reference_driver.R <output_dir>

suppressMessages(library(alakazam))

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
op <- function(f) file.path(out_dir, f)

data(ExampleDb)
data(ExampleTrees)
data(SingleDb)

# -- gene annotation -------------------------------------------------------
gene_df <- data.frame(
    v_call  = ExampleDb$v_call,
    gene    = getGene(ExampleDb$v_call),
    family  = getFamily(ExampleDb$v_call),
    allele  = getAllele(ExampleDb$v_call),
    locus   = getLocus(ExampleDb$v_call),
    chain   = getChain(ExampleDb$v_call),
    stringsAsFactors = FALSE)
write.table(gene_df, op("gene_calls.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# -- countGenes ------------------------------------------------------------
cg <- countGenes(ExampleDb, gene = "v_call", groups = "sample_id",
                 mode = "family")
write.table(as.data.frame(cg), op("count_genes.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

cg_gene <- countGenes(ExampleDb, gene = "v_call", mode = "gene")
write.table(as.data.frame(cg_gene), op("count_genes_gene.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# -- groupGenes ------------------------------------------------------------
gg <- groupGenes(ExampleDb)
write.table(data.frame(sequence_id = gg$sequence_id,
                       vj_group    = gg$vj_group,
                       stringsAsFactors = FALSE),
            op("group_genes.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

gg_jl <- groupGenes(ExampleDb, junc_len = "junction_length")
write.table(data.frame(sequence_id = gg_jl$sequence_id,
                       vj_group    = gg_jl$vj_group,
                       stringsAsFactors = FALSE),
            op("group_genes_junclen.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# -- countClones -----------------------------------------------------------
cc <- countClones(ExampleDb, groups = "sample_id")
write.table(as.data.frame(cc), op("count_clones.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# -- amino-acid properties -------------------------------------------------
aa <- aminoAcidProperties(ExampleDb[, c("sequence_id", "junction")],
                          seq = "junction")
write.table(as.data.frame(aa), op("aa_properties.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# -- translateDNA ----------------------------------------------------------
tr <- data.frame(sequence_id = ExampleDb$sequence_id,
                 junction_aa = translateDNA(ExampleDb$junction),
                 cdr3_aa     = translateDNA(ExampleDb$junction, trim = TRUE),
                 stringsAsFactors = FALSE)
write.table(tr, op("translate.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# -- sequence distances ----------------------------------------------------
sub_seq <- ExampleDb$junction[1:25]
names(sub_seq) <- ExampleDb$sequence_id[1:25]
# pad to equal length for pairwiseDist
maxlen <- max(nchar(sub_seq))
sub_seq <- padSeqEnds(sub_seq)
names(sub_seq) <- ExampleDb$sequence_id[1:25]
pd <- pairwiseDist(sub_seq)
write.table(as.data.frame(pd), op("pairwise_dist.tsv"), sep = "\t",
            quote = TRUE, row.names = TRUE)

# -- diversity / abundance -------------------------------------------------
set.seed(42)
abund <- estimateAbundance(ExampleDb, group = "sample_id", nboot = 200)
write.table(as.data.frame(abund@abundance), op("abundance.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

set.seed(42)
div <- alphaDiversity(ExampleDb, group = "sample_id", min_q = 0, max_q = 4,
                      step_q = 1, nboot = 200)
write.table(as.data.frame(div@diversity), op("diversity.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# -- coverage --------------------------------------------------------------
clones <- countClones(ExampleDb, groups = "sample_id")
cov_df <- data.frame(
    sample_id = c("+7d", "-1h"),
    coverage  = c(calcCoverage(clones$seq_count[clones$sample_id == "+7d"]),
                  calcCoverage(clones$seq_count[clones$sample_id == "-1h"])),
    stringsAsFactors = FALSE)
write.table(cov_df, op("coverage.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# -- topology --------------------------------------------------------------
g <- ExampleTrees[[23]]
pl <- getPathLengths(g, root = "Germline")
write.table(pl, op("path_lengths.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

st <- summarizeSubtrees(g, fields = "c_call", root = "Germline")
write.table(st[, c("name", "outdegree", "size", "depth", "pathlength")],
            op("subtrees.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

te <- tableEdges(g, "c_call", exclude = c("Germline", NA))
write.table(as.data.frame(te), op("table_edges.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

# -- Change-O database I/O -------------------------------------------------
ext <- system.file("extdata", package = "alakazam")
changeo <- suppressWarnings(
    readChangeoDb(file.path(ext, "example_changeo.tab.gz")))
write.table(as.data.frame(changeo), op("changeo_db.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE, na = "NA")
# round-trip
tmp_changeo <- op("changeo_roundtrip.tab")
writeChangeoDb(changeo, tmp_changeo)
changeo_rt <- suppressWarnings(readChangeoDb(tmp_changeo))
write.table(as.data.frame(changeo_rt), op("changeo_db_roundtrip.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")

# -- sequencing quality ---------------------------------------------------
qdb <- suppressWarnings(readChangeoDb(file.path(ext, "example_quality.tsv")))
fdb <- readFastqDb(qdb, file.path(ext, "example_quality.fastq"),
                   style = "both", quality_sequence = TRUE)
write.table(data.frame(quality_alignment_num = fdb$quality_alignment_num,
                       stringsAsFactors = FALSE),
            op("fastq_quality.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)
pq <- getPositionQuality(fdb)
write.table(as.data.frame(pq), op("position_quality.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE, na = "NA")

# -- junction alignment ---------------------------------------------------
germline_db <- list(
"IGHV3-11*05"=paste0("CAGGTGCAGCTGGTGGAGTCTGGGGGA...GGCTTGGTCAAGCCTGGAGGG",
"TCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTC............AGTGACTACTACATGAGCT",
"GGATCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTTTCATACATTAGTAGTAGT......AGTAGTTA",
"CACAAACTACGCAGACTCTGTGAAG...GGCCGATTCACCATCTCCAGAGACAACGCCAAGAACTCACTG",
"TATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTGTATTACTGTGCGAGAGA"),
"IGHD3-10*01"="GTATTACTATGGTTCGGGGAGTTATTATAAC",
"IGHJ5*02"="ACAACTGGTTCGACCCCTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG")
ja <- junctionAlignment(SingleDb, germline_db)
write.table(as.data.frame(ja[, c("sequence_id", "e3v_length", "e5d_length",
            "e3d_length", "e5j_length", "v_cdr3_length", "j_cdr3_length")]),
            op("junction_alignment.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)

cat("R reference driver complete.\n")
