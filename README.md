# pyalakazam

**Pure-Python port of the R/CRAN package [alakazam](https://alakazam.readthedocs.io)** — the core
package of the [Immcantation](https://immcantation.readthedocs.io) framework for
adaptive immune receptor repertoire (AIRR-seq) analysis
(Gupta *et al.*, *Bioinformatics* 2015).

`pyalakazam` re-implements the full computational API of alakazam in pure
Python (numpy / scipy / pandas / matplotlib) — **no rpy2, no R, no external
PHYLIP binary**. It operates on AIRR-format `pandas.DataFrame`s with the
standard columns `sequence_id`, `v_call`, `d_call`, `j_call`, `junction`,
`junction_aa`, `clone_id`, … and aims for numerical parity with alakazam 1.4.3.

## Installation

```bash
pip install pyalakazam
```

Or from source:

```bash
git clone https://github.com/omicverse/py-alakazam
cd py-alakazam && pip install .
```

## Coverage

| Area | Functions |
|------|-----------|
| **Gene annotation** | `getSegment`, `getAllele`, `getGene`, `getFamily`, `getLocus`, `getChain`, `countGenes`, `sortGenes` |
| **Diversity / abundance** | `alphaDiversity`, `rarefyDiversity`, `testDiversity`, `calcDiversity`, `calcCoverage`, `estimateAbundance`, `countClones` |
| **Amino-acid properties** | `aminoAcidProperties`, `gravy`, `bulk`, `polar`, `aliphatic`, `charge`, `isValidAASeq`, `countPatterns` |
| **Sequence utilities** | `translateDNA`, `maskSeqGaps`, `maskSeqEnds`, `padSeqEnds`, `extractVRegion`, `collapseDuplicates` |
| **Sequence distances** | `seqDist`, `seqEqual`, `pairwiseDist`, `pairwiseEqual`, `nonsquareDist`, `getDNAMatrix`, `getAAMatrix` |
| **Lineage** | `makeChangeoClone`, `buildPhylipLineage`, `graphToPhylo`, `phyloToGraph` |
| **Topology** | `getPathLengths`, `getMRCA`, `tableEdges`, `summarizeSubtrees`, `permuteLabels`, `testEdges`, `testMRCA` |
| **Plotting** | `plotGeneUsage`, `plotDiversityCurve`, `plotAbundanceCurve`, `plotLineageTree` |
| **Core / data** | `translateStrings`, `checkColumns`, `stoufferMeta`, `load_example_db`, `load_single_db`, `load_example_trees` |

## Quick-start

```python
import pyalakazam as ak

# bundled example B-cell repertoire (alakazam ExampleDb)
db = ak.load_example_db()

# gene-usage frequencies
genes = ak.countGenes(db, gene="v_call", groups="sample_id", mode="family")

# clonal diversity (Hill numbers) with bootstrap CIs
div = ak.alphaDiversity(db, group="sample_id", min_q=0, max_q=4, nboot=200)
ak.plotDiversityCurve(div, legend_title="Sample")

# CDR3 physicochemical descriptors
aa = ak.aminoAcidProperties(db[["sequence_id", "junction"]], seq="junction")

# Ig lineage tree (maximum parsimony, no external PHYLIP)
clone = ak.makeChangeoClone(db[db.clone_id == 3138],
                            text_fields=["sample_id", "c_call"],
                            num_fields=["duplicate_count"])
tree = ak.buildPhylipLineage(clone)
ak.plotLineageTree(tree, label_field="c_call")
```

## Numerical parity with R alakazam 1.4.3

`tests/test_r_parity.py` runs alakazam on its bundled `ExampleDb` /
`ExampleTrees` datasets and compares against pyalakazam:

* **Deterministic functions** — gene parsing, `countGenes`, `countClones`,
  `aminoAcidProperties`, `translateDNA`, sequence distances, `calcCoverage`
  and the topology analyses (`getPathLengths`, `summarizeSubtrees`,
  `tableEdges`) — agree **bit-exactly / to rel-diff < 1e-6**.
* **Bootstrap functions** — `estimateAbundance` and `alphaDiversity` use
  multinomial resampling. R's and NumPy's RNGs differ, so the bootstrap
  *realisations* are not bit-identical; the **point estimates agree to
  Pearson r > 0.999** and the confidence intervals to within a few percent.

## The lineage caveat (no PHYLIP)

alakazam's `buildPhylipLineage` shells out to the external PHYLIP `dnapars`
binary. To stay self-contained, pyalakazam re-implements the maximum-parsimony
search in pure Python — greedy stepwise addition, NNI + SPR hill-climbing, and
a vectorised Fitch / Sankoff small-parsimony reconstruction. The germline is
rooted as the outgroup and zero-weight inferred parents are collapsed exactly
as alakazam does.

For the 49 trees in `ExampleTrees`, pyalakazam recovers an **identical-length
tree for 31 trees**; for the rest the topology and total tree length are very
close. Two effects explain the small differences: (1) PHYLIP allows ambiguous
characters (`N`) in *inferred* internal nodes, which can lower the recomputed
`seqDist` edge-weight sum below the strict-parsimony length, and (2) for very
large clones the heuristic search may settle in a near- rather than
globally-optimal topology. The trees produced are always valid maximum-
parsimony reconstructions.

## License

AGPL-3, preserving the license of the original alakazam package. The original
alakazam is developed by the Kleinstein Lab (Yale University) as part of the
Immcantation framework. See `LICENSE`.

## Citation

If you use this port, please cite the original alakazam publication:

> Gupta NT, Vander Heiden JA, Uduman M, Gadala-Maria D, Yaari G, Kleinstein SH.
> Change-O: a toolkit for analyzing large-scale B cell immunoglobulin
> repertoire sequencing data. *Bioinformatics* 2015.
