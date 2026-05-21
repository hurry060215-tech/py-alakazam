"""pyalakazam: pure-Python port of the R/CRAN package *alakazam*.

A faithful, dependency-light Python re-implementation of the core
package of the Immcantation framework for adaptive immune receptor
repertoire (AIRR-seq) analysis (Gupta et al, *Bioinformatics* 2015).

It operates on AIRR-format data frames (columns ``sequence_id``,
``v_call``, ``d_call``, ``j_call``, ``junction``, ``junction_aa``,
``clone_id``, ...) and covers the full computational API of alakazam.

Gene annotation
---------------
* :func:`getSegment`, :func:`getAllele`, :func:`getGene`,
  :func:`getFamily`, :func:`getLocus`, :func:`getChain` --- IMGT
  segment-call parsing.
* :func:`countGenes` --- V(D)J gene-usage frequencies (copy/clone
  weighted, per-locus).
* :func:`sortGenes` --- sort gene names by name or position.

Diversity & abundance
---------------------
* :func:`alphaDiversity` / :func:`rarefyDiversity` / :func:`testDiversity`
  --- Hill-number diversity curves with uniform resampling and bootstrap
  confidence intervals.
* :func:`calcDiversity`, :func:`calcCoverage`.
* :func:`estimateAbundance`, :func:`countClones` --- clonal abundance.
* :class:`AbundanceCurve`, :class:`DiversityCurve`.

Amino-acid properties
---------------------
* :func:`aminoAcidProperties` --- per-CDR3 physicochemical descriptors.
* :func:`gravy`, :func:`bulk`, :func:`polar`, :func:`aliphatic`,
  :func:`charge`, :func:`isValidAASeq`, :func:`countPatterns`.

Sequence utilities & distances
------------------------------
* :func:`translateDNA`, :func:`maskSeqGaps`, :func:`maskSeqEnds`,
  :func:`padSeqEnds`, :func:`extractVRegion`, :func:`collapseDuplicates`.
* :func:`seqDist`, :func:`seqEqual`, :func:`pairwiseDist`,
  :func:`pairwiseEqual`, :func:`nonsquareDist`, :func:`getDNAMatrix`,
  :func:`getAAMatrix`.

Lineage reconstruction & topology
---------------------------------
* :func:`makeChangeoClone`, :func:`buildPhylipLineage` --- pure-Python
  maximum-parsimony lineage trees (no external PHYLIP binary).
* :func:`getPathLengths`, :func:`getMRCA`, :func:`tableEdges`,
  :func:`summarizeSubtrees`, :func:`permuteLabels`, :func:`testEdges`,
  :func:`testMRCA`.
* :class:`ChangeoClone`, :class:`LineageTree`.

Plotting
--------
* :func:`plotGeneUsage`, :func:`plotDiversityCurve`,
  :func:`plotAbundanceCurve`, :func:`plotLineageTree`.

Example data
------------
* :func:`load_example_db`, :func:`load_single_db`,
  :func:`load_example_trees`.
"""
from __future__ import annotations

from .aminoacids import (aliphatic, aminoAcidProperties, bulk, charge,
                         countPatterns, gravy, isValidAASeq, polar)
from .constants import (ABBREV_AA, BULKINESS_ZIMJ68, DNA_IUPAC,
                        HYDROPATHY_KYTJ82, IMGT_REGIONS, IUPAC_AA, IUPAC_DNA,
                        PK_EMBOSS, POLARITY_GRAR74)
from .core import checkColumns, stoufferMeta, translateStrings
from .data import load_example_db, load_example_trees, load_single_db
from .diversity import (AbundanceCurve, DiversityCurve, alphaDiversity,
                        calcCoverage, calcDiversity, countClones,
                        estimateAbundance, rarefyDiversity, testDiversity)
from .gene import (countGenes, getAllele, getChain, getFamily, getGene,
                   getLocus, getSegment, sortGenes)
from .graph import LineageTree
from .lineage import (ChangeoClone, buildPhylipLineage, graphToPhylo,
                      makeChangeoClone, phyloToGraph)
from .plotting import (plotAbundanceCurve, plotDiversityCurve, plotGeneUsage,
                       plotLineageTree)
from .sequence import (collapseDuplicates, extractVRegion, getAAMatrix,
                       getDNAMatrix, maskSeqEnds, maskSeqGaps, nonsquareDist,
                       padSeqEnds, pairwiseDist, pairwiseEqual, seqDist,
                       seqEqual, translateDNA)
from .topology import (EdgeTest, MRCATest, getMRCA, getPathLengths,
                       permuteLabels, summarizeSubtrees, tableEdges,
                       testEdges, testMRCA)

__version__ = "0.1.0"

__all__ = [
    # gene annotation
    "getSegment", "getAllele", "getGene", "getFamily", "getLocus",
    "getChain", "countGenes", "sortGenes",
    # diversity & abundance
    "calcDiversity", "calcCoverage", "alphaDiversity", "rarefyDiversity",
    "testDiversity", "estimateAbundance", "countClones",
    "AbundanceCurve", "DiversityCurve",
    # amino-acid properties
    "aminoAcidProperties", "gravy", "bulk", "polar", "aliphatic", "charge",
    "isValidAASeq", "countPatterns",
    # sequence utilities & distances
    "translateDNA", "maskSeqGaps", "maskSeqEnds", "padSeqEnds",
    "extractVRegion", "collapseDuplicates", "seqDist", "seqEqual",
    "pairwiseDist", "pairwiseEqual", "nonsquareDist", "getDNAMatrix",
    "getAAMatrix",
    # lineage & topology
    "makeChangeoClone", "buildPhylipLineage", "ChangeoClone", "LineageTree",
    "graphToPhylo", "phyloToGraph",
    "getPathLengths", "getMRCA", "tableEdges", "summarizeSubtrees",
    "permuteLabels", "testEdges", "testMRCA", "EdgeTest", "MRCATest",
    # plotting
    "plotGeneUsage", "plotDiversityCurve", "plotAbundanceCurve",
    "plotLineageTree",
    # core utilities
    "translateStrings", "checkColumns", "stoufferMeta",
    # example data
    "load_example_db", "load_single_db", "load_example_trees",
    # constants
    "IUPAC_DNA", "IUPAC_AA", "DNA_IUPAC", "ABBREV_AA", "IMGT_REGIONS",
    "HYDROPATHY_KYTJ82", "BULKINESS_ZIMJ68", "POLARITY_GRAR74", "PK_EMBOSS",
]
