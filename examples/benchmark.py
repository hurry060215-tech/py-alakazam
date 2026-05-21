"""Benchmark pyalakazam against the alakazam example datasets.

Times the main computational entry points on ``ExampleDb`` /
``ExampleTrees`` and prints a small summary table. Run with::

    python examples/benchmark.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import pyalakazam as ak


def _time(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"  {label:<34s} {dt * 1e3:9.1f} ms")
    return out, dt


def main():
    print("=" * 56)
    print("pyalakazam benchmark  (alakazam ExampleDb / ExampleTrees)")
    print("=" * 56)

    db = ak.load_example_db()
    trees = ak.load_example_trees()
    print(f"ExampleDb: {db.shape[0]} sequences, "
          f"{db['clone_id'].nunique()} clones")
    print(f"ExampleTrees: {len(trees)} lineage trees\n")

    print("Gene annotation")
    _time("getGene (1999 calls)", ak.getGene, db["v_call"].tolist())
    _time("getFamily (1999 calls)", ak.getFamily, db["v_call"].tolist())
    _time("countGenes (family, by sample)", ak.countGenes, db,
          gene="v_call", groups="sample_id", mode="family")

    print("\nAmino-acid properties")
    _time("translateDNA (1999 junctions)", ak.translateDNA,
          db["junction"].tolist())
    _time("aminoAcidProperties (1999 seqs)", ak.aminoAcidProperties,
          db[["sequence_id", "junction"]], seq="junction")

    print("\nSequence distances")
    seqs = ak.padSeqEnds(db["junction"].iloc[:60].tolist())
    seq_map = dict(zip(db["sequence_id"].iloc[:60], seqs))
    _time("pairwiseDist (60x60)", ak.pairwiseDist, seq_map)

    print("\nClonal abundance & diversity")
    _time("countClones (by sample)", ak.countClones, db, groups="sample_id")
    abund, _ = _time("estimateAbundance (nboot=200)", ak.estimateAbundance,
                     db, group="sample_id", nboot=200, seed=1)
    _time("alphaDiversity (q 0-4, nboot=200)", ak.alphaDiversity, abund,
          min_q=0, max_q=4, step_q=0.5)

    print("\nLineage reconstruction (maximum parsimony)")
    big = db["clone_id"].value_counts()
    cid = big.index[5]
    sub = db[db["clone_id"] == cid].copy()
    clone = ak.makeChangeoClone(sub, text_fields=["sample_id", "c_call"],
                                num_fields=["duplicate_count"])
    g, _ = _time(f"buildPhylipLineage (clone {cid})",
                 ak.buildPhylipLineage, clone)
    if g is not None:
        print(f"    -> {len(g.vertices)} vertices, {len(g.edges)} edges")

    print("\nTopology analysis")
    _time("getPathLengths", ak.getPathLengths, trees[22], root="Germline")
    _time("summarizeSubtrees", ak.summarizeSubtrees, trees[22],
          fields=["c_call"])
    _time("testEdges (8 trees, nperm=50)", ak.testEdges, trees[:8],
          "c_call", nperm=50, seed=1)

    print("\nDone.")


if __name__ == "__main__":
    main()
