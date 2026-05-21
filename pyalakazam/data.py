"""Bundled example datasets (the alakazam ``data/`` objects).

``ExampleDb`` and ``SingleDb`` are AIRR-format B-cell repertoire tables;
``ExampleTrees`` is a list of annotated lineage trees.
"""
from __future__ import annotations

import gzip
import os

import pandas as pd

from .graph import LineageTree

__all__ = ["load_example_db", "load_single_db", "load_example_trees"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_example_db() -> pd.DataFrame:
    """Load ``ExampleDb`` --- a 1999-sequence B-cell AIRR data frame."""
    path = os.path.join(_DATA_DIR, "ExampleDb.tsv.gz")
    with gzip.open(path, "rt") as fh:
        return pd.read_csv(fh, sep="\t", dtype={"sequence_id": str})


def load_single_db() -> pd.DataFrame:
    """Load ``SingleDb`` --- a single-sequence AIRR example record."""
    path = os.path.join(_DATA_DIR, "SingleDb.tsv.gz")
    with gzip.open(path, "rt") as fh:
        return pd.read_csv(fh, sep="\t", dtype={"sequence_id": str})


def load_example_trees() -> list[LineageTree]:
    """Load ``ExampleTrees`` --- 49 annotated Ig lineage trees."""
    path = os.path.join(_DATA_DIR, "ExampleTrees.tsv.gz")
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str)

    trees: dict[int, LineageTree] = {}
    for idx, sub in df.groupby("tree_idx"):
        g = LineageTree()
        verts = sub[sub["record_type"] == "V"]
        edges = sub[sub["record_type"] == "E"]
        for _, r in verts.iterrows():
            attrs = {}
            seq = r.get("sequence")
            attrs["sequence"] = seq if (isinstance(seq, str) and seq) else None
            for col in ("sample_id", "c_call"):
                v = r.get(col)
                attrs[col] = v if (isinstance(v, str) and v != "") else None
            dc = r.get("duplicate_count")
            attrs["duplicate_count"] = (float(dc) if (isinstance(dc, str)
                                        and dc != "") else None)
            attrs["label"] = r["name"]
            g.add_vertex(r["name"], **attrs)
        for _, r in edges.iterrows():
            w = float(r["weight"])
            g.add_edge(r["from"], r["to"], weight=w, label=w)
        trees[int(idx)] = g
    return [trees[i] for i in sorted(trees)]
