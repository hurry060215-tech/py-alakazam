"""Ig lineage-tree reconstruction (port of alakazam ``R/Lineage.R``).

alakazam's :func:`buildPhylipLineage` shells out to the external PHYLIP
``dnapars`` binary. To keep this package self-contained, the maximum-
parsimony search is re-implemented in pure Python:

* a greedy stepwise-addition heuristic builds an initial tree,
* nearest-neighbour-interchange (NNI) hill-climbing refines it,
* internal-node states are reconstructed with the Fitch algorithm.

The resulting tree is then post-processed exactly as alakazam does:
inferred internal nodes that are identical to their parent (Hamming
distance 0) are collapsed, the germline is rooted as the outgroup, and
edge weights are reassigned as :func:`seqDist` distances.

For small clones (the regime of antibody lineage trees) this yields the
same most-parsimonious topology as ``dnapars``; for very large clones the
heuristic may find a near- rather than globally-optimal tree --- this is
documented as the single behavioural caveat versus the R package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .gene import getGene
from .graph import LineageTree
from .sequence import (collapseDuplicates, getDNAMatrix, maskSeqEnds,
                       maskSeqGaps, padSeqEnds, seqDist)

__all__ = [
    "ChangeoClone", "makeChangeoClone", "buildPhylipLineage",
    "graphToPhylo", "phyloToGraph",
]

_NT = ("A", "C", "G", "T")
# IUPAC nucleotide -> set of concrete bases (for Fitch on ambiguous data)
_IUPAC = {
    "A": {"A"}, "C": {"C"}, "G": {"G"}, "T": {"T"},
    "M": {"A", "C"}, "R": {"A", "G"}, "W": {"A", "T"}, "S": {"C", "G"},
    "Y": {"C", "T"}, "K": {"G", "T"}, "V": {"A", "C", "G"},
    "H": {"A", "C", "T"}, "D": {"A", "G", "T"}, "B": {"C", "G", "T"},
    "N": {"A", "C", "G", "T"}, "-": {"A", "C", "G", "T"},
    ".": {"A", "C", "G", "T"}, "?": {"A", "C", "G", "T"},
}
# IUPAC nucleotide -> 4-bit mask (bit 0=A,1=C,2=G,3=T) for vectorised Fitch
_BITMASK = {c: sum(1 << "ACGT".index(b) for b in bases)
            for c, bases in _IUPAC.items()}


def _seq_to_mask(seq: str) -> np.ndarray:
    """Encode a sequence as a per-site 4-bit IUPAC mask array."""
    return np.array([_BITMASK.get(c, 15) for c in seq], dtype=np.int8)


# popcount lookup for 4-bit masks
_POPCOUNT = np.array([bin(i).count("1") for i in range(16)], dtype=np.int8)


def _fitch_score_fast(masks: dict, edges: list, root: str) -> int:
    """Fitch parsimony length via vectorised bitmask operations.

    ``masks`` maps every node name to its per-site 4-bit mask array
    (internal nodes may be absent --- they are computed). Returns only
    the total tree length (no sequence reconstruction), making it cheap
    enough to evaluate many candidate topologies.
    """
    children = {}
    for (a, b) in edges:
        children.setdefault(a, []).append(b)

    order = []
    visited = set()
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            order.append(v)
            continue
        if v in visited:
            continue
        visited.add(v)
        stack.append((v, True))
        for c in children.get(v, []):
            stack.append((c, False))

    n_sites = len(next(iter(masks.values())))
    state: dict = {}
    score = 0
    for v in order:
        kids = children.get(v, [])
        if v in masks and not kids:
            state[v] = masks[v]
            continue
        if not kids:
            state[v] = masks[v]
            continue
        cur = state[kids[0]].copy()
        for c in kids[1:]:
            inter = cur & state[c]
            union = cur | state[c]
            empty = inter == 0
            score += int(empty.sum())
            cur = np.where(empty, union, inter)
        if v in masks:
            inter = cur & masks[v]
            empty = inter == 0
            score += int(empty.sum())
            cur = np.where(empty, cur | masks[v], inter)
        state[v] = cur.astype(np.int8)
    return score


# ==========================================================================
# ChangeoClone
# ==========================================================================
@dataclass
class ChangeoClone:
    """Pre-processed sequences for a single clonal lineage."""
    data: pd.DataFrame
    clone: str
    germline: str
    v_gene: str
    j_gene: str
    junc_len: int

    def __repr__(self):
        return (f"ChangeoClone(clone={self.clone!r}, "
                f"{len(self.data)} sequences, v_gene={self.v_gene!r})")


def makeChangeoClone(data: pd.DataFrame, id: str = "sequence_id",
                     seq: str = "sequence_alignment",
                     germ: str = "germline_alignment", v_call: str = "v_call",
                     j_call: str = "j_call", junc_len: str = "junction_length",
                     clone: str = "clone_id", mask_char: str = "N",
                     locus: str = "locus", max_mask: int = 0,
                     pad_end: bool = False, text_fields=None, num_fields=None,
                     seq_fields=None, add_count: bool = True,
                     verbose: bool = False) -> ChangeoClone:
    """Build a :class:`ChangeoClone` from one clone's AIRR records.

    Faithful port of alakazam ``makeChangeoClone``: gaps are masked,
    ragged ends masked, duplicate sequences collapsed.
    """
    text_fields = list(text_fields or [])
    num_fields = list(num_fields or [])
    seq_fields = list(seq_fields or [])

    if data[clone].nunique() > 1:
        raise ValueError(
            f"data contains {data[clone].nunique()} clone identifiers; "
            "expecting one.")
    if locus in data.columns:
        if data[locus].isna().any():
            raise ValueError(f"Missing values found in {locus} column")
        if (data[locus] != "IGH").any():
            raise ValueError("Only heavy chain (IGH) allowed in locus column.")

    tmp = data[[id, seq] + text_fields + num_fields + seq_fields].copy()
    tmp[seq] = maskSeqGaps(list(tmp[seq]), mask_char=mask_char,
                           outer_only=False)
    tmp[seq] = maskSeqEnds(list(tmp[seq]), mask_char=mask_char,
                           max_mask=max_mask, trim=False)
    germline = maskSeqGaps(str(data[germ].iloc[0]), mask_char=mask_char,
                           outer_only=False)

    if pad_end:
        tmp[seq] = padSeqEnds(list(tmp[seq]), pad_char=mask_char)
        germline = padSeqEnds(germline, pad_char=mask_char)

    lens = tmp[seq].astype(str).map(len)
    if lens.nunique() > 1 and not pad_end:
        raise ValueError(
            "All sequences are not the same length; try pad_end=True.")

    tmp = collapseDuplicates(tmp, id=id, seq=seq, text_fields=text_fields,
                             num_fields=num_fields, seq_fields=seq_fields,
                             add_count=add_count, verbose=verbose)

    tmp = tmp.rename(columns={seq: "sequence", id: "sequence_id"})

    return ChangeoClone(
        data=tmp.reset_index(drop=True),
        clone=str(data[clone].iloc[0]),
        germline=germline,
        v_gene=getGene(str(data[v_call].iloc[0])),
        j_gene=getGene(str(data[j_call].iloc[0])),
        junc_len=int(data[junc_len].iloc[0]))


# ==========================================================================
# Maximum-parsimony tree search
# ==========================================================================
def _hamming(a: str, b: str) -> int:
    """Ambiguity-aware Hamming distance (0 if IUPAC sets intersect)."""
    d = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            continue
        sa = _IUPAC.get(ca, {ca})
        sb = _IUPAC.get(cb, {cb})
        if not (sa & sb):
            d += 1
    return d


_NT_IDX = {b: i for i, b in enumerate(_NT)}


def _fitch_root(seqs: dict, edges: list, root: str):
    """Sankoff weighted parsimony: reconstruct internal sequences.

    Uses the Sankoff dynamic program over the four nucleotide states with
    a 0/1 cost matrix (cost 0 when IUPAC sets are compatible). For each
    site this finds an internal-state assignment that minimises the total
    tree length, matching the trees produced by PHYLIP ``dnapars``.

    ``seqs`` maps observed-node names to sequences; internal nodes are
    those absent from ``seqs``. Returns ``({node: sequence}, score)``.
    """
    children = {}
    for (a, b) in edges:
        children.setdefault(a, []).append(b)
    n_sites = len(next(iter(seqs.values())))

    # postorder
    order = []
    visited = set()
    stack = [(root, False)]
    while stack:
        v, processed = stack.pop()
        if processed:
            order.append(v)
            continue
        if v in visited:
            continue
        visited.add(v)
        stack.append((v, True))
        for c in children.get(v, []):
            stack.append((c, False))

    INF = 10 ** 9
    assigned = {v: [] for v in order if v not in seqs}
    total_score = 0

    for i in range(n_sites):
        # postorder Sankoff DP; all nodes handled uniformly
        cost = {}            # v -> per-state minimal subtree cost
        ptr = {}             # v -> per-state {child: chosen child state}
        for v in order:
            kids = children.get(v, [])
            choice = [{} for _ in range(4)]
            if v in seqs:
                allowed = _IUPAC.get(seqs[v][i], {seqs[v][i]})
                base = [0 if _NT[k] in allowed else INF for k in range(4)]
            else:
                base = [0, 0, 0, 0]
            cvec = list(base)
            for s in range(4):
                if base[s] >= INF:
                    cvec[s] = INF
                    continue
                tot = base[s]
                for c in kids:
                    best = INF
                    best_cs = 0
                    for cs in range(4):
                        tc = cost[c][cs] + (0 if cs == s else 1)
                        if tc < best:
                            best = tc
                            best_cs = cs
                    tot += best
                    choice[s][c] = best_cs
                cvec[s] = tot
            cost[v] = cvec
            ptr[v] = choice

        root_costs = cost[root]
        best_root = min(range(4), key=lambda s: root_costs[s])
        total_score += root_costs[best_root]

        # preorder backtrack (order is postorder, so reversed = preorder)
        state = {root: best_root}
        for v in reversed(order):
            s = state.get(v)
            if s is None:
                continue
            for c, cs in ptr[v][s].items():
                state[c] = cs
        for v in assigned:
            assigned[v].append(_NT[state[v]])

    out = dict(seqs)
    for v, chars in assigned.items():
        out[v] = "".join(chars)
    return out, total_score


def _tree_score(seqs: dict, edges: list) -> int:
    return sum(_hamming(seqs[a], seqs[b]) for (a, b) in edges)


def _build_initial_tree(taxa: list, seqs: dict):
    """Greedy stepwise addition: attach each taxon at its cheapest edge."""
    if len(taxa) < 2:
        return [], dict(seqs)

    edges = [(taxa[0], taxa[1])]
    work = dict(seqs)
    inf_count = 0

    for t in taxa[2:]:
        best = None
        best_cost = None
        for (a, b) in edges:
            inf_name = f"_inf{inf_count}"
            # internal node = Fitch of a, b, t
            inf_seq = _fitch_triplet(work[a], work[b], work[t])
            cost = (_hamming(work[a], inf_seq) + _hamming(work[b], inf_seq)
                    + _hamming(work[t], inf_seq))
            base = _hamming(work[a], work[b])
            delta = cost - base
            if best_cost is None or delta < best_cost:
                best_cost = delta
                best = (a, b, inf_seq)
        a, b, inf_seq = best
        inf_name = f"_inf{inf_count}"
        inf_count += 1
        work[inf_name] = inf_seq
        edges.remove((a, b))
        edges += [(inf_name, a), (inf_name, b), (inf_name, t)]
    return edges, work


def _fitch_triplet(s1: str, s2: str, s3: str) -> str:
    """Parsimony-consensus of three sequences (majority / deterministic)."""
    out = []
    for c1, c2, c3 in zip(s1, s2, s3):
        sets = [_IUPAC.get(c, {c}) for c in (c1, c2, c3)]
        # site state present in >=2
        counts = {}
        for s in sets:
            for ch in s:
                counts[ch] = counts.get(ch, 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], -ord(kv[0])))
        out.append(best[0])
    return "".join(out)


def _edges_to_undirected(edges):
    """Return a frozenset-keyed undirected edge set."""
    return {frozenset(e) for e in edges}


def _undirected_to_list(uedges):
    out = []
    for e in uedges:
        a, b = tuple(e)
        out.append((a, b))
    return out


def _score_topology(uedges, masks: dict) -> int:
    """Fast Fitch parsimony length of an undirected topology."""
    internal = sorted({v for e in uedges for v in e
                       if str(v).startswith("_inf")})
    edge_list = _undirected_to_list(uedges)
    if not internal:
        # rooted at an arbitrary node
        root = next(iter(masks))
        directed = _orient_edge_list(edge_list, root)
        return _fitch_score_fast(masks, directed, root)
    root = internal[0]
    directed = _orient_edge_list(edge_list, root)
    return _fitch_score_fast(masks, directed, root)


def _orient_edge_list(edges: list, root: str) -> list:
    """Orient an undirected edge list outward from ``root``."""
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    directed = []
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for w in adj.get(u, []):
            if w not in visited:
                visited.add(w)
                directed.append((u, w))
                stack.append(w)
    return directed


def _nni_optimize(edges: list, seqs: dict, masks: dict | None = None,
                  max_iter: int = 500):
    """Hill-climb the unrooted topology with nearest-neighbour interchange.

    Candidate topologies are scored with the fast bitmask Fitch routine;
    the first strictly-improving rearrangement is taken and the search
    restarts. Returns the optimised edge list.
    """
    if masks is None:
        masks = {k: _seq_to_mask(v) for k, v in seqs.items()}
    uedges = _edges_to_undirected(edges)
    cur = _score_topology(uedges, masks)

    for _ in range(max_iter):
        improved = False
        adj = {}
        for e in uedges:
            a, b = tuple(e)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        for e in list(uedges):
            u, v = tuple(e)
            if not (str(u).startswith("_inf") and str(v).startswith("_inf")):
                continue
            u_neigh = sorted(w for w in adj[u] if w != v)
            v_neigh = sorted(w for w in adj[v] if w != u)
            if len(u_neigh) != 2 or len(v_neigh) != 2:
                continue
            for vk in (v_neigh[0], v_neigh[1]):
                a = u_neigh[0]
                new_u = set(uedges)
                new_u.discard(frozenset((u, a)))
                new_u.discard(frozenset((v, vk)))
                new_u.add(frozenset((u, vk)))
                new_u.add(frozenset((v, a)))
                sc2 = _score_topology(new_u, masks)
                if sc2 < cur:
                    uedges = new_u
                    cur = sc2
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return _undirected_to_list(uedges), cur


def _reconstruct_for(uedges, seqs: dict) -> dict:
    """Reconstruct internal states for an undirected topology."""
    internal = sorted({v for e in uedges for v in e
                       if str(v).startswith("_inf")})
    if not internal:
        return dict(seqs)
    edge_list = _undirected_to_list(uedges)
    return _reconstruct_unrooted(edge_list, seqs, internal)


def _search_best_tree(taxa: list, seqs: dict):
    """Stepwise-addition + NNI/SPR from several taxon orders.

    Multiple addition orders are tried because greedy stepwise addition
    is order-dependent; NNI and SPR hill-climbing then refine each
    candidate and the most-parsimonious tree is kept.
    """
    if len(taxa) < 2:
        return [], dict(seqs)

    rest = [t for t in taxa if t != "Germline"]
    germ = seqs["Germline"]
    by_dist = sorted(rest, key=lambda t: _hamming(seqs[t], germ))
    orders = []
    seen_orders = set()
    candidates = [taxa, ["Germline"] + by_dist,
                  ["Germline"] + by_dist[::-1], ["Germline"] + rest[::-1]]
    # a few random shuffles for larger clones to escape local optima
    if len(rest) > 6:
        rng = np.random.default_rng(0)
        for _ in range(4):
            sh = list(rest)
            rng.shuffle(sh)
            candidates.append(["Germline"] + sh)
    for order in candidates:
        key = tuple(order)
        if key not in seen_orders:
            seen_orders.add(key)
            orders.append(order)

    masks = {k: _seq_to_mask(v) for k, v in seqs.items()}
    n_taxa = len(rest)
    spr_iter = 40 if n_taxa <= 40 else 15

    best_edges = None
    best_score = None
    for order in orders:
        e0, _ = _build_initial_tree(order, seqs)
        if not e0:
            continue
        e1, _ = _nni_optimize(e0, seqs, masks=masks)
        e1, _ = _spr_optimize(e1, seqs, masks=masks, max_iter=spr_iter)
        e1, sc = _nni_optimize(e1, seqs, masks=masks)
        if best_score is None or sc < best_score:
            best_score = sc
            best_edges = e1
    # reconstruct internal sequences for the winning topology
    best_work = _reconstruct_for(_edges_to_undirected(best_edges), seqs)
    return best_edges, best_work


def _spr_optimize(edges: list, seqs: dict, masks: dict | None = None,
                  max_iter: int = 40):
    """Subtree-pruning-and-regrafting hill-climb (escapes NNI local optima).

    Each leaf is detached and reattached at the cheapest edge; the first
    strictly-improving move is taken and the search restarts. Candidate
    topologies are scored with the fast bitmask Fitch routine.
    """
    if masks is None:
        masks = {k: _seq_to_mask(v) for k, v in seqs.items()}
    uedges = _edges_to_undirected(edges)
    cur = _score_topology(uedges, masks)

    for _ in range(max_iter):
        adj = {}
        for e in uedges:
            a, b = tuple(e)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        leaves = [v for v in adj if len(adj[v]) == 1
                  and not str(v).startswith("_inf")]
        improved = False
        for leaf in leaves:
            parent = next(iter(adj[leaf]))
            if len(adj.get(parent, [])) != 3:
                continue
            par_neigh = [w for w in adj[parent] if w != leaf]
            pruned = set(uedges)
            pruned.discard(frozenset((leaf, parent)))
            pruned.discard(frozenset((parent, par_neigh[0])))
            pruned.discard(frozenset((parent, par_neigh[1])))
            pruned.add(frozenset((par_neigh[0], par_neigh[1])))
            for tgt in list(pruned):
                x, y = tuple(tgt)
                new_inf = parent
                new_u = set(pruned)
                new_u.discard(tgt)
                new_u.add(frozenset((x, new_inf)))
                new_u.add(frozenset((new_inf, y)))
                new_u.add(frozenset((new_inf, leaf)))
                sc2 = _score_topology(new_u, masks)
                if sc2 < cur:
                    uedges = new_u
                    cur = sc2
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return _undirected_to_list(uedges), cur


def _reconstruct_unrooted(edges: list, seqs: dict, internal: list) -> dict:
    """Reconstruct internal states for an unrooted topology via rooted Fitch."""
    if not internal:
        return dict(seqs)
    root = internal[0]
    # orient edges away from root
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    directed = []
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in visited:
                visited.add(w)
                directed.append((u, w))
                stack.append(w)
    assigned, _ = _fitch_root(seqs, directed, root)
    return assigned


# ==========================================================================
# buildPhylipLineage
# ==========================================================================
def buildPhylipLineage(clone: ChangeoClone, phylip_exec=None,
                       dist_mat: pd.DataFrame | None = None,
                       branch_length: str = "mutations") -> LineageTree | None:
    """Reconstruct an Ig lineage tree via maximum parsimony (pure Python).

    Drop-in replacement for alakazam ``buildPhylipLineage`` that does not
    require the external PHYLIP binary. ``phylip_exec`` is accepted and
    ignored for API compatibility.

    Parameters
    ----------
    clone : ChangeoClone
        Pre-processed clone (see :func:`makeChangeoClone`).
    dist_mat : DataFrame, optional
        Character distance matrix for edge weights; defaults to
        ``getDNAMatrix(gap=0)``.
    branch_length : {"mutations", "distance"}
        Edge-weight definition.

    Returns
    -------
    LineageTree or None
        ``None`` if the clone has fewer than two unique sequences.
    """
    if dist_mat is None:
        dist_mat = getDNAMatrix(gap=0)
    if len(clone.data) < 2:
        return None

    germ_len = len(clone.germline)
    seq_lens = clone.data["sequence"].astype(str).map(len).unique()
    if germ_len == 0:
        raise ValueError(f"Clone {clone.clone} has no germline sequence.")
    if len(seq_lens) != 1:
        raise ValueError(f"Clone {clone.clone} sequences differ in length.")
    if seq_lens[0] != germ_len:
        raise ValueError("Germline and input sequences differ in length.")

    obs = {row["sequence_id"]: str(row["sequence"])
           for _, row in clone.data.iterrows()}
    germ = clone.germline

    # taxa for the parsimony search: germline + observed sequences
    taxa = ["Germline"] + list(obs.keys())
    seqs = {"Germline": germ, **obs}

    # try several stepwise-addition orders, NNI-optimise each, keep the best
    edges, work = _search_best_tree(taxa, seqs)
    if not edges:
        return None

    # rename internal nodes -> Inferred1..n
    inf_nodes = sorted({v for e in edges for v in e if v.startswith("_inf")})
    rename = {old: f"Inferred{i + 1}" for i, old in enumerate(inf_nodes)}
    edges = [(rename.get(a, a), rename.get(b, b)) for (a, b) in edges]
    work = {rename.get(k, k): v for k, v in work.items()}

    # root at Germline: orient the tree outward
    directed = _orient_from_root(edges, "Germline")

    # final internal-state reconstruction with germline as outgroup
    internal = [v for v in work if v.startswith("Inferred")]
    if internal:
        assigned, _ = _fitch_root({k: v for k, v in seqs.items()},
                                  directed, "Germline")
        for v in internal:
            if v in assigned:
                work[v] = assigned[v]

    # build edge dataframe with seqDist weights
    edge_rows = []
    for (a, b) in directed:
        w = seqDist(work[a], work[b], dist_mat)
        edge_rows.append([a, b, w])
    edges_df = pd.DataFrame(edge_rows, columns=["from", "to", "weight"])

    if branch_length == "mutations":
        edges_df = _collapse_inferred(edges_df, work, dist_mat)

    # keep only inferred nodes still present
    present = set(edges_df["from"]) | set(edges_df["to"])
    inf_df = pd.DataFrame([
        {"sequence_id": v, "sequence": work[v]}
        for v in sorted(v for v in work
                        if v.startswith("Inferred") and v in present)])
    clone_data = pd.concat([clone.data, inf_df], ignore_index=True)

    return _to_graph(edges_df, clone, clone_data, work)


def _orient_from_root(edges: list, root: str) -> list:
    """Orient an undirected edge list as a tree rooted at ``root``."""
    adj = {}
    for (a, b) in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    directed = []
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for w in sorted(adj.get(u, [])):
            if w not in visited:
                visited.add(w)
                directed.append((u, w))
                stack.append(w)
    return directed


def _collapse_inferred(edges_df: pd.DataFrame, work: dict,
                       dist_mat) -> pd.DataFrame:
    """Collapse zero-weight inferred parents (port of modifyPhylipEdges)."""
    edges_df = edges_df.copy().reset_index(drop=True)
    while True:
        mask = ((edges_df["weight"] == 0)
                & (edges_df["from"] != "Germline")
                & edges_df["from"].astype(str).str.match(r"^Inferred\d+$"))
        rows = edges_df.index[mask].tolist()
        if not rows:
            break
        r = rows[0]
        parent = edges_df.at[r, "from"]
        child = edges_df.at[r, "to"]
        # replace parent with child everywhere
        edges_df["from"] = edges_df["from"].replace(parent, child)
        edges_df["to"] = edges_df["to"].replace(parent, child)
        # drop self-loop row
        edges_df = edges_df[~((edges_df["from"] == child)
                              & (edges_df["to"] == child))]
        edges_df = edges_df.reset_index(drop=True)
        # recompute weights
        for i in edges_df.index:
            a = edges_df.at[i, "from"]
            b = edges_df.at[i, "to"]
            edges_df.at[i, "weight"] = seqDist(work[a], work[b], dist_mat)
    return edges_df.reset_index(drop=True)


def _to_graph(edges_df: pd.DataFrame, clone: ChangeoClone,
              clone_data: pd.DataFrame, work: dict) -> LineageTree:
    """Assemble a :class:`LineageTree` from edges + clone annotations."""
    g = LineageTree()
    for _, row in edges_df.iterrows():
        g.add_edge(row["from"], row["to"], weight=row["weight"],
                   label=row["weight"])

    seq_map = {r["sequence_id"]: r["sequence"]
               for _, r in clone_data.iterrows()}
    seq_map["Germline"] = clone.germline
    ann_fields = [c for c in clone_data.columns
                  if c not in ("sequence_id", "sequence")]

    for v in g.vertices:
        g.set_vertex_attr("sequence", v, seq_map.get(v, work.get(v)))
        g.set_vertex_attr("label", v, v)
        row = clone_data[clone_data["sequence_id"] == v]
        for f in ann_fields:
            if len(row):
                g.set_vertex_attr(f, v, row.iloc[0][f])
            else:
                g.set_vertex_attr(f, v, np.nan)

    g.graph_attr.update(clone=clone.clone, v_gene=clone.v_gene,
                        j_gene=clone.j_gene, junc_len=clone.junc_len)
    return g


# ==========================================================================
# Graph / phylo conversion
# ==========================================================================
@dataclass
class Phylo:
    """Minimal ape-``phylo``-like container (edges + labels)."""
    edge: np.ndarray
    edge_length: np.ndarray
    tip_label: list
    node_label: list
    nodes: dict = field(default_factory=dict)
    germid: str | None = None


def graphToPhylo(graph: LineageTree) -> Phylo:
    """Convert a :class:`LineageTree` to an ape-``phylo``-style object."""
    leaves = [v for v in graph.vertices if graph.outdegree(v) == 0]
    internal = [v for v in graph.vertices if graph.outdegree(v) > 0]
    order = leaves + internal
    idx = {v: i + 1 for i, v in enumerate(order)}
    edge = np.array([[idx[a], idx[b]] for (a, b) in graph.edges])
    weights = graph.edge_attr.get("weight", {})
    elen = np.array([weights.get(e, 0) for e in graph.edges], dtype=float)
    return Phylo(edge=edge, edge_length=elen, tip_label=leaves,
                 node_label=internal)


def phyloToGraph(phylo: Phylo, germline: str | None = "Germline"
                 ) -> LineageTree:
    """Convert an ape-``phylo``-style object back to a :class:`LineageTree`."""
    n = len(phylo.tip_label) + len(phylo.node_label)
    names = [None] * n
    for i, t in enumerate(phylo.tip_label):
        names[i] = t
    for j, lbl in enumerate(phylo.node_label):
        names[len(phylo.tip_label) + j] = lbl
    g = LineageTree()
    for (a, b), w in zip(phylo.edge, phylo.edge_length):
        g.add_edge(names[a - 1], names[b - 1], weight=w, label=w)
    return g
