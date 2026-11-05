#!/usr/bin/env python3
"""Diagnostic: look inside the PCST call on a few real questions.

Usage (from repo root):
  python scripts/debug_pcst.py
  python scripts/debug_pcst.py --n 3 --base-edge-cost 0.05 --anchor-prize 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import src.retrieval.pcst_retriever as m
from src.acquisition.dataset_loader import BenchmarkDatasetLoader
from src.retrieval.embedder import GraphEmbedder
from src.utils.device import get_device

_orig = m.pcst_fast.pcst_fast


def _wrapped(edges, prizes, costs, root, num_clusters, pruning, verbosity):
    sel_v, sel_e = _orig(edges, prizes, costs, root, num_clusters, pruning, verbosity)
    n_real_nodes = None  # filled by caller
    print("  --- inside pcst_fast ---")
    print(f"  input : {len(prizes)} vertices, {len(edges)} edges")
    pos = prizes > 0
    print(f"  prizes: {int(pos.sum())} vertices with prize>0, max={prizes.max():.3f}, "
          f"top5={np.sort(prizes)[-5:][::-1].round(3).tolist()}")
    print(f"  costs : min={costs.min():.4f} max={costs.max():.4f} "
          f"zero-cost edges={int((costs == 0).sum())}")
    print(f"  output: {len(sel_v)} selected vertices, {len(sel_e)} selected edges")
    print(f"          pruning={pruning!r} num_clusters={num_clusters} root={root}")
    print(f"  sel_v[:15]          = {sel_v[:15].tolist()}")
    print(f"  sel_e[:15]          = {sel_e[:15].tolist()}")
    print(f"  edges[sel_e][:15]   = {edges[sel_e[:15]].tolist()}")
    print(f"  costs[sel_e][:15]   = {costs[sel_e[:15]].round(3).tolist()}")
    print(f"  distinct edge ids selected: {len(set(sel_e.tolist()))}")
    return sel_v, sel_e


m.pcst_fast.pcst_fast = _wrapped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--topk-nodes", type=int, default=10)
    p.add_argument("--topk-edges", type=int, default=10)
    p.add_argument("--base-edge-cost", type=float, default=0.5)
    p.add_argument("--anchor-prize", type=float, default=4.0)
    args = p.parse_args()

    n_lines = len(Path(m.__file__).read_text(encoding="utf-8").splitlines())
    print(f"pcst_retriever.py: {n_lines} lines at {m.__file__}")
    embedder = GraphEmbedder(device=get_device(), use_mock=False)
    retriever = m.PCSTRetriever(
        embedder=embedder,
        topk_nodes=args.topk_nodes,
        topk_edges=args.topk_edges,
        base_edge_cost=args.base_edge_cost,
        anchor_prize=args.anchor_prize,
    )
    loader = BenchmarkDatasetLoader(dataset_name="webqsp", split="test", streaming=True)

    for sg in loader.iter_subgraphs(max_samples=args.n):
        nodes = {t.head for t in sg.triples} | {t.tail for t in sg.triples}
        print("=" * 70)
        print(f"Q: {sg.question}")
        print(f"q_entities={sg.q_entities}  in graph? {[q in nodes for q in sg.q_entities]}")
        print(f"a_entities={sg.a_entities[:5]}  in graph? {[a in nodes for a in sg.a_entities[:5]]}")
        print(f"graph: {len(sg.triples)} triples, {len(nodes)} nodes")
        out = retriever.retrieve(sg)
        print(f"RESULT: {len(out.triples)} triples kept")
        for t in out.triples[:5]:
            print("   ", t.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())