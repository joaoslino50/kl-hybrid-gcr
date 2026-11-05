#!/usr/bin/env python3
"""Measure PCST retrieval on a real benchmark (SBERT real, no LLM, no API key needed).

Reports, over N questions:
  - avg triples/nodes before and after PCST (how much the graph shrinks)
  - answer-path recall on the FULL subgraph (upper bound)
  - answer-path recall on the PCST subgraph (what we actually keep)

Usage (from repo root):
  python scripts/eval_retrieval.py --source webqsp --max-samples 200
  python scripts/eval_retrieval.py --source webqsp --max-samples 200 --anchor-prize 8 --topk-nodes 20
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.acquisition.dataset_loader import BenchmarkDatasetLoader
from src.evaluation.metrics import compute_answer_path_recall
from src.retrieval.embedder import GraphEmbedder
from src.retrieval.pcst_retriever import PCSTRetriever
from src.utils.device import get_device


def n_nodes(subgraph) -> int:
    nodes = set()
    for t in subgraph.triples:
        nodes.add(t.head)
        nodes.add(t.tail)
    return len(nodes)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    p.add_argument("--source", choices=["webqsp", "cwq"], default="webqsp")
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--topk-nodes", type=int, default=None)
    p.add_argument("--topk-edges", type=int, default=None)
    p.add_argument("--base-edge-cost", type=float, default=None)
    p.add_argument("--anchor-prize", type=float, default=None)
    p.add_argument("--out", default=None, help="JSONL with per-question results")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    ret = cfg.get("retrieval", {})
    pc = ret.get("pcst", {})

    def pick(cli, key, default):
        return cli if cli is not None else pc.get(key, default)

    device = get_device()
    embedder = GraphEmbedder(
        model_name=ret.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=device,
        batch_size=int(ret.get("batch_size", 64)),
        use_mock=False,  # REAL SBERT
    )
    retriever = PCSTRetriever(
        embedder=embedder,
        topk_nodes=int(pick(args.topk_nodes, "topk_nodes", 10)),
        topk_edges=int(pick(args.topk_edges, "topk_edges", 10)),
        base_edge_cost=float(pick(args.base_edge_cost, "base_edge_cost", 0.5)),
        anchor_prize=float(pick(args.anchor_prize, "anchor_prize", 4.0)),
        num_clusters=int(pc.get("num_clusters", 1)),
        pruning=str(pc.get("pruning", "gw")),
    )
    print(
        f"PCST params: topk_nodes={retriever.topk_nodes} topk_edges={retriever.topk_edges} "
        f"edge_cost={retriever.base_edge_cost} anchor_prize={retriever.anchor_prize}"
    )

    loader = BenchmarkDatasetLoader(dataset_name=args.source, split=args.split, streaming=True)

    rows = []
    t_start = time.perf_counter()
    for i, full in enumerate(loader.iter_subgraphs(max_samples=args.max_samples)):
        pruned = retriever.retrieve(full)
        row = {
            "id": full.question_id,
            "question": full.question,
            "triples_full": len(full.triples),
            "triples_pruned": len(pruned.triples),
            "nodes_full": n_nodes(full),
            "nodes_pruned": n_nodes(pruned),
            "has_gold": bool(full.a_entities),
            "recall_full": compute_answer_path_recall(full),
            "recall_pruned": compute_answer_path_recall(pruned),
        }
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1} questions...")

    if not rows:
        print("No samples loaded.")
        return 1

    n = len(rows)
    mean = lambda k: statistics.fmean(r[k] for r in rows)  # noqa: E731
    print("\n" + "=" * 60)
    print(f"PCST retrieval on {args.source}/{args.split}, N={n} ({time.perf_counter() - t_start:.1f}s)")
    print("=" * 60)
    print(f"  Triples  full -> pruned : {mean('triples_full'):.1f} -> {mean('triples_pruned'):.1f}")
    print(f"  Nodes    full -> pruned : {mean('nodes_full'):.1f} -> {mean('nodes_pruned'):.1f}")
    print(f"  Recall (full subgraph, upper bound) : {mean('recall_full'):.4f}")
    print(f"  Recall (PCST subgraph)              : {mean('recall_pruned'):.4f}")
    lost = sum(1 for r in rows if r["recall_full"] > 0 and r["recall_pruned"] == 0)
    reach = sum(1 for r in rows if r["recall_full"] > 0)
    print(f"  Questions where PCST lost the answer: {lost} of {reach} reachable ({100 * lost / max(reach, 1):.1f}%)")

    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / f"retrieval_{args.source}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPer-question results saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())