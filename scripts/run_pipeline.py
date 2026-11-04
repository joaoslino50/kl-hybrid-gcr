#!/usr/bin/env python3
"""CLI entry point for the 3-stage Hybrid PCST + GCR KGQA pipeline and baselines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.acquisition.dataset_loader import BenchmarkDatasetLoader, load_mock_subgraphs
from src.acquisition.sparql_client import WikidataSPARQLClient
from src.acquisition.visualizer import export_subgraph_html
from src.baselines.llm_only import LLMOnlyBaseline
from src.baselines.standard_rag import StandardRAGBaseline
from src.decoding.constrained_llm import ConstrainedPathGenerator
from src.decoding.final_reasoner import FinalAnswerReasoner
from src.evaluation.metrics import (
    compute_answer_path_recall,
    compute_f1,
    compute_faithful_reasoning_ratio,
    compute_hit_at_1,
    compute_latency_stats,
)
from src.retrieval.embedder import GraphEmbedder
from src.retrieval.pcst_retriever import PCSTRetriever
from src.utils.device import get_device
from src.utils.types import ReasoningOutput, SubgraphData


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description="Run the Hybrid PCST + Graph-Constrained Reasoning (GCR) KGQA Pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--source",
        choices=["mock", "webqsp", "cwq", "wikidata"],
        default=None,
        help="Override acquisition source (default read from YAML config).",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "full_gcr", "standard_rag", "llm_only_zero_shot", "llm_only_cot"],
        default="hybrid",
        help="Pipeline variant or baseline to run.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2,
        help="Maximum number of QA examples to process.",
    )
    parser.add_argument(
        "--use-mock-models",
        action="store_true",
        help="Use lightweight deterministic mock embedder/LLMs instead of downloading weights.",
    )
    parser.add_argument(
        "--export-html",
        type=str,
        default=None,
        help="Optional path to export an interactive PyVis HTML visualization of the first sample.",
    )
    return parser.parse_args()


def acquire_subgraphs(
    source: str,
    cfg: dict[str, Any],
    max_samples: int,
    use_mock: bool,
) -> list[SubgraphData]:
    """Stage 1: Acquire L-hop subgraphs from mock data, HF benchmarks, or Wikidata SPARQL."""
    acq_cfg = cfg.get("acquisition", {})
    if source == "mock":
        return load_mock_subgraphs()[:max_samples]

    if source in {"webqsp", "cwq"}:
        loader = BenchmarkDatasetLoader(
            dataset_name=source,
            split=acq_cfg.get("dataset_split", "test"),
            streaming=True,
        )
        return loader.load(max_samples=max_samples)

    if source == "wikidata":
        sparql_cfg = acq_cfg.get("sparql", {})
        client = WikidataSPARQLClient(
            endpoint_url=sparql_cfg.get("endpoint_url", "https://query.wikidata.org/sparql"),
            user_agent=sparql_cfg.get(
                "user_agent",
                "KL-Hybrid-GCR-Research/0.1 (University of Coimbra; KGQA Project)",
            ),
            timeout_sec=int(sparql_cfg.get("timeout_sec", 30)),
            max_triples_per_hop=int(sparql_cfg.get("max_triples_per_hop", 50)),
        )
        return [
            client.fetch_l_hop_subgraph(
                question_id="wikidata_001",
                question="What is the capital of Portugal?",
                seed_entities=["Q45"],
                num_hops=int(acq_cfg.get("max_hops", 2)),
                use_mock=use_mock,
            )
        ]

    raise ValueError(f"Unsupported source: {source}")


def main() -> int:
    """Execute the configured KGQA pipeline and print evaluation metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    source = args.source or cfg.get("acquisition", {}).get("source", "mock")
    subgraphs = acquire_subgraphs(
        source=source,
        cfg=cfg,
        max_samples=args.max_samples,
        use_mock=args.use_mock_models,
    )

    ret_cfg = cfg.get("retrieval", {})
    pcst_cfg = ret_cfg.get("pcst", {})
    dec_cfg = cfg.get("decoding", {})
    pgen_cfg = dec_cfg.get("path_generator", {})
    fres_cfg = dec_cfg.get("final_reasoner", {})
    base_cfg = cfg.get("baselines", {})

    embedder = GraphEmbedder(
        model_name=ret_cfg.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=device,
        batch_size=int(ret_cfg.get("batch_size", 64)),
        use_mock=args.use_mock_models,
    )
    pcst_retriever = PCSTRetriever(
        embedder=embedder,
        topk_nodes=int(pcst_cfg.get("topk_nodes", 10)),
        topk_edges=int(pcst_cfg.get("topk_edges", 10)),
        base_edge_cost=float(pcst_cfg.get("base_edge_cost", 0.5)),
        anchor_prize=float(pcst_cfg.get("anchor_prize", 4.0)),
        num_clusters=int(pcst_cfg.get("num_clusters", 1)),
        pruning=str(pcst_cfg.get("pruning", "gw")),
    )
    path_generator = ConstrainedPathGenerator(
        model_name=pgen_cfg.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct"),
        max_path_hops=int(pgen_cfg.get("max_path_hops", 2)),
        max_paths_in_trie=int(pgen_cfg.get("max_paths_in_trie", 512)),
        num_beams=int(pgen_cfg.get("num_beams", 3)),
        max_new_tokens=int(pgen_cfg.get("max_new_tokens", 64)),
        device=device,
        use_mock=args.use_mock_models,
    )
    final_reasoner = FinalAnswerReasoner(
        model_name=fres_cfg.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct"),
        temperature=float(fres_cfg.get("temperature", 0.0)),
        max_tokens=int(fres_cfg.get("max_tokens", 64)),
        device=device,
        use_mock=args.use_mock_models,
    )

    outputs: list[ReasoningOutput] = []
    hits: list[float] = []
    f1s: list[float] = []
    faithfulness_scores: list[float] = []
    recalls: list[float] = []

    for idx, full_subgraph in enumerate(subgraphs):
        t0 = time.perf_counter()
        active_subgraph = full_subgraph

        if args.mode == "hybrid":
            active_subgraph = pcst_retriever.retrieve(full_subgraph)
            paths = path_generator.generate_paths(active_subgraph)
            prior_ms = (time.perf_counter() - t0) * 1000.0
            result = final_reasoner.deduce_answer(
                active_subgraph, paths, prior_latency_ms=prior_ms
            )
        elif args.mode == "full_gcr":
            paths = path_generator.generate_paths(full_subgraph)
            prior_ms = (time.perf_counter() - t0) * 1000.0
            result = final_reasoner.deduce_answer(
                full_subgraph, paths, prior_latency_ms=prior_ms
            )
        elif args.mode == "standard_rag":
            rag = StandardRAGBaseline(
                embedder=embedder,
                reasoner=final_reasoner,
                top_k=int(base_cfg.get("standard_rag", {}).get("top_k", 10)),
            )
            result = rag.answer(full_subgraph)
        elif args.mode in {"llm_only_zero_shot", "llm_only_cot"}:
            llm_mode = "cot" if args.mode.endswith("cot") else "zero_shot"
            llm_only = LLMOnlyBaseline(
                model_name=base_cfg.get("llm_only", {}).get(
                    "model_name", "meta-llama/Meta-Llama-3-8B-Instruct"
                ),
                mode=llm_mode,  # type: ignore[arg-type]
                device=device,
                use_mock=args.use_mock_models,
            )
            result = llm_only.answer(full_subgraph)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        outputs.append(result)
        hits.append(compute_hit_at_1(result.final_answer, full_subgraph.a_entities))
        f1s.append(compute_f1(result.final_answer, full_subgraph.a_entities))
        faithfulness_scores.append(
            compute_faithful_reasoning_ratio(result.generated_paths, full_subgraph)
        )
        recalls.append(compute_answer_path_recall(active_subgraph))

        if idx == 0 and args.export_html:
            export_subgraph_html(
                active_subgraph,
                output_path=args.export_html,
                highlight_paths=result.generated_paths,
            )

        print(f"\n--- Sample [{full_subgraph.question_id}] ---")
        print(f"Question      : {full_subgraph.question}")
        print(f"Gold Answers  : {full_subgraph.a_entities}")
        print(f"Triples (in->active): {len(full_subgraph.triples)} -> {len(active_subgraph.triples)}")
        print(f"Predicted     : {result.final_answer}")
        print(f"Paths Count   : {len(result.generated_paths)}")
        print(
            f"Hit@1={hits[-1]:.2f} | F1={f1s[-1]:.2f} | "
            f"Faithfulness={faithfulness_scores[-1]:.2f} | "
            f"PathRecall={recalls[-1]:.2f} | Latency={result.latency_ms:.2f}ms"
        )

    if outputs:
        n = len(outputs)
        latency_stats = compute_latency_stats(outputs)
        print("\n" + "=" * 64)
        print(f"Aggregate Summary (mode={args.mode}, source={source}, N={n})")
        print("=" * 64)
        print(f"  Hit@1                     : {sum(hits) / n:.4f}")
        print(f"  F1                        : {sum(f1s) / n:.4f}")
        print(f"  Faithful Reasoning Ratio  : {sum(faithfulness_scores) / n:.4f}")
        print(f"  Answer-Path Recall        : {sum(recalls) / n:.4f}")
        print(f"  Mean Latency (ms)         : {latency_stats['mean_ms']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

