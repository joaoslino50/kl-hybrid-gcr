#!/usr/bin/env python3
"""Batch evaluation of the Checkpoint-1 baselines on WebQSP / CWQ.

Modes:
  llm_only_zero_shot | llm_only_cot  -> local HF model, no KG (runs on your GPU)
  standard_rag                        -> top-k triples + GPT-4o-mini (needs OPENAI_API_KEY in .env)

Models are loaded ONCE. Every question is appended to a JSONL file as soon as it is
done, so a crash / Colab disconnect loses nothing: re-run with --resume to continue.

Examples (from repo root):
  python scripts/eval_baselines.py --mode llm_only_zero_shot --model Qwen/Qwen2-1.5B-Instruct --max-samples 100
  python scripts/eval_baselines.py --mode llm_only_cot --model Qwen/Qwen2-1.5B-Instruct --max-samples 100
  python scripts/eval_baselines.py --mode standard_rag --max-samples 100
  python scripts/eval_baselines.py --mode standard_rag --max-samples 500 --resume
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

from src.acquisition.dataset_loader import BenchmarkDatasetLoader
from src.baselines.llm_only import LLMOnlyBaseline
from src.baselines.standard_rag import StandardRAGBaseline
from src.decoding.final_reasoner import FinalAnswerReasoner
from src.evaluation.metrics import compute_f1, compute_hit_at_1
from src.retrieval.embedder import GraphEmbedder
from src.utils.device import get_device


def read_done(path: Path) -> list[dict]:
    rows: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["llm_only_zero_shot", "llm_only_cot", "standard_rag"])
    p.add_argument("--source", choices=["webqsp", "cwq"], default="webqsp")
    p.add_argument("--split", default="test")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct",
                   help="HF model for llm_only modes (must fit in GPU memory)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--top-k", type=int, default=10, help="triples retrieved by standard_rag")
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / f"{args.mode}_{args.source}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = read_done(out) if args.resume else []
    done_ids = {r["id"] for r in rows}
    if not args.resume and out.exists():
        out.unlink()
    if done_ids:
        print(f"Resuming: {len(done_ids)} questions already in {out}")

    device = get_device()
    print(f"Device: {device} | mode: {args.mode} | source: {args.source}/{args.split}")

    if args.mode.startswith("llm_only"):
        runner = LLMOnlyBaseline(
            model_name=args.model,
            mode="cot" if args.mode.endswith("cot") else "zero_shot",
            max_new_tokens=args.max_new_tokens,
            device=device,
            use_mock=False,
        )
        run = runner.answer
        model_label = args.model
    else:
        reasoner = FinalAnswerReasoner(model_name=args.openai_model, use_mock=False)
        if not reasoner.api_key:
            print("ERROR: OPENAI_API_KEY not found. Create a .env file in the repo root with:\n"
                  "  OPENAI_API_KEY=sk-...\n(never commit it)")
            return 1
        embedder = GraphEmbedder(device=device, use_mock=False)
        runner = StandardRAGBaseline(embedder=embedder, reasoner=reasoner, top_k=args.top_k)
        run = runner.answer
        model_label = f"{args.openai_model} + top-{args.top_k} triples"

    loader = BenchmarkDatasetLoader(dataset_name=args.source, split=args.split, streaming=True)

    t_start = time.perf_counter()
    new = 0
    with out.open("a", encoding="utf-8") as fh:
        for i, sg in enumerate(loader.iter_subgraphs(max_samples=args.max_samples)):
            if sg.question_id in done_ids:
                continue
            try:
                res = run(sg)
                pred = res.final_answer
                latency = res.latency_ms
                err = None
            except Exception as exc:  # keep going; log the failure for the error analysis
                pred, latency, err = "", 0.0, f"{type(exc).__name__}: {exc}"
                print(f"  [warn] question {sg.question_id} failed: {err}")
            row = {
                "id": sg.question_id,
                "question": sg.question,
                "gold": sg.a_entities,
                "prediction": pred,
                "hit1": compute_hit_at_1(pred, sg.a_entities) if pred else 0.0,
                "f1": compute_f1(pred, sg.a_entities) if pred else 0.0,
                "latency_ms": latency,
                "error": err,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(row)
            new += 1
            if new % 10 == 0:
                print(f"  {len(rows)} done (hit@1 so far {statistics.fmean(r['hit1'] for r in rows):.3f})")

    if not rows:
        print("No results.")
        return 1

    n = len(rows)
    ok = [r for r in rows if not r["error"]]
    print("\n" + "=" * 60)
    print(f"{args.mode} | {model_label} | {args.source}/{args.split} | N={n} "
          f"({time.perf_counter() - t_start:.0f}s this run)")
    print("=" * 60)
    print(f"  Hit@1        : {statistics.fmean(r['hit1'] for r in rows):.4f}")
    print(f"  F1           : {statistics.fmean(r['f1'] for r in rows):.4f}")
    print(f"  Mean latency : {statistics.fmean(r['latency_ms'] for r in ok):.0f} ms" if ok else "")
    print(f"  Failed       : {n - len(ok)}")
    print(f"\nPer-question results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())