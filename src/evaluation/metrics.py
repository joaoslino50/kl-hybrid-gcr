"""Evaluation metrics: Hit@1, F1, Faithful Reasoning Ratio, Answer-Path Recall, and Latency."""

from __future__ import annotations

from collections import Counter
import re
import statistics
import string
from typing import Sequence

import networkx as nx

from src.utils.types import ReasoningOutput, SubgraphData, Triple

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation and English articles, and collapse whitespace."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    no_articles = _ARTICLES_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", no_articles).strip()


def _contains_token_subsequence(haystack_tokens: list[str], needle_tokens: list[str]) -> bool:
    """Return True if `needle_tokens` occurs as a contiguous token subsequence in `haystack_tokens`."""
    if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
        return False
    window = len(needle_tokens)
    return any(
        haystack_tokens[i : i + window] == needle_tokens
        for i in range(len(haystack_tokens) - window + 1)
    )


def _entity_matches(prediction: str, gold: str) -> bool:
    """Check whether normalized `gold` matches normalized `prediction` at word boundaries."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold)
    if not norm_pred or not norm_gold:
        return False
    if norm_pred == norm_gold:
        return True
    return _contains_token_subsequence(norm_pred.split(), norm_gold.split())


def compute_hit_at_1(prediction: str | Sequence[str], gold_answers: Sequence[str]) -> float:
    """Compute Hit@1 accuracy (1.0 if top predicted answer matches any gold answer, else 0.0).

    Args:
        prediction: Predicted answer string (or ranked list of answer strings).
        gold_answers: Ground-truth answer entities/strings.

    Returns:
        float: `1.0` if the top prediction matches at least one gold answer, else `0.0`.
    """
    if not gold_answers:
        return 0.0

    if isinstance(prediction, str):
        lines = [seg.strip() for seg in re.split(r"[\n|]", prediction) if seg.strip()]
        top_pred = lines[0] if lines else ""
    else:
        top_pred = str(prediction[0]).strip() if prediction else ""

    if not top_pred:
        return 0.0

    for gold in gold_answers:
        if _entity_matches(top_pred, gold):
            return 1.0
    return 0.0


def _token_f1_single(prediction: str, gold: str) -> float:
    """Compute SQuAD-style token-level F1 between two single answer strings."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def compute_f1(predictions: str | Sequence[str], gold_answers: Sequence[str]) -> float:
    """Compute F1 score supporting both multi-entity KGQA sets and single-string QA.

    When multiple predicted or gold entities are evaluated, computes entity-level
    Precision/Recall/F1 (standard RoG / WebQSP / CWQ protocol). When a single
    predicted string has partial lexical overlap with a single gold answer, falls
    back to token-level F1.

    Args:
        predictions: Predicted answer string (optionally delimited by newline, `|`, or `,`)
            or sequence of predicted entity strings.
        gold_answers: Sequence of ground-truth answer entities.

    Returns:
        float: F1 score in `[0.0, 1.0]`.
    """
    cleaned_golds = [g.strip() for g in gold_answers if g and g.strip()]
    if not cleaned_golds:
        return 0.0

    if isinstance(predictions, str):
        if "\n" in predictions or "|" in predictions:
            pred_list = [p.strip() for p in re.split(r"[\n|]", predictions) if p.strip()]
        elif "," in predictions and len(cleaned_golds) > 1:
            pred_list = [p.strip() for p in predictions.split(",") if p.strip()]
        else:
            pred_list = [predictions.strip()] if predictions.strip() else []
    else:
        pred_list = [str(p).strip() for p in predictions if str(p).strip()]

    if not pred_list:
        return 0.0

    # Entity-level matching across predicted and gold answer sets
    matched_preds = sum(
        1 for pred in pred_list if any(_entity_matches(pred, gold) for gold in cleaned_golds)
    )
    matched_golds = sum(
        1 for gold in cleaned_golds if any(_entity_matches(pred, gold) for pred in pred_list)
    )

    if matched_preds > 0 and matched_golds > 0:
        precision = matched_preds / len(pred_list)
        recall = matched_golds / len(cleaned_golds)
        entity_f1 = (2.0 * precision * recall) / (precision + recall)
        if len(pred_list) == 1 and len(cleaned_golds) == 1:
            # If prediction contains extra noisy words around gold, reflect token precision
            norm_p = normalize_answer(pred_list[0])
            norm_g = normalize_answer(cleaned_golds[0])
            if norm_p == norm_g:
                return 1.0
            return _token_f1_single(pred_list[0], cleaned_golds[0])
        return entity_f1

    # Partial token-overlap fallback when no full entity matched
    if len(pred_list) == 1:
        return max(_token_f1_single(pred_list[0], gold) for gold in cleaned_golds)
    return 0.0


def compute_faithful_reasoning_ratio(
    generated_paths: Sequence[Sequence[Triple]],
    ground_truth_subgraph: SubgraphData,
) -> float:
    """Compute share of generated path triples that exist in the ground-truth `SubgraphData`.

    Args:
        generated_paths: Generated multi-hop reasoning paths (`list[list[Triple]]`).
        ground_truth_subgraph: Reference `SubgraphData` representing valid KG triples.

    Returns:
        float: Ratio in `[0.0, 1.0]` of generated triples grounded in `ground_truth_subgraph`.
    """
    all_generated_triples: list[Triple] = [
        triple for path in generated_paths for triple in path
    ]
    if not all_generated_triples:
        return 0.0

    valid_triples: set[tuple[str, str, str]] = {
        triple.to_tuple() for triple in ground_truth_subgraph.triples
    }
    grounded_count = sum(
        1 for triple in all_generated_triples if triple.to_tuple() in valid_triples
    )
    return grounded_count / len(all_generated_triples)


def compute_path_level_faithfulness(
    generated_paths: Sequence[Sequence[Triple]],
    ground_truth_subgraph: SubgraphData,
) -> float:
    """Compute share of complete generated paths that are contiguous and 100% KG-grounded."""
    non_empty_paths = [path for path in generated_paths if path]
    if not non_empty_paths:
        return 0.0

    valid_triples: set[tuple[str, str, str]] = {
        triple.to_tuple() for triple in ground_truth_subgraph.triples
    }
    valid_path_count = 0
    for path in non_empty_paths:
        all_in_kg = all(triple.to_tuple() in valid_triples for triple in path)
        contiguous = all(
            path[idx].tail == path[idx + 1].head for idx in range(len(path) - 1)
        )
        if all_in_kg and contiguous:
            valid_path_count += 1

    return valid_path_count / len(non_empty_paths)


def compute_answer_path_recall(
    pruned_subgraph: SubgraphData,
    directed: bool = False,
) -> float:
    """Measure share of gold `a_entities` reachable from `q_entities` in the pruned subgraph.

    Args:
        pruned_subgraph: Retrieved/pruned `SubgraphData` (e.g., output of `PCSTRetriever`).
        directed: If False (default), checks KG connectivity ignoring relation orientation
            (matching undirected PCST trees); if True, requires strictly forward directed paths.

    Returns:
        float: Recall in `[0.0, 1.0]` of gold `a_entities` reachable from at least one `q_entity`.
    """
    if not pruned_subgraph.a_entities or not pruned_subgraph.q_entities:
        return 0.0

    di_graph = pruned_subgraph.to_networkx()
    graph: nx.Graph = di_graph if directed else di_graph.to_undirected()

    valid_sources = [q for q in pruned_subgraph.q_entities if q in graph]
    if not valid_sources:
        return 0.0

    reachable_gold = 0
    for answer_entity in pruned_subgraph.a_entities:
        if answer_entity not in graph:
            continue
        if any(nx.has_path(graph, source, answer_entity) for source in valid_sources):
            reachable_gold += 1

    return reachable_gold / len(pruned_subgraph.a_entities)


def compute_latency_stats(outputs: Sequence[ReasoningOutput]) -> dict[str, float]:
    """Compute summary latency statistics (mean, median, min, max in milliseconds)."""
    if not outputs:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    latencies = [float(out.latency_ms) for out in outputs]
    return {
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }

