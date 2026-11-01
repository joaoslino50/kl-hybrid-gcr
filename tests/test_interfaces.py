"""Pytest unit tests for Subgraph Interface contracts, KGTrie, PCST, Visualizer, and Metrics."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
import pytest
import torch

from src.acquisition.dataset_loader import BenchmarkDatasetLoader, load_mock_subgraphs
from src.acquisition.visualizer import export_subgraph_html
from src.decoding.constrained_llm import ConstrainedPathGenerator
from src.decoding.final_reasoner import FinalAnswerReasoner
from src.decoding.kg_trie import KGTrie, KGTrieLogitsProcessor, enumerate_kg_paths
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
from src.utils.types import ReasoningOutput, SubgraphData, Triple


@pytest.fixture
def sample_subgraph() -> SubgraphData:
    """Return a canonical 2-hop KG `SubgraphData` fixture."""
    return SubgraphData(
        question_id="q_test_01",
        question="Which university is in the historic capital of Portugal?",
        q_entities=["Portugal"],
        a_entities=["University of Coimbra"],
        triples=[
            Triple(head="Portugal", relation="historic_capital", tail="Coimbra"),
            Triple(head="Coimbra", relation="institution", tail="University of Coimbra"),
            Triple(head="Portugal", relation="capital", tail="Lisbon"),
            Triple(head="Spain", relation="capital", tail="Madrid"),
        ],
    )


def test_device_selection_returns_valid_torch_device() -> None:
    """`get_device()` should return a working `torch.device` (`cuda`, `mps`, or `cpu`)."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}
    tensor = torch.ones((4, 4), device=device)
    assert (tensor @ tensor).shape == (4, 4)


def test_subgraph_interface_dataclasses_and_to_networkx(sample_subgraph: SubgraphData) -> None:
    """Verify `Triple`, `SubgraphData.to_networkx()`, and `ReasoningOutput` contracts."""
    # Triple immutability & hashing
    t1 = Triple("Portugal", "capital", "Lisbon")
    t2 = Triple("Portugal", "capital", "Lisbon")
    assert t1 == t2
    assert len({t1, t2}) == 1
    assert t1.to_tuple() == ("Portugal", "capital", "Lisbon")

    # Parallel edges between same head/tail should aggregate cleanly in nx.DiGraph
    sample_subgraph.triples.append(Triple("Portugal", "largest_city", "Lisbon"))
    graph = sample_subgraph.to_networkx()
    assert isinstance(graph, nx.DiGraph)
    assert graph.nodes["Portugal"]["is_q_entity"] is True
    assert graph.nodes["Portugal"]["is_a_entity"] is False
    assert graph.nodes["University of Coimbra"]["is_a_entity"] is True
    assert graph.has_edge("Portugal", "Lisbon")
    assert graph["Portugal"]["Lisbon"]["relations"] == ["capital", "largest_city"]

    output = ReasoningOutput(
        question_id=sample_subgraph.question_id,
        generated_paths=[[t1]],
        final_answer="University of Coimbra",
        latency_ms=12.5,
    )
    assert output.question_id == "q_test_01"
    assert output.latency_ms == pytest.approx(12.5)


def test_dataset_loader_parse_record() -> None:
    """`BenchmarkDatasetLoader.parse_record` should parse RoG HF dict records into `SubgraphData`."""
    raw = {
        "id": "WebQTest-42",
        "question": "what is the capital of portugal",
        "q_entity": ["Portugal"],
        "a_entity": ["Lisbon"],
        "graph": [
            ["Portugal", "location.country.capital", "Lisbon"],
            ["Lisbon", "location.location.containedby", "Portugal"],
        ],
    }
    parsed = BenchmarkDatasetLoader.parse_record(raw)
    assert parsed.question_id == "WebQTest-42"
    assert parsed.q_entities == ["Portugal"]
    assert parsed.a_entities == ["Lisbon"]
    assert len(parsed.triples) == 2
    assert parsed.triples[0] == Triple("Portugal", "location.country.capital", "Lisbon")


def test_kg_trie_prefix_lookup_and_logits_processor_masking() -> None:
    """`KGTrie` and `KGTrieLogitsProcessor` should strictly allow valid next tokens and `-inf` mask others."""
    trie = KGTrie()
    trie.insert([5, 12, 19])
    trie.insert([5, 14, 21])
    trie.insert([8, 30])

    assert len(trie) == 3
    assert trie.get_allowed_next_tokens([]) == [5, 8]
    assert trie.get_allowed_next_tokens([5]) == [12, 14]
    assert trie.get_allowed_next_tokens([5, 12]) == [19]
    assert trie.get_allowed_next_tokens([5, 12, 19]) == []
    assert trie.get_allowed_next_tokens([999]) == []
    assert trie.is_terminal([5, 12, 19]) is True
    assert trie.is_terminal([5, 12]) is False

    # Test LogitsProcessor with prompt_length=2 and eos_token_id=2
    processor = KGTrieLogitsProcessor(trie=trie, prompt_length=2, eos_token_id=2)
    input_ids = torch.tensor(
        [
            [101, 102, 5],          # prefix after prompt is [5] -> allowed: [12, 14]
            [101, 102, 5],          # second row overridden below
        ],
        dtype=torch.long,
    )
    input_ids[1] = torch.tensor([101, 102, 8])  # prefix [8] -> allowed: [30]
    raw_scores = torch.ones((2, 40), dtype=torch.float32) * 3.5

    masked = processor(input_ids, raw_scores)
    row0_valid = torch.nonzero(torch.isfinite(masked[0]), as_tuple=False).squeeze(-1).tolist()
    row1_valid = torch.nonzero(torch.isfinite(masked[1]), as_tuple=False).squeeze(-1).tolist()

    assert row0_valid == [12, 14]
    assert row1_valid == [30]
    assert masked[0, 12].item() == pytest.approx(3.5)
    assert masked[0, 0].item() == float("-inf")

    # Terminal prefix should allow eos_token_id
    terminal_ids = torch.tensor([[101, 102, 5, 12, 19]], dtype=torch.long)
    terminal_masked = processor(terminal_ids, torch.zeros((1, 40), dtype=torch.float32))
    term_valid = torch.nonzero(torch.isfinite(terminal_masked[0]), as_tuple=False).squeeze(-1).tolist()
    assert term_valid == [2]


def test_pcst_retriever_preserves_subgraph_interface(sample_subgraph: SubgraphData) -> None:
    """`PCSTRetriever.retrieve` should prune via `pcst_fast` and return a valid `SubgraphData`."""
    embedder = GraphEmbedder(use_mock=True)
    retriever = PCSTRetriever(embedder=embedder, topk_nodes=3, topk_edges=3, base_edge_cost=0.6)
    pruned = retriever.retrieve(sample_subgraph)

    assert isinstance(pruned, SubgraphData)
    assert pruned.question_id == sample_subgraph.question_id
    assert pruned.q_entities == sample_subgraph.q_entities
    assert pruned.a_entities == sample_subgraph.a_entities
    assert 1 <= len(pruned.triples) <= len(sample_subgraph.triples)
    assert set(pruned.triples).issubset(set(sample_subgraph.triples))


def test_visualizer_exports_pyvis_html(sample_subgraph: SubgraphData, tmp_path: Path) -> None:
    """`export_subgraph_html` should generate a valid interactive HTML document."""
    html_file = tmp_path / "subgraph_preview.html"
    paths = enumerate_kg_paths(sample_subgraph, max_hops=2)
    written_path = export_subgraph_html(sample_subgraph, html_file, highlight_paths=paths[:1])

    assert written_path.exists()
    content = written_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "University of Coimbra" in content
    assert "historic_capital" in content


def test_evaluation_metrics(sample_subgraph: SubgraphData) -> None:
    """Verify Hit@1, F1, Faithful Reasoning Ratio, Answer-Path Recall, and Latency metrics."""
    # Hit@1 & F1
    assert compute_hit_at_1("University of Coimbra", ["University of Coimbra"]) == 1.0
    assert compute_hit_at_1("The University of Coimbra!", ["University of Coimbra"]) == 1.0
    assert compute_hit_at_1("University of Porto", ["University of Coimbra"]) == 0.0

    assert compute_f1("University of Coimbra", ["University of Coimbra"]) == pytest.approx(1.0)
    assert compute_f1(["Spain"], ["France", "Spain"]) == pytest.approx(2.0 / 3.0)
    assert compute_f1("Lisbon", ["University of Coimbra"]) == pytest.approx(0.0)

    # Faithful Reasoning Ratio (2 valid KG triples + 1 hallucinated triple -> 2/3)
    generated_paths = [
        [
            Triple("Portugal", "historic_capital", "Coimbra"),
            Triple("Coimbra", "institution", "University of Coimbra"),
        ],
        [
            Triple("Coimbra", "hallucinated_relation", "Atlantis"),
        ],
    ]
    ratio = compute_faithful_reasoning_ratio(generated_paths, sample_subgraph)
    assert ratio == pytest.approx(2.0 / 3.0)

    # Answer-Path Recall
    assert compute_answer_path_recall(sample_subgraph) == pytest.approx(1.0)
    disconnected = SubgraphData(
        question_id="q_disc",
        question=sample_subgraph.question,
        q_entities=["Portugal"],
        a_entities=["University of Coimbra"],
        triples=[Triple("Portugal", "capital", "Lisbon")],
    )
    assert compute_answer_path_recall(disconnected) == pytest.approx(0.0)

    # Latency stats
    stats = compute_latency_stats(
        [
            ReasoningOutput("q1", [], "A", 10.0),
            ReasoningOutput("q2", [], "B", 30.0),
        ]
    )
    assert stats["mean_ms"] == pytest.approx(20.0)
    assert stats["median_ms"] == pytest.approx(20.0)


def test_end_to_end_mock_hybrid_pipeline() -> None:
    """Verify all 3 stages connect seamlessly using `SubgraphData` and `ReasoningOutput`."""
    subgraphs = load_mock_subgraphs()
    embedder = GraphEmbedder(use_mock=True)
    retriever = PCSTRetriever(embedder=embedder)
    generator = ConstrainedPathGenerator(use_mock=True, max_path_hops=2)
    reasoner = FinalAnswerReasoner(use_mock=True)

    pruned = retriever.retrieve(subgraphs[0])
    paths = generator.generate_paths(pruned)
    output = reasoner.deduce_answer(pruned, paths)

    assert isinstance(output, ReasoningOutput)
    assert output.question_id == subgraphs[0].question_id
    assert len(output.generated_paths) >= 1
    assert compute_faithful_reasoning_ratio(output.generated_paths, subgraphs[0]) == pytest.approx(
        1.0
    )

