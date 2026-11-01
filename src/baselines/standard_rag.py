"""Standard RAG baseline retrieving top-k KG triples via SentenceBERT cosine similarity."""

from __future__ import annotations

import time

import torch

from src.decoding.final_reasoner import FinalAnswerReasoner
from src.retrieval.embedder import GraphEmbedder
from src.utils.types import ReasoningOutput, SubgraphData, Triple


class StandardRAGBaseline:
    """Top-k single-triple semantic retrieval followed by unconstrained LLM answer generation."""

    def __init__(
        self,
        embedder: GraphEmbedder | None = None,
        reasoner: FinalAnswerReasoner | None = None,
        top_k: int = 10,
    ) -> None:
        """Initialize the Standard Triple-RAG baseline.

        Args:
            embedder: `GraphEmbedder` encoding verbalized triples and questions.
            reasoner: `FinalAnswerReasoner` generating an unconstrained final answer.
            top_k: Number of highest-similarity triples retrieved into context.
        """
        self.embedder = embedder if embedder is not None else GraphEmbedder(use_mock=True)
        self.reasoner = reasoner if reasoner is not None else FinalAnswerReasoner(use_mock=True)
        self.top_k = top_k

    def retrieve_top_k_triples(self, subgraph: SubgraphData) -> list[Triple]:
        """Rank `subgraph.triples` by SentenceBERT cosine similarity to `subgraph.question`."""
        if not subgraph.triples or self.top_k <= 0:
            return []

        verbalized = [
            f"{t.head} {t.relation.replace('_', ' ').replace('.', ' ')} {t.tail}"
            for t in subgraph.triples
        ]
        sims = self.embedder.compute_similarities(subgraph.question, verbalized).detach().cpu()
        k = min(self.top_k, len(subgraph.triples))
        _, top_indices = torch.topk(sims, k=k)
        return [subgraph.triples[idx] for idx in top_indices.tolist()]

    def answer(self, subgraph: SubgraphData) -> ReasoningOutput:
        """Retrieve top-k triples as 1-hop paths and deduce an unconstrained answer."""
        start_time = time.perf_counter()
        top_triples = self.retrieve_top_k_triples(subgraph)
        retrieval_ms = (time.perf_counter() - start_time) * 1000.0

        pseudo_paths: list[list[Triple]] = [[triple] for triple in top_triples]
        return self.reasoner.deduce_answer(
            subgraph=subgraph,
            generated_paths=pseudo_paths,
            prior_latency_ms=retrieval_ms,
        )

