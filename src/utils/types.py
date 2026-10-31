"""Shared data contracts enforcing the strict Subgraph Interface across modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


@dataclass(frozen=True)
class Triple:
    """A directed Knowledge Graph RDF-style triple (head, relation, tail)."""

    head: str
    relation: str
    tail: str

    def to_tuple(self) -> tuple[str, str, str]:
        """Return the triple as a standard 3-tuple of strings."""
        return (self.head, self.relation, self.tail)

    def to_text(self) -> str:
        """Verbalize the triple into a linear text representation."""
        return f"{self.head} -> {self.relation} -> {self.tail}"


@dataclass
class SubgraphData:
    """Canonical subgraph representation exchanged between acquisition, retrieval, and decoding.

    Attributes:
        question_id: Unique identifier of the QA example.
        question: Natural-language question string.
        q_entities: Topic/anchor entities linked in the question.
        a_entities: Ground-truth answer entities (empty list during live open-world inference).
        triples: Directed KG triples constituting the subgraph.
    """

    question_id: str
    question: str
    q_entities: list[str] = field(default_factory=list)
    a_entities: list[str] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)

    def to_networkx(self) -> nx.DiGraph:
        """Convert the subgraph triples into an annotated NetworkX directed graph.

        Nodes receive boolean attributes `is_q_entity` and `is_a_entity`.
        Parallel relations between the same `(head, tail)` pair are preserved in
        the edge's `relations` list attribute and joined in `relation`.

        Returns:
            nx.DiGraph: Directed graph representation of this subgraph.
        """
        graph = nx.DiGraph(question_id=self.question_id, question=self.question)
        q_set = set(self.q_entities)
        a_set = set(self.a_entities)

        for triple in self.triples:
            for entity in (triple.head, triple.tail):
                if entity not in graph:
                    graph.add_node(
                        entity,
                        label=entity,
                        is_q_entity=entity in q_set,
                        is_a_entity=entity in a_set,
                    )

            if graph.has_edge(triple.head, triple.tail):
                existing_relations: list[str] = graph[triple.head][triple.tail]["relations"]
                if triple.relation not in existing_relations:
                    existing_relations.append(triple.relation)
                joined = " | ".join(existing_relations)
                graph[triple.head][triple.tail]["relation"] = joined
                graph[triple.head][triple.tail]["label"] = joined
            else:
                graph.add_edge(
                    triple.head,
                    triple.tail,
                    relation=triple.relation,
                    relations=[triple.relation],
                    label=triple.relation,
                )

        return graph


@dataclass
class ReasoningOutput:
    """Output contract produced by decoding pipelines and baselines.

    Attributes:
        question_id: Unique identifier matching the input `SubgraphData.question_id`.
        generated_paths: List of multi-hop reasoning paths, each a sequence of `Triple` objects.
        final_answer: Deduced natural-language or entity answer string.
        latency_ms: Wall-clock inference time in milliseconds.
    """

    question_id: str
    generated_paths: list[list[Triple]]
    final_answer: str
    latency_ms: float
