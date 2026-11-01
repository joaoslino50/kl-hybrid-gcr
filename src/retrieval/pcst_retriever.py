"""Prize-Collecting Steiner Tree (PCST) subgraph pruning inspired by G-Retriever."""

from __future__ import annotations

import logging

import numpy as np
import pcst_fast
import torch

from src.retrieval.embedder import GraphEmbedder
from src.utils.types import SubgraphData, Triple

logger = logging.getLogger(__name__)


class PCSTRetriever:
    """Prunes an L-hop `SubgraphData` into a compact, question-relevant Steiner subgraph."""

    def __init__(
        self,
        embedder: GraphEmbedder | None = None,
        topk_nodes: int = 10,
        topk_edges: int = 10,
        base_edge_cost: float = 0.5,
        anchor_prize: float = 4.0,
        num_clusters: int = 1,
        pruning: str = "gw",
    ) -> None:
        """Initialize the PCST subgraph retriever.

        Args:
            embedder: `GraphEmbedder` instance for node/relation similarity scoring.
            topk_nodes: Number of highest-similarity nodes assigned positive semantic prizes.
            topk_edges: Number of highest-similarity edges receiving prize-adjusted cost discounts.
            base_edge_cost: Baseline traversal cost $C_e$ per undirected KG edge.
            anchor_prize: Fixed high prize assigned to `q_entities` to anchor the Steiner tree.
            num_clusters: Target number of connected clusters in `pcst_fast`.
            pruning: Pruning heuristic for `pcst_fast` (`"gw"`, `"strong"`, `"simple"`, `"none"`).
        """
        self.embedder = embedder if embedder is not None else GraphEmbedder(use_mock=True)
        self.topk_nodes = topk_nodes
        self.topk_edges = topk_edges
        self.base_edge_cost = base_edge_cost
        self.anchor_prize = anchor_prize
        self.num_clusters = num_clusters
        self.pruning = pruning

    def retrieve(self, subgraph: SubgraphData) -> SubgraphData:
        """Extract a compact question-relevant `SubgraphData` via PCST optimization.

        Strictly preserves the `SubgraphData -> SubgraphData` interface contract.
        """
        if len(subgraph.triples) <= 1:
            return SubgraphData(
                question_id=subgraph.question_id,
                question=subgraph.question,
                q_entities=list(subgraph.q_entities),
                a_entities=list(subgraph.a_entities),
                triples=list(subgraph.triples),
            )

        # Build deterministic node index mapping from triples
        nodes: list[str] = []
        node_to_idx: dict[str, int] = {}
        for triple in subgraph.triples:
            for entity in (triple.head, triple.tail):
                if entity not in node_to_idx:
                    node_to_idx[entity] = len(nodes)
                    nodes.append(entity)

        # Filter out self-loops for pcst_fast while tracking original triple indices
        valid_edge_pairs: list[list[int]] = []
        valid_triple_indices: list[int] = []
        relation_texts: list[str] = []
        for idx, triple in enumerate(subgraph.triples):
            u, v = node_to_idx[triple.head], node_to_idx[triple.tail]
            if u != v:
                valid_edge_pairs.append([u, v])
                valid_triple_indices.append(idx)
                relation_texts.append(triple.relation.replace("_", " ").replace(".", " "))

        if not valid_edge_pairs:
            return SubgraphData(
                question_id=subgraph.question_id,
                question=subgraph.question,
                q_entities=list(subgraph.q_entities),
                a_entities=list(subgraph.a_entities),
                triples=list(subgraph.triples),
            )

        # Compute node prizes from SentenceBERT similarities (G-Retriever rank-weighted schedule)
        node_sims = self.embedder.compute_similarities(subgraph.question, nodes).detach().cpu()
        prizes = np.zeros(len(nodes), dtype=np.float64)

        k_v = min(self.topk_nodes, len(nodes))
        if k_v > 0:
            topk_vals, topk_indices = torch.topk(node_sims, k=k_v)
            for rank, (node_idx, sim_val) in enumerate(
                zip(topk_indices.tolist(), topk_vals.tolist())
            ):
                rank_prize = max(0.0, float(k_v - rank) / float(k_v)) * max(0.0, float(sim_val) + 1.0)
                prizes[node_idx] = max(prizes[node_idx], rank_prize)

        # Anchor question entities so the Steiner tree connects around topic entities
        for q_ent in subgraph.q_entities:
            if q_ent in node_to_idx:
                prizes[node_to_idx[q_ent]] = max(prizes[node_to_idx[q_ent]], self.anchor_prize)

        # Compute edge costs, discounting high-similarity relations via virtual nodes when prize > cost
        rel_sims = self.embedder.compute_similarities(subgraph.question, relation_texts).detach().cpu()
        edge_prizes = np.zeros(len(valid_edge_pairs), dtype=np.float64)
        k_e = min(self.topk_edges, len(valid_edge_pairs))
        if k_e > 0:
            topk_e_vals, topk_e_indices = torch.topk(rel_sims, k=k_e)
            for rank, (edge_idx, sim_val) in enumerate(
                zip(topk_e_indices.tolist(), topk_e_vals.tolist())
            ):
                edge_prizes[edge_idx] = (
                    max(0.0, float(k_e - rank) / float(k_e))
                    * max(0.0, float(sim_val) + 1.0)
                    * 0.5
                )

        pcst_edges: list[list[int]] = []
        pcst_costs: list[float] = []
        pcst_prizes: list[float] = prizes.tolist()
        pcst_edge_to_triple_idx: dict[int, int] = {}
        virtual_node_to_triple_idx: dict[int, int] = {}

        for local_e_idx, ((u, v), orig_t_idx) in enumerate(
            zip(valid_edge_pairs, valid_triple_indices)
        ):
            e_prize = float(edge_prizes[local_e_idx])
            if e_prize <= self.base_edge_cost:
                edge_id = len(pcst_edges)
                pcst_edges.append([u, v])
                pcst_costs.append(max(1e-4, self.base_edge_cost - e_prize))
                pcst_edge_to_triple_idx[edge_id] = orig_t_idx
            else:
                # Subdivide high-prize edge with a virtual node carrying net surplus prize
                v_node = len(pcst_prizes)
                pcst_prizes.append(e_prize - self.base_edge_cost)
                virtual_node_to_triple_idx[v_node] = orig_t_idx

                e1 = len(pcst_edges)
                pcst_edges.append([u, v_node])
                pcst_costs.append(0.0)
                pcst_edge_to_triple_idx[e1] = orig_t_idx

                e2 = len(pcst_edges)
                pcst_edges.append([v_node, v])
                pcst_costs.append(0.0)
                pcst_edge_to_triple_idx[e2] = orig_t_idx

        edges_arr = np.asarray(pcst_edges, dtype=np.int64)
        prizes_arr = np.asarray(pcst_prizes, dtype=np.float64)
        costs_arr = np.asarray(pcst_costs, dtype=np.float64)

        selected_vertices, selected_edges = pcst_fast.pcst_fast(
            edges_arr,
            prizes_arr,
            costs_arr,
            -1,
            self.num_clusters,
            self.pruning,
            0,
        )

        chosen_triple_indices: set[int] = set()
        for e_id in selected_edges.tolist():
            if e_id in pcst_edge_to_triple_idx:
                chosen_triple_indices.add(pcst_edge_to_triple_idx[e_id])
        for v_id in selected_vertices.tolist():
            if v_id in virtual_node_to_triple_idx:
                chosen_triple_indices.add(virtual_node_to_triple_idx[v_id])

        # Fallback: if PCST selected only an isolated anchor vertex without edges, keep its top edges
        if not chosen_triple_indices:
            selected_node_names = {nodes[v] for v in selected_vertices.tolist() if v < len(nodes)}
            anchor_names = selected_node_names or set(subgraph.q_entities)
            for idx, triple in enumerate(subgraph.triples):
                if triple.head in anchor_names or triple.tail in anchor_names:
                    chosen_triple_indices.add(idx)
            if not chosen_triple_indices:
                chosen_triple_indices.add(0)

        pruned_triples: list[Triple] = [
            subgraph.triples[idx] for idx in sorted(chosen_triple_indices)
        ]

        logger.info(
            "PCST pruned subgraph '%s': %d -> %d triples",
            subgraph.question_id,
            len(subgraph.triples),
            len(pruned_triples),
        )
        return SubgraphData(
            question_id=subgraph.question_id,
            question=subgraph.question,
            q_entities=list(subgraph.q_entities),
            a_entities=list(subgraph.a_entities),
            triples=pruned_triples,
        )

