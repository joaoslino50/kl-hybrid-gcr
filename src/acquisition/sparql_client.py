"""Live Wikidata L-hop neighborhood acquisition via SPARQLWrapper."""

from __future__ import annotations

import logging
import re
from typing import Any

from SPARQLWrapper import JSON, SPARQLWrapper

from src.utils.types import SubgraphData, Triple

logger = logging.getLogger(__name__)

_QID_PATTERN = re.compile(r"^Q\d+$")


class WikidataSPARQLClient:
    """Client for acquiring L-hop RDF subgraphs around seed entities from Wikidata."""

    def __init__(
        self,
        endpoint_url: str = "https://query.wikidata.org/sparql",
        user_agent: str = "KL-Hybrid-GCR-Research/0.1 (University of Coimbra; KGQA Project)",
        timeout_sec: int = 30,
        max_triples_per_hop: int = 50,
        language: str = "en",
    ) -> None:
        """Initialize the Wikidata SPARQL client.

        Args:
            endpoint_url: SPARQL query service URL.
            user_agent: Compliant User-Agent header required by Wikidata policy.
            timeout_sec: Request timeout in seconds.
            max_triples_per_hop: Cap on triples fetched per hop expansion.
            language: Preferred label language code (default `"en"`).
        """
        self.endpoint_url = endpoint_url
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self.max_triples_per_hop = max_triples_per_hop
        self.language = language

    def _execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT query and return JSON result bindings."""
        sparql = SPARQLWrapper(self.endpoint_url, agent=self.user_agent)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(self.timeout_sec)
        sparql.setQuery(query)
        response = sparql.query().convert()
        if isinstance(response, dict):
            return list(response.get("results", {}).get("bindings", []))
        return []

    def resolve_entity_qid(self, entity_or_qid: str) -> tuple[str, str] | None:
        """Resolve an entity label or Wikidata QID into `(qid, canonical_label)`."""
        candidate = entity_or_qid.strip()
        if _QID_PATTERN.match(candidate):
            query = f"""
            SELECT ?label WHERE {{
              wd:{candidate} rdfs:label ?label .
              FILTER(LANG(?label) = "{self.language}")
            }}
            LIMIT 1
            """
            bindings = self._execute_query(query)
            label = bindings[0]["label"]["value"] if bindings else candidate
            return candidate, label

        escaped = candidate.replace('"', '\\"')
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          ?item rdfs:label "{escaped}"@{self.language} .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
        }}
        LIMIT 1
        """
        bindings = self._execute_query(query)
        if not bindings:
            return None
        uri = bindings[0]["item"]["value"]
        qid = uri.rsplit("/", maxsplit=1)[-1]
        label = bindings[0].get("itemLabel", {}).get("value", candidate)
        return qid, label

    def _fetch_one_hop(self, seed_qids: list[str]) -> tuple[list[Triple], dict[str, str]]:
        """Fetch 1-hop outgoing entity-to-entity truthy Wikidata triples for `seed_qids`."""
        if not seed_qids:
            return [], {}

        values_clause = " ".join(f"wd:{qid}" for qid in seed_qids if _QID_PATTERN.match(qid))
        if not values_clause:
            return [], {}

        query = f"""
        SELECT ?head ?headLabel ?propLabel ?tail ?tailLabel WHERE {{
          VALUES ?head {{ {values_clause} }}
          ?head ?wdt ?tail .
          FILTER(ISIRI(?tail) && STRSTARTS(STR(?tail), "http://www.wikidata.org/entity/Q"))
          ?prop wikibase:directClaim ?wdt .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
        }}
        LIMIT {self.max_triples_per_hop}
        """
        bindings = self._execute_query(query)

        triples: list[Triple] = []
        discovered_qids: dict[str, str] = {}
        for row in bindings:
            head_label = row.get("headLabel", {}).get("value", "").strip()
            rel_label = row.get("propLabel", {}).get("value", "").strip()
            tail_uri = row.get("tail", {}).get("value", "").strip()
            tail_label = row.get("tailLabel", {}).get("value", "").strip()
            tail_qid = tail_uri.rsplit("/", maxsplit=1)[-1]

            if head_label and rel_label and tail_label:
                triples.append(Triple(head=head_label, relation=rel_label, tail=tail_label))
                if _QID_PATTERN.match(tail_qid):
                    discovered_qids[tail_qid] = tail_label

        return triples, discovered_qids

    def fetch_l_hop_subgraph(
        self,
        question_id: str,
        question: str,
        seed_entities: list[str],
        num_hops: int = 2,
        use_mock: bool = False,
    ) -> SubgraphData:
        """Acquire an L-hop Wikidata neighborhood around `seed_entities` as `SubgraphData`.

        Args:
            question_id: Unique question identifier.
            question: Natural-language question text.
            seed_entities: Question entities (labels such as `"Portugal"` or QIDs such as `"Q45"`).
            num_hops: Number of neighborhood expansion hops $L$.
            use_mock: If True, return a deterministic mock Wikidata subgraph without network I/O.

        Returns:
            SubgraphData: Populated L-hop subgraph adhering to the Subgraph Interface.
        """
        if use_mock:
            anchor = seed_entities[0] if seed_entities else "Portugal"
            mock_triples = [
                Triple(anchor, "capital", "Lisbon"),
                Triple(anchor, "official language", "Portuguese"),
                Triple("Lisbon", "located in administrative territorial entity", anchor),
                Triple("University of Coimbra", "country", anchor),
            ]
            return SubgraphData(
                question_id=question_id,
                question=question,
                q_entities=[anchor],
                a_entities=[],
                triples=mock_triples,
            )

        frontier_qids: list[str] = []
        visited_qids: set[str] = set()
        resolved_q_labels: list[str] = []

        for entity in seed_entities:
            resolved = self.resolve_entity_qid(entity)
            if resolved is not None:
                qid, label = resolved
                resolved_q_labels.append(label)
                if qid not in visited_qids:
                    frontier_qids.append(qid)
                    visited_qids.add(qid)
            else:
                logger.warning("Could not resolve Wikidata entity '%s'", entity)

        collected_triples: list[Triple] = []
        seen_triples: set[Triple] = set()

        for hop in range(max(1, num_hops)):
            if not frontier_qids:
                break
            logger.info("Fetching Wikidata hop %d/%d for %d frontier entities", hop + 1, num_hops, len(frontier_qids))
            hop_triples, next_qids = self._fetch_one_hop(frontier_qids)
            for triple in hop_triples:
                if triple not in seen_triples:
                    seen_triples.add(triple)
                    collected_triples.append(triple)

            frontier_qids = [qid for qid in next_qids if qid not in visited_qids]
            visited_qids.update(frontier_qids)

        return SubgraphData(
            question_id=question_id,
            question=question,
            q_entities=resolved_q_labels or list(seed_entities),
            a_entities=[],
            triples=collected_triples,
        )
