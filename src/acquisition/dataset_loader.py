"""Hugging Face dataset loader for RoG-WebQSP and RoG-CWQ Freebase subgraphs."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from datasets import load_dataset

from src.utils.types import SubgraphData, Triple

logger = logging.getLogger(__name__)

DATASET_ALIASES: dict[str, str] = {
    "webqsp": "rmanluo/RoG-webqsp",
    "cwq": "rmanluo/RoG-cwq",
    "rmanluo/RoG-webqsp": "rmanluo/RoG-webqsp",
    "rmanluo/RoG-cwq": "rmanluo/RoG-cwq",
}


def _ensure_str_list(value: Any) -> list[str]:
    """Coerce dataset entity/answer fields into a deduplicated list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result
    return [str(value).strip()]


def load_mock_subgraphs() -> list[SubgraphData]:
    """Return synthetic multi-hop KGQA subgraphs adhering to `SubgraphData` for offline testing."""
    return [
        SubgraphData(
            question_id="mock_001",
            question="Which university is located in the former capital of Portugal founded in 1290?",
            q_entities=["Portugal"],
            a_entities=["University of Coimbra"],
            triples=[
                Triple("Portugal", "capital", "Lisbon"),
                Triple("Portugal", "historic_capital", "Coimbra"),
                Triple("Coimbra", "institution", "University of Coimbra"),
                Triple("University of Coimbra", "inception", "1290"),
                Triple("Lisbon", "country", "Portugal"),
                Triple("Portugal", "continent", "Europe"),
                Triple("Europe", "has_country", "Spain"),
                Triple("Porto", "country", "Portugal"),
            ],
        ),
        SubgraphData(
            question_id="mock_002",
            question="Where was the author of Os Lusiadas born?",
            q_entities=["Os Lusiadas"],
            a_entities=["Lisbon"],
            triples=[
                Triple("Os Lusiadas", "author", "Luis de Camoes"),
                Triple("Luis de Camoes", "place_of_birth", "Lisbon"),
                Triple("Luis de Camoes", "language", "Portuguese"),
                Triple("Os Lusiadas", "publication_date", "1572"),
                Triple("Lisbon", "located_in", "Portugal"),
            ],
        ),
    ]


class BenchmarkDatasetLoader:
    """Loader converting `rmanluo/RoG-webqsp` and `rmanluo/RoG-cwq` examples to `SubgraphData`."""

    def __init__(
        self,
        dataset_name: str = "rmanluo/RoG-webqsp",
        split: str = "test",
        streaming: bool = False,
    ) -> None:
        """Initialize the Hugging Face KGQA dataset loader.

        Args:
            dataset_name: Shorthand (`"webqsp"`, `"cwq"`) or full HF dataset ID.
            split: Dataset split (`"train"`, `"validation"`, `"test"`).
            streaming: Stream records without downloading the entire split upfront.
        """
        self.hf_dataset_id = DATASET_ALIASES.get(dataset_name, dataset_name)
        self.split = split
        self.streaming = streaming

    @staticmethod
    def parse_record(record: dict[str, Any], fallback_idx: int = 0) -> SubgraphData:
        """Parse a single Hugging Face RoG dataset dictionary into `SubgraphData`."""
        question_id = str(record.get("id", f"sample_{fallback_idx}"))
        question = str(record.get("question", "")).strip()

        q_entities = _ensure_str_list(record.get("q_entity", record.get("q_entities")))
        a_entities = _ensure_str_list(
            record.get("a_entity", record.get("a_entities", record.get("answer")))
        )

        raw_graph = record.get("graph", record.get("triples", []))
        triples: list[Triple] = []
        for entry in raw_graph:
            if isinstance(entry, Triple):
                triples.append(entry)
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                head, relation, tail = str(entry[0]).strip(), str(entry[1]).strip(), str(entry[2]).strip()
                if head and relation and tail:
                    triples.append(Triple(head=head, relation=relation, tail=tail))
            elif isinstance(entry, dict):
                head = str(entry.get("head", "")).strip()
                relation = str(entry.get("relation", "")).strip()
                tail = str(entry.get("tail", "")).strip()
                if head and relation and tail:
                    triples.append(Triple(head=head, relation=relation, tail=tail))

        return SubgraphData(
            question_id=question_id,
            question=question,
            q_entities=q_entities,
            a_entities=a_entities,
            triples=triples,
        )

    def iter_subgraphs(self, max_samples: int | None = None) -> Iterator[SubgraphData]:
        """Yield `SubgraphData` instances from the configured Hugging Face dataset."""
        logger.info(
            "Loading dataset '%s' (split='%s', streaming=%s)",
            self.hf_dataset_id,
            self.split,
            self.streaming,
        )
        dataset = load_dataset(self.hf_dataset_id, split=self.split, streaming=self.streaming)

        for idx, record in enumerate(dataset):
            if max_samples is not None and idx >= max_samples:
                break
            yield self.parse_record(record, fallback_idx=idx)

    def load(self, max_samples: int | None = None) -> list[SubgraphData]:
        """Materialize a list of `SubgraphData` instances."""
        return list(self.iter_subgraphs(max_samples=max_samples))
