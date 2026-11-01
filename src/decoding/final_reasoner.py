"""Final answer deduction over KG-constrained reasoning paths via GPT-4o-mini."""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence

from dotenv import load_dotenv

from src.decoding.kg_trie import format_reasoning_path
from src.utils.types import ReasoningOutput, SubgraphData, Triple

logger = logging.getLogger(__name__)


class FinalAnswerReasoner:
    """Inductive general-purpose LLM (`gpt-4o-mini`) producing the final answer from KG paths."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 128,
        api_key: str | None = None,
        use_mock: bool = False,
    ) -> None:
        """Initialize the OpenAI final answer reasoner.

        Args:
            model_name: OpenAI chat completion model name (default `"gpt-4o-mini"`).
            temperature: Sampling temperature (`0.0` for deterministic QA evaluation).
            max_tokens: Maximum response tokens.
            api_key: Explicit OpenAI API key; falls back to `OPENAI_API_KEY` env var / `.env`.
            use_mock: Deduce answer heuristically from the top generated path without API calls.
        """
        load_dotenv()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_mock = use_mock
        self._client = None

    def _get_client(self):
        """Lazily initialize `openai.OpenAI` client."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to `.env` or pass `use_mock=True`."
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @staticmethod
    def build_messages(
        question: str,
        generated_paths: Sequence[Sequence[Triple]],
    ) -> list[dict[str, str]]:
        """Construct grounded system and user messages for final answer deduction."""
        if generated_paths:
            formatted_paths = "\n".join(
                f"{idx + 1}. {format_reasoning_path(path)}"
                for idx, path in enumerate(generated_paths)
            )
        else:
            formatted_paths = "(No Knowledge Graph paths retrieved)"

        return [
            {
                "role": "system",
                "content": (
                    "You are a precise Knowledge Graph Question Answering assistant. "
                    "Use the provided Knowledge Graph reasoning paths to answer the question. "
                    "Return ONLY the concise final answer entity or value without extra commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Knowledge Graph Reasoning Paths:\n{formatted_paths}\n\n"
                    "Final Answer:"
                ),
            },
        ]

    @staticmethod
    def _mock_deduce(
        subgraph: SubgraphData,
        generated_paths: Sequence[Sequence[Triple]],
    ) -> str:
        """Heuristic mock deduction selecting a gold-matching or terminal path entity."""
        a_set = set(subgraph.a_entities)
        for path in generated_paths:
            for triple in reversed(path):
                if triple.tail in a_set:
                    return triple.tail
                if triple.head in a_set:
                    return triple.head

        if generated_paths and generated_paths[0]:
            return generated_paths[0][-1].tail
        return "Unknown"

    def deduce_answer(
        self,
        subgraph: SubgraphData,
        generated_paths: list[list[Triple]],
        prior_latency_ms: float = 0.0,
    ) -> ReasoningOutput:
        """Produce a `ReasoningOutput` from `subgraph` and constrained `generated_paths`."""
        start_time = time.perf_counter()

        if self.use_mock:
            answer = self._mock_deduce(subgraph, generated_paths)
        else:
            client = self._get_client()
            messages = self.build_messages(subgraph.question, generated_paths)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = (response.choices[0].message.content or "").strip()

        deduction_ms = (time.perf_counter() - start_time) * 1000.0
        return ReasoningOutput(
            question_id=subgraph.question_id,
            generated_paths=generated_paths,
            final_answer=answer,
            latency_ms=prior_latency_ms + deduction_ms,
        )

