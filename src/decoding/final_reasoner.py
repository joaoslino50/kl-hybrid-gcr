"""Final answer deduction over KG-constrained reasoning paths via Llama-3-8B (HF) or OpenAI."""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence

from dotenv import load_dotenv
import torch

from src.decoding.constrained_llm import load_shared_hf_causal_lm
from src.decoding.kg_trie import format_reasoning_path
from src.utils.device import get_device
from src.utils.types import ReasoningOutput, SubgraphData, Triple

logger = logging.getLogger(__name__)


class FinalAnswerReasoner:
    """Inductive LLM producing the final answer from faithful KG paths on GPU (or OpenAI API)."""

    def __init__(
        self,
        model_name: str = "NousResearch/Meta-Llama-3-8B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 64,
        device: torch.device | None = None,
        api_key: str | None = None,
        use_mock: bool = False,
    ) -> None:
        """Initialize the final answer reasoner.

        By default, uses Hugging Face (`meta-llama/Meta-Llama-3-8B-Instruct`) on the active
        GPU (`cuda` / `mps`), sharing VRAM weights with `ConstrainedPathGenerator`.
        If `model_name` starts with `"gpt-"` (e.g. `"gpt-4o-mini"`), uses the OpenAI client.

        Args:
            model_name: Hugging Face model ID (default `"meta-llama/Meta-Llama-3-8B-Instruct"`)
                or OpenAI model ID (`"gpt-4o-mini"`).
            temperature: Sampling temperature (`0.0` for greedy deterministic QA).
            max_tokens: Maximum generated response tokens.
            device: Target PyTorch device for Hugging Face models; defaults to `get_device()`.
            api_key: Optional OpenAI API key (only used if `model_name` is an OpenAI `gpt-*` model).
            use_mock: Deduce answer heuristically from the top generated path without LLM calls.
        """
        load_dotenv()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device if device is not None else get_device()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_mock = use_mock
        self._openai_client = None

    @property
    def is_openai_model(self) -> bool:
        """Return True only when configured with an OpenAI `gpt-*` / `o1-*` model."""
        lower = self.model_name.lower()
        return lower.startswith(("gpt-", "o1-", "o3-"))

    def _get_openai_client(self):
        """Lazily initialize `openai.OpenAI` client when using an OpenAI model."""
        if self._openai_client is None:
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Use a Hugging Face model such as "
                    "'meta-llama/Meta-Llama-3-8B-Instruct' or set OPENAI_API_KEY."
                )
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.api_key)
        return self._openai_client

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
    def _clean_generated_answer(raw_text: str) -> str:
        """Strip boilerplate prefixes and keep the concise entity answer line."""
        cleaned = raw_text.strip()
        marker = "final answer:"
        if marker in cleaned.lower():
            idx = cleaned.lower().rfind(marker) + len(marker)
            cleaned = cleaned[idx:].strip()
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return lines[0].strip(" .\"'") if lines else "Unknown"

    def _deduce_with_hf(
        self,
        question: str,
        generated_paths: Sequence[Sequence[Triple]],
    ) -> str:
        """Run final answer deduction on GPU using the shared Hugging Face CausalLM."""
        tokenizer, model = load_shared_hf_causal_lm(self.model_name, self.device)
        messages = self.build_messages(question, generated_paths)

        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"{messages[0]['content']}\n\n{messages[1]['content']} "

        inputs = tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
        return self._clean_generated_answer(decoded)

    @staticmethod
    def _mock_deduce(
        subgraph: SubgraphData,
        generated_paths: Sequence[Sequence[Triple]],
    ) -> str:
        """Heuristic mock deduction selecting a gold-matching or non-topic path entity."""
        a_set = set(subgraph.a_entities)
        q_set = set(subgraph.q_entities)
        for path in generated_paths:
            for triple in reversed(path):
                if triple.tail in a_set:
                    return triple.tail
                if triple.head in a_set:
                    return triple.head

        if generated_paths and generated_paths[0]:
            last_triple = generated_paths[0][-1]
            if last_triple.tail not in q_set:
                return last_triple.tail
            return last_triple.head
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
        elif self.is_openai_model:
            client = self._get_openai_client()
            messages = self.build_messages(subgraph.question, generated_paths)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = self._clean_generated_answer(response.choices[0].message.content or "")
        else:
            answer = self._deduce_with_hf(subgraph.question, generated_paths)

        deduction_ms = (time.perf_counter() - start_time) * 1000.0
        return ReasoningOutput(
            question_id=subgraph.question_id,
            generated_paths=generated_paths,
            final_answer=answer,
            latency_ms=prior_latency_ms + deduction_ms,
        )
