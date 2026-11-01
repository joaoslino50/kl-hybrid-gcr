"""LLM-only baseline (Zero-shot and Chain-of-Thought prompting without KG access)."""

from __future__ import annotations

import logging
import time
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.device import get_device
from src.utils.types import ReasoningOutput, SubgraphData

logger = logging.getLogger(__name__)


class LLMOnlyBaseline:
    """Parametric-memory LLM baseline operating without Knowledge Graph context."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        mode: Literal["zero_shot", "cot"] = "zero_shot",
        max_new_tokens: int = 128,
        device: torch.device | None = None,
        use_mock: bool = True,
    ) -> None:
        """Initialize the LLM-only baseline.

        Args:
            model_name: Hugging Face CausalLM checkpoint.
            mode: `"zero_shot"` for direct answer or `"cot"` for Chain-of-Thought prompting.
            max_new_tokens: Generation token budget.
            device: PyTorch device; defaults to `get_device()`.
            use_mock: Return deterministic mock parametric predictions without loading weights.
        """
        if mode not in {"zero_shot", "cot"}:
            raise ValueError(f"Unsupported LLMOnlyBaseline mode: {mode!r}")
        self.model_name = model_name
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.device = device if device is not None else get_device()
        self.use_mock = use_mock

        self._tokenizer = None
        self._model = None

    def _load_hf_model(self):
        """Lazily load Hugging Face tokenizer and CausalLM."""
        if self._tokenizer is None or self._model is None:
            logger.info("Loading LLMOnlyBaseline '%s' on %s", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            dtype = torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(self.device)
            self._model.eval()
        return self._tokenizer, self._model

    def build_prompt(self, question: str) -> str:
        """Build either a zero-shot or Chain-of-Thought prompt without KG triples."""
        if self.mode == "cot":
            return (
                "Answer the following question step by step, then conclude with "
                "'Final Answer: <entity>'.\n"
                f"Question: {question}\n"
                "Reasoning:"
            )
        return (
            "Answer the following question concisely with only the entity name.\n"
            f"Question: {question}\n"
            "Final Answer:"
        )

    @staticmethod
    def _extract_final_answer(raw_text: str) -> str:
        """Extract the answer span following `'Final Answer:'` if present."""
        marker = "final answer:"
        lower_text = raw_text.lower()
        if marker in lower_text:
            idx = lower_text.rfind(marker) + len(marker)
            return raw_text[idx:].strip().splitlines()[0].strip()
        lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def answer(self, subgraph: SubgraphData) -> ReasoningOutput:
        """Answer `subgraph.question` without inspecting `subgraph.triples`."""
        start_time = time.perf_counter()

        if self.use_mock:
            # Deliberately ignores subgraph.triples to simulate parametric-only prediction
            guess = subgraph.q_entities[0] if subgraph.q_entities else "Unknown"
            final_answer = f"Unverified ({guess})"
        else:
            tokenizer, model = self._load_hf_model()
            prompt = self.build_prompt(subgraph.question)
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = int(inputs["input_ids"].shape[1])

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            decoded = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
            final_answer = self._extract_final_answer(decoded)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return ReasoningOutput(
            question_id=subgraph.question_id,
            generated_paths=[],
            final_answer=final_answer,
            latency_ms=elapsed_ms,
        )

