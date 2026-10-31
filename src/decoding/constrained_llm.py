"""Graph-constrained reasoning path generator combining Llama-3-8B and `KGTrieLogitsProcessor`."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from src.decoding.kg_trie import (
    KGTrie,
    KGTrieLogitsProcessor,
    enumerate_kg_paths,
    format_reasoning_path,
    parse_reasoning_path,
)
from src.retrieval.embedder import GraphEmbedder
from src.utils.device import get_device
from src.utils.types import SubgraphData, Triple

logger = logging.getLogger(__name__)


class ConstrainedPathGenerator:
    """Generates faithful KG reasoning paths constrained by a localized `KGTrie`."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_path_hops: int = 2,
        max_paths_in_trie: int = 512,
        num_beams: int = 3,
        max_new_tokens: int = 64,
        device: torch.device | None = None,
        use_mock: bool = True,
    ) -> None:
        """Initialize the KG-constrained LLM path generator.

        Args:
            model_name: Hugging Face CausalLM checkpoint (e.g., Llama-3-8B-Instruct).
            max_path_hops: Maximum hop length $L$ indexed into the localized `KGTrie`.
            max_paths_in_trie: Maximum number of candidate paths inserted into the Trie.
            num_beams: Number of beam search reasoning paths returned.
            max_new_tokens: Maximum generated token budget per path.
            device: Target PyTorch device; defaults to `get_device()`.
            use_mock: Exercise `KGTrie` + `KGTrieLogitsProcessor` with a lightweight mock LM.
        """
        self.model_name = model_name
        self.max_path_hops = max_path_hops
        self.max_paths_in_trie = max_paths_in_trie
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.device = device if device is not None else get_device()
        self.use_mock = use_mock

        self._tokenizer = None
        self._model = None
        self._mock_embedder = GraphEmbedder(device=self.device, use_mock=True) if use_mock else None

    def _load_hf_model(self):
        """Lazily load Hugging Face tokenizer and CausalLM onto `self.device`."""
        if self._tokenizer is None or self._model is None:
            logger.info("Loading constrained path LLM '%s' on %s", self.model_name, self.device)
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

    @staticmethod
    def _build_prompt(question: str) -> str:
        """Format the reasoning path generation prompt prefix."""
        return (
            "Generate a faithful Knowledge Graph reasoning path answering the question.\n"
            f"Question: {question}\n"
            "Reasoning Path: "
        )

    def _generate_mock_constrained(
        self,
        subgraph: SubgraphData,
        candidate_paths: list[list[Triple]],
    ) -> list[list[Triple]]:
        """Run token-by-token constrained decoding through `KGTrieLogitsProcessor` using mock logits."""
        eos_id = 1
        vocab: dict[str, int] = {"<pad>": 0, "<eos>": eos_id}
        id_to_token: dict[int, str] = {0: "<pad>", eos_id: "<eos>"}

        def _encode(text: str) -> list[int]:
            ids: list[int] = []
            for tok in text.split(" "):
                if tok not in vocab:
                    new_id = len(vocab)
                    vocab[tok] = new_id
                    id_to_token[new_id] = tok
                ids.append(vocab[tok])
            return ids

        trie = KGTrie.from_paths(candidate_paths, encode_fn=_encode, eos_token_id=eos_id)
        processor = KGTrieLogitsProcessor(trie=trie, prompt_length=0, eos_token_id=eos_id)

        # Score candidate paths by lexical similarity to bias mock token logits realistically
        path_texts = [format_reasoning_path(p) for p in candidate_paths]
        assert self._mock_embedder is not None
        sims = self._mock_embedder.compute_similarities(subgraph.question, path_texts).detach().cpu()
        ranked_indices = torch.argsort(sims, descending=True).tolist()

        decoded_paths: list[list[Triple]] = []
        vocab_size = len(vocab)
        for rank_idx in ranked_indices[: max(1, self.num_beams)]:
            target_tokens = _encode(path_texts[rank_idx]) + [eos_id]
            generated: list[int] = []
            for step_tok in target_tokens:
                input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)
                raw_logits = torch.zeros((1, vocab_size), dtype=torch.float32, device=self.device)
                raw_logits[0, step_tok] = 5.0
                masked_logits = processor(input_ids, raw_logits)
                next_tok = int(torch.argmax(masked_logits[0]).item())
                if next_tok == eos_id:
                    break
                generated.append(next_tok)

            decoded_text = " ".join(id_to_token[t] for t in generated)
            parsed = parse_reasoning_path(decoded_text)
            if parsed:
                decoded_paths.append(parsed)

        return decoded_paths

    def generate_paths(self, subgraph: SubgraphData) -> list[list[Triple]]:
        """Enumerate valid paths, build `KGTrie`, and decode top faithful reasoning paths."""
        candidate_paths = enumerate_kg_paths(
            subgraph,
            max_hops=self.max_path_hops,
            max_paths=self.max_paths_in_trie,
        )
        if not candidate_paths:
            return []

        if self.use_mock:
            return self._generate_mock_constrained(subgraph, candidate_paths)

        tokenizer, model = self._load_hf_model()
        eos_id = tokenizer.eos_token_id

        def _hf_encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        trie = KGTrie.from_paths(candidate_paths, encode_fn=_hf_encode, eos_token_id=eos_id)
        prompt = self._build_prompt(subgraph.question)
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        processor = KGTrieLogitsProcessor(
            trie=trie,
            prompt_length=prompt_len,
            eos_token_id=eos_id,
        )

        num_return = min(self.num_beams, len(candidate_paths))
        with torch.no_grad():
            outputs = model.generate(
                **prompt_inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=max(1, num_return),
                num_return_sequences=max(1, num_return),
                do_sample=False,
                logits_processor=LogitsProcessorList([processor]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
            )

        generated_paths: list[list[Triple]] = []
        for seq in outputs:
            gen_ids = seq[prompt_len:].tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            parsed = parse_reasoning_path(text)
            if parsed:
                generated_paths.append(parsed)

        return generated_paths
