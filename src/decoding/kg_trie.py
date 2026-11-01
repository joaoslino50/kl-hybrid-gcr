"""Prefix Trie over tokenized Knowledge Graph paths and Hugging Face LogitsProcessor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
from transformers import LogitsProcessor

from src.utils.types import SubgraphData, Triple


@dataclass
class _TrieNode:
    """Internal node of `KGTrie` mapping token IDs to child nodes."""

    children: dict[int, _TrieNode] = field(default_factory=dict)
    is_end: bool = False


class KGTrie:
    """Prefix Trie over token ID sequences representing valid KG reasoning paths."""

    def __init__(self) -> None:
        """Initialize an empty token-level prefix Trie."""
        self._root = _TrieNode()
        self._num_sequences = 0

    def __len__(self) -> int:
        """Return the number of distinct sequences inserted into the Trie."""
        return self._num_sequences

    def insert(self, token_ids: list[int]) -> None:
        """Insert a tokenized KG path sequence into the prefix Trie.

        Args:
            token_ids: Sequence of integer token IDs for a valid reasoning path.
        """
        node = self._root
        for token_id in token_ids:
            tok = int(token_id)
            if tok not in node.children:
                node.children[tok] = _TrieNode()
            node = node.children[tok]

        if not node.is_end:
            node.is_end = True
            self._num_sequences += 1

    def insert_batch(self, sequences: Iterable[Sequence[int]]) -> None:
        """Insert multiple token ID sequences into the Trie."""
        for seq in sequences:
            self.insert(list(seq))

    def _traverse(self, prefix_ids: Sequence[int]) -> _TrieNode | None:
        """Walk the Trie along `prefix_ids`, returning the reached node or `None`."""
        node = self._root
        for token_id in prefix_ids:
            tok = int(token_id)
            child = node.children.get(tok)
            if child is None:
                return None
            node = child
        return node

    def get_allowed_next_tokens(self, prefix_ids: list[int]) -> list[int]:
        """Return sorted valid next token IDs continuing `prefix_ids` in the KG-Trie.

        Args:
            prefix_ids: Generated prefix token IDs so far.

        Returns:
            list[int]: Allowed next token IDs, or `[]` if `prefix_ids` is off-trie or at a leaf.
        """
        node = self._traverse(prefix_ids)
        if node is None:
            return []
        return sorted(node.children.keys())

    def is_terminal(self, prefix_ids: Sequence[int]) -> bool:
        """Check whether `prefix_ids` corresponds to a complete inserted KG path."""
        node = self._traverse(prefix_ids)
        return bool(node is not None and node.is_end)

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[Sequence[Triple]],
        encode_fn: Callable[[str], list[int]],
        eos_token_id: int | None = None,
    ) -> KGTrie:
        """Build a `KGTrie` by verbalizing and tokenizing KG reasoning paths."""
        trie = cls()
        for path in paths:
            if not path:
                continue
            text = format_reasoning_path(path)
            token_ids = list(encode_fn(text))
            if eos_token_id is not None and (not token_ids or token_ids[-1] != eos_token_id):
                token_ids.append(eos_token_id)
            trie.insert(token_ids)
        return trie


class KGTrieLogitsProcessor(LogitsProcessor):
    """Hugging Face `LogitsProcessor` masking tokens outside `KGTrie` with `-inf`."""

    def __init__(
        self,
        trie: KGTrie,
        prompt_length: int = 0,
        eos_token_id: int | None = None,
    ) -> None:
        """Initialize the KG-Trie constrained decoding processor.

        Args:
            trie: Populated `KGTrie` of valid tokenized reasoning paths.
            prompt_length: Number of prompt tokens preceding constrained path generation.
            eos_token_id: Optional EOS token ID permitted once a terminal path node is reached.
        """
        super().__init__()
        self.trie = trie
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Mask invalid next-token logits with `-inf` for every sequence in the batch.

        Args:
            input_ids: Current token IDs of shape `(batch_size, sequence_length)`.
            scores: Unnormalized next-token logits of shape `(batch_size, vocab_size)`.

        Returns:
            torch.FloatTensor: Masked logits tensor with `-inf` on all invalid tokens.
        """
        masked_scores = torch.full_like(scores, float("-inf"))
        vocab_size = scores.shape[-1]

        for batch_idx in range(scores.shape[0]):
            if input_ids.shape[1] > self.prompt_length:
                prefix_ids = input_ids[batch_idx, self.prompt_length :].tolist()
            else:
                prefix_ids = []

            allowed = self.trie.get_allowed_next_tokens(prefix_ids)

            if self.eos_token_id is not None:
                if self.trie.is_terminal(prefix_ids) and self.eos_token_id not in allowed:
                    allowed = [*allowed, self.eos_token_id]
                elif not allowed:
                    allowed = [self.eos_token_id]

            valid_indices = [tok for tok in allowed if 0 <= tok < vocab_size]
            if valid_indices:
                idx_tensor = torch.tensor(valid_indices, dtype=torch.long, device=scores.device)
                masked_scores[batch_idx, idx_tensor] = scores[batch_idx, idx_tensor]

        return masked_scores


def format_reasoning_path(path: Sequence[Triple]) -> str:
    """Serialize a sequence of `Triple` hops into a canonical GCR path string."""
    return " ; ".join(triple.to_text() for triple in path)


def parse_reasoning_path(path_text: str) -> list[Triple]:
    """Parse a canonical `"head -> rel -> tail ; ..."` string back into `list[Triple]`."""
    triples: list[Triple] = []
    for segment in path_text.split(";"):
        parts = [part.strip() for part in segment.strip().split("->")]
        if len(parts) == 3 and all(parts):
            triples.append(Triple(head=parts[0], relation=parts[1], tail=parts[2]))
    return triples


def enumerate_kg_paths(
    subgraph: SubgraphData,
    max_hops: int = 2,
    max_paths: int = 512,
) -> list[list[Triple]]:
    """Enumerate valid simple multi-hop KG paths starting from `q_entities` (or all heads).

    Args:
        subgraph: Source `SubgraphData` (typically pruned by `PCSTRetriever`).
        max_hops: Maximum reasoning path length $L$.
        max_paths: Safety cap on total enumerated paths.

    Returns:
        list[list[Triple]]: Valid KG paths grounded strictly in `subgraph.triples`.
    """
    if not subgraph.triples or max_hops < 1:
        return []

    outgoing: dict[str, list[Triple]] = {}
    incoming_inverted: dict[str, list[Triple]] = {}
    for triple in subgraph.triples:
        outgoing.setdefault(triple.head, []).append(triple)
        incoming_inverted.setdefault(triple.tail, []).append(triple)

    seed_entities = [e for e in subgraph.q_entities if e in outgoing or e in incoming_inverted]
    if not seed_entities:
        seed_entities = list(outgoing.keys())

    paths: list[list[Triple]] = []

    def _dfs(current_entity: str, current_path: list[Triple], visited: set[str]) -> None:
        if len(paths) >= max_paths:
            return
        if current_path:
            paths.append(list(current_path))
        if len(current_path) >= max_hops:
            return

        # Prioritize forward directed triples from current_entity
        for triple in outgoing.get(current_entity, []):
            if triple.tail not in visited:
                visited.add(triple.tail)
                current_path.append(triple)
                _dfs(triple.tail, current_path, visited)
                current_path.pop()
                visited.remove(triple.tail)
                if len(paths) >= max_paths:
                    return

    for seed in seed_entities:
        _dfs(seed, [], {seed})
        if len(paths) >= max_paths:
            break

    # If q_entities only had incoming edges, include 1-hop incoming triples so Trie is non-empty
    if not paths:
        for triple in subgraph.triples[:max_paths]:
            paths.append([triple])

    return paths

