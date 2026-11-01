#!/usr/bin/env python3
"""End-to-end environment, compute device, SubgraphData, and KGTrie verification script."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Ensure repository root is importable regardless of current working directory
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
import pcst_fast
import sentence_transformers
import torch
import transformers

from src.decoding.kg_trie import KGTrie, KGTrieLogitsProcessor
from src.utils.device import get_device
from src.utils.types import SubgraphData, Triple


def main() -> int:
    """Execute the 3-step backbone smoke test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("=" * 72)
    print("GCR-PCST-KGQA Backbone Setup Verification")
    print("=" * 72)

    # 1. Check and print core library versions
    print("\n[1/3] Checking core package versions...")
    print(f"  - python                : {sys.version.split()[0]}")
    print(f"  - torch                 : {torch.__version__}")
    print(f"  - transformers          : {transformers.__version__}")
    print(f"  - sentence_transformers : {sentence_transformers.__version__}")
    print(f"  - networkx              : {nx.__version__}")
    print(f"  - pcst_fast             : {getattr(pcst_fast, '__name__', 'pcst_fast')} (loaded)")

    # 2. Detect active compute device and run tensor matrix multiplication
    print("\n[2/3] Detecting compute device and running matrix multiplication...")
    device = get_device()
    torch.manual_seed(42)
    mat_a = torch.randn((128, 128), dtype=torch.float32, device=device)
    mat_b = torch.randn((128, 128), dtype=torch.float32, device=device)
    product = torch.matmul(mat_a, mat_b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    assert product.shape == (128, 128), f"Unexpected product shape: {product.shape}"
    assert torch.isfinite(product).all().item(), "Non-finite values in matrix multiplication!"
    print(
        f"  - Active device         : {device}"
        f"\n  - MatMul output shape   : {tuple(product.shape)} (norm={product.norm().item():.4f})"
    )

    # 3. Verify SubgraphData -> NetworkX, KGTrie, and KGTrieLogitsProcessor masking
    print("\n[3/3] Testing SubgraphData, NetworkX conversion, KGTrie & KGTrieLogitsProcessor...")
    dummy_subgraph = SubgraphData(
        question_id="verify_001",
        question="Which university is in the historic capital of Portugal?",
        q_entities=["Portugal"],
        a_entities=["University of Coimbra"],
        triples=[
            Triple(head="Portugal", relation="historic_capital", tail="Coimbra"),
            Triple(head="Coimbra", relation="institution", tail="University of Coimbra"),
            Triple(head="Portugal", relation="capital", tail="Lisbon"),
        ],
    )
    nx_graph = dummy_subgraph.to_networkx()
    assert isinstance(nx_graph, nx.DiGraph)
    assert nx_graph.number_of_nodes() == 4
    assert nx_graph.number_of_edges() == 3
    print(
        f"  - SubgraphData -> DiGraph: {nx_graph.number_of_nodes()} nodes, "
        f"{nx_graph.number_of_edges()} edges"
    )

    trie = KGTrie()
    trie.insert([10, 20, 30])
    trie.insert([10, 25, 40])
    allowed_after_10 = trie.get_allowed_next_tokens([10])
    assert allowed_after_10 == [20, 25], f"Expected [20, 25], got {allowed_after_10}"

    processor = KGTrieLogitsProcessor(trie=trie, prompt_length=0)
    mock_input_ids = torch.tensor([[10]], dtype=torch.long, device=device)
    mock_logits = torch.arange(50, dtype=torch.float32, device=device).unsqueeze(0)
    masked_logits = processor(mock_input_ids, mock_logits)

    finite_mask = torch.isfinite(masked_logits[0])
    surviving_tokens = torch.nonzero(finite_mask, as_tuple=False).squeeze(-1).tolist()
    assert surviving_tokens == [20, 25], f"Masking failed! Surviving tokens: {surviving_tokens}"
    assert masked_logits[0, 0].item() == float("-inf")
    assert masked_logits[0, 20].item() == 20.0
    assert masked_logits[0, 25].item() == 25.0
    print(f"  - Allowed next tokens   : {allowed_after_10}")
    print(f"  - Unmasked logit tokens : {surviving_tokens} (all invalid tokens masked to -inf)")

    print("\n" + "=" * 72)
    print("SUCCESS: All environment, device, and KG-Trie interface checks passed!")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

