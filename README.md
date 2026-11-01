# Graph-Constrained Reasoning for Reliable LLM Question Answering (`kl-hybrid-gcr`)

A 3-stage hybrid **Knowledge Graph Question Answering (KGQA)** research pipeline combining **G-Retriever's** Prize-Collecting Steiner Tree (PCST) subgraph retrieval with **Graph-Constrained Reasoning (GCR)** via a localized prefix `KGTrie`.

---

## 1. Architecture Overview

```text
  [Question + Entities]
           │
           ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Stage 1: Graph Acquisition (`src/acquisition/`)                   │
  │  • Benchmarks: `rmanluo/RoG-webqsp` & `rmanluo/RoG-cwq` (Freebase)│
  │  • Open-world: Live L-hop Wikidata SPARQL + PyVis HTML export     │
  └───────────────────────────────────────────────────────────────────┘
           │  SubgraphData (Full L-hop neighborhood)
           ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Stage 2: Subgraph Retrieval (`src/retrieval/`)                    │
  │  • SentenceBERT question/node/relation embeddings                 │
  │  • Prize-Collecting Steiner Tree (`pcst_fast`) pruning            │
  └───────────────────────────────────────────────────────────────────┘
           │  SubgraphData (Compact, question-relevant subgraph)
           ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Stage 3: Graph-Constrained Decoding (`src/decoding/`)             │
  │  • Prefix `KGTrie` + HF `KGTrieLogitsProcessor` (-inf masking)    │
  │  • `Llama-3-8B` faithful path generation -> `GPT-4o-mini` answer  │
  └───────────────────────────────────────────────────────────────────┘
           │  ReasoningOutput (Generated KG paths, final answer, ms)
           ▼
  [Evaluation (`src/evaluation/metrics.py`)]
  Hit@1 | F1 | Faithful Reasoning Ratio | Answer-Path Recall | Latency
```

### Strict Subgraph Interface (`src/utils/types.py`)
All modules communicate exclusively through shared `@dataclass` contracts so team members can develop and test stages independently:
- `Triple(head, relation, tail)`
- `SubgraphData(question_id, question, q_entities, a_entities, triples)` with `.to_networkx() -> nx.DiGraph`
- `ReasoningOutput(question_id, generated_paths, final_answer, latency_ms)`

---

## 2. Repository Structure

```text
kl-hybrid-gcr/
├── .gitignore
├── README.md
├── environment.yml
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── device.py          # Hardware-agnostic device selection (CUDA -> MPS -> CPU)
│   │   └── types.py           # Shared dataclasses for the Subgraph Interface
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py  # WebQSP / CWQ loader (Hugging Face datasets)
│   │   ├── sparql_client.py   # Live Wikidata L-hop SPARQL acquisition
│   │   └── visualizer.py      # NetworkX & PyVis interactive HTML graph export
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py        # SentenceBERT node/edge/question encoding
│   │   └── pcst_retriever.py  # PCST optimization pruning
│   ├── decoding/
│   │   ├── __init__.py
│   │   ├── kg_trie.py         # Prefix Trie built from tokenized KG paths
│   │   ├── constrained_llm.py # HF LogitsProcessor + Llama-3-8B path generation
│   │   └── final_reasoner.py  # GPT-4o-mini final answer deduction
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── llm_only.py        # Zero-shot & CoT without KG
│   │   └── standard_rag.py    # Top-k triple retrieval via SentenceBERT
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py         # Hit@1, F1, Faithful Reasoning Ratio, Subgraph Recall, Latency
├── scripts/
│   ├── verify_setup.py        # End-to-end smoke test for packages, GPU, and dummy pipeline
│   └── run_pipeline.py        # CLI entry point
└── tests/
    └── test_interfaces.py     # Pytest unit test for data contracts and KGTrie
```

---

## 3. Environment Setup

Compatible with **NVIDIA GPUs (`cuda`)**, **Apple Silicon (`mps`)**, and **CPU** on Python 3.11.

### Option A: Conda
```bash
conda env create -f environment.yml
conda activate kl-hybrid-gcr
```

### Option B: `uv` or `venv` + `pip`
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file at the repository root when running live LLM APIs or gated Hugging Face models:
```ini
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

---

## 4. Verification & Usage

### Run Setup Verification & Unit Tests
```bash
python scripts/verify_setup.py
pytest
```

### Run the Pipeline CLI
```bash
# End-to-end smoke run using mock models & synthetic KG sample
python scripts/run_pipeline.py --source mock --mode hybrid --use-mock-models

# Export an interactive PyVis HTML visualization of the pruned subgraph & reasoning paths
python scripts/run_pipeline.py --source mock --mode hybrid --use-mock-models --export-html outputs/demo_subgraph.html

# Run baselines (standard_rag, full_gcr, llm_only_zero_shot, llm_only_cot)
python scripts/run_pipeline.py --source mock --mode standard_rag --use-mock-models
```

---

## 5. References

1. L. Luo et al., *Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models*, ICML 2025.
2. X. He et al., *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering*, NeurIPS 2024.
3. L. Chen et al., *Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs*, NeurIPS 2024.
4. L. Luo et al., *Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning*, ICLR 2024.

