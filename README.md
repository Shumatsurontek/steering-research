# Steering Vectors vs. Prompt Engineering for Agentic Calendar Tasks

A comparative study of **activation steering** versus **prompt engineering** for improving task-specific behavior in instruction-tuned LLMs, focusing on an agentic calendar scheduling use case.

> **Paper:** [`article/main.pdf`](article/main.pdf) — 8-page arxiv-style article with full results

## Key Findings

| Finding | Detail |
|---------|--------|
| **Last-layer steering is ineffective on instruct models** | Even at 50× coefficient, Qwen3-4B-Instruct generation is unchanged. Instruction tuning saturates the behavior. |
| **Llama-3.2-3B has the best tokenizer for agentic tasks** | Overall score 0.551 — best JSON compression (106 tokens vs 125-136) and temporal handling |
| **All tokenizers fragment ISO 8601 timestamps** | Temporal integrity < 0.32 across all 4 models — a universal bottleneck |
| **Logit lens confirms calendar features in layers 33-35** | Promoted tokens: `schedule`, `agenda`, `attendees`, `RSVP`, `calendar` |
| **Cross-lingual steering signal detected** | Chinese calendar tokens promoted alongside English/French equivalents |

## Architecture

```
steering-research/
├── article/
│   └── main.tex / main.pdf          # LaTeX article (8 pages)
├── src/
│   ├── tokenizers/
│   │   ├── compare.py                # Tokenizer comparison + quality scoring
│   │   └── visualize.py              # Publication-quality plots
│   ├── analysis/
│   │   └── sae_features.py           # Contrastive activation extraction + logit lens
│   ├── steering/
│   │   └── apply_vectors.py          # Steering vector application + coefficient sweep
│   └── agents/
│       └── prompt_baselines.py       # 5 prompt strategies + 29-case eval dataset
├── results/
│   ├── token_counts_heatmap.png      # Tokenizer comparison heatmap
│   ├── compression_ratio_bar.png     # Compression ratios by model
│   ├── quality_scores_radar.png      # Radar chart of quality metrics
│   ├── tokenization_detail.png       # Visual token segmentation
│   ├── fragmentation_focus.png       # Date/JSON fragmentation analysis
│   ├── steering_vectors.pt           # 36 per-layer steering vectors
│   ├── layer_importance.json         # Layer ranking by discriminative power
│   ├── logit_lens_results.json       # Promoted/suppressed tokens per layer
│   └── steering_results.json         # Coefficient sweep results
├── data/
│   └── calendar_eval_dataset.json    # 29 bilingual test cases
├── PLAN.md                           # Research plan with status
└── requirements.txt                  # Python dependencies
```

## Methodology

### 1. Tokenizer Analysis

Comparative analysis of 4 tokenizers on 9 prompt categories (calendar requests, temporal expressions, JSON tool calls):

| Model | Vocab Size | Overall Score | Best At |
|-------|-----------|--------------|---------|
| Qwen3-4B-Instruct-2507 | 151,643 | 0.483 | Multilingual parity |
| Gemma-3-1B-IT | 262,144 | 0.511 | Semantic coherence |
| **Llama-3.2-3B-Instruct** | **128,000** | **0.551** | **JSON + temporal** |
| Phi-3-mini-4k-instruct | 32,000 | 0.438 | — |

Scoring dimensions: `temporal_integrity`, `semantic_coherence`, `json_efficiency`, `multilingual_parity`

### 2. Contrastive Feature Extraction

SAE-inspired approach without training autoencoders:
- **10 calendar** vs **10 neutral** contrastive prompt pairs
- Hook-based residual stream extraction at all 36 layers
- Mean-difference steering vectors with L2 norm ranking
- Logit lens projection through unembedding matrix

```
Layer 35: L2=361.0  →  Promotes: schedule, agenda, attendees, RSVP
Layer 34: L2=218.6  →  Promotes: schedule, agenda, calendar, attendees
Layer 33: L2=179.4  →  Promotes: agenda, schedule, invite, attendees
```

### 3. Steering Vector Application

Coefficient sweep (α ∈ {0, 5, 15, 30, 50}) on layer 35:

```
Result: NO CHANGE at any coefficient.
Instruction tuning has already saturated calendar behavior.
```

### 4. Prompt Engineering Baselines

5 strategies on a 29-case bilingual dataset (18 FR / 11 EN):

| Strategy | Messages | Description |
|----------|----------|-------------|
| Zero-shot | 2 | System prompt only |
| Few-shot (3) | 8 | 3 example pairs |
| Few-shot (5) | 12 | 5 example pairs |
| Chain-of-thought | 2 | Step-by-step extraction |
| Tool use | 2 + tool | Function calling schema |

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1: Tokenizer analysis
python -m src.tokenizers.compare
python -m src.tokenizers.visualize

# Phase 2: Feature extraction (downloads Qwen3-4B ~8GB)
python -m src.analysis.sae_features

# Phase 3: Steering vectors
python -m src.steering.apply_vectors

# Phase 4: Prompt baselines (creates eval dataset)
python -m src.agents.prompt_baselines
```

## Models

| Model | Params | Context | Role |
|-------|--------|---------|------|
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4.0B | 256K | Primary: steering + generation |
| [Gemma-3-1B-IT](https://huggingface.co/google/gemma-3-1b-it) | 1.0B | 32K | Tokenizer comparison |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3.2B | 128K | Tokenizer comparison |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 3.8B | 4K | Tokenizer comparison |

## References

- [Steering LLM Reasoning Through Bias-Only Adaptation](https://arxiv.org/abs/2505.18706) (EMNLP 2025)
- [Steering LLM Thinking with Budget Guidance](https://arxiv.org/abs/2506.13752)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Neuronpedia — Qwen3-4B Circuit Tracer](https://www.neuronpedia.org/qwen3-4b/graph)
- [Gemma Scope 2 Tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r)

## Future Work

- [ ] Mid-layer steering (layers 15-25) where representations are more malleable
- [ ] Steering on base (non-instruct) Qwen3-4B for comparison
- [ ] Full inference evaluation with Ollama/llama.cpp + LangChain
- [ ] Budget guidance integration for reasoning length control
- [ ] Train task-specific SAEs using Neuronpedia framework

## License

MIT
