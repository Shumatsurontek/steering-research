# Steering Vectors vs. Prompt Engineering for Agentic Calendar Tasks

A comparative study of **activation steering** versus **prompt engineering** for improving task-specific behavior in instruction-tuned LLMs, focusing on an agentic calendar scheduling use case.

> **Paper:** [`article/main.pdf`](article/main.pdf) — arxiv-style article with full results

## Key Findings

| Finding | Detail |
|---------|--------|
| **Mid-layer steering is the sweet spot** | Layers 15–18 at α=30 achieve 100% behavioral change on Qwen3-4B-Instruct. Late layers (33–35) are rigid. |
| **Base models are more steerable but fragile** | Layer 15, α=30 boosts calendar score from 0.70→0.97 on base model, but α≥60 causes degeneration to 0.0 |
| **Late-layer rigidity is architectural** | Both instruct and base models show the same pattern — not caused by instruction tuning |
| **Budget guidance has no effect on compact outputs** | Instruct model already generates ~75 tokens of JSON — 0% savings at any budget level |
| **SAE features confirm extreme sparsity** | 9/450 sampled Neuronpedia features (0.07%) are calendar-related; 115 found via keyword search |
| **Llama-3.2-3B has the best tokenizer** | Overall score 0.551 — best JSON compression (106 tokens vs 125-136) and temporal handling |
| **All tokenizers fragment ISO 8601** | Temporal integrity < 0.32 across all 4 models — a universal bottleneck |
| **Cross-lingual steering signal** | Chinese calendar tokens promoted alongside English/French equivalents in logit lens |

## Architecture

```
steering-research/
├── article/
│   └── main.tex / main.pdf              # LaTeX article (~10 pages)
├── src/
│   ├── tokenizers/
│   │   ├── compare.py                    # Tokenizer comparison + quality scoring
│   │   └── visualize.py                  # Publication-quality plots (5 figures)
│   ├── analysis/
│   │   ├── sae_features.py              # Contrastive activation extraction + logit lens
│   │   └── neuronpedia_features.py      # Neuronpedia API feature exploration
│   ├── steering/
│   │   ├── apply_vectors.py             # Initial steering (layer 35 only)
│   │   ├── midlayer_sweep.py            # Mid-layer sweep (11 layers × 6 coefficients)
│   │   ├── base_model_steering.py       # Base model (non-instruct) steering
│   │   └── budget_guidance.py           # Budget guidance (Gamma predictor)
│   └── agents/
│       └── prompt_baselines.py          # 5 prompt strategies + 29-case eval dataset
├── results/
│   ├── token_counts_heatmap.png         # Tokenizer comparison heatmap
│   ├── compression_ratio_bar.png        # Compression ratios by model
│   ├── quality_scores_radar.png         # Radar chart of quality metrics
│   ├── tokenization_detail.png          # Visual token segmentation
│   ├── fragmentation_focus.png          # Date/JSON fragmentation analysis
│   ├── steering_vectors.pt             # 36 per-layer instruct steering vectors
│   ├── base_steering_vectors.pt        # 36 per-layer base model steering vectors
│   ├── layer_importance.json           # Layer ranking by discriminative power
│   ├── logit_lens_results.json         # Promoted/suppressed tokens per layer
│   ├── steering_results.json           # Initial layer-35 sweep results
│   ├── midlayer_steering_results.json  # Full 11-layer × 6-coefficient sweep
│   ├── base_model_steering_results.json # Base model sweep results
│   ├── budget_guidance_results.json    # Budget guidance results
│   └── neuronpedia_features.json       # SAE feature exploration results
├── data/
│   └── calendar_eval_dataset.json       # 29 bilingual test cases
├── PLAN.md                              # Research plan with status
└── requirements.txt                     # Python dependencies
```

## Methodology

### 1. Tokenizer Analysis

Comparative analysis of 4 tokenizers on 9 prompt categories:

| Model | Vocab Size | Overall Score | Best At |
|-------|-----------|--------------|---------|
| Qwen3-4B-Instruct-2507 | 151,643 | 0.483 | Multilingual parity |
| Gemma-3-1B-IT | 262,144 | 0.511 | Semantic coherence |
| **Llama-3.2-3B-Instruct** | **128,000** | **0.551** | **JSON + temporal** |
| Phi-3-mini-4k-instruct | 32,000 | 0.438 | — |

### 2. Contrastive Feature Extraction

SAE-inspired approach: 10 calendar vs 10 neutral prompts, hook-based residual stream extraction at all 36 layers, mean-difference steering vectors + logit lens projection.

```
Layer 35: L2=361.0  →  Promotes: schedule, agenda, attendees, RSVP
Layer 34: L2=218.6  →  Promotes: schedule, agenda, calendar, attendees
Layer 33: L2=179.4  →  Promotes: agenda, schedule, invite, attendees
```

### 3. Mid-Layer Steering Sweep

11 layers × 6 coefficients on Qwen3-4B-Instruct:

```
Layer 35 @ α=200:  17% change  ← RIGID
Layer 33 @ α=200:  17% change  ← RIGID
Layer 15 @ α=30:  100% change  ← SWEET SPOT ★
Layer 18 @ α=30:  100% change  ← SWEET SPOT ★
Layer 5  @ α=60:  100% change  ← DEGENERATE (empty outputs)
```

### 4. Base Model Steering

Qwen3-4B (non-instruct) confirms the pattern but with a narrower effective range:

```
Layer 15, α=30:  cal_score 0.70 → 0.97 (+0.27)  ★ Effective
Layer 15, α=60:  cal_score 0.70 → 0.33 (-0.37)  ✗ Degenerate
Layer 35, α=200: cal_score 0.70 → 0.77 (+0.07)  ~ Minimal effect
```

### 5. Budget Guidance

Gamma-distribution EOS bias for controlling generation length:

```
Result: 0% token savings at ALL budget levels (32-512).
The instruct model already produces minimal outputs (~75 tokens).
Budget guidance targets thinking overhead, not structured output verbosity.
```

### 6. Neuronpedia Feature Exploration

Transcoder-HP dictionary (163,840 features/layer):
- 115 calendar features found via keyword search across layers 20-35
- 9/450 random samples = 0.07% density (extreme sparsity)
- Features specialize in later layers ("scheduling appointments" vs "time references")

### 7. Prompt Engineering Baselines

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

# Phase 3a: Initial steering (layer 35 only)
python -m src.steering.apply_vectors

# Phase 3b: Mid-layer sweep (11 layers × 6 coefficients)
python -m src.steering.midlayer_sweep

# Phase 3c: Base model steering (downloads Qwen3-4B base ~8GB)
python -m src.steering.base_model_steering

# Phase 3d: Budget guidance
python -m src.steering.budget_guidance

# Phase 3e: Neuronpedia exploration (API calls, no GPU needed)
python -m src.analysis.neuronpedia_features

# Phase 4: Prompt baselines (creates eval dataset)
python -m src.agents.prompt_baselines
```

## Models

| Model | Params | Context | Role |
|-------|--------|---------|------|
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4.0B | 256K | Instruct: steering + generation |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4.0B | 256K | Base: steering comparison |
| [Gemma-3-1B-IT](https://huggingface.co/google/gemma-3-1b-it) | 1.0B | 32K | Tokenizer comparison |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3.2B | 128K | Tokenizer comparison |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 3.8B | 4K | Tokenizer comparison |

## References

- [Steering LLM Reasoning Through Bias-Only Adaptation](https://arxiv.org/abs/2505.18706) (EMNLP 2025)
- [Steering LLM Thinking with Budget Guidance](https://arxiv.org/abs/2506.13752)
- [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248)
- [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Neuronpedia — Qwen3-4B Circuit Tracer](https://www.neuronpedia.org/qwen3-4b/graph)
- [Gemma Scope 2 Tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r)

## Future Work

- [ ] Full inference evaluation with Ollama/llama.cpp + LangChain agent
- [ ] Test steering on smaller SLM (Qwen or LFM2) on GSM8K benchmark vs prompt engineering
- [ ] Train task-specific SAEs using Neuronpedia framework
- [ ] Sampling-based generation (temperature > 0) to detect subtler steering effects

## License

MIT
