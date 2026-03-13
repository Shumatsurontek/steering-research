<p align="center">
  <img src="https://img.shields.io/badge/🧠_Mechanistic-Interpretability-blueviolet?style=for-the-badge" alt="Mechanistic Interpretability"/>
  <img src="https://img.shields.io/badge/🎯_Activation-Steering-ff6b6b?style=for-the-badge" alt="Activation Steering"/>
  <img src="https://img.shields.io/badge/📅_Agentic-Calendar_Tasks-00b894?style=for-the-badge" alt="Agentic Calendar"/>
  <img src="https://img.shields.io/badge/🔬_Research-Paper-fdcb6e?style=for-the-badge" alt="Research Paper"/>
</p>

<h1 align="center">
  🧬 Steering Vectors vs. Prompt Engineering<br/>
  <sub>for Agentic Calendar Tasks</sub>
</h1>

<p align="center">
  <strong>Arthur Edmond</strong> · LLM Engineer @ <a href="https://swapn.com">Swapn</a><br/>
  <em>A deep dive into where, how, and why activation steering works (or doesn't) on instruction-tuned LLMs</em>
</p>

<p align="center">
  <a href="article/main.pdf"><img src="https://img.shields.io/badge/📄_Read_the_Paper-PDF-red?style=flat-square" alt="Paper PDF"/></a>
  <img src="https://img.shields.io/badge/Model-Qwen3--4B-blue?style=flat-square" alt="Qwen3-4B"/>
  <img src="https://img.shields.io/badge/Layers-36-green?style=flat-square" alt="36 Layers"/>
  <img src="https://img.shields.io/badge/Experiments-9-orange?style=flat-square" alt="7 Experiments"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="MIT License"/>
</p>

---

## ⚡ TL;DR

> **Mid-layer steering (layers 15–18) at moderate coefficients (α=30) achieves 100% behavioral change on instruction-tuned models**, while the "obvious" choice — steering at the most discriminative final layers — does absolutely nothing. This paper maps the full landscape of when steering works, when it breaks, and when you should just write a better prompt.

---

## 🔥 Key Findings

<table>
<tr>
<td width="50%">

### 🎯 The Steering Sweet Spot
Layers 15–18 at α=30 → **100% change rate** on Qwen3-4B-Instruct. Non-calendar prompts get reinterpreted as calendar tasks. Late layers (33–35) → **0% change** even at α=200.

### 🧊 Late-Layer Rigidity is Architectural
Both instruct AND base models show the same rigidity pattern at layers 30–35. This is a **transformer property**, not an instruction-tuning artifact.

### 💥 Base Models: Powerful but Fragile
Layer 15 at α=30 boosts calendar score from 0.70→**0.97** on base model. But α≥60 → **total degeneration** (score 0.0). The effective window is razor-thin.

</td>
<td width="50%">

### 📊 Budget Guidance: A Null Result
Instruct model already produces **~75 tokens** of compact JSON. Budget guidance (32–512 token limits) achieves **0% savings**. It targets thinking overhead, not output verbosity.

### 🔬 SAE Features: Extreme Sparsity
Neuronpedia transcoder analysis: only **9/450 random samples** (0.07%) are calendar-related. Yet 115 features found via keyword search across layers 20–35.

### 🧮 SLM Steering: GSM8K Math Reasoning
Steering on Qwen3-0.6B **doubles** base model GSM8K accuracy (20%→**40%**). Instruct model gains only +10%. Sweet spot shifts to 64–89% depth on the 28-layer model.

### 📊 KL Divergence: 3-Order Gap
Mid-layers: **1–48 bits** of KL divergence. Late layers: **<0.01 bits**. Steering increases sampling diversity from 20%→**100%** while preserving JSON output type.

</td>
</tr>
</table>

---

## 📐 The Three-Zone Steerability Hierarchy

```
                    ┌─────────────────────────────────────────────────┐
                    │         STEERABILITY vs. LAYER DEPTH            │
                    │                                                 │
  Change Rate (%)   │   ██                                            │
       100 ─────────│   ██ ████                                       │
                    │   ██ ████ ██                                    │
        80 ─────────│   ██ ████ ████                                  │
                    │   ██ ████ ████ ██                               │
        60 ─────────│   ██ ████ ████ ████                             │
                    │   ██ ████ ████ ████ ██                          │
        40 ─────────│   ██ ████ ████ ████ ████                        │
                    │   ██ ████ ████ ████ ████ ██                     │
        20 ─────────│   ██ ████ ████ ████ ████ ████ ██ ██            │
                    │   ██ ████ ████ ████ ████ ████ ████ ████ ██ ██  │
         0 ─────────│───██─████─████─████─████─████─████─████─██─██──│
                    │   5   10   15   18   20   22   25   30  33  35 │
                    │  ⚠️UNSTABLE │ ★ SWEET SPOT │  ❄️ FROZEN       │
                    └─────────────────────────────────────────────────┘
                              Layer Index (α = 30)
```

| Zone | Layers | Behavior | Why |
|------|--------|----------|-----|
| ⚠️ **Unstable** | 1–10 | High change but degenerate outputs | Representations too raw — syntactic, not semantic |
| ★ **Sweet Spot** | 15–18 | **100% change, coherent outputs** | Syntactic→semantic transition point — malleable yet structured |
| ❄️ **Frozen** | 30–35 | 0–17% change even at α=200 | Already committed to output distribution — architectural rigidity |

---

## 🏗️ Architecture

```
steering-research/
│
├── 📄 article/
│   ├── main.tex                          # LaTeX source (~10 pages, arxiv-ready)
│   └── main.pdf                          # Compiled paper
│
├── 🔬 src/
│   ├── tokenizers/
│   │   ├── compare.py                    # 4-model tokenizer comparison + quality scoring
│   │   └── visualize.py                  # 5 publication-quality matplotlib figures
│   │
│   ├── analysis/
│   │   ├── sae_features.py              # Contrastive activation extraction + logit lens
│   │   └── neuronpedia_features.py      # Neuronpedia API: 163,840 SAE features/layer
│   │
│   ├── steering/
│   │   ├── apply_vectors.py             # Phase 3a: Initial layer-35 steering (null result)
│   │   ├── midlayer_sweep.py            # Phase 3b: 11 layers × 6 coefficients → sweet spot
│   │   ├── base_model_steering.py       # Phase 3c: Base model — steerable but fragile
│   │   ├── budget_guidance.py           # Phase 3d: Gamma predictor (null result)
│   │   ├── slm_gsm8k_steering.py       # Phase 3e: SLM steering on GSM8K (Qwen3-0.6B) ★
│   │   └── sampling_steering.py         # Phase 3f: KL divergence + sampling diversity ★
│   │
│   └── agents/
│       └── prompt_baselines.py          # 5 strategies × 29 bilingual eval cases
│
├── 📊 results/                           # All JSON results + steering vectors (.pt)
├── 📁 data/                              # Evaluation datasets
├── 📋 PLAN.md                            # Research plan with status tracking
└── 📦 requirements.txt
```

---

## 🧪 Experiments

### 1️⃣ Tokenizer Analysis — 4 Models, 9 Prompt Categories

| Model | Vocab | Overall Score | Temporal | JSON | Multilingual |
|-------|-------|:---:|:---:|:---:|:---:|
| **Llama-3.2-3B** | 128K | **0.551** ★ | 0.316 | 0.375 | 0.816 |
| Gemma-3-1B | 262K | 0.511 | 0.226 | 0.288 | 0.845 |
| Qwen3-4B | 152K | 0.483 | 0.205 | 0.288 | 0.802 |
| Phi-3-mini | 32K | 0.438 | 0.173 | 0.249 | 0.814 |

> **All tokenizers score < 0.32 on temporal integrity** — ISO 8601 timestamps are universally fragmented into individual characters.

### 2️⃣ Contrastive Feature Extraction + Logit Lens

10 calendar vs 10 neutral prompts → per-layer mean-difference vectors → logit lens projection:

```python
Layer 35: L2=361.0  →  ✅ schedule, agenda, attendees, RSVP    ❌ magnitude, density
Layer 34: L2=218.6  →  ✅ schedule, agenda, calendar            ❌ licking, entropy
Layer 33: L2=179.4  →  ✅ agenda, schedule, invite              ❌ entropy, Measured
```

### 3️⃣ Mid-Layer Steering Sweep

**The experiment that changed everything.** 11 layers × 6 coefficients:

| Layer | α=0 | α=10 | α=30 | α=60 | α=100 | α=200 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **15** | 0% | 17% | **100%** ★ | **100%** | 83% | 100% |
| **18** | 0% | 33% | **100%** ★ | **100%** | **100%** | 67% |
| 20 | 0% | 17% | 33% | 100% | 100% | 67% |
| 33 | 0% | 0% | 17% | 17% | 17% | 17% |
| 35 | 0% | 0% | 0% | 0% | 0% | 17% |

### 4️⃣ Base Model Steering — Calendar Score (0→1)

| Layer | α=0 | α=10 | α=30 | α=60 | α=100 | α=200 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **15** | 0.70 | 0.70 | **0.97** ★ | 0.33 💀 | 0.03 💀 | 0.00 💀 |
| 18 | 0.70 | 0.73 | 0.73 | 0.13 💀 | 0.00 💀 | 0.00 💀 |
| 35 | 0.70 | 0.70 | 0.70 | 0.70 | 0.73 | 0.77 |

> Sweet spot → degeneration in **one coefficient step**. Instruction tuning acts as a stabilizer.

### 5️⃣ Budget Guidance — Token-Level Control

| Budget | Avg Tokens | Valid | Savings |
|:---:|:---:|:---:|:---:|
| ∞ (baseline) | 74.6 | 100% | — |
| 32 | 74.6 | 100% | **0%** |
| 512 | 74.6 | 100% | **0%** |

> The instruct model is already Pareto-optimal for structured output. Budget guidance targets *thinking overhead*, not output verbosity.

### 6️⃣ Neuronpedia SAE Feature Mapping

- **163,840** transcoder features per layer
- **115** calendar features via keyword search
- **9/450** random samples (0.07% density)
- Features specialize in later layers: generic "time" → specific "scheduling appointments"

### 7️⃣ Prompt Engineering Baselines

| Strategy | Messages | Description |
|----------|:---:|-------------|
| Zero-shot | 2 | System prompt only |
| Few-shot (3) | 8 | 3 example input/output pairs |
| Few-shot (5) | 12 | 5 example pairs |
| Chain-of-thought | 2 | Step-by-step reasoning |
| Tool use | 2 + tool | Function calling schema |

> 29 bilingual test cases (18 FR / 11 EN) × 4 complexity levels. Framework ready — full inference evaluation pending.

### 8️⃣ SLM Steering on GSM8K — Qwen3-0.6B (28 layers)

| Model | Strategy | Baseline | Best Steering | Layer | α |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Instruct** | zero_shot | 20% | **30%** (+10%) | 25 | 60 |
| **Instruct** | cot | 10% | 10% (no gain) | — | — |
| **Instruct** | few_shot | 20% | 20% (no gain) | — | — |
| **Base** | zero_shot | 0% | 10% | 18 | 30 |
| **Base** | cot | 0% | 0% | — | — |
| **Base** | few_shot | 20% | **40%** (+20%) ★ | 20 | 100 |

> Sweet spot shifts to 64–89% depth (layers 18–25) on the 28-layer model. Base model doubles accuracy with steering. CoT *hurts* the 0.6B model (10% < 20% zero-shot).

### 9️⃣ Sampling-Based Steering — KL Divergence + Diversity

**KL Divergence (bits) — Mid-layers vs. Late layers:**

| Prompt | L15@α=30 | L18@α=30 | L35@α=30 | L35@α=60 |
|:---:|:---:|:---:|:---:|:---:|
| calendar_fr | 4.20 | 0.61 | 0.00 | 0.00 |
| calendar_en | 1.51 | 2.10 | 0.00 | 0.00 |
| ambiguous_en | 11.0 | 1.50 | 0.00 | 0.00 |
| non_cal_fr | **21.2** | **23.5** | 0.00 | 0.01 |
| non_cal_en | **20.5** | **17.8** | 0.00 | 0.00 |

**Sampling Diversity (L15@α=30 vs. baseline):**

| Prompt | Baseline T=0.3 | Steered T=0.3 | Baseline T=1.0 | Steered T=1.0 |
|:---:|:---:|:---:|:---:|:---:|
| calendar_fr | 20% | **100%** | 60% | **100%** |
| calendar_en | 20% | **80%** | 40% | **80%** |
| ambiguous_fr | 20% | **80%** | 60% | **100%** |

> 3-order-of-magnitude KL gap between mid-layers and late layers. Steering increases diversity 2–5× while preserving JSON output type. Late-layer rigidity is **distributional**, not just behavioral.

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Shumatsurontek/steering-research.git
cd steering-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline
python -m src.tokenizers.compare          # Phase 1: Tokenizer analysis
python -m src.tokenizers.visualize        # Phase 1: Generate plots
python -m src.analysis.sae_features       # Phase 2: Contrastive extraction
python -m src.steering.apply_vectors      # Phase 3a: Layer-35 (null result)
python -m src.steering.midlayer_sweep     # Phase 3b: Sweet spot discovery ★
python -m src.steering.base_model_steering # Phase 3c: Base model fragility
python -m src.steering.budget_guidance    # Phase 3d: Budget guidance (null)
python -m src.steering.slm_gsm8k_steering  # Phase 3e: SLM GSM8K steering ★
python -m src.steering.sampling_steering   # Phase 3f: KL divergence + diversity ★
python -m src.analysis.neuronpedia_features # Phase 3g: SAE features (API)
python -m src.agents.prompt_baselines     # Phase 4: Eval dataset
```

---

## 🤖 Models

| Model | Params | Role | Key Insight |
|-------|:---:|------|-------------|
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4.0B | Primary: steering + generation | Mid-layer sweet spot at L15–18 |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4.0B | Base model comparison | More steerable, more fragile |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | SLM instruct: GSM8K steering | Sweet spot at 64–89% depth |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 0.6B | SLM base: GSM8K steering | Doubles accuracy with steering |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3.2B | Tokenizer comparison | Best tokenizer for agentic tasks |
| [Gemma-3-1B-IT](https://huggingface.co/google/gemma-3-1b-it) | 1.0B | Tokenizer comparison | Best semantic coherence |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 3.8B | Tokenizer comparison | Smallest vocab (32K) |

---

## 📚 References

| Paper | Key Idea |
|-------|----------|
| [Activation Addition](https://arxiv.org/abs/2308.10248) (Turner et al., 2023) | Contrastive mean-difference vectors for steering |
| [Bias-Only Adaptation](https://arxiv.org/abs/2505.18706) (Gao et al., EMNLP 2025) | Per-layer RL steering matching full fine-tuning |
| [Budget Guidance](https://arxiv.org/abs/2506.13752) (Li et al., 2025) | Gamma-distribution predictor for reasoning length |
| [Representation Engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023) | Top-down approach to AI transparency |
| [Neuronpedia](https://www.neuronpedia.org/qwen3-4b/graph) | Circuit-level attribution for Qwen3-4B |
| [Gemma Scope 2](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r) | SAEs, Transcoders, Crosscoders tutorial |

---

## 🔮 Future Work

- [ ] Full agentic evaluation with Ollama/llama.cpp + LangChain
- [x] SLM steering on GSM8K (Qwen3-0.6B) — **done**
- [x] Sampling-based analysis (T>0, KL divergence) — **done**
- [ ] Train task-specific SAEs via Neuronpedia
- [ ] Multi-task steering (calendar + code + reasoning simultaneously)

---

<p align="center">
  <strong>Arthur Edmond</strong> · <a href="https://swapn.com">Swapn</a><br/>
  <sub>Built with PyTorch, HuggingFace Transformers, and an unhealthy obsession with residual streams</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="MIT"/>
</p>
