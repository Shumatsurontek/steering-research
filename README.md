<p align="center">
  <img src="https://img.shields.io/badge/🧠_Mechanistic-Interpretability-blueviolet?style=for-the-badge" alt="Mechanistic Interpretability"/>
  <img src="https://img.shields.io/badge/🎯_Activation-Steering-ff6b6b?style=for-the-badge" alt="Activation Steering"/>
  <img src="https://img.shields.io/badge/🤖_Multi--Agent-Orchestration-00b894?style=for-the-badge" alt="Multi-Agent Orchestration"/>
  <img src="https://img.shields.io/badge/🔬_Research-Paper-fdcb6e?style=for-the-badge" alt="Research Paper"/>
</p>

<h1 align="center">
  🧬 Activation Steering for Small Language Models<br/>
  <sub>From Mid-Layer Sweet Spots to Dynamic Multi-Agent Orchestration</sub>
</h1>

<p align="center">
  <strong>Arthur Edmond</strong> · LLM Engineer @ <a href="https://swapn.com">Swapn</a><br/>
  <em>A deep dive into where, how, and why activation steering works (or doesn't) on SLMs — from mechanistic analysis to SWE-bench</em>
</p>

<p align="center">
  <a href="article/main.pdf"><img src="https://img.shields.io/badge/📄_Read_the_Paper-PDF-red?style=flat-square" alt="Paper PDF"/></a>
  <img src="https://img.shields.io/badge/Models-Qwen3_·_LFM2.5_·_Llama3-blue?style=flat-square" alt="Multi-Model"/>
  <img src="https://img.shields.io/badge/Benchmarks-GSM8K_·_SWE--bench_·_MMLU--Pro-green?style=flat-square" alt="Benchmarks"/>
  <img src="https://img.shields.io/badge/Experiments-19-orange?style=flat-square" alt="19 Experiments"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="MIT License"/>
</p>

---

## ⚡ TL;DR

> **Mid-layer steering (layers 15–18) at moderate coefficients (α=30) achieves 100% behavioral change on SLMs**, while late layers do nothing. Steering boosts zero-shot by +16pp on GSM8K but *hurts* when combined with few-shot or RAG context. On MMLU-Pro (n=200), steering consistently degrades accuracy — initial n=20 "improvements" were false positives. SAE feature decomposition reveals why: contrastive vectors activate 50–60% of features diffusely, with near-zero overlap with domain-specific features. A **Streamlit demo** lets you explore baseline vs steered outputs live.

---

## 🔥 Key Findings

<table>
<tr>
<td width="50%">

### 🎯 The Steering Sweet Spot
Layers 15–18 at α=30 → **100% change rate** on Qwen3-4B-Instruct. Non-target prompts get reinterpreted toward the steering direction. Late layers (33–35) → **0% change** even at α=200.

### 🧊 Late-Layer Rigidity is Architectural
Both instruct AND base models show the same rigidity pattern at layers 30–35. This is a **transformer property**, not an instruction-tuning artifact.

### 💥 Base Models: Powerful but Fragile
Layer 15 at α=30 boosts task score from 0.70→**0.97** on base model. But α≥60 → **total degeneration** (score 0.0). The effective window is razor-thin.

</td>
<td width="50%">

### 🧮 SLM Steering: GSM8K +16pp Zero-Shot
Zero-shot CoT + steering on Qwen3-0.6B-Instruct: **46%→62% (+16pp)**. But **5-shot + steering = interference** (-8pp). Adaptive `α = f(n_few_shot)` needed.

### 🔧 SWE-bench: RAG + Steering
RAG raises path validity from **0%→68%**. But steering *degrades* RAG patches (68%→15%→0%). **Steering is optimal in zero-shot regime only**.

### 🤖 Dynamic Multi-Agent Orchestrator
Sequential vector switching: **4–7× more domain-relevant output** vs baseline. Vector composition dilutes signal. Each domain has a distinct (layer, α) sweet spot.

### 📊 KL Divergence: 3-Order Gap
Mid-layers: **1–48 bits** of KL divergence. Late layers: **<0.01 bits**. Steering increases sampling diversity from 20%→**100%** while preserving output type.

### 🧬 SAE: Style ≠ Knowledge
Custom SAE (8192 features, 20M tokens) reveals contrastive vectors activate **50–60% of features diffusely**. Domain-specific features are sparse & localized. **Overlap: 0–4 / 20** → contrastive vectors ≠ domain knowledge. Finding **robust across sparsity regimes** (10× L1 penalty yields same pattern).

### 📉 MMLU-Pro n=200: False Positives
n=20 suggested +15pp on history. n=200 (stderr ±2.5pp) **reversed every positive finding**. Steering degrades all domains by -2 to -6pp.

### 🔀 LFM2-700M: Cross-Architecture Validation
Full SAE pipeline on hybrid conv+attention model (10 LIV conv + 6 GQA blocks). Contrastive vectors are **least diffuse** (26–38% vs 47–59% Qwen), but overlap remains tiny (1–2/20). **Style-not-knowledge is architecture-invariant.**

### 🎯 Output-Score Selection (n=200)
Selecting SAE features by **output influence** (W_dec projected through unembedding) instead of input activation. Features are entirely disjoint from input-diff features. At α=10, output-score causes less harm than all other methods. History shows +1.5pp (within stderr). **Improvement is methodological, not substantive.**

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
│   ├── main.tex                          # LaTeX source (~26 pages, arxiv-ready)
│   ├── main.pdf                          # Compiled paper
│   └── figures/                          # Publication-quality figures (PDF + PNG)
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
│   │   ├── sampling_steering.py         # Phase 3f: KL divergence + sampling diversity ★
│   │   ├── gsm8k_benchmark.py           # Phase 3g: lm-eval validation (5-shot + 0-shot) ★
│   │   ├── mmlu_pro_benchmark_mc.py     # MMLU-Pro loglikelihood benchmark (3 models)
│   │   ├── mmlu_pro_figures.py          # Publication figures for MMLU-Pro results
│   │   ├── mmlu_pro_samples.py          # Per-sample output capture (baseline vs steered)
│   │   ├── mmlu_pro_sample_figures.py   # Log-likelihood & probability distribution figures
│   │   ├── train_sae.py                 # SAE training (multi-model: --model Qwen/Qwen3-4B)
│   │   ├── analyze_sae_features.py      # SAE domain analysis + contrastive overlap (multi-model)
│   │   ├── feature_targeted_steering.py # Feature-targeted vectors from SAE decoder columns ★
│   │   ├── app_steering_demo.py         # 🎮 Streamlit demo: live comparison of steering methods
│   │   ├── domain_vectors.py            # Phase 6a: Domain-specific vector extraction ★
│   │   ├── vector_composition.py        # Phase 6b: Composition tests (add vs switch) ★
│   │   └── swebench_domain_vectors.py   # Phase 6c: SWE-bench cluster vectors ★
│   │
│   └── agents/
│       ├── prompt_baselines.py          # 5 strategies × 29 bilingual eval cases
│       ├── steering_orchestrator.py     # Phase 6d: Dynamic steering orchestrator ★
│       ├── swebench_pipeline.py         # Phase 6e: SWE-bench eval pipeline + RAG ★
│       └── swebench_rag.py             # Phase 6e: Repo checkout + file retrieval ★
│
├── 🌐 web/
│   ├── api/                              # FastAPI backend (SSE streaming, model management)
│   │   ├── main.py                       # App, CORS, lifespan, request logging
│   │   ├── deps.py                       # ModelManager singleton (1 model in RAM)
│   │   ├── steering.py                   # Extracted hook + generation logic
│   │   └── routers/                      # generate, models, vectors endpoints
│   └── frontend/                         # React + TypeScript + Vite
│       └── src/components/               # Sidebar, ArenaPanel, ChatCard, VectorViz
│
├── 📊 results/                           # All JSON results + steering vectors (.pt) + SAE weights
├── 📁 data/                              # Evaluation datasets
├── 📋 PLAN.md                            # Research plan with status tracking
├── 🐳 Dockerfile                         # Multi-stage build (Node + Python)
├── 📋 justfile                           # Task runner (just up, just dev, etc.)
└── 📦 requirements.txt
```

---

## 🌐 Steering Arena (Web App)

Interactive React + FastAPI app for real-time comparison of steering methods. Streams baseline, contrastive, and SAE feature-targeted generation side by side.

**Features:**
- Real-time SSE streaming with 3 methods running sequentially
- Dynamic layer selection with cross-layer SAE support (warning when not at trained layer)
- Vector space visualizations (PCA, cosine similarity heatmap, L2 norms)
- Model hot-swapping (Qwen3-0.6B / Qwen3-4B)

**Quick start with Docker:**

```bash
just up    # builds image + runs on :8000 (mount results/ for vectors)
```

**Dev mode (hot reload):**

```bash
just dev-backend   # uvicorn on :8000
just dev-frontend  # vite on :5173 (proxies /api → :8000)
```

*Developed by Arthur EDMOND*

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

**lm-eval Validation (n=50) — 5-shot vs. Zero-shot CoT:**

| Model | Condition | 5-shot Strict | 5-shot Flex | 0-shot Strict | 0-shot Flex |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Instruct** | Baseline | 48% | 48% | 38% | 46% |
| **Instruct** | Steered | 44% (-4) | 40% (-8) | 40% (+2) | **62% (+16)** ★ |
| **Base** | Baseline | 48% | 48% | 36% | 28% |
| **Base** | Steered | 26% (-22) | 34% (-14) | 8% (-28) | 22% (-6) |

> **Zero-shot + steering = synergy** (+16pp on instruct), **5-shot + steering = interference** (-8pp). Steering improves *reasoning* without improving *format compliance*. This motivates adaptive coefficient selection: `α = f(n_few_shot, model_type)` for dynamic multi-agent steering.

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

### 🔟 Dynamic Steering Orchestrator

**Vector Composition (can we add vectors?):**

| Composition | Strategy | Coherent | Avg Score |
|:---:|:---:|:---:|:---:|
| code + bug | baseline | 3/3 | **4.0** |
| code + bug | addition | 3/3 | 2.0 (diluted) |
| code + bug + patch | addition (3 vec) | 3/3 | 5.7 (= baseline) |

> **Addition dilutes signal** — no degeneration but no gain. Sequential switching is the right architecture.

**Orchestrator — Dynamic vs Static vs Baseline:**

| Scenario | Variant | KW Hits | Tokens | Coherence |
|:---:|:---:|:---:|:---:|:---:|
| bug fix | **dynamic** | **4** (vs 1) | 768 | 100% |
| test failure | **dynamic** | **7** (vs 1) | 818 (-20%) | 100% |
| feature regression | **dynamic** | **8** (vs 9) | 1280 | 100% |

> **Dynamic switching = 4–7× more domain-relevant output** vs baseline. Static steering *hurts* on heterogeneous tasks (4 vs 9 hits). Zero degeneration across all conditions.

**SWE-bench Cluster Vectors:**

| Cluster | Best Config | Score | Cosine to Generic |
|:---:|:---:|:---:|:---:|
| django_web (46%) | L18@α=10 | **13** | 0.865 (bug_analysis) |
| scientific (37%) | L18@α=60 | 6 | 0.844 |
| dev_tooling (15%) | L25@α=10 | 8 | 0.849 |

> Each cluster has a **distinct sweet spot**. Cluster-specific vectors capture additional signal beyond generic domains (cosine 0.84–0.87, not 1.0).

### 1️⃣1️⃣ SWE-bench Verified — RAG + Steering (Qwen3-0.6B)

**Without RAG (n=20): all patches fail** — 0% resolved, model invents file paths.

**With RAG (n=20):**

| Variant | Valid Diffs | Path Validity | Avg Quality |
|:---:|:---:|:---:|:---:|
| **rag_baseline** | **95%** | **68%** | **0.655** |
| rag_static (code_reading) | 100% | 15% | 0.276 |
| rag_dynamic (3-step) | 90% | 0% | 0.205 |

> **Steering degrades RAG performance** — same pattern as GSM8K: RAG context acts like implicit few-shot, and steering on top causes destructive interference. The unsteered rag_baseline is the best variant. **Steering is optimal in zero-shot regime only.**

### 1️⃣2️⃣ MMLU-Pro Multi-Model Benchmark (n=200 validation)

**Dual-mode evaluation** on 3 models × 3 domains (math, law, history):

| Domain | Baseline | α=10 | α=30 | α=60 |
|:---:|:---:|:---:|:---:|:---:|
| math | 25.5% (±3.1) | 27.0% (+1.5) | 24.0% (-1.5) | 20.0% (-5.5) |
| law | 17.0% (±2.7) | 14.5% (-2.5) | 11.0% (-6.0) | 8.5% (-8.5) |
| history | 19.5% (±2.8) | 15.5% (-4.0) | 16.5% (-3.0) | 14.0% (-5.5) |

> **n=20 produced false positives** (history +15pp with stderr ±9pp). **n=200 reversed every positive finding** (stderr ±2.5pp). Steering consistently degrades all domains. Generate-until mode is catastrophic (0% at α=60).

### 1️⃣3️⃣ Cross-Architecture Geometry

| Model Pair | Spearman ρ | Pearson r |
|:---:|:---:|:---:|
| Qwen3-0.6B vs Llama-3.2-3B | 0.893 | 0.920 |
| Qwen3-0.6B vs LFM2.5-1.2B | 0.936 | 0.957 |
| Llama-3.2-3B vs LFM2.5-1.2B | 0.888 | 0.909 |

> **Domain geometry is architecture-invariant** (ρ > 0.88). History = universal outlier. STEM clusters tightly. Topology preserved across Transformer, larger Transformer, and hybrid SSM+Attention.

### 1️⃣4️⃣ SAE Feature Decomposition — Why Contrastive Vectors Fail

Custom SAE: 8192 features (8× expansion), 20M tokens OpenWebText, layer 14.

| Domain | Contrastive activates | Overlap (top-20 ∩ top-20) |
|:---:|:---:|:---:|
| Math | 4,664 / 8,192 (57%) | **2 features** |
| Law | 4,845 / 8,192 (59%) | **0 features** |
| History | 3,857 / 8,192 (47%) | **4 features** |

> Contrastive vectors = **diffuse directions** activating majority of SAE dictionary. Domain-specific features are **sparse & localized**. Near-zero overlap explains "domain style vs domain knowledge" distinction.

### 1️⃣5️⃣ Feature-Targeted Steering vs Contrastive (n=50)

Vectors built from SAE decoder columns instead of contrastive means:

| Domain | Method | α=3 | α=5 | α=10 |
|:---:|:---:|:---:|:---:|:---:|
| math | Baseline | 18.0% | — | — |
| math | Contrastive | 18.0% | 18.0% | 20.0% |
| math | **Feature uniform k20** | 18.0% | 18.0% | **22.0%** |
| law | Baseline | 24.0% | — | — |
| law | Contrastive | 22.0% | 22.0% | 18.0% |
| law | **Feature uniform k20** | **24.0%** | **24.0%** | 16.0% |

> Feature-targeted vectors **preserve baseline accuracy** better at low α (law: 24% maintained vs 22% contrastive). Marginal improvements within noise — the 0.6B model simply lacks deep domain knowledge to unlock. Feature-targeted steering degrades less aggressively than contrastive at moderate α.

### 1️⃣6️⃣ SAE Sparsity Robustness (L1=0.05)

Retrained with 10× higher L1 penalty to verify findings aren't artifacts:

| Metric | L1=0.005 | L1=0.05 |
|:---:|:---:|:---:|
| MSE | 26.9 | 71.4 |
| L0 (active features) | 7,131 (87%) | 5,793 (71%) |
| Contrastive activations | 47–59% | 48–56% |
| Overlap (max) | 4/20 | 3/20 |

> **Finding is robust**: 10× L1 reduces L0 by only 19% — still far from truly sparse (ideal: L0 < 200). Contrastive vectors remain diffuse across sparsity regimes. Achieving interpretable sparsity would require L1 ≥ 1.0 or a larger SAE (32K+ features).

### 1️⃣7️⃣ Qwen3-4B SAE Comparison — Does Scale Help?

SAE (20,480 features, 8×2560) trained on layer 18 of Qwen3-4B. Feature-targeted benchmark n=50.

**SAE Analysis — 0.6B vs 4B:**

| Metric | Qwen3-0.6B | Qwen3-4B |
|:---:|:---:|:---:|
| Contrastive diffusion | 47–59% | **41–51%** |
| Overlap law | 0/20 | **5/20** ★ |
| Overlap math | 2/20 | 1/20 |
| Overlap history | 4/20 | 0/20 |

**Feature-Targeted Benchmark — Qwen3-4B (n=50):**

| Domain | Baseline | Contrastive best | Single-feature best |
|:---:|:---:|:---:|:---:|
| Math | **48.0%** | 44.0% (α=10) | 44.0% (α=10) |
| Law | 22.0% | 24.0% (α=10) | **24.0%** (α=30, +8pp vs contrastive) |
| History | **34.0%** | 34.0% (α=10-30) | 32.0% (α=10) |

> **4B baselines are 2-3× higher** (math 48% vs 18%, history 34% vs 14%) but **no steering method improves over baseline**. Contrastive vectors are less diffuse on 4B (41-51% vs 47-59%) and law shows 5/20 overlap, yet this doesn't translate to accuracy gains. **Single-feature steering preserves baseline best** — at α=30 on law: 24% (single) vs 16% (contrastive). Scaling doesn't solve the fundamental limitation: steering ≠ knowledge injection.

### 🎮 Streamlit Demo

```bash
streamlit run src/steering/app_steering_demo.py
```

Interactive comparison of baseline, contrastive, and feature-targeted steering with adjustable domain, coefficient, and strategy. Includes word-level diff highlighting and batch mode for all domain prompts.

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

| Model | Params | Architecture | Role |
|-------|:---:|:---:|------|
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4.0B | Transformer (36L) | Primary: steering sweet spot discovery |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4.0B | Transformer (36L) | Base model fragility + SAE comparison |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | Transformer (28L) | SLM: GSM8K, SWE-bench, MMLU-Pro |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 0.6B | Transformer (28L) | SLM base: GSM8K steering |
| [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 1.2B | **Hybrid SSM+Attn (16L)** | MMLU-Pro: cross-architecture steering |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3.2B | Transformer (32L) | Tokenizer + MMLU-Pro scaling test |
| [Gemma-3-1B-IT](https://huggingface.co/google/gemma-3-1b-it) | 1.0B | Transformer | Tokenizer comparison |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 3.8B | Transformer | Tokenizer comparison |

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
| [SAELens](https://github.com/jbloomAus/SAELens) (Bloom et al., 2024) | SAE training & analysis library |
| [MMLU-Pro](https://arxiv.org/abs/2406.01574) (Wang et al., 2024) | 10-way MC benchmark (12K questions, 14 domains) |

---

## 🔮 Future Work

- [x] SLM steering on GSM8K (Qwen3-0.6B) — **+16pp zero-shot**
- [x] Sampling-based analysis (T>0, KL divergence) — **3-order gap confirmed**
- [x] Dynamic multi-agent orchestrator — **4–7× domain relevance**
- [x] SWE-bench Verified + RAG — **0%→68% path validity**
- [x] **MMLU-Pro multi-model benchmark** — n=200 validation revealed false positives, steering degrades all domains
- [x] **Cross-architecture geometry** — domain similarity invariant across Transformer/SSM (ρ > 0.88)
- [x] **SAE feature decomposition** — near-zero overlap between contrastive vectors and domain features
- [x] **Feature-targeted steering** — SAE decoder column vectors preserve baseline better than contrastive
- [x] **Streamlit demo** — live comparison of steering methods with word-level diff
- [x] Higher sparsity SAE (L1=0.05) — finding robust: contrastive still diffuse, overlap 0–3/20
- [x] **Qwen3-4B SAE comparison** — contrastive less diffuse (41-51%), law overlap 5/20, but still no accuracy gains
- [ ] Cross-model steering via learned linear projections (leveraging geometric invariance)

---

<p align="center">
  <strong>Arthur Edmond</strong> · <a href="https://swapn.com">Swapn</a><br/>
  <sub>Built with PyTorch, HuggingFace Transformers, and an unhealthy obsession with residual streams</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="MIT"/>
</p>
