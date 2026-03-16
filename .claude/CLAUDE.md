# Steering Research — Claude Code Context

## Project Overview
Research comparing **activation steering vectors** vs **prompt engineering** on SLMs.
Goal: understand mechanistically when/why steering helps or hurts, across models and benchmarks.

## Architecture

```
steering-research/
├── article/main.tex          # LaTeX paper (~26 pages, arxiv-ready)
├── src/
│   ├── tokenizers/           # Tokenizer comparison (Phase 1)
│   ├── analysis/             # Feature extraction: contrastive activations, Neuronpedia (Phase 2)
│   ├── steering/             # All steering experiments (Phases 3-9)
│   │   ├── train_sae.py              # SAE training (multi-model: --model, --layer, --d_in)
│   │   ├── analyze_sae_features.py   # SAE domain analysis + contrastive overlap (multi-model)
│   │   ├── feature_targeted_steering.py  # Feature-targeted vectors from SAE decoder columns
│   │   ├── mmlu_pro_vectors.py       # Contrastive vectors for MMLU-Pro domains (--model filter)
│   │   ├── mmlu_pro_benchmark_mc.py  # MMLU-Pro loglikelihood evaluation
│   │   ├── app_steering_demo.py      # Streamlit demo app
│   │   └── tasks/mmlu_pro_mc/        # Custom lm-eval tasks
│   └── agents/               # Dynamic steering orchestrator, SWE-bench pipeline
├── results/                  # JSON results, .pt vectors, SAE weights (gitignored)
├── PLAN.md                   # Research plan with phase tracking
└── README.md                 # Project README with all experiment results
```

## Models & Presets

| Model | Layers | Hidden | SAE Layer | SAE Dir |
|-------|--------|--------|-----------|---------|
| Qwen/Qwen3-0.6B | 28 | 1024 | 14 | sae_qwen3_0.6b_L14_8x |
| Qwen/Qwen3-4B | 36 | 2560 | 18 | sae_qwen3_4b_L18_8x |

Scripts use `MODEL_PRESETS` dicts for automatic config resolution.

## Key Conventions

- **Python venv**: always use `.venv/bin/python` (not system python) for running scripts
- **Script invocation**: `python -m src.steering.<script>` from project root
- **Results**: saved to `results/` directory (gitignored — too large for git)
- **SAE pipeline order**: train_sae → analyze_sae_features → feature_targeted_steering
- **Contrastive vectors must exist before SAE analysis** (mmlu_pro_vectors.py generates them)
- **Device**: MPS on macOS, CUDA on Linux, CPU fallback — all scripts auto-detect
- **Article**: single file `article/main.tex`, compile with pdflatex
- **Language**: code/comments in English, PLAN.md in French, README in English

## Important Patterns

- All SAE scripts accept `--model` to select which model to analyze
- `stop_at_layer` is derived dynamically from hook_name (no hardcoding)
- Steering hook: `hidden + coeff * (vector / ||vector||)` applied via `register_forward_hook`
- Feature-targeted vectors: `v = Σ W_dec[feature_i] * weight_i` for top-k differential features
- MMLU-Pro uses custom lm-eval tasks in `src/steering/tasks/mmlu_pro_mc/`

## What NOT to Do

- Don't commit .pt files, .safetensors, or results/ (gitignored)
- Don't hardcode layer numbers — use MODEL_PRESETS
- Don't run scripts with system python — always `.venv/bin/python`
- Don't amend commits — user manages git themselves
- Don't create documentation files unless asked
