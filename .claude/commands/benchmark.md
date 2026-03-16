---
description: Run steering benchmark — feature-targeted vs contrastive on MMLU-Pro or GSM8K
argument-hint: [model] [--task mmlu|gsm8k] [--limit N]
---

Run a steering benchmark for the model specified in $ARGUMENTS.

## Argument parsing
- First arg: model name (e.g. `Qwen/Qwen3-4B`)
- `--task mmlu` (default): MMLU-Pro MC feature-targeted vs contrastive
- `--task gsm8k`: GSM8K zero-shot + steering validation
- `--limit N`: number of samples (default: 50)

## MMLU-Pro Pipeline
1. Verify SAE weights + contrastive vectors exist for the model
2. Run: `.venv/bin/python -m src.steering.feature_targeted_steering --model <model> --limit <N>`
3. Report summary table: baseline vs contrastive vs feature-targeted (weighted, uniform, single) at each alpha
4. Compare with existing results from other models

## GSM8K Pipeline
1. Run: `.venv/bin/python -m src.steering.gsm8k_benchmark --model <model> --limit <N>`
2. Report: 5-shot vs 0-shot, strict vs flex accuracy, steered vs baseline
3. Key metric: zero-shot flex accuracy delta (the synergy indicator)

## After benchmark
- Save results to `results/`
- Print comparison table if results exist for multiple models
- Suggest updating docs with `/update-docs`

## Important
- Always use `.venv/bin/python`
- Benchmarks are long-running — use `run_in_background` for large N
- If SAE missing, suggest `/train-sae` first
