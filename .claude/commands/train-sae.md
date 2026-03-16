---
description: Train a Sparse Autoencoder on a model's residual stream (20M tokens, wandb)
argument-hint: [model_name] e.g. Qwen/Qwen3-4B
---

Train an SAE for the model specified in $ARGUMENTS.

## Steps

1. Resolve model preset from MODEL_PRESETS in `src/steering/train_sae.py` to get the correct layer and d_in
2. Check if SAE already exists in `results/` for this model — warn user if overwriting
3. Check if contrastive vectors exist (`results/mmlu_pro_vectors_*.pt`) — if not, generate them first with `mmlu_pro_vectors.py --model <key>`
4. Run: `.venv/bin/python -m src.steering.train_sae --model $ARGUMENTS --training_tokens 20000000 --wandb`
5. Monitor output — report MSE, explained variance, and L0 when done
6. Update PLAN.md with training results

## Defaults
- Training tokens: 20M
- Expansion factor: 8x
- L1 coefficient: 0.005
- Always enable wandb logging

## Important
- Always use `.venv/bin/python`, never system python
- Training is long-running — use `run_in_background` and notify when complete
- If MPS OOMs, suggest reducing `--context_size` to 64
