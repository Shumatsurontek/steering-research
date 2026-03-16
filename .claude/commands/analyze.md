---
description: Run SAE domain analysis + contrastive overlap for a model
argument-hint: [model_name] e.g. Qwen/Qwen3-4B
---

Run the full SAE analysis pipeline for $ARGUMENTS.

## Steps

1. Verify SAE weights exist in `results/` for this model (check MODEL_PRESETS in analyze_sae_features.py)
2. Verify contrastive vectors exist (`results/mmlu_pro_vectors_*.pt`)
3. Run: `.venv/bin/python -m src.steering.analyze_sae_features --model $ARGUMENTS`
4. Report key metrics:
   - Domain-specific features (top-5 per domain with differential activation)
   - Contrastive vector activation count (X / total features)
   - Overlap analysis (top-20 intersection)
5. Compare with existing results from other models if available
6. Update PLAN.md with analysis results

## Important
- Always use `.venv/bin/python`
- If SAE doesn't exist, suggest running `/train-sae` first
- Results saved to `results/sae_domain_analysis_<model>.json`
