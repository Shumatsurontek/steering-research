"""
MMLU-Pro Benchmark — Domain-Specific Steering on 3 Models × 3 Domains.

Tests whether domain-specific steering vectors improve zero-shot performance
on the 3 most dissimilar MMLU-Pro domains: math, law, history.

Models: Qwen3-0.6B, Llama-3.2-3B-Instruct, LFM2.5-1.2B-Instruct
Methodology: subclass HFLM, inject normalized steering vector via forward hook.

Usage:
    python -m src.steering.mmlu_pro_benchmark --limit 50
    python -m src.steering.mmlu_pro_benchmark --limit 50 --model qwen3_0.6b
"""

import argparse
import gc
import json
import functools
from pathlib import Path
from contextlib import contextmanager

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# 3 most dissimilar domains (avg cosine 0.249 on Qwen3-0.6B mid-layer)
TARGET_DOMAINS = ["math", "law", "history"]

# Model configs: id, vector file key, mid-layer for steering
MODELS = {
    "qwen3_0.6b": {
        "model_id": "Qwen/Qwen3-0.6B",
        "vectors_file": "mmlu_pro_vectors_qwen3_0.6b.pt",
        "mid_layer": 14,
        "coefficients": [10, 30, 60],
    },
    "llama3_3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "vectors_file": "mmlu_pro_vectors_llama3_3b.pt",
        "mid_layer": 14,
        "coefficients": [10, 30, 60],
    },
    "lfm2_1.2b": {
        "model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
        "vectors_file": "mmlu_pro_vectors_lfm2_1.2b.pt",
        "mid_layer": 8,
        "coefficients": [10, 30, 60],
    },
}


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def _steering_hook(module, input, output, *, vector, coeff):
    hidden = output[0] if isinstance(output, tuple) else output
    vec_normed = vector / (vector.norm() + 1e-8)
    steered = hidden + coeff * vec_normed.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


def get_layers(model):
    """Get transformer layers, handling different architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


@contextmanager
def apply_steering(model, layer, vector, coeff):
    layers = get_layers(model)
    handle = layers[layer].register_forward_hook(
        functools.partial(_steering_hook, vector=vector, coeff=coeff)
    )
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# HFLM subclasses
# ---------------------------------------------------------------------------
class ExtendedHFLM(HFLM):
    """HFLM with extended generation budget for thinking models."""
    @property
    def max_gen_toks(self):
        return 2048


class SteeredHFLM(ExtendedHFLM):
    """HFLM with steering hook on a specific layer."""

    def __init__(self, steering_vector, steering_layer, steering_coeff, **kwargs):
        super().__init__(**kwargs)
        self._steering_vector = steering_vector
        self._steering_layer = steering_layer
        self._steering_coeff = steering_coeff

    def _model_call(self, *args, **kwargs):
        with apply_steering(
            self.model, self._steering_layer,
            self._steering_vector, self._steering_coeff
        ):
            return super()._model_call(*args, **kwargs)

    def _model_generate(self, *args, **kwargs):
        with apply_steering(
            self.model, self._steering_layer,
            self._steering_vector, self._steering_coeff
        ):
            return super()._model_generate(*args, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_domain_eval(model_key, domain, vectors, limit, device, dtype):
    """Run baseline + steered evaluations for one model × one domain."""
    cfg = MODELS[model_key]
    model_id = cfg["model_id"]
    mid_layer = cfg["mid_layer"]
    task = f"mmlu_pro_{domain}"
    dtype_str = {torch.float16: "float16", torch.bfloat16: "bfloat16",
                 torch.float32: "float32"}[dtype]

    results = {"model": model_key, "domain": domain, "task": task}

    # --- Baseline ---
    print(f"\n  [{model_key}] [{domain}] Baseline (n={limit})...")
    baseline = ExtendedHFLM(
        pretrained=model_id, device=device, dtype=dtype_str, batch_size=4,
    )
    eval_out = lm_eval.simple_evaluate(
        model=baseline, tasks=[task], batch_size=4, limit=limit,
    )
    task_res = eval_out["results"].get(task, {})
    results["baseline"] = {k: v for k, v in task_res.items()
                           if not k.startswith("alias")}
    print(f"    Baseline: {results['baseline']}")

    del baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- Steered at mid-layer with best coefficient ---
    vec = vectors[domain][mid_layer]
    for coeff in cfg["coefficients"]:
        label = f"L{mid_layer}_a{coeff}"
        print(f"\n  [{model_key}] [{domain}] Steered {label} (n={limit})...")
        steered = SteeredHFLM(
            steering_vector=vec,
            steering_layer=mid_layer,
            steering_coeff=float(coeff),
            pretrained=model_id, device=device, dtype=dtype_str, batch_size=4,
        )
        eval_out = lm_eval.simple_evaluate(
            model=steered, tasks=[task], batch_size=4, limit=limit,
        )
        task_res = eval_out["results"].get(task, {})
        results[label] = {k: v for k, v in task_res.items()
                          if not k.startswith("alias")}
        print(f"    Steered {label}: {results[label]}")

        del steered
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MMLU-Pro benchmark with domain-specific steering"
    )
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of examples per domain (default: 50)")
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODELS.keys()) + ["all"],
                        help="Which model to benchmark")
    parser.add_argument("--domain", type=str, default="all",
                        choices=TARGET_DOMAINS + ["all"],
                        help="Which domain to evaluate")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    print("=" * 60)
    print("MMLU-PRO BENCHMARK — DOMAIN-SPECIFIC STEERING")
    print("=" * 60)
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Limit: {args.limit} per domain")
    print(f"Domains: {TARGET_DOMAINS}")

    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    domains = TARGET_DOMAINS if args.domain == "all" else [args.domain]

    all_results = {}

    for model_key in model_keys:
        cfg = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"MODEL: {cfg['model_id']}")
        print(f"{'='*60}")

        # Load pre-extracted vectors
        vec_path = RESULTS_DIR / cfg["vectors_file"]
        if not vec_path.exists():
            print(f"  SKIP: vectors not found at {vec_path}")
            print(f"  Run: python -m src.steering.mmlu_pro_vectors first")
            continue

        vectors = torch.load(vec_path, map_location="cpu", weights_only=True)
        print(f"  Loaded vectors: {vec_path}")

        model_results = {}
        for domain in domains:
            try:
                res = run_domain_eval(
                    model_key, domain, vectors, args.limit, device, dtype
                )
                model_results[domain] = res
            except Exception as e:
                print(f"  ERROR on {domain}: {e}")
                model_results[domain] = {"error": str(e)}

        all_results[model_key] = model_results

    # Save
    out_path = RESULTS_DIR / f"mmlu_pro_benchmark_n{args.limit}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15s} {'Domain':<10s} {'Baseline':>10s} {'Best Steered':>14s} {'Delta':>8s}")
    print("-" * 60)
    for model_key, model_res in all_results.items():
        for domain, res in model_res.items():
            if "error" in res:
                print(f"{model_key:<15s} {domain:<10s} {'ERROR':>10s}")
                continue
            base_acc = res.get("baseline", {}).get("acc,none", "?")
            # Find best steered
            best_label, best_acc = "none", 0
            for k, v in res.items():
                if k.startswith("L") and isinstance(v, dict):
                    acc = v.get("acc,none", 0)
                    if isinstance(acc, (int, float)) and acc > best_acc:
                        best_acc = acc
                        best_label = k
            if isinstance(base_acc, (int, float)):
                delta = best_acc - base_acc
                print(f"{model_key:<15s} {domain:<10s} {base_acc:>9.1%} {best_acc:>9.1%} ({best_label}) {delta:>+7.1%}")
            else:
                print(f"{model_key:<15s} {domain:<10s} {str(base_acc):>10s}")


if __name__ == "__main__":
    main()
