"""
Feature-targeted steering: construct steering vectors from SAE decoder columns
instead of contrastive mean differences.

The hypothesis: targeting sparse, domain-specific SAE features should produce
more precise steering that improves knowledge-level QA (not just style).

Three vector construction strategies:
  1. top-k weighted: sum of W_dec[i] * differential_activation[i] for top-k features
  2. top-k uniform: sum of W_dec[i] for top-k features (equal weight)
  3. single-feature: W_dec[best_feature] only (maximally sparse)

Benchmark: MMLU-Pro loglikelihood mode, comparing contrastive vs feature-targeted.

Usage:
    python -m src.steering.feature_targeted_steering --limit 50
    python -m src.steering.feature_targeted_steering --model Qwen/Qwen3-4B --limit 50
"""

import argparse
import gc
import json
import functools
from pathlib import Path
from contextlib import contextmanager

import torch
import numpy as np
import lm_eval
from lm_eval.models.huggingface import HFLM
from sae_lens import SAE
from transformer_lens import HookedTransformer

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TASKS_DIR = Path(__file__).resolve().parent / "tasks" / "mmlu_pro_mc"
TARGET_DOMAINS = ["math", "law", "history"]

MODEL_PRESETS = {
    "Qwen/Qwen3-0.6B": {"layer": 14, "sae_dir": "sae_qwen3_0.6b_L14_8x",
                          "vectors": "mmlu_pro_vectors_qwen3_0.6b.pt"},
    "Qwen/Qwen3-4B": {"layer": 18, "sae_dir": "sae_qwen3_4b_L18_8x",
                        "vectors": "mmlu_pro_vectors_qwen3_4b.pt"},
    "LiquidAI/LFM2-700M": {"layer": 8, "sae_dir": "sae_lfm2_700m_L8_8x",
                             "vectors": "mmlu_pro_vectors_lfm2_700m.pt"},
}

# Same domain prompts as analyze_sae_features.py for consistency
DOMAIN_PROMPTS = {
    "math": [
        "Solve the equation 3x + 7 = 22 for x.",
        "What is the derivative of sin(x) * cos(x)?",
        "Prove that the square root of 2 is irrational.",
        "Calculate the integral of e^x from 0 to 1.",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "What is the probability of rolling two sixes with two dice?",
        "Simplify the expression (x^2 - 4)/(x - 2).",
        "How many ways can you arrange 5 books on a shelf?",
        "What is the Taylor series expansion of ln(1+x)?",
        "Solve the differential equation dy/dx = 2xy.",
    ],
    "law": [
        "What is the difference between civil and criminal law?",
        "Explain the concept of habeas corpus.",
        "What are the elements of a valid contract?",
        "Define the legal principle of stare decisis.",
        "What is the Miranda warning and when must it be given?",
        "Explain the doctrine of sovereign immunity.",
        "What constitutes negligence in tort law?",
        "What is the difference between a felony and a misdemeanor?",
        "Explain the concept of due process under the 14th Amendment.",
        "What are the requirements for obtaining a patent?",
    ],
    "history": [
        "What caused the fall of the Roman Empire?",
        "Describe the main events of the French Revolution.",
        "What was the significance of the Magna Carta?",
        "Explain the causes of World War I.",
        "What was the impact of the Industrial Revolution on society?",
        "Describe the civil rights movement in the United States.",
        "What were the consequences of the Treaty of Versailles?",
        "Explain the rise and fall of the Ottoman Empire.",
        "What was the significance of the Silk Road?",
        "Describe the colonization of the Americas by European powers.",
    ],
}


# ---------------------------------------------------------------------------
# Build feature-targeted vectors from SAE
# ---------------------------------------------------------------------------
def build_feature_vectors(sae, model_id, layer, device, top_k=20):
    """
    Build feature-targeted steering vectors for each domain.

    Returns dict with three strategies per domain:
      - 'weighted_k{top_k}': weighted sum of top-k decoder columns
      - 'uniform_k{top_k}': equal-weight sum of top-k decoder columns
      - 'single_best': single best feature's decoder column
    """
    from .sae_utils import compute_all_domain_activations

    print("\n  Computing SAE activations per domain...")
    raw_activations = compute_all_domain_activations(model_id, sae, DOMAIN_PROMPTS, layer, device)
    activations = {}
    for domain, acts in raw_activations.items():
        activations[domain] = acts.mean(dim=0)
        print(f"    {domain}: mean act = {activations[domain].mean():.4f}")

    domains = list(activations.keys())
    W_dec = sae.W_dec.detach().cpu()  # (d_sae, d_in)

    vectors = {}
    for domain in TARGET_DOMAINS:
        domain_mean = activations[domain]
        other_mean = torch.stack(
            [activations[d] for d in domains if d != domain]
        ).mean(dim=0)
        differential = domain_mean - other_mean

        # Top-k features by differential activation
        topk = differential.topk(top_k)
        top_indices = topk.indices
        top_values = topk.values

        # Strategy 1: weighted sum (differential activation as weight)
        weighted_vec = torch.zeros(W_dec.shape[1])
        for idx, val in zip(top_indices, top_values):
            weighted_vec += val.item() * W_dec[idx]

        # Strategy 2: uniform sum
        uniform_vec = W_dec[top_indices].sum(dim=0)

        # Strategy 3: single best feature
        single_vec = W_dec[top_indices[0]].clone()

        vectors[domain] = {
            f"weighted_k{top_k}": weighted_vec,
            f"uniform_k{top_k}": uniform_vec,
            "single_best": single_vec,
            "top_features": top_indices.tolist(),
            "top_diff_values": top_values.tolist(),
        }

        print(f"\n  [{domain}] Feature-targeted vectors built:")
        print(f"    Top-5 features: {top_indices[:5].tolist()}")
        print(f"    Top-5 diffs: {[f'{v:.3f}' for v in top_values[:5].tolist()]}")
        print(f"    Weighted vec norm: {weighted_vec.norm():.4f}")
        print(f"    Uniform vec norm: {uniform_vec.norm():.4f}")
        print(f"    Single vec norm: {single_vec.norm():.4f}")

    return vectors


# ---------------------------------------------------------------------------
# Steering hook (same as mmlu_pro_benchmark_mc.py)
# ---------------------------------------------------------------------------
def _steering_hook(module, input, output, *, vector, coeff, mode="additive"):
    hidden = output[0] if isinstance(output, tuple) else output
    v = vector.to(hidden.device, dtype=hidden.dtype)
    vec_normed = v / (v.norm() + 1e-8)
    if mode == "multiplicative":
        steered = hidden * (1.0 + coeff * vec_normed)
    else:
        steered = hidden + coeff * vec_normed
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


@contextmanager
def apply_steering(model, layer, vector, coeff, mode="additive"):
    layers = get_layers(model)
    handle = layers[layer].register_forward_hook(
        functools.partial(_steering_hook, vector=vector, coeff=coeff, mode=mode)
    )
    try:
        yield
    finally:
        handle.remove()


class SteeredHFLM(HFLM):
    """HFLM subclass with dynamic steering. Set _steering_vector=None for baseline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._steering_vector = None
        self._steering_layer = 14
        self._steering_coeff = 0.0
        self._steering_mode = "additive"

    def _model_call(self, *args, **kwargs):
        if self._steering_vector is not None and self._steering_coeff > 0:
            with apply_steering(
                self.model, self._steering_layer,
                self._steering_vector, self._steering_coeff, self._steering_mode
            ):
                return super()._model_call(*args, **kwargs)
        return super()._model_call(*args, **kwargs)

    def _model_generate(self, *args, **kwargs):
        if self._steering_vector is not None and self._steering_coeff > 0:
            with apply_steering(
                self.model, self._steering_layer,
                self._steering_vector, self._steering_coeff, self._steering_mode
            ):
                return super()._model_generate(*args, **kwargs)
        return super()._model_generate(*args, **kwargs)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_eval(hflm, domain, vector, coeff, label, limit, layer=14, mode="additive"):
    """Run a single MMLU-Pro MC evaluation, reusing a pre-loaded HFLM."""
    task = f"mmlu_pro_mc_{domain}"

    # Configure steering on the shared model
    hflm._steering_vector = vector
    hflm._steering_layer = layer
    hflm._steering_coeff = float(coeff) if vector is not None else 0.0
    hflm._steering_mode = mode

    eval_out = lm_eval.simple_evaluate(
        model=hflm, tasks=[task], batch_size=8, limit=limit,
        task_manager=lm_eval.tasks.TaskManager(include_path=str(TASKS_DIR)),
        log_samples=True,
    )
    task_res = eval_out["results"].get(task, {})
    acc = task_res.get("acc,none", None)
    stderr = task_res.get("acc_stderr,none", None)

    # Extract per-sample predictions with options and log-likelihoods
    samples = []
    for sample in eval_out.get("samples", {}).get(task, []):
        doc = sample.get("doc", {})
        target = sample.get("target", "")
        resps = sample.get("resps", [])
        options = doc.get("options", [])
        if resps and options:
            lls = [r[0] if isinstance(r, (list, tuple)) else r for r in resps]
            best_idx = max(range(len(lls)), key=lambda i: lls[i]) if lls else -1
            answer_key = chr(65 + best_idx) if 0 <= best_idx < len(options) else "?"

            # Build per-option scores
            option_scores = []
            for j, opt in enumerate(options):
                ll_val = lls[j] if j < len(lls) else None
                if isinstance(ll_val, (list, tuple)):
                    ll_val = ll_val[0]
                option_scores.append({
                    "key": chr(65 + j),
                    "text": opt[:200],
                    "loglikelihood": round(float(ll_val), 3) if ll_val is not None else None,
                    "selected": j == best_idx,
                })

            samples.append({
                "question": doc.get("question", ""),
                "answer": answer_key,
                "answer_text": options[best_idx][:200] if 0 <= best_idx < len(options) else "",
                "correct": str(target),
                "correct_text": options[ord(str(target)) - 65][:200] if str(target).isalpha() and ord(str(target)) - 65 < len(options) else "",
                "is_correct": answer_key == str(target),
                "options": option_scores,
            })

    return {
        "label": label, "acc": acc, "stderr": stderr,
        "raw": {k: v for k, v in task_res.items() if not k.startswith("alias")},
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feature-targeted steering vs contrastive on MMLU-Pro MC"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Steering layer (default: from preset)")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--domain", type=str, default="all",
                        choices=TARGET_DOMAINS + ["all"])
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of SAE features for targeted vectors")
    parser.add_argument("--coefficients", type=str, default="10,30,60",
                        help="Comma-separated steering coefficients")
    args = parser.parse_args()

    preset = MODEL_PRESETS.get(args.model, MODEL_PRESETS["Qwen/Qwen3-0.6B"])
    mid_layer = args.layer if args.layer is not None else preset["layer"]
    sae_path = RESULTS_DIR / preset["sae_dir"]
    contrastive_file = preset["vectors"]
    hook_name = f"blocks.{mid_layer}.hook_resid_post"
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    coefficients = [int(c) for c in args.coefficients.split(",")]
    domains = TARGET_DOMAINS if args.domain == "all" else [args.domain]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device, dtype_str = "cuda", "bfloat16"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype_str = "mps", "float16"
    else:
        device, dtype_str = "cpu", "float16"  # float16 on CPU to save RAM

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]

    print("=" * 60)
    print(f"FEATURE-TARGETED STEERING — {args.model} MMLU-Pro MC")
    print("=" * 60)
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Layer: {mid_layer} | Limit: {args.limit} | top_k: {args.top_k}")
    print(f"Coefficients: {coefficients}")
    print(f"SAE: {sae_path}")

    # --- Step 1: Build or load cached feature-targeted vectors ---
    cache_path = RESULTS_DIR / f"feature_vectors_{model_short}_L{mid_layer}_k{args.top_k}.pt"

    if cache_path.exists():
        print(f"\n  Loading cached feature vectors from {cache_path}")
        feature_vectors = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        print("\n" + "=" * 60)
        print("BUILDING FEATURE-TARGETED VECTORS")
        print("=" * 60)

        from .sae_utils import load_sae

        sae = load_sae(str(sae_path), device=device)
        print(f"  SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

        feature_vectors = build_feature_vectors(sae, args.model, mid_layer, device, top_k=args.top_k)

        del sae
        cleanup_memory()

        # Cache for next run
        torch.save(feature_vectors, cache_path)
        print(f"  Cached feature vectors to {cache_path}")

    # --- Step 2: Load contrastive vectors + output-score vectors ---
    contrastive_path = RESULTS_DIR / contrastive_file
    contrastive_vectors = torch.load(contrastive_path, map_location="cpu", weights_only=True)
    print(f"\n  Contrastive vectors loaded from {contrastive_path}")

    # Output-score vectors (optional)
    osv_path = RESULTS_DIR / f"output_score_vectors_{model_short}.pt"
    output_score_vectors = None
    if osv_path.exists():
        output_score_vectors = torch.load(osv_path, map_location="cpu", weights_only=True)
        print(f"  Output-score vectors loaded from {osv_path}")

    # --- Step 3: Benchmark ---
    print("\n" + "=" * 60)
    print("MMLU-Pro MC EVALUATION")
    print("=" * 60)

    # Load model ONCE and reuse across all configs
    print(f"\n  Loading model for evaluation ({dtype_str} on {device})...")
    hflm = SteeredHFLM(pretrained=args.model, device=device, dtype=dtype_str, batch_size=8)
    print(f"  Model loaded.")

    def _eval(vec, coeff, label, mode="additive"):
        print(f"\n  [{label}] n={args.limit}...")
        res = run_eval(hflm, domain, vec, coeff, label, args.limit, layer=mid_layer, mode=mode)
        print(f"    acc={res['acc']:.3f} ± {res['stderr']:.3f}" if res['acc'] is not None else "    ERROR")
        return res

    all_results = {}

    for domain in domains:
        print(f"\n{'─'*60}")
        print(f"DOMAIN: {domain}")
        print(f"{'─'*60}")

        domain_results = []

        # Baseline
        domain_results.append(_eval(None, 0, "baseline"))

        # Contrastive steering — additive
        contrastive_vec = contrastive_vectors[domain][mid_layer]
        for coeff in coefficients:
            domain_results.append(_eval(contrastive_vec, coeff, f"contrastive_a{coeff}"))

        # Contrastive — multiplicative
        for coeff in coefficients:
            domain_results.append(_eval(contrastive_vec, coeff, f"contrastive_mult_a{coeff}", mode="multiplicative"))

        # Feature-targeted: weighted (additive)
        fv = feature_vectors[domain]
        for coeff in coefficients:
            vec = fv[f"weighted_k{args.top_k}"]
            domain_results.append(_eval(vec, coeff, f"feat_weighted_k{args.top_k}_a{coeff}"))

        # Feature-targeted: uniform (additive)
        for coeff in coefficients:
            vec = fv[f"uniform_k{args.top_k}"]
            domain_results.append(_eval(vec, coeff, f"feat_uniform_k{args.top_k}_a{coeff}"))

        # Feature-targeted: single best (additive)
        for coeff in coefficients:
            vec = fv["single_best"]
            domain_results.append(_eval(vec, coeff, f"feat_single_a{coeff}"))

        # Output-score vectors (if available)
        if output_score_vectors and domain in output_score_vectors:
            osv = output_score_vectors[domain]
            for key, label_prefix in [
                ("output_weighted", "outscore_weighted"),
                ("output_single", "outscore_single"),
            ]:
                if key in osv:
                    for coeff in coefficients:
                        domain_results.append(_eval(osv[key], coeff, f"{label_prefix}_a{coeff}"))
                    # Also test multiplicative
                    for coeff in coefficients:
                        domain_results.append(_eval(osv[key], coeff, f"{label_prefix}_mult_a{coeff}", mode="multiplicative"))

        all_results[domain] = domain_results

    del hflm
    cleanup_memory()

    # --- Save (merge with existing domains) ---
    out_path = RESULTS_DIR / f"feature_targeted_benchmark_{model_short}_n{args.limit}.json"

    # Load existing data to merge
    save_data = {}
    if out_path.exists():
        with open(out_path) as f:
            save_data = json.load(f)
        print(f"  Merging with existing domains: {list(save_data.keys())}")

    for domain, results in all_results.items():
        # Strip per-sample data from main results (saved separately)
        clean_results = [{k: v for k, v in r.items() if k != "samples"} for r in results]
        save_data[domain] = {
            "results": clean_results,
            "top_features": feature_vectors[domain].get("top_features", []),
            "top_diff_values": feature_vectors[domain].get("top_diff_values", []),
        }

    # Save per-sample data separately for the sample viewer
    samples_data = {}
    for domain, results in all_results.items():
        # Get baseline samples as reference (first result)
        baseline_samples = results[0].get("samples", [])
        if not baseline_samples:
            continue
        merged = []
        for i, bs in enumerate(baseline_samples):
            entry = {
                "question": bs["question"],
                "correct": bs["correct"],
                "correct_text": bs.get("correct_text", ""),
                "options": bs.get("options", []),
                "results": {},
            }
            for res in results:
                if i < len(res.get("samples", [])):
                    s = res["samples"][i]
                    entry["results"][res["label"]] = {
                        "answer": s["answer"],
                        "answer_text": s.get("answer_text", ""),
                        "correct": s["is_correct"],
                        "options": s.get("options", []),
                    }
            merged.append(entry)
        samples_data[domain] = merged

    samples_path = RESULTS_DIR / f"benchmark_samples_{model_short}_n{args.limit}.json"
    # Merge samples too
    existing_samples = {}
    if samples_path.exists():
        with open(samples_path) as f:
            existing_samples = json.load(f)
    existing_samples.update(samples_data)
    with open(samples_path, "w") as f:
        json.dump(existing_samples, f, indent=2, default=str)
    print(f"Saved samples: {samples_path}")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("SUMMARY — Feature-Targeted vs Contrastive Steering")
    print(f"{'='*80}")
    print(f"{'Domain':<10s} {'Method':<30s} {'Acc':>8s} {'Stderr':>8s} {'Delta':>8s}")
    print("─" * 80)
    for domain, results in all_results.items():
        baseline_acc = results[0]["acc"] if results[0]["acc"] is not None else 0
        for res in results:
            acc = res["acc"] if res["acc"] is not None else 0
            stderr = res["stderr"] if res["stderr"] is not None else 0
            delta = acc - baseline_acc
            marker = " ★" if delta > 0 and res["label"] != "baseline" else ""
            print(f"{domain:<10s} {res['label']:<30s} {acc:>7.1%} {stderr:>7.3f} {delta:>+7.1%}{marker}")
        print()


if __name__ == "__main__":
    main()
