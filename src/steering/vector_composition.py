"""
Vector Composition Tests for Domain Steering Vectors.

Tests whether steering vectors from different domains can be composed
(added together) without degeneration. Three strategies:
1. Addition (normalize after sum)
2. Weighted addition (0.5/0.5 and 0.7/0.3)
3. Sequential switching (apply one vector, then another)

Evaluates on instruct model only (Qwen3-0.6B).
"""

import json
import functools
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.steering.domain_vectors import (
    SLM_INSTRUCT_ID,
    DOMAIN_KEYWORDS,
    _steering_hook,
    get_device_and_dtype,
    load_model,
    cleanup_model,
    score_domain_flavor,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
VECTORS_PATH = RESULTS_DIR / "domain_steering_vectors.pt"

# Best layer/alpha per domain (from sweep results)
BEST_CONFIG = {
    "code_reading":   {"layer": 15, "alpha": 30.0},
    "bug_analysis":   {"layer": 18, "alpha": 60.0},
    "patch_writing":  {"layer": 15, "alpha": 30.0},
    "test_reasoning": {"layer": 15, "alpha": 30.0},  # not used, kept for reference
}

# Compositions to test
COMPOSITIONS = [
    {
        "name": "code_reading + bug_analysis",
        "domains": ["code_reading", "bug_analysis"],
    },
    {
        "name": "bug_analysis + patch_writing",
        "domains": ["bug_analysis", "patch_writing"],
    },
    {
        "name": "code_reading + bug_analysis + patch_writing",
        "domains": ["code_reading", "bug_analysis", "patch_writing"],
    },
]

# Evaluation prompts
EVAL_PROMPTS = [
    (
        "Look at this function and tell me what's wrong with it:\n"
        "def calculate_total(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item.price\n"
        "    return total"
    ),
    "This code crashes with AttributeError on line 5. How would you fix it?",
    "Review this code, identify the bug, and write a patch",
]

MAX_NEW_TOKENS = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_composed_layer_and_alpha(domains):
    """Average layer (rounded to nearest tested) and half average alpha."""
    tested_layers = [15, 18, 20, 22, 25]
    avg_layer = sum(BEST_CONFIG[d]["layer"] for d in domains) / len(domains)
    # Round to nearest tested layer
    best_layer = min(tested_layers, key=lambda l: abs(l - avg_layer))
    avg_alpha = sum(BEST_CONFIG[d]["alpha"] for d in domains) / len(domains)
    composed_alpha = avg_alpha / 2.0
    return best_layer, composed_alpha


def is_coherent(text):
    """Simple coherence check: not too short, not too repetitive."""
    if len(text.strip()) < 10:
        return False
    words = text.split()
    if len(words) < 3:
        return False
    # Check for excessive repetition (degeneration sign)
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.15:
            return False
    return True


def count_tokens(text, tokenizer):
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_prompt_with_chat_template(tokenizer, prompt):
    """Apply chat template for instruct model."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_with_hooks(model, tokenizer, prompt, device, hooks_spec,
                        max_tokens=MAX_NEW_TOKENS):
    """Generate with one or more steering hooks.

    hooks_spec: list of (layer_idx, vector, coeff) tuples
    """
    text = format_prompt_with_chat_template(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    handles = []
    try:
        for layer_idx, vec, coeff in hooks_spec:
            vec_normed = vec / (vec.norm() + 1e-8)
            h = model.model.layers[layer_idx].register_forward_hook(
                functools.partial(_steering_hook, vector=vec_normed, coeff=coeff)
            )
            handles.append(h)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1e-7,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in handles:
            h.remove()

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    return resp.strip()


def generate_sequential(model, tokenizer, prompt, device, domains, vectors,
                        layer, alpha, max_tokens=MAX_NEW_TOKENS):
    """Sequential switching: generate half tokens with v1, half with v2 (etc.)."""
    tokens_per_domain = max_tokens // len(domains)
    full_response = ""

    # Build up context incrementally
    for i, domain in enumerate(domains):
        vec = vectors["instruct"][domain][layer]
        is_last = (i == len(domains) - 1)
        gen_tokens = max_tokens - count_tokens(full_response, tokenizer) if is_last else tokens_per_domain

        # For first domain, use original prompt; for subsequent, append prior output
        if i == 0:
            current_prompt = prompt
        else:
            current_prompt = prompt + "\n\n" + full_response

        part = generate_with_hooks(
            model, tokenizer, current_prompt, device,
            [(layer, vec, alpha)],
            max_tokens=gen_tokens,
        )
        full_response += (" " if full_response else "") + part

    return full_response.strip()


def evaluate_generation(text, domains, tokenizer):
    """Evaluate a single generation against all relevant domains."""
    result = {
        "text": text,
        "token_count": count_tokens(text, tokenizer),
        "coherent": is_coherent(text),
        "domain_scores": {},
    }
    for domain in domains:
        result["domain_scores"][domain] = score_domain_flavor(text, domain)
    return result


# ---------------------------------------------------------------------------
# Composition strategies
# ---------------------------------------------------------------------------

def test_addition(model, tokenizer, device, vectors, composition):
    """Strategy 1: Simple addition with normalization."""
    domains = composition["domains"]
    layer, alpha = compute_composed_layer_and_alpha(domains)

    # Sum vectors and normalize
    vecs = [vectors["instruct"][d][layer] for d in domains]
    combined = sum(vecs)
    combined = combined / (combined.norm() + 1e-8) * vecs[0].norm()  # scale to original norm

    results = []
    for prompt in EVAL_PROMPTS:
        text = generate_with_hooks(
            model, tokenizer, prompt, device,
            [(layer, combined, alpha)],
        )
        results.append(evaluate_generation(text, domains, tokenizer))

    return {
        "strategy": "addition",
        "layer": layer,
        "alpha": alpha,
        "results": results,
    }


def test_weighted_addition(model, tokenizer, device, vectors, composition,
                           weights):
    """Strategy 2: Weighted addition with normalization."""
    domains = composition["domains"]
    layer, alpha = compute_composed_layer_and_alpha(domains)

    vecs = [vectors["instruct"][d][layer] for d in domains]
    combined = sum(w * v for w, v in zip(weights, vecs))
    combined = combined / (combined.norm() + 1e-8) * vecs[0].norm()

    results = []
    for prompt in EVAL_PROMPTS:
        text = generate_with_hooks(
            model, tokenizer, prompt, device,
            [(layer, combined, alpha)],
        )
        results.append(evaluate_generation(text, domains, tokenizer))

    weight_str = "/".join(f"{w}" for w in weights)
    return {
        "strategy": f"weighted_{weight_str}",
        "weights": weights,
        "layer": layer,
        "alpha": alpha,
        "results": results,
    }


def test_sequential(model, tokenizer, device, vectors, composition):
    """Strategy 3: Sequential switching between vectors."""
    domains = composition["domains"]
    layer, alpha = compute_composed_layer_and_alpha(domains)

    results = []
    for prompt in EVAL_PROMPTS:
        text = generate_sequential(
            model, tokenizer, prompt, device,
            domains, vectors, layer, alpha,
        )
        results.append(evaluate_generation(text, domains, tokenizer))

    return {
        "strategy": "sequential",
        "layer": layer,
        "alpha": alpha,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Baseline: no steering
# ---------------------------------------------------------------------------

def test_baseline(model, tokenizer, device, domains):
    """Generate without any steering for comparison."""
    results = []
    for prompt in EVAL_PROMPTS:
        text = generate_with_hooks(
            model, tokenizer, prompt, device, [],  # no hooks
        )
        results.append(evaluate_generation(text, domains, tokenizer))

    return {
        "strategy": "baseline (no steering)",
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VECTOR COMPOSITION TEST")
    print("=" * 70)

    device, dtype = get_device_and_dtype()
    print(f"Device: {device}  |  dtype: {dtype}")

    # Load vectors
    print(f"\nLoading vectors from {VECTORS_PATH}...")
    all_vectors = torch.load(VECTORS_PATH, map_location="cpu", weights_only=True)
    print(f"  Domains available: {list(all_vectors['instruct'].keys())}")

    # Load model once
    print(f"\nLoading model: {SLM_INSTRUCT_ID}...")
    model, tokenizer = load_model(SLM_INSTRUCT_ID, device, dtype)
    print(f"  Loaded: {model.config.num_hidden_layers} layers")

    all_results = {
        "model": SLM_INSTRUCT_ID,
        "device": device,
        "max_new_tokens": MAX_NEW_TOKENS,
        "best_config_per_domain": {d: BEST_CONFIG[d] for d in BEST_CONFIG},
        "compositions": [],
    }

    # ------------------------------------------------------------------
    # Run tests for each composition
    # ------------------------------------------------------------------
    for comp in COMPOSITIONS:
        print(f"\n{'='*70}")
        print(f"COMPOSITION: {comp['name']}")
        print(f"{'='*70}")

        domains = comp["domains"]
        layer, alpha = compute_composed_layer_and_alpha(domains)
        print(f"  Composed layer={layer}, alpha={alpha:.1f}")

        comp_result = {
            "name": comp["name"],
            "domains": domains,
            "composed_layer": layer,
            "composed_alpha": alpha,
            "strategies": [],
        }

        # Baseline
        print("\n  [Baseline - no steering]")
        baseline = test_baseline(model, tokenizer, device, domains)
        comp_result["strategies"].append(baseline)
        for i, r in enumerate(baseline["results"]):
            print(f"    Prompt {i+1}: coherent={r['coherent']}, "
                  f"tokens={r['token_count']}, scores={r['domain_scores']}")

        # Strategy 1: Addition
        print("\n  [Addition]")
        addition = test_addition(model, tokenizer, device, all_vectors, comp)
        comp_result["strategies"].append(addition)
        for i, r in enumerate(addition["results"]):
            print(f"    Prompt {i+1}: coherent={r['coherent']}, "
                  f"tokens={r['token_count']}, scores={r['domain_scores']}")

        # Strategy 2a: Weighted 0.5/0.5
        weights_equal = [0.5] * len(domains)
        print(f"\n  [Weighted {weights_equal}]")
        weighted_eq = test_weighted_addition(
            model, tokenizer, device, all_vectors, comp, weights_equal
        )
        comp_result["strategies"].append(weighted_eq)
        for i, r in enumerate(weighted_eq["results"]):
            print(f"    Prompt {i+1}: coherent={r['coherent']}, "
                  f"tokens={r['token_count']}, scores={r['domain_scores']}")

        # Strategy 2b: Weighted 0.7/0.3 (for 2-domain compositions)
        if len(domains) == 2:
            weights_skewed = [0.7, 0.3]
            print(f"\n  [Weighted {weights_skewed}]")
            weighted_sk = test_weighted_addition(
                model, tokenizer, device, all_vectors, comp, weights_skewed
            )
            comp_result["strategies"].append(weighted_sk)
            for i, r in enumerate(weighted_sk["results"]):
                print(f"    Prompt {i+1}: coherent={r['coherent']}, "
                      f"tokens={r['token_count']}, scores={r['domain_scores']}")

        # Strategy 3: Sequential
        print("\n  [Sequential switching]")
        sequential = test_sequential(model, tokenizer, device, all_vectors, comp)
        comp_result["strategies"].append(sequential)
        for i, r in enumerate(sequential["results"]):
            print(f"    Prompt {i+1}: coherent={r['coherent']}, "
                  f"tokens={r['token_count']}, scores={r['domain_scores']}")

        all_results["compositions"].append(comp_result)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    cleanup_model(model, device)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_path = RESULTS_DIR / "vector_composition_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results: {output_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    header = f"{'Composition':<45} {'Strategy':<22} {'Coherent':>8} {'AvgTok':>7} {'AvgScore':>9}"
    print(header)
    print("-" * len(header))

    for comp_res in all_results["compositions"]:
        comp_name = comp_res["name"]
        domains = comp_res["domains"]

        for strat in comp_res["strategies"]:
            strat_name = strat["strategy"]
            n_results = len(strat["results"])

            coherent_count = sum(1 for r in strat["results"] if r["coherent"])
            avg_tokens = sum(r["token_count"] for r in strat["results"]) / n_results
            # Average total domain score across all prompts
            avg_score = sum(
                sum(r["domain_scores"].values())
                for r in strat["results"]
            ) / n_results

            print(f"{comp_name:<45} {strat_name:<22} "
                  f"{coherent_count}/{n_results:>5} {avg_tokens:>7.0f} "
                  f"{avg_score:>9.1f}")

    # ------------------------------------------------------------------
    # Degeneration analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DEGENERATION ANALYSIS")
    print("=" * 70)

    for comp_res in all_results["compositions"]:
        comp_name = comp_res["name"]
        print(f"\n  {comp_name}:")

        for strat in comp_res["strategies"]:
            strat_name = strat["strategy"]
            incoherent = [r for r in strat["results"] if not r["coherent"]]
            very_short = [r for r in strat["results"] if r["token_count"] < 10]
            very_long = [r for r in strat["results"] if r["token_count"] > 200]

            issues = []
            if incoherent:
                issues.append(f"{len(incoherent)} incoherent")
            if very_short:
                issues.append(f"{len(very_short)} very short (<10 tok)")
            if very_long:
                issues.append(f"{len(very_long)} very long (>200 tok)")

            status = ", ".join(issues) if issues else "OK"
            print(f"    {strat_name:<22}: {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
