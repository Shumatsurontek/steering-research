"""
Sampling-Based Steering — Detecting Subtle Steering Effects

Tests whether steering effects that are invisible under greedy decoding
manifest as probability redistribution under sampling (T>0).

Key metrics:
- KL divergence between steered/unsteered logit distributions
- Top-k token probability shifts
- Response diversity under sampling
"""

import json
import functools
import math
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = (
    "You are a calendar assistant. Extract event details from the user's message "
    "and return a JSON object with: title, date, start_time, end_time, location, "
    "attendees. Today is 2026-03-13 (Friday)."
)

TEST_PROMPTS = [
    ("calendar_fr", "Crée un rendez-vous demain à 14h avec Marie pour discuter du projet."),
    ("calendar_en", "Schedule a meeting next Monday at 10am with the team in Room 301."),
    ("ambiguous_en", "I need to see Marie tomorrow about the project."),
    ("ambiguous_fr", "Faut que je vois Antoine la semaine prochaine."),
    ("non_calendar_fr", "Quelle est la capitale de la France ?"),
    ("non_calendar_en", "Explain how transformers work."),
]

# Layers to test: the sweet spot (15, 18) + rigid (35) for comparison
LAYERS_TO_TEST = [15, 18, 35]
COEFFICIENTS = [0.0, 10.0, 30.0, 60.0]
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
N_SAMPLES = 5  # Samples per (prompt, layer, coeff, temp) combination


def steering_hook(module, input, output, *, vector, coeff):
    hidden = output[0] if isinstance(output, tuple) else output
    steered = hidden + coeff * vector.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


def get_first_token_logits(model, tokenizer, prompt, device,
                           vector=None, layer=None, coeff=0.0):
    """Get logits for the first generated token (no actual generation)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    handle = None
    try:
        if vector is not None and layer is not None and coeff != 0:
            vec_normed = vector / (vector.norm() + 1e-8)
            handle = model.model.layers[layer].register_forward_hook(
                functools.partial(steering_hook, vector=vec_normed, coeff=coeff)
            )
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # (vocab_size,)
    finally:
        if handle:
            handle.remove()

    return logits


def compute_kl_divergence(logits_base, logits_steered):
    """KL(steered || base) in bits."""
    p = F.softmax(logits_base.float(), dim=-1)
    q = F.softmax(logits_steered.float(), dim=-1)
    # KL(q || p) = sum(q * log(q/p))
    kl = F.kl_div(p.log(), q, reduction="sum", log_target=False)
    return kl.item() / math.log(2)  # Convert to bits


def top_k_comparison(logits_base, logits_steered, tokenizer, k=10):
    """Compare top-k tokens between base and steered distributions."""
    probs_base = F.softmax(logits_base.float(), dim=-1)
    probs_steered = F.softmax(logits_steered.float(), dim=-1)

    top_base = torch.topk(probs_base, k)
    top_steered = torch.topk(probs_steered, k)

    base_tokens = set(top_base.indices.tolist())
    steered_tokens = set(top_steered.indices.tolist())
    overlap = base_tokens & steered_tokens

    base_top = []
    for idx, prob in zip(top_base.indices.tolist(), top_base.values.tolist()):
        tok = tokenizer.decode([idx])
        steered_prob = probs_steered[idx].item()
        base_top.append({
            "token": tok, "base_prob": round(prob, 4),
            "steered_prob": round(steered_prob, 4),
            "delta": round(steered_prob - prob, 4),
        })

    steered_top = []
    for idx, prob in zip(top_steered.indices.tolist(), top_steered.values.tolist()):
        tok = tokenizer.decode([idx])
        base_prob = probs_base[idx].item()
        steered_top.append({
            "token": tok, "steered_prob": round(prob, 4),
            "base_prob": round(base_prob, 4),
            "delta": round(prob - base_prob, 4),
        })

    return {
        "overlap_ratio": round(len(overlap) / k, 2),
        "base_top_tokens": base_top[:5],
        "steered_top_tokens": steered_top[:5],
    }


def generate_samples(model, tokenizer, prompt, device,
                     vector=None, layer=None, coeff=0.0,
                     temperature=0.7, n_samples=5):
    """Generate multiple samples to measure diversity."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    responses = []
    for _ in range(n_samples):
        handle = None
        try:
            if vector is not None and layer is not None and coeff != 0:
                vec_normed = vector / (vector.norm() + 1e-8)
                handle = model.model.layers[layer].register_forward_hook(
                    functools.partial(steering_hook, vector=vec_normed, coeff=coeff)
                )
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200,
                    do_sample=True if temperature > 0 else False,
                    temperature=max(temperature, 1e-7),
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
        finally:
            if handle:
                handle.remove()

        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if "</think>" in resp:
            resp = resp.split("</think>")[-1].strip()
        responses.append(resp.strip())

    # Diversity: unique responses / total
    unique = len(set(responses))
    return {
        "responses": [r[:200] for r in responses],
        "n_unique": unique,
        "diversity": round(unique / n_samples, 2),
    }


def classify_response(response: str) -> str:
    """Classify response type."""
    r = response.lower()
    if "{" in response and "date" in r:
        return "JSON"
    if any(w in r for w in ["schedule", "agenda", "rendez", "réunion", "meeting", "calendar"]):
        return "CALENDAR_TEXT"
    if any(w in r for w in ["besoin", "need", "information", "quel jour", "précis"]):
        return "CLARIFY"
    if len(response) < 10:
        return "DEGENERATE"
    return "OTHER"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAMPLING-BASED STEERING ANALYSIS")
    print("=" * 60)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Device: {device}")

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()

    # Load steering vectors
    vectors = torch.load(RESULTS_DIR / "steering_vectors.pt", map_location="cpu", weights_only=True)
    print(f"Loaded {len(vectors)} steering vectors")

    # ========================================
    # PART 1: Logit-level analysis (KL divergence)
    # ========================================
    print("\n--- PART 1: LOGIT DISTRIBUTION ANALYSIS ---")
    kl_results = {}

    for tag, prompt in TEST_PROMPTS:
        print(f"\n  [{tag}]")
        kl_results[tag] = {}

        # Get baseline logits
        baseline_logits = get_first_token_logits(model, tokenizer, prompt, device)

        for layer in LAYERS_TO_TEST:
            vec_key = f"layer_{layer}"
            if vec_key not in vectors:
                continue
            vec = vectors[vec_key]
            kl_results[tag][layer] = {}

            for coeff in COEFFICIENTS:
                if coeff == 0:
                    continue
                steered_logits = get_first_token_logits(
                    model, tokenizer, prompt, device, vec, layer, coeff
                )

                kl = compute_kl_divergence(baseline_logits, steered_logits)
                top_k = top_k_comparison(baseline_logits, steered_logits, tokenizer)

                kl_results[tag][layer][coeff] = {
                    "kl_bits": round(kl, 4),
                    "top10_overlap": top_k["overlap_ratio"],
                    "base_top5": top_k["base_top_tokens"],
                    "steered_top5": top_k["steered_top_tokens"],
                }
                print(f"    L{layer} α={coeff:>5.0f}: KL={kl:>8.4f} bits, "
                      f"top-10 overlap={top_k['overlap_ratio']:.0%}")

    # ========================================
    # PART 2: Sampling diversity analysis
    # ========================================
    print("\n--- PART 2: SAMPLING DIVERSITY ---")
    sampling_results = {}

    for tag, prompt in TEST_PROMPTS[:4]:  # Calendar + ambiguous only
        print(f"\n  [{tag}]")
        sampling_results[tag] = {}

        for temp in TEMPERATURES:
            if temp == 0:
                continue
            sampling_results[tag][temp] = {}

            # Baseline (no steering)
            baseline = generate_samples(
                model, tokenizer, prompt, device,
                temperature=temp, n_samples=N_SAMPLES,
            )
            types_base = [classify_response(r) for r in baseline["responses"]]
            type_dist_base = dict(Counter(types_base))

            sampling_results[tag][temp]["baseline"] = {
                "diversity": baseline["diversity"],
                "type_distribution": type_dist_base,
            }
            print(f"    T={temp:.1f} baseline: diversity={baseline['diversity']:.0%}, "
                  f"types={type_dist_base}")

            # Steered (layer 15 and 18 at α=30)
            for layer in [15, 18]:
                vec_key = f"layer_{layer}"
                if vec_key not in vectors:
                    continue
                vec = vectors[vec_key]

                steered = generate_samples(
                    model, tokenizer, prompt, device,
                    vector=vec, layer=layer, coeff=30.0,
                    temperature=temp, n_samples=N_SAMPLES,
                )
                types_steered = [classify_response(r) for r in steered["responses"]]
                type_dist_steered = dict(Counter(types_steered))

                sampling_results[tag][temp][f"L{layer}_a30"] = {
                    "diversity": steered["diversity"],
                    "type_distribution": type_dist_steered,
                }
                print(f"    T={temp:.1f} L{layer}@30: diversity={steered['diversity']:.0%}, "
                      f"types={type_dist_steered}")

    # ========================================
    # PART 3: KL divergence summary
    # ========================================
    print("\n" + "=" * 60)
    print("KL DIVERGENCE SUMMARY (bits)")
    print("=" * 60)
    print(f"{'Prompt':>15s} | ", end="")
    for layer in LAYERS_TO_TEST:
        for coeff in [10, 30, 60]:
            print(f"L{layer}@{coeff:>3d} | ", end="")
    print()
    print("─" * 100)

    kl_matrix = {}
    for tag in kl_results:
        print(f"{tag:>15s} | ", end="")
        kl_matrix[tag] = {}
        for layer in LAYERS_TO_TEST:
            for coeff in [10.0, 30.0, 60.0]:
                kl = kl_results.get(tag, {}).get(layer, {}).get(coeff, {}).get("kl_bits", 0)
                kl_matrix[tag][f"L{layer}_a{int(coeff)}"] = kl
                print(f"{kl:>7.3f} | ", end="")
        print()

    # --- Save ---
    output = {
        "model": MODEL_ID,
        "layers_tested": LAYERS_TO_TEST,
        "coefficients": COEFFICIENTS,
        "temperatures": TEMPERATURES,
        "n_samples": N_SAMPLES,
        "kl_divergence": {
            tag: {
                str(l): {str(c): {k: v for k, v in data.items() if k not in ("base_top5", "steered_top5")}
                         for c, data in ldata.items()}
                for l, ldata in tdata.items()
            }
            for tag, tdata in kl_results.items()
        },
        "kl_matrix": kl_matrix,
        "sampling": sampling_results,
    }
    path = RESULTS_DIR / "sampling_steering_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
