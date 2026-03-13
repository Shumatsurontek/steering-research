"""
Mid-Layer Steering Sweep — Test layers 15-25 where representations
are transitioning from syntactic to semantic and should be more malleable.
"""

import json
import functools
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

TEST_PROMPTS = [
    ("calendar_fr", "Crée un rendez-vous demain à 14h avec Marie pour discuter du projet."),
    ("calendar_en", "Schedule a meeting next Monday at 10am with the team."),
    ("ambiguous_en", "I need to see Marie tomorrow about the project."),
    ("ambiguous_fr", "Faut que je vois Antoine la semaine prochaine."),
    ("non_calendar_fr", "Quelle est la capitale de la France ?"),
    ("non_calendar_en", "Explain how transformers work."),
]

SYSTEM_PROMPT = (
    "You are a calendar assistant. When the user wants to schedule something, "
    "extract the event details as JSON with fields: title, date, start_time, "
    "end_time, location, attendees. Today is 2026-03-13 (Friday). "
    "If the request is not about scheduling, respond normally."
)

# Layers to test: mid-range (15-25) + a few early + the best late for comparison
LAYERS_TO_TEST = [5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35]
COEFFICIENTS = [0.0, 10.0, 30.0, 60.0, 100.0, 200.0]


def steering_hook(module, input, output, *, vector, coeff):
    hidden = output[0] if isinstance(output, tuple) else output
    steered = hidden + coeff * vector.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


def generate(model, tokenizer, prompt, device, vector=None, layer=None, coeff=0.0):
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
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle:
            handle.remove()

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    return resp.strip()


def classify_response(response: str) -> str:
    """Classify response type for comparison."""
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
    print("MID-LAYER STEERING SWEEP")
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

    # Baseline
    print("\n--- BASELINE (no steering) ---")
    baselines = {}
    for tag, prompt in TEST_PROMPTS:
        resp = generate(model, tokenizer, prompt, device)
        baselines[tag] = {"response": resp[:300], "type": classify_response(resp)}
        print(f"  [{tag}] → {baselines[tag]['type']}: {resp[:100]}...")

    # Sweep
    print("\n--- MID-LAYER SWEEP ---")
    sweep = {}
    for layer in LAYERS_TO_TEST:
        vec_key = f"layer_{layer}"
        if vec_key not in vectors:
            print(f"  Layer {layer}: no vector, skipping")
            continue

        vec = vectors[vec_key]
        sweep[layer] = {}
        print(f"\n  Layer {layer} (vec norm={vec.norm():.1f}):")

        for coeff in COEFFICIENTS:
            sweep[layer][coeff] = {}
            changes = 0

            for tag, prompt in TEST_PROMPTS:
                resp = generate(model, tokenizer, prompt, device, vec, layer, coeff)
                rtype = classify_response(resp)
                baseline_type = baselines[tag]["type"]
                changed = rtype != baseline_type
                if changed:
                    changes += 1

                sweep[layer][coeff][tag] = {
                    "response": resp[:300],
                    "type": rtype,
                    "changed": changed,
                }

            n = len(TEST_PROMPTS)
            print(f"    coeff={coeff:>6.1f}: {changes}/{n} responses changed "
                  f"({changes/n*100:.0f}%)")

    # Summary table
    print("\n" + "=" * 60)
    print("CHANGE RATE MATRIX (% of responses that changed vs baseline)")
    print("=" * 60)
    header = f"{'Layer':>5s} | " + " | ".join(f"{c:>6.0f}" for c in COEFFICIENTS)
    print(header)
    print("─" * len(header))

    summary = {}
    for layer in LAYERS_TO_TEST:
        if layer not in sweep:
            continue
        rates = []
        for coeff in COEFFICIENTS:
            entries = sweep[layer].get(coeff, {})
            n_changed = sum(1 for v in entries.values() if v.get("changed"))
            rate = n_changed / len(TEST_PROMPTS) * 100 if TEST_PROMPTS else 0
            rates.append(rate)
        summary[layer] = rates
        print(f"{layer:>5d} | " + " | ".join(f"{r:>5.0f}%" for r in rates))

    # Save
    output = {
        "model": MODEL_ID,
        "layers_tested": LAYERS_TO_TEST,
        "coefficients": COEFFICIENTS,
        "baselines": baselines,
        "sweep": {
            str(l): {str(c): v for c, v in cv.items()}
            for l, cv in sweep.items()
        },
        "change_rate_matrix": {str(l): r for l, r in summary.items()},
    }
    path = RESULTS_DIR / "midlayer_steering_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")

    # Find optimal layer/coeff
    best_layer, best_coeff, best_rate = 0, 0, 0
    for layer, rates in summary.items():
        for i, rate in enumerate(rates):
            if rate > best_rate and COEFFICIENTS[i] > 0:
                best_rate = rate
                best_layer = layer
                best_coeff = COEFFICIENTS[i]

    print(f"\nBest: layer={best_layer}, coeff={best_coeff}, change_rate={best_rate:.0f}%")


if __name__ == "__main__":
    main()
