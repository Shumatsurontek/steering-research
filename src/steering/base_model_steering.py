"""
Phase 3b: Steering on Base (Non-Instruct) Qwen3-4B

Tests if steering vectors are effective on the base model where
instruction tuning hasn't saturated the calendar behavior.
Compares with instruct model results.
"""

import json
import functools
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BASE_MODEL_ID = "Qwen/Qwen3-4B"
INSTRUCT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# For base model: use completion-style prompts (no chat template)
BASE_PROMPTS = [
    ("calendar_fr", "Create a calendar event: Crée un rendez-vous demain à 14h avec Marie pour discuter du projet.\nEvent JSON:"),
    ("calendar_en", "Create a calendar event: Schedule a meeting next Monday at 10am with the team.\nEvent JSON:"),
    ("ambiguous_en", "Create a calendar event: I need to see Marie tomorrow about the project.\nEvent JSON:"),
    ("ambiguous_fr", "Create a calendar event: Faut que je vois Antoine la semaine prochaine.\nEvent JSON:"),
    ("non_calendar_fr", "Question: Quelle est la capitale de la France?\nAnswer:"),
    ("non_calendar_en", "Question: Explain how transformers work.\nAnswer:"),
]

# Contrastive prompts for base model (different format than instruct)
CALENDAR_PROMPTS_BASE = [
    "The calendar event details are: meeting with Marie tomorrow at 2pm to discuss the project.",
    "Schedule: team standup every Monday at 9:30am in the main conference room.",
    "Event: sprint retrospective on March 24th from 10am to 11:30am with Jean-Pierre and Fatima.",
    "Appointment: dentist on Thursday at 8am.",
    "Booking: lunch with Antoine on Friday at noon at Le Petit Zinc restaurant.",
    "Reminder: submit quarterly report by end of day March 31st.",
    "Calendar invite: architecture review on April 2nd at 2pm in Room 301 with David and Emma.",
    "Meeting request: product demo on March 28th from 10am to 11:30am with the sales team.",
    "Block time: brainstorming session on March 24th from 2pm to 4pm in Innovation room.",
    "Schedule call: sync with New York office on March 20th at 3pm CET.",
]

NEUTRAL_PROMPTS_BASE = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Pythagorean theorem states that a² + b² = c² for right triangles.",
    "Python is a high-level programming language known for its readability.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "Machine learning is a subset of artificial intelligence that learns from data.",
    "The solar system has eight planets orbiting the Sun.",
    "DNA stands for deoxyribonucleic acid, which carries genetic information.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "Quantum computing uses qubits that can exist in superposition states.",
]

LAYERS_TO_TEST = [10, 15, 18, 20, 22, 25, 30, 35]
COEFFICIENTS = [0.0, 10.0, 30.0, 60.0, 100.0, 200.0]


def gather_hook(module, input, output, *, cache, layer_idx):
    hidden = output[0] if isinstance(output, tuple) else output
    cache[layer_idx] = hidden.detach().cpu()


def steering_hook(module, input, output, *, vector, coeff):
    hidden = output[0] if isinstance(output, tuple) else output
    steered = hidden + coeff * vector.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


def extract_base_activations(model, tokenizer, prompts, device):
    """Extract last-token activations from base model."""
    n_layers = model.config.num_hidden_layers
    all_acts = {i: [] for i in range(n_layers)}

    for prompt in prompts:
        cache = {}
        handles = []
        try:
            for i, layer in enumerate(model.model.layers):
                handles.append(layer.register_forward_hook(
                    functools.partial(gather_hook, cache=cache, layer_idx=i)
                ))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
            for i in range(n_layers):
                all_acts[i].append(cache[i][0, -1, :])
        finally:
            for h in handles:
                h.remove()

    return {i: torch.stack(acts) for i, acts in all_acts.items()}


def generate_base(model, tokenizer, prompt, device, vector=None, layer=None, coeff=0.0):
    """Generate from base model with optional steering."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    handle = None
    try:
        if vector is not None and layer is not None and coeff != 0:
            vec_normed = vector / (vector.norm() + 1e-8)
            handle = model.model.layers[layer].register_forward_hook(
                functools.partial(steering_hook, vector=vec_normed, coeff=coeff)
            )
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0,
            )
    finally:
        if handle:
            handle.remove()

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def has_calendar_signal(text: str) -> float:
    """Score how much the response contains calendar-related content (0-1)."""
    text_lower = text.lower()
    calendar_keywords = [
        "date", "time", "meeting", "schedule", "calendar", "event",
        "rendez-vous", "réunion", "heure", "jour", "appointment",
        "attendees", "location", "title", "pm", "am",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi",
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "{", "}", ":", "\"title\"", "\"date\"",
    ]
    hits = sum(1 for kw in calendar_keywords if kw in text_lower)
    return min(1.0, hits / 5.0)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BASE MODEL STEERING — Qwen3-4B (non-instruct)")
    print("=" * 60)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Device: {device}")

    # --- Load base model ---
    print(f"\nLoading base model: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=dtype,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Loaded: {n_layers} layers, hidden_dim={model.config.hidden_size}")

    # --- Extract contrastive activations from BASE model ---
    print(f"\nExtracting base model activations...")
    print(f"  Calendar prompts: {len(CALENDAR_PROMPTS_BASE)}")
    cal_acts = extract_base_activations(model, tokenizer, CALENDAR_PROMPTS_BASE, device)
    print(f"  Neutral prompts: {len(NEUTRAL_PROMPTS_BASE)}")
    neu_acts = extract_base_activations(model, tokenizer, NEUTRAL_PROMPTS_BASE, device)

    # --- Compute base model steering vectors ---
    print("\nComputing base model steering vectors...")
    base_vectors = {}
    layer_norms = {}
    for i in range(n_layers):
        diff = cal_acts[i].mean(dim=0) - neu_acts[i].mean(dim=0)
        base_vectors[i] = diff
        layer_norms[i] = diff.norm().item()

    # Print layer norms
    sorted_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)
    print("\nTop-10 layers by L2 norm (base model):")
    for rank, (layer, norm) in enumerate(sorted_layers[:10], 1):
        print(f"  {rank}. Layer {layer}: {norm:.2f}")

    # Save base vectors
    torch.save(
        {f"layer_{i}": v for i, v in base_vectors.items()},
        RESULTS_DIR / "base_steering_vectors.pt"
    )

    # --- Baseline generation (no steering) ---
    print("\n--- BASELINE (no steering, base model) ---")
    baselines = {}
    for tag, prompt in BASE_PROMPTS:
        resp = generate_base(model, tokenizer, prompt, device)
        cal_score = has_calendar_signal(resp)
        baselines[tag] = {"response": resp[:300], "cal_score": round(cal_score, 2)}
        print(f"  [{tag}] cal_score={cal_score:.2f}: {resp[:120]}...")

    # --- Steering sweep ---
    print("\n--- BASE MODEL STEERING SWEEP ---")
    sweep = {}
    for layer in LAYERS_TO_TEST:
        vec = base_vectors[layer]
        sweep[layer] = {}
        print(f"\n  Layer {layer} (norm={layer_norms[layer]:.1f}):")

        for coeff in COEFFICIENTS:
            sweep[layer][coeff] = {}
            cal_scores = []

            for tag, prompt in BASE_PROMPTS:
                resp = generate_base(model, tokenizer, prompt, device, vec, layer, coeff)
                cal_score = has_calendar_signal(resp)
                cal_scores.append(cal_score)
                baseline_score = baselines[tag]["cal_score"]
                delta = cal_score - baseline_score

                sweep[layer][coeff][tag] = {
                    "response": resp[:300],
                    "cal_score": round(cal_score, 2),
                    "delta": round(delta, 2),
                }

            avg_score = sum(cal_scores) / len(cal_scores)
            avg_baseline = sum(b["cal_score"] for b in baselines.values()) / len(baselines)
            print(f"    coeff={coeff:>6.1f}: avg_cal_score={avg_score:.2f} "
                  f"(baseline={avg_baseline:.2f}, delta={avg_score - avg_baseline:+.2f})")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CALENDAR SCORE MATRIX (avg across prompts, 0=no calendar, 1=strong calendar)")
    print("=" * 60)
    header = f"{'Layer':>5s} | " + " | ".join(f"{c:>6.0f}" for c in COEFFICIENTS)
    print(header)
    print("─" * len(header))

    score_matrix = {}
    for layer in LAYERS_TO_TEST:
        if layer not in sweep:
            continue
        scores = []
        for coeff in COEFFICIENTS:
            entries = sweep[layer].get(coeff, {})
            avg = sum(e["cal_score"] for e in entries.values()) / len(entries) if entries else 0
            scores.append(avg)
        score_matrix[layer] = scores
        print(f"{layer:>5d} | " + " | ".join(f"{s:>6.2f}" for s in scores))

    # --- Save ---
    output = {
        "base_model": BASE_MODEL_ID,
        "instruct_model": INSTRUCT_MODEL_ID,
        "layers_tested": LAYERS_TO_TEST,
        "coefficients": COEFFICIENTS,
        "layer_norms": {str(k): round(v, 2) for k, v in sorted_layers},
        "baselines": baselines,
        "sweep": {
            str(l): {str(c): v for c, v in cv.items()}
            for l, cv in sweep.items()
        },
        "score_matrix": {str(l): [round(s, 3) for s in scores]
                         for l, scores in score_matrix.items()},
    }
    path = RESULTS_DIR / "base_model_steering_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
