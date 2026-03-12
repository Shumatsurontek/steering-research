"""
Phase 3: Apply Steering Vectors to Qwen3-4B for Calendar Tasks

Tests the effect of steering vectors (from Phase 2) on model generation.
Compares steered vs unsteered outputs on calendar extraction prompts.
"""

import json
import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Test prompts — mix of calendar and ambiguous
TEST_PROMPTS = [
    # Clear calendar intent
    "Crée un rendez-vous demain à 14h avec Marie pour discuter du projet.",
    "Schedule a meeting next Monday at 10am with the team.",
    # Ambiguous — could be calendar or not
    "I need to see Marie tomorrow about the project.",
    "Faut que je vois Antoine la semaine prochaine.",
    # Non-calendar (should NOT be steered into calendar)
    "Quelle est la capitale de la France ?",
    "Explain how transformers work.",
]

SYSTEM_PROMPT = (
    "You are a calendar assistant. When the user wants to schedule something, "
    "extract the event details as JSON with fields: title, date, start_time, "
    "end_time, location, attendees. Today is 2026-03-12 (Thursday). "
    "If the request is not about scheduling, respond normally."
)


def steering_hook(module, input, output, *, vector: torch.Tensor, coeff: float, device: str):
    """Forward hook that adds a scaled steering vector to residual stream."""
    hidden = output[0] if isinstance(output, tuple) else output
    # Add steering vector to ALL token positions
    steered = hidden + coeff * vector.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


def generate_response(
    model, tokenizer, prompt: str, device: str,
    steering_vector: torch.Tensor = None,
    steering_layer: int = None,
    steering_coeff: float = 0.0,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response, optionally with steering."""
    # Format as chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Disable thinking for cleaner output
    if "<|im_start|>assistant" in text and "/no_think" not in text:
        text = text.rstrip()
        if text.endswith("<|im_start|>assistant"):
            text += "\n<|tool_call|>" if "schedule" in prompt.lower() or "rendez" in prompt.lower() else "\n"

    inputs = tokenizer(text, return_tensors="pt").to(device)

    handle = None
    try:
        if steering_vector is not None and steering_layer is not None and steering_coeff != 0:
            # Normalize steering vector to unit norm, then scale by coeff * avg_activation_norm
            vec_normed = steering_vector / (steering_vector.norm() + 1e-8)
            layer = model.model.layers[steering_layer]
            handle = layer.register_forward_hook(
                functools.partial(
                    steering_hook, vector=vec_normed, coeff=steering_coeff, device=device
                )
            )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()

    # Decode only new tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Clean thinking tags if present
    if "<think>" in response:
        parts = response.split("</think>")
        response = parts[-1].strip() if len(parts) > 1 else response
    return response.strip()


def sweep_coefficients(
    model, tokenizer, device: str,
    steering_vectors: dict, top_layers: list[int],
    coefficients: list[float],
) -> dict:
    """Sweep steering coefficients across top layers."""
    results = {}

    for layer_idx in top_layers:
        vec = steering_vectors[f"layer_{layer_idx}"]
        results[layer_idx] = {}

        for coeff in coefficients:
            results[layer_idx][coeff] = []
            print(f"\n  Layer {layer_idx}, coeff={coeff:.1f}:")

            for prompt in TEST_PROMPTS:
                response = generate_response(
                    model, tokenizer, prompt, device,
                    steering_vector=vec,
                    steering_layer=layer_idx,
                    steering_coeff=coeff,
                )
                results[layer_idx][coeff].append({
                    "prompt": prompt,
                    "response": response[:300],  # truncate for readability
                })
                print(f"    [{prompt[:50]}...]")
                print(f"    → {response[:150]}...")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3: STEERING VECTOR APPLICATION — Qwen3-4B")
    print("=" * 60)

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"Device: {device}")

    # --- Load model ---
    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()

    # --- Load steering vectors ---
    vectors_path = RESULTS_DIR / "steering_vectors.pt"
    if not vectors_path.exists():
        print(f"ERROR: {vectors_path} not found. Run Phase 2 first.")
        return
    steering_vectors = torch.load(vectors_path, map_location="cpu", weights_only=True)
    print(f"Loaded {len(steering_vectors)} steering vectors")

    # --- Load layer importance ---
    with open(RESULTS_DIR / "layer_importance.json") as f:
        layer_metrics = json.load(f)
    top_layers = [m["layer"] for m in layer_metrics[:3]]
    print(f"Top layers: {top_layers}")

    # --- Baseline (no steering) ---
    print("\n" + "─" * 60)
    print("BASELINE (no steering)")
    print("─" * 60)
    baselines = []
    for prompt in TEST_PROMPTS:
        response = generate_response(model, tokenizer, prompt, device)
        baselines.append({"prompt": prompt, "response": response[:300]})
        print(f"\n  [{prompt[:60]}]")
        print(f"  → {response[:200]}")

    # --- Sweep coefficients ---
    print("\n" + "─" * 60)
    print("STEERING SWEEP")
    print("─" * 60)
    # Use moderate coefficients — too high causes degeneration
    coefficients = [0.0, 5.0, 15.0, 30.0, 50.0]
    sweep_results = sweep_coefficients(
        model, tokenizer, device, steering_vectors,
        top_layers=[top_layers[0]],  # focus on best layer
        coefficients=coefficients,
    )

    # --- Save all results ---
    output = {
        "model": MODEL_ID,
        "device": device,
        "top_layers": top_layers,
        "baselines": baselines,
        "sweep": {
            str(layer): {
                str(coeff): entries
                for coeff, entries in coeff_results.items()
            }
            for layer, coeff_results in sweep_results.items()
        },
    }

    output_path = RESULTS_DIR / "steering_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")

    print("\n" + "=" * 60)
    print("Phase 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
