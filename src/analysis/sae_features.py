"""
Phase 2: SAE-Inspired Feature Extraction on Qwen3-4B-Instruct-2507

Extracts residual stream activations using contrastive prompt pairs
(calendar vs neutral) to identify steering-relevant layers and directions.
Applies logit lens to interpret what the difference vectors encode.
"""

import json
import os
import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Contrastive prompt pairs
CALENDAR_PROMPTS = [
    "Crée un rendez-vous demain à 14h avec Marie pour discuter du projet.",
    "Schedule a meeting tomorrow at 2pm with the marketing team.",
    "Bloque le créneau de 10h à 11h30 lundi pour la rétrospective.",
    "Can you set up a call with Jean-Pierre on Friday at 3pm?",
    "Planifie une réunion d'équipe mercredi prochain dans la salle Confluence.",
    "Book a 30-minute slot for a 1:1 with Fatima next Tuesday morning.",
    "Ajoute un événement le 24 mars à 16h : revue trimestrielle.",
    "Reserve the conference room for a demo on March 28th at 10am.",
    "Cale-moi un déjeuner avec Antoine vendredi midi.",
    "Create a recurring weekly standup every Monday at 9:30am.",
]

NEUTRAL_PROMPTS = [
    "Quelle est la capitale de la France ?",
    "Explain how photosynthesis works in simple terms.",
    "Écris un poème sur la pluie d'automne.",
    "What are the main differences between Python and Rust?",
    "Résume l'histoire de la Révolution française en 3 phrases.",
    "How does a neural network learn through backpropagation?",
    "Donne-moi une recette de tarte aux pommes.",
    "What is the speed of light in vacuum?",
    "Explique le théorème de Pythagore à un enfant de 10 ans.",
    "List the planets of the solar system in order.",
]


# ---------------------------------------------------------------------------
# Activation extraction via hooks
# ---------------------------------------------------------------------------

def gather_residual_hook(module, input, output, *, cache: dict, layer_idx: int):
    """Forward hook to capture residual stream output at a given layer."""
    # output is a tuple; first element is the hidden state
    hidden = output[0] if isinstance(output, tuple) else output
    cache[layer_idx] = hidden.detach().cpu()


def extract_activations(model, tokenizer, prompts: list[str], device: str) -> dict[int, torch.Tensor]:
    """
    Extract last-token residual stream activations for each layer.

    Returns: dict[layer_idx -> Tensor of shape (n_prompts, hidden_dim)]
    """
    n_layers = model.config.num_hidden_layers
    all_activations = {i: [] for i in range(n_layers)}

    for prompt in prompts:
        cache = {}
        handles = []

        try:
            # Register hooks on all decoder layers
            for i, layer in enumerate(model.model.layers):
                handle = layer.register_forward_hook(
                    functools.partial(gather_residual_hook, cache=cache, layer_idx=i)
                )
                handles.append(handle)

            # Forward pass
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)

            # Collect last-token activation from each layer
            for i in range(n_layers):
                last_token_act = cache[i][0, -1, :]  # (hidden_dim,)
                all_activations[i].append(last_token_act)

        finally:
            for h in handles:
                h.remove()

    # Stack into tensors: (n_prompts, hidden_dim)
    return {i: torch.stack(acts) for i, acts in all_activations.items()}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_steering_vectors(
    cal_acts: dict[int, torch.Tensor],
    neu_acts: dict[int, torch.Tensor],
) -> dict[int, dict]:
    """
    Compute per-layer steering vectors (mean difference) and metrics.
    """
    results = {}
    for layer_idx in cal_acts:
        cal_mean = cal_acts[layer_idx].mean(dim=0)  # (hidden_dim,)
        neu_mean = neu_acts[layer_idx].mean(dim=0)

        diff = cal_mean - neu_mean
        l2_norm = diff.norm().item()

        cos_sim = F.cosine_similarity(
            cal_mean.unsqueeze(0), neu_mean.unsqueeze(0)
        ).item()

        results[layer_idx] = {
            "steering_vector": diff,
            "cal_mean": cal_mean,
            "neu_mean": neu_mean,
            "l2_norm": l2_norm,
            "cosine_similarity": cos_sim,
            "cosine_distance": 1.0 - cos_sim,
        }

    return results


def logit_lens(model, steering_results: dict, top_k: int = 20) -> dict[int, dict]:
    """
    Project steering vectors through the unembedding matrix to find
    which tokens each vector promotes/suppresses.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Get unembedding weights: lm_head.weight is (vocab_size, hidden_dim)
    unembed = model.lm_head.weight.detach().float().cpu()

    # If model has a final layer norm, apply it
    if hasattr(model.model, "norm"):
        norm = model.model.norm
        # We'll apply the norm weights manually for efficiency
        if hasattr(norm, "weight"):
            norm_weight = norm.weight.detach().float().cpu()
        else:
            norm_weight = None
    else:
        norm_weight = None

    results = {}
    for layer_idx, data in steering_results.items():
        vec = data["steering_vector"].float()

        # Apply RMSNorm weights if available (element-wise multiply)
        if norm_weight is not None:
            vec_normed = vec * norm_weight
        else:
            vec_normed = vec

        # Project through unembedding: (vocab_size,)
        logits = unembed @ vec_normed

        # Top promoted tokens
        top_vals, top_ids = logits.topk(top_k)
        promoted = [
            {"token": tokenizer.decode([tid]), "token_id": tid.item(), "logit": val.item()}
            for tid, val in zip(top_ids, top_vals)
        ]

        # Top suppressed tokens
        bot_vals, bot_ids = logits.topk(top_k, largest=False)
        suppressed = [
            {"token": tokenizer.decode([tid]), "token_id": tid.item(), "logit": val.item()}
            for tid, val in zip(bot_ids, bot_vals)
        ]

        results[layer_idx] = {"promoted": promoted, "suppressed": suppressed}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 2: SAE-INSPIRED FEATURE EXTRACTION — Qwen3-4B")
    print("=" * 60)

    # --- Device selection ---
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("Using Apple MPS")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU (this will be slow)")

    # --- Load model ---
    print(f"\nLoading {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map=device if device != "mps" else None,
            low_cpu_mem_usage=True,
        )
        if device == "mps":
            model = model.to(device)
        model.eval()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"\nOOM Error: {e}")
        print("Try: pip install bitsandbytes && use load_in_4bit=True")
        return

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Model loaded: {n_layers} layers, hidden_dim={hidden_dim}")

    # --- Extract activations ---
    print(f"\nExtracting activations for {len(CALENDAR_PROMPTS)} calendar prompts...")
    cal_acts = extract_activations(model, tokenizer, CALENDAR_PROMPTS, device)

    print(f"Extracting activations for {len(NEUTRAL_PROMPTS)} neutral prompts...")
    neu_acts = extract_activations(model, tokenizer, NEUTRAL_PROMPTS, device)

    # --- Compute steering vectors ---
    print("\nComputing per-layer steering vectors...")
    steering_results = compute_steering_vectors(cal_acts, neu_acts)

    # --- Layer importance ranking ---
    layer_metrics = []
    for idx, data in steering_results.items():
        layer_metrics.append({
            "layer": idx,
            "l2_norm": round(data["l2_norm"], 4),
            "cosine_similarity": round(data["cosine_similarity"], 6),
            "cosine_distance": round(data["cosine_distance"], 6),
        })

    # Sort by L2 norm (highest = most discriminative)
    layer_metrics.sort(key=lambda x: x["l2_norm"], reverse=True)
    for i, m in enumerate(layer_metrics):
        m["rank"] = i + 1

    print("\n" + "─" * 60)
    print("LAYER IMPORTANCE (by L2 norm of steering vector)")
    print("─" * 60)
    print(f"{'Rank':>4s}  {'Layer':>5s}  {'L2 Norm':>10s}  {'Cos Dist':>10s}")
    print("─" * 40)
    for m in layer_metrics[:10]:
        print(f"{m['rank']:>4d}  {m['layer']:>5d}  {m['l2_norm']:>10.4f}  {m['cosine_distance']:>10.6f}")

    # Save layer importance
    with open(RESULTS_DIR / "layer_importance.json", "w") as f:
        json.dump(layer_metrics, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'layer_importance.json'}")

    # --- Save steering vectors ---
    vectors_to_save = {
        f"layer_{idx}": data["steering_vector"]
        for idx, data in steering_results.items()
    }
    torch.save(vectors_to_save, RESULTS_DIR / "steering_vectors.pt")
    print(f"Saved: {RESULTS_DIR / 'steering_vectors.pt'}")

    # --- Logit lens on top-3 layers ---
    top_3_layers = [m["layer"] for m in layer_metrics[:3]]
    print(f"\nApplying logit lens to top-3 layers: {top_3_layers}")

    top_3_results = {
        idx: steering_results[idx] for idx in top_3_layers
    }
    lens_results = logit_lens(model, top_3_results)

    print("\n" + "─" * 60)
    print("LOGIT LENS — Top promoted/suppressed tokens per layer")
    print("─" * 60)
    for layer_idx in top_3_layers:
        print(f"\n  Layer {layer_idx}:")
        print(f"    Promoted:  {[t['token'] for t in lens_results[layer_idx]['promoted'][:10]]}")
        print(f"    Suppressed: {[t['token'] for t in lens_results[layer_idx]['suppressed'][:10]]}")

    # Save logit lens (convert for JSON serialization)
    lens_json = {str(k): v for k, v in lens_results.items()}
    with open(RESULTS_DIR / "logit_lens_results.json", "w") as f:
        json.dump(lens_json, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {RESULTS_DIR / 'logit_lens_results.json'}")

    print("\n" + "=" * 60)
    print("Phase 2 complete.")
    print("=" * 60)

    return steering_results, layer_metrics, lens_results


if __name__ == "__main__":
    main()
