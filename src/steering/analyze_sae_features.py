"""
Analyze SAE features by domain: identify which features activate
differentially on math, law, and history prompts.

Compares SAE-based domain decomposition to our contrastive steering vectors.

Usage:
    python -m src.steering.analyze_sae_features
    python -m src.steering.analyze_sae_features --model Qwen/Qwen3-4B --layer 18
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MODEL_PRESETS = {
    "Qwen/Qwen3-0.6B": {"layer": 14, "sae_dir": "sae_qwen3_0.6b_L14_8x",
                          "vectors": "mmlu_pro_vectors_qwen3_0.6b.pt"},
    "Qwen/Qwen3-4B": {"layer": 18, "sae_dir": "sae_qwen3_4b_L18_8x",
                        "vectors": "mmlu_pro_vectors_qwen3_4b.pt"},
}

# Domain-specific prompts for activation analysis
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


def compute_domain_activations(sae, model, prompts, hook_name, stop_layer=None):
    """
    Compute mean SAE feature activations for a list of prompts.

    Args:
        sae: Trained SAE model (from SAE.from_pretrained or SAE.load_from_disk)
        model: HookedTransformer model
        prompts: list of strings
        hook_name: TransformerLens hook point (e.g. "blocks.14.hook_resid_post")
        stop_layer: stop forward pass after this layer (default: inferred from hook_name)

    Returns:
        torch.Tensor of shape (n_prompts, n_features) with mean feature
        activations per prompt.
    """
    if stop_layer is None:
        # Extract layer number from hook_name and add 1
        stop_layer = int(hook_name.split(".")[1]) + 1
    all_acts = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, stop_at_layer=stop_layer)
        residual = cache[hook_name]  # (1, seq_len, d_in)

        # Encode each position through SAE, then average over sequence
        # This preserves per-token sparsity before aggregating
        flat = residual.squeeze(0)  # (seq_len, d_in)
        feat_acts = sae.encode(flat)  # (seq_len, d_sae)
        mean_acts = feat_acts.mean(dim=0)  # (d_sae,)
        all_acts.append(mean_acts.detach().cpu())

    return torch.stack(all_acts)  # (n_prompts, d_sae)


def find_domain_specific_features(activations_by_domain, top_k=20):
    """
    Find features that activate most differentially for each domain.

    For each domain, compute: mean_activation_in_domain - mean_activation_other_domains
    Return the top_k most differential features per domain.
    """
    domains = list(activations_by_domain.keys())
    all_acts = {d: acts.mean(dim=0) for d, acts in activations_by_domain.items()}

    results = {}
    for domain in domains:
        domain_mean = all_acts[domain]
        other_mean = torch.stack([all_acts[d] for d in domains if d != domain]).mean(dim=0)
        differential = domain_mean - other_mean

        top_indices = differential.topk(top_k).indices.tolist()
        top_values = differential.topk(top_k).values.tolist()

        results[domain] = {
            "top_features": top_indices,
            "differential_activation": top_values,
            "mean_activation": domain_mean[top_indices].tolist(),
        }

    return results


def compare_with_contrastive_vectors(sae, contrastive_vectors, domain, layer):
    """
    Project contrastive steering vectors into SAE feature space.
    This shows which SAE features align with our contrastive vectors.
    """
    vec = contrastive_vectors[domain][layer]
    vec = vec.to(sae.W_enc.device, dtype=sae.W_enc.dtype)

    # Project into SAE space: feature_acts = ReLU(vec @ W_enc + b_enc)
    projected = torch.nn.functional.relu(vec @ sae.W_enc + sae.b_enc)

    top_k = 20
    top_indices = projected.topk(top_k).indices.tolist()
    top_values = projected.topk(top_k).values.tolist()

    return {
        "top_features": top_indices,
        "activation_strength": top_values,
        "n_active": (projected > 0).sum().item(),
        "total_features": projected.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE features by domain")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Hook layer (default: from preset)")
    parser.add_argument("--sae_dir", type=str, default=None,
                        help="SAE directory name in results/ (default: from preset)")
    args = parser.parse_args()

    preset = MODEL_PRESETS.get(args.model, MODEL_PRESETS["Qwen/Qwen3-0.6B"])
    layer = args.layer if args.layer is not None else preset["layer"]
    sae_dir = args.sae_dir or preset["sae_dir"]
    hook_name = f"blocks.{layer}.hook_resid_post"
    sae_path = RESULTS_DIR / sae_dir
    vec_file = preset.get("vectors", "")
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    print("=" * 60)
    print(f"SAE FEATURE ANALYSIS — {args.model} Layer {layer}")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model
    print(f"\nLoading model on {device}...")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, device=device
    )

    # Load SAE
    print(f"Loading SAE from {sae_path}...")
    sae = SAE.load_from_disk(str(sae_path), device=device)
    print(f"  SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # Compute activations per domain
    activations_by_domain = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        print(f"\n  Computing activations for {domain} ({len(prompts)} prompts)...")
        acts = compute_domain_activations(sae, model, prompts, hook_name)
        activations_by_domain[domain] = acts
        print(f"    Shape: {acts.shape}, Mean activation: {acts.mean():.4f}")

    # Find domain-specific features
    print("\n" + "=" * 60)
    print("DOMAIN-SPECIFIC FEATURES")
    print("=" * 60)
    domain_features = find_domain_specific_features(activations_by_domain)

    for domain, info in domain_features.items():
        print(f"\n  [{domain.upper()}] Top 10 differential features:")
        for i in range(10):
            feat_id = info["top_features"][i]
            diff = info["differential_activation"][i]
            mean = info["mean_activation"][i]
            print(f"    Feature {feat_id:5d}: diff={diff:.4f}, mean={mean:.4f}")

    # Compare with contrastive vectors
    print("\n" + "=" * 60)
    print("CONTRASTIVE VECTOR → SAE PROJECTION")
    print("=" * 60)

    vec_path = RESULTS_DIR / vec_file
    if vec_path.exists():
        vectors = torch.load(vec_path, map_location="cpu", weights_only=True)
        for domain in ["math", "law", "history"]:
            proj = compare_with_contrastive_vectors(sae, vectors, domain, layer)
            print(f"\n  [{domain.upper()}] Contrastive vector activates {proj['n_active']}/{proj['total_features']} SAE features")
            print(f"    Top 5 features: {proj['top_features'][:5]}")
            print(f"    Top 5 strengths: {[f'{v:.3f}' for v in proj['activation_strength'][:5]]}")

        # Check overlap between domain-specific and contrastive-projected features
        print("\n" + "=" * 60)
        print("OVERLAP ANALYSIS")
        print("=" * 60)
        for domain in ["math", "law", "history"]:
            proj = compare_with_contrastive_vectors(sae, vectors, domain, layer)
            domain_top = set(domain_features[domain]["top_features"])
            contrastive_top = set(proj["top_features"])
            overlap = domain_top & contrastive_top
            print(f"  [{domain}] Domain top-20 ∩ Contrastive top-20: {len(overlap)} features — {overlap if overlap else '∅'}")
    else:
        print(f"  SKIP: contrastive vectors not found at {vec_path}")
        print(f"  (Run MMLU-Pro benchmark first to generate contrastive vectors for this model)")

    # Save results
    results = {
        "domain_features": {
            d: {k: v if not isinstance(v, list) else v
                for k, v in info.items()}
            for d, info in domain_features.items()
        },
        "sae_path": str(sae_path),
        "model": args.model,
        "hook": hook_name,
        "n_features": sae.cfg.d_sae,
    }
    out_path = RESULTS_DIR / f"sae_domain_analysis_{model_short}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
