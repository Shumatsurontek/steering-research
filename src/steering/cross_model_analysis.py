"""
Cross-Model Steering Vector Analysis.

Compares the geometry of steering vectors across 3 architectures:
- Qwen3-0.6B (Transformer, 896d, 28 layers)
- Llama-3.2-3B-Instruct (Transformer, 3072d, 28 layers)
- LFM2.5-1.2B-Instruct (Hybrid SSM+Attention, 2048d, 16 layers)

Key questions:
1. Is the domain similarity structure preserved across architectures?
   (Do all models agree on which domains are "close" vs "far"?)
2. Can steering vectors transfer directly between same-dim models?
3. Does a linear projection (Procrustes) enable cross-architecture transfer?

Usage:
    python -m src.steering.cross_model_analysis
"""

import json
from pathlib import Path

import torch
import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MODELS = {
    "qwen3_0.6b": {"mid_layer": 14, "n_layers": 28},
    "llama3_3b": {"mid_layer": 14, "n_layers": 28},
    "lfm2_1.2b": {"mid_layer": 8, "n_layers": 16},
}

DOMAINS = [
    "math", "physics", "chemistry", "law", "engineering", "economics",
    "health", "psychology", "business", "biology", "philosophy",
    "computer_science", "history", "other",
]


def load_vectors(model_key):
    path = RESULTS_DIR / f"mmlu_pro_vectors_{model_key}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def cosine_matrix(vectors, layer):
    """Compute 14×14 pairwise cosine similarity matrix."""
    vecs = [vectors[d][layer] for d in DOMAINS]
    n = len(vecs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos = torch.nn.functional.cosine_similarity(
                vecs[i].unsqueeze(0).float(), vecs[j].unsqueeze(0).float()
            ).item()
            mat[i, j] = cos
    return mat


def upper_triangle(mat):
    """Extract upper triangle (excluding diagonal) as flat array."""
    n = mat.shape[0]
    indices = np.triu_indices(n, k=1)
    return mat[indices]


def main():
    print("=" * 60)
    print("CROSS-MODEL STEERING VECTOR GEOMETRY ANALYSIS")
    print("=" * 60)

    # Step 1: Load vectors and compute cosine matrices at mid-layer
    matrices = {}
    hidden_dims = {}
    for model_key, cfg in MODELS.items():
        vecs = load_vectors(model_key)
        layer = cfg["mid_layer"]
        matrices[model_key] = cosine_matrix(vecs, layer)
        # Get hidden dimension from any vector
        sample_vec = vecs[DOMAINS[0]][layer]
        hidden_dims[model_key] = sample_vec.shape[0]
        print(f"\n{model_key}: hidden_dim={sample_vec.shape[0]}, "
              f"mid_layer={layer}")

    # Step 2: Compare geometry — correlate upper triangles
    print(f"\n{'='*60}")
    print("GEOMETRY COMPARISON (Spearman correlation of cosine matrices)")
    print(f"{'='*60}")
    print("If high (>0.7): models share similar domain geometry")
    print("If low (<0.3): geometry is architecture-specific\n")

    model_keys = list(MODELS.keys())
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            m1, m2 = model_keys[i], model_keys[j]
            ut1 = upper_triangle(matrices[m1])
            ut2 = upper_triangle(matrices[m2])
            rho, pval = stats.spearmanr(ut1, ut2)
            pearson_r, _ = stats.pearsonr(ut1, ut2)
            print(f"  {m1} vs {m2}:")
            print(f"    Spearman ρ = {rho:.4f} (p={pval:.2e})")
            print(f"    Pearson  r = {pearson_r:.4f}")
            print()

    # Step 3: Per-domain agreement — rank domains by avg cosine to others
    print(f"{'='*60}")
    print("DOMAIN RANKING COMPARISON (avg cosine to all other domains)")
    print(f"{'='*60}\n")

    rankings = {}
    for model_key in model_keys:
        mat = matrices[model_key]
        avg_cos = []
        for i in range(len(DOMAINS)):
            # Average cosine to all other domains (excluding self)
            others = [mat[i, j] for j in range(len(DOMAINS)) if j != i]
            avg_cos.append(np.mean(others))
        # Rank: most similar first
        ranked = sorted(range(len(DOMAINS)), key=lambda i: -avg_cos[i])
        rankings[model_key] = [DOMAINS[i] for i in ranked]

        print(f"  {model_key} (most → least similar to others):")
        for rank, idx in enumerate(ranked):
            print(f"    {rank+1:2d}. {DOMAINS[idx]:<18s} avg_cos={avg_cos[idx]:.4f}")
        print()

    # Step 4: Rank correlation of domain orderings
    print(f"{'='*60}")
    print("DOMAIN RANK CORRELATION (Kendall τ)")
    print(f"{'='*60}\n")

    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            m1, m2 = model_keys[i], model_keys[j]
            # Convert rankings to numerical ranks
            rank1 = [rankings[m1].index(d) for d in DOMAINS]
            rank2 = [rankings[m2].index(d) for d in DOMAINS]
            tau, pval = stats.kendalltau(rank1, rank2)
            print(f"  {m1} vs {m2}: τ = {tau:.4f} (p={pval:.4f})")
    print()

    # Step 5: Focus on our 3 target domains (math, law, history)
    print(f"{'='*60}")
    print("TARGET DOMAINS: math, law, history — pairwise cosines")
    print(f"{'='*60}\n")

    target = ["math", "law", "history"]
    target_idx = [DOMAINS.index(d) for d in target]

    for model_key in model_keys:
        mat = matrices[model_key]
        print(f"  {model_key}:")
        for a in range(len(target)):
            for b in range(a + 1, len(target)):
                ia, ib = target_idx[a], target_idx[b]
                print(f"    {target[a]:>8s} — {target[b]:<8s}: {mat[ia, ib]:.4f}")
        print()

    # Step 6: Hidden dimension analysis — can we do direct transfer?
    print(f"{'='*60}")
    print("DIRECT TRANSFER FEASIBILITY")
    print(f"{'='*60}\n")

    for model_key, dim in hidden_dims.items():
        print(f"  {model_key}: {dim}d")
    print()

    dims = list(hidden_dims.values())
    if len(set(dims)) == len(dims):
        print("  → All dimensions differ. Direct vector transfer impossible.")
        print("    Options: Procrustes alignment or train a linear projection.")
    else:
        same_dim = [(k1, k2) for i, (k1, d1) in enumerate(hidden_dims.items())
                     for k2, d2 in list(hidden_dims.items())[i+1:]
                     if d1 == d2]
        if same_dim:
            print(f"  → Same dimension pairs: {same_dim}")
            print("    Direct vector transfer possible between these!")

    # Step 7: Norm scaling analysis
    print(f"\n{'='*60}")
    print("NORM SCALING AT MID-LAYER (target domains)")
    print(f"{'='*60}\n")

    for model_key in model_keys:
        vecs = load_vectors(model_key)
        layer = MODELS[model_key]["mid_layer"]
        print(f"  {model_key} (L{layer}):")
        for domain in target:
            norm = vecs[domain][layer].norm().item()
            print(f"    {domain:>8s}: L2={norm:.2f}")
        print()

    # Save analysis
    analysis = {
        "hidden_dims": hidden_dims,
        "geometry_correlations": {},
        "domain_rankings": rankings,
        "target_pairwise": {},
    }

    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            m1, m2 = model_keys[i], model_keys[j]
            ut1 = upper_triangle(matrices[m1])
            ut2 = upper_triangle(matrices[m2])
            rho, _ = stats.spearmanr(ut1, ut2)
            pearson_r, _ = stats.pearsonr(ut1, ut2)
            analysis["geometry_correlations"][f"{m1}_vs_{m2}"] = {
                "spearman": round(rho, 4),
                "pearson": round(pearson_r, 4),
            }

    for model_key in model_keys:
        mat = matrices[model_key]
        pw = {}
        for a in range(len(target)):
            for b in range(a + 1, len(target)):
                ia, ib = target_idx[a], target_idx[b]
                pw[f"{target[a]}_vs_{target[b]}"] = round(mat[ia, ib], 4)
        analysis["target_pairwise"][model_key] = pw

    out_path = RESULTS_DIR / "cross_model_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
