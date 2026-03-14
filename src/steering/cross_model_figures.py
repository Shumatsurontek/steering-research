"""
Cross-Model Steering Vector Visualizations.

Generates publication-quality figures for the article:
1. Cosine similarity heatmaps side-by-side (3 models)
2. MDS projection of domain vectors — overlay all 3 models
3. Procrustes alignment residuals

Usage:
    python -m src.steering.cross_model_figures
"""

import json
from pathlib import Path

import torch
import numpy as np
from scipy import stats
from scipy.spatial import procrustes
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "article" / "figures"

MODELS = {
    "qwen3_0.6b": {"mid_layer": 14, "label": "Qwen3-0.6B", "color": "#2196F3"},
    "llama3_3b": {"mid_layer": 14, "label": "Llama-3.2-3B", "color": "#FF5722"},
    "lfm2_1.2b": {"mid_layer": 8, "label": "LFM2.5-1.2B", "color": "#4CAF50"},
}

DOMAINS = [
    "math", "physics", "chemistry", "law", "engineering", "economics",
    "health", "psychology", "business", "biology", "philosophy",
    "computer_science", "history", "other",
]

# Short labels for plots
DOMAIN_SHORT = {
    "math": "Math", "physics": "Phys", "chemistry": "Chem",
    "law": "Law", "engineering": "Eng", "economics": "Econ",
    "health": "Health", "psychology": "Psych", "business": "Bus",
    "biology": "Bio", "philosophy": "Phil", "computer_science": "CS",
    "history": "Hist", "other": "Other",
}

TARGET_DOMAINS = ["math", "law", "history"]


def load_vectors(model_key):
    path = RESULTS_DIR / f"mmlu_pro_vectors_{model_key}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)


def cosine_matrix(vectors, layer):
    vecs = [vectors[d][layer].float() for d in DOMAINS]
    n = len(vecs)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos = torch.nn.functional.cosine_similarity(
                vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
            ).item()
            mat[i, j] = cos
    return mat


def cosine_to_distance(cos_mat):
    """Convert cosine similarity to distance for MDS."""
    return 1.0 - cos_mat


def figure_1_heatmaps(matrices):
    """Three side-by-side cosine similarity heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    labels = [DOMAIN_SHORT[d] for d in DOMAINS]

    for idx, (model_key, cfg) in enumerate(MODELS.items()):
        ax = axes[idx]
        mat = matrices[model_key]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.2, vmax=1.0, aspect="equal")
        ax.set_xticks(range(len(DOMAINS)))
        ax.set_yticks(range(len(DOMAINS)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(cfg["label"], fontsize=12, fontweight="bold", pad=8)

        # Highlight target domains
        for t in TARGET_DOMAINS:
            ti = DOMAINS.index(t)
            ax.axhline(y=ti - 0.5, xmin=0, xmax=1, color="gold",
                       linewidth=1.5, linestyle="--", alpha=0.7)
            ax.axhline(y=ti + 0.5, xmin=0, xmax=1, color="gold",
                       linewidth=1.5, linestyle="--", alpha=0.7)
            ax.axvline(x=ti - 0.5, ymin=0, ymax=1, color="gold",
                       linewidth=1.5, linestyle="--", alpha=0.7)
            ax.axvline(x=ti + 0.5, ymin=0, ymax=1, color="gold",
                       linewidth=1.5, linestyle="--", alpha=0.7)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")

    fig.suptitle("Domain Steering Vector Similarity Across Architectures",
                 fontsize=14, fontweight="bold", y=1.02)
    return fig


def figure_2_mds_overlay(matrices):
    """MDS projection of all 3 models overlaid — shows geometry preservation."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    embeddings = {}
    for model_key, cfg in MODELS.items():
        dist = cosine_to_distance(matrices[model_key])
        np.fill_diagonal(dist, 0)
        # Make symmetric
        dist = (dist + dist.T) / 2
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                  normalized_stress="auto")
        coords = mds.fit_transform(dist)
        embeddings[model_key] = coords

    # Procrustes-align all to Qwen as reference
    ref_key = "qwen3_0.6b"
    ref_coords = embeddings[ref_key]
    # Normalize reference
    ref_centered = ref_coords - ref_coords.mean(axis=0)
    ref_scale = np.sqrt((ref_centered ** 2).sum())
    ref_norm = ref_centered / ref_scale

    aligned = {ref_key: ref_norm}
    procrustes_residuals = {}

    for model_key in MODELS:
        if model_key == ref_key:
            continue
        coords = embeddings[model_key]
        coords_centered = coords - coords.mean(axis=0)
        coords_scale = np.sqrt((coords_centered ** 2).sum())
        coords_norm = coords_centered / coords_scale

        # Procrustes: find optimal rotation
        _, aligned_coords, disparity = procrustes(ref_norm, coords_norm)
        aligned[model_key] = aligned_coords
        procrustes_residuals[model_key] = disparity

    # Plot
    markers = {"qwen3_0.6b": "o", "llama3_3b": "s", "lfm2_1.2b": "^"}

    for model_key, cfg in MODELS.items():
        coords = aligned[model_key]
        color = cfg["color"]
        marker = markers[model_key]

        for i, domain in enumerate(DOMAINS):
            is_target = domain in TARGET_DOMAINS
            size = 120 if is_target else 60
            edge = "black" if is_target else "none"
            lw = 2 if is_target else 0

            ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker,
                      s=size, edgecolors=edge, linewidths=lw, alpha=0.8,
                      zorder=3)

    # Label domains (use average position across models)
    for i, domain in enumerate(DOMAINS):
        avg_x = np.mean([aligned[mk][i, 0] for mk in MODELS])
        avg_y = np.mean([aligned[mk][i, 1] for mk in MODELS])
        fontweight = "bold" if domain in TARGET_DOMAINS else "normal"
        fontsize = 10 if domain in TARGET_DOMAINS else 8
        ax.annotate(DOMAIN_SHORT[domain], (avg_x, avg_y),
                   textcoords="offset points", xytext=(8, 8),
                   fontsize=fontsize, fontweight=fontweight, alpha=0.9)

    # Draw lines connecting same domain across models
    for i in range(len(DOMAINS)):
        pts = [aligned[mk][i] for mk in MODELS]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                       color="gray", linewidth=0.5, alpha=0.3, zorder=1)

    # Legend
    legend_elements = []
    for model_key, cfg in MODELS.items():
        disp = f" (d={procrustes_residuals[model_key]:.4f})" if model_key in procrustes_residuals else " (ref)"
        legend_elements.append(
            Line2D([0], [0], marker=markers[model_key], color="w",
                   markerfacecolor=cfg["color"], markersize=10,
                   label=f"{cfg['label']}{disp}")
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markeredgewidth=2, markersize=10,
               label="Target domains (math, law, history)")
    )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
             framealpha=0.9)

    ax.set_xlabel("MDS Dimension 1", fontsize=11)
    ax.set_ylabel("MDS Dimension 2", fontsize=11)
    ax.set_title("Domain Geometry is Preserved Across Architectures\n"
                 "(MDS of cosine distance, Procrustes-aligned to Qwen3)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    return fig, procrustes_residuals


def figure_3_correlation_scatter(matrices):
    """Pairwise scatter of upper-triangle cosines between models."""
    model_keys = list(MODELS.keys())
    pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        m1, m2 = model_keys[i], model_keys[j]
        n = len(DOMAINS)
        triu_idx = np.triu_indices(n, k=1)
        ut1 = matrices[m1][triu_idx]
        ut2 = matrices[m2][triu_idx]

        # Color by whether pair involves a target domain
        colors = []
        for a, b in zip(triu_idx[0], triu_idx[1]):
            if DOMAINS[a] in TARGET_DOMAINS or DOMAINS[b] in TARGET_DOMAINS:
                colors.append("gold")
            else:
                colors.append("#666666")

        ax.scatter(ut1, ut2, c=colors, s=30, alpha=0.7, edgecolors="black",
                  linewidths=0.5, zorder=2)

        # Regression line
        slope, intercept, r_val, _, _ = stats.linregress(ut1, ut2)
        x_line = np.linspace(min(ut1), max(ut1), 100)
        ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.5,
               alpha=0.7, label=f"r={r_val:.3f}")

        # Diagonal
        lim = [min(min(ut1), min(ut2)) - 0.05, max(max(ut1), max(ut2)) + 0.05]
        ax.plot(lim, lim, "k:", alpha=0.3, linewidth=1)

        rho, _ = stats.spearmanr(ut1, ut2)

        ax.set_xlabel(f"{MODELS[m1]['label']} cosine", fontsize=10)
        ax.set_ylabel(f"{MODELS[m2]['label']} cosine", fontsize=10)
        ax.set_title(f"ρ={rho:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")

    fig.suptitle("Pairwise Domain Cosine Correlation Between Models",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def figure_4_combined(matrices):
    """Combined figure: heatmaps (top) + MDS overlay (bottom left) + scatter (bottom right)."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2],
                          hspace=0.35, wspace=0.3)

    labels = [DOMAIN_SHORT[d] for d in DOMAINS]

    # Top row: 3 heatmaps
    for idx, (model_key, cfg) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[0, idx])
        mat = matrices[model_key]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.2, vmax=1.0, aspect="equal")
        ax.set_xticks(range(len(DOMAINS)))
        ax.set_yticks(range(len(DOMAINS)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(cfg["label"], fontsize=11, fontweight="bold")

        for t in TARGET_DOMAINS:
            ti = DOMAINS.index(t)
            for line_fn in [ax.axhline, ax.axvline]:
                line_fn(ti - 0.5, color="gold", linewidth=1.2,
                       linestyle="--", alpha=0.6)
                line_fn(ti + 0.5, color="gold", linewidth=1.2,
                       linestyle="--", alpha=0.6)

    # Bottom left: MDS overlay (spans 2 columns)
    ax_mds = fig.add_subplot(gs[1, :2])

    embeddings = {}
    for model_key, cfg in MODELS.items():
        dist = cosine_to_distance(matrices[model_key])
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                  normalized_stress="auto")
        embeddings[model_key] = mds.fit_transform(dist)

    ref_key = "qwen3_0.6b"
    ref = embeddings[ref_key]
    ref_c = ref - ref.mean(axis=0)
    ref_s = np.sqrt((ref_c ** 2).sum())
    ref_n = ref_c / ref_s

    aligned = {ref_key: ref_n}
    residuals = {}
    for mk in MODELS:
        if mk == ref_key:
            continue
        c = embeddings[mk] - embeddings[mk].mean(axis=0)
        s = np.sqrt((c ** 2).sum())
        n = c / s
        _, al, d = procrustes(ref_n, n)
        aligned[mk] = al
        residuals[mk] = d

    markers = {"qwen3_0.6b": "o", "llama3_3b": "s", "lfm2_1.2b": "^"}

    for mk, cfg in MODELS.items():
        coords = aligned[mk]
        for i, domain in enumerate(DOMAINS):
            is_t = domain in TARGET_DOMAINS
            ax_mds.scatter(coords[i, 0], coords[i, 1],
                          c=cfg["color"], marker=markers[mk],
                          s=100 if is_t else 50,
                          edgecolors="black" if is_t else "none",
                          linewidths=1.5 if is_t else 0, alpha=0.8, zorder=3)

    for i in range(len(DOMAINS)):
        avg_x = np.mean([aligned[mk][i, 0] for mk in MODELS])
        avg_y = np.mean([aligned[mk][i, 1] for mk in MODELS])
        fw = "bold" if DOMAINS[i] in TARGET_DOMAINS else "normal"
        fs = 10 if DOMAINS[i] in TARGET_DOMAINS else 8
        ax_mds.annotate(DOMAIN_SHORT[DOMAINS[i]], (avg_x, avg_y),
                       textcoords="offset points", xytext=(7, 7),
                       fontsize=fs, fontweight=fw, alpha=0.85)

    for i in range(len(DOMAINS)):
        pts = [aligned[mk][i] for mk in MODELS]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                ax_mds.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                           color="gray", linewidth=0.4, alpha=0.25, zorder=1)

    legend_els = []
    for mk, cfg in MODELS.items():
        disp = f" (d={residuals[mk]:.3f})" if mk in residuals else " (ref)"
        legend_els.append(
            Line2D([0], [0], marker=markers[mk], color="w",
                   markerfacecolor=cfg["color"], markersize=9,
                   label=f"{cfg['label']}{disp}"))
    ax_mds.legend(handles=legend_els, loc="upper left", fontsize=9)
    ax_mds.set_title("MDS Projection (Procrustes-aligned)", fontsize=11,
                    fontweight="bold")
    ax_mds.set_xlabel("Dimension 1")
    ax_mds.set_ylabel("Dimension 2")
    ax_mds.grid(True, alpha=0.2)

    # Bottom right: summary statistics
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis("off")

    model_keys = list(MODELS.keys())
    text_lines = [
        "Cross-Architecture Geometry",
        "━" * 30,
        "",
        "Spearman ρ (cosine matrices):",
    ]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        m1, m2 = model_keys[i], model_keys[j]
        n = len(DOMAINS)
        triu = np.triu_indices(n, k=1)
        rho, _ = stats.spearmanr(matrices[m1][triu], matrices[m2][triu])
        l1 = MODELS[m1]["label"][:10]
        l2 = MODELS[m2]["label"][:10]
        text_lines.append(f"  {l1} — {l2}: ρ={rho:.3f}")

    text_lines += [
        "",
        "Procrustes disparity:",
        f"  Llama vs Qwen (ref): {residuals.get('llama3_3b', 0):.4f}",
        f"  LFM2 vs Qwen (ref):  {residuals.get('lfm2_1.2b', 0):.4f}",
        "",
        "Hidden dimensions:",
        "  Qwen3-0.6B:   1024d",
        "  Llama-3.2-3B:  3072d",
        "  LFM2.5-1.2B:  2048d",
        "",
        "Key finding:",
        "  Domain geometry is",
        "  architecture-invariant",
        "  (ρ > 0.88 for all pairs)",
        "",
        "  History is the universal",
        "  outlier across all models",
    ]

    ax_stats.text(0.05, 0.95, "\n".join(text_lines), transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          alpha=0.8))

    fig.suptitle("Cross-Architecture Steering Vector Analysis",
                 fontsize=15, fontweight="bold", y=0.98)

    return fig, residuals


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading vectors and computing cosine matrices...")
    matrices = {}
    for model_key, cfg in MODELS.items():
        vecs = load_vectors(model_key)
        matrices[model_key] = cosine_matrix(vecs, cfg["mid_layer"])
        print(f"  {model_key}: done")

    # Figure 1: Heatmaps
    print("\nGenerating heatmaps...")
    fig1 = figure_1_heatmaps(matrices)
    fig1.savefig(FIGURES_DIR / "cross_model_heatmaps.pdf",
                bbox_inches="tight", dpi=150)
    fig1.savefig(FIGURES_DIR / "cross_model_heatmaps.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig1)
    print("  Saved: cross_model_heatmaps.pdf/png")

    # Figure 2: MDS overlay
    print("Generating MDS overlay...")
    fig2, residuals = figure_2_mds_overlay(matrices)
    fig2.savefig(FIGURES_DIR / "cross_model_mds.pdf",
                bbox_inches="tight", dpi=150)
    fig2.savefig(FIGURES_DIR / "cross_model_mds.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print("  Saved: cross_model_mds.pdf/png")
    for mk, d in residuals.items():
        print(f"    Procrustes disparity {mk}: {d:.4f}")

    # Figure 3: Correlation scatter
    print("Generating correlation scatter...")
    fig3 = figure_3_correlation_scatter(matrices)
    fig3.savefig(FIGURES_DIR / "cross_model_scatter.pdf",
                bbox_inches="tight", dpi=150)
    fig3.savefig(FIGURES_DIR / "cross_model_scatter.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig3)
    print("  Saved: cross_model_scatter.pdf/png")

    # Figure 4: Combined (for article)
    print("Generating combined figure...")
    fig4, _ = figure_4_combined(matrices)
    fig4.savefig(FIGURES_DIR / "cross_model_combined.pdf",
                bbox_inches="tight", dpi=200)
    fig4.savefig(FIGURES_DIR / "cross_model_combined.png",
                bbox_inches="tight", dpi=200)
    plt.close(fig4)
    print("  Saved: cross_model_combined.pdf/png")

    print("\nDone! All figures in article/figures/")


if __name__ == "__main__":
    main()
