"""
Generate publication-quality figures for MMLU-Pro benchmark results.

Produces:
  1. Grouped bar chart: Qwen3-0.6B n=200 (baseline vs steered per domain)
  2. Heatmap: accuracy by (model × domain) for each coefficient
  3. n=20 vs n=200 comparison showing noise reduction
  4. Combined figure for the article
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "article" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "baseline": "#2c3e50",
    "a10": "#3498db",
    "a30": "#e67e22",
    "a60": "#e74c3c",
}

DOMAIN_COLORS = {
    "math": "#2ecc71",
    "law": "#9b59b6",
    "history": "#e67e22",
}


def load_results():
    with open(RESULTS_DIR / "mmlu_pro_mc_benchmark_n200.json") as f:
        n200 = json.load(f)
    with open(RESULTS_DIR / "mmlu_pro_mc_benchmark_n20.json") as f:
        n20 = json.load(f)
    return n200, n20


def get_acc(res, key):
    """Extract accuracy from result dict."""
    if isinstance(res, dict) and key in res:
        val = res[key]
        if isinstance(val, dict):
            return val.get("acc,none", None)
    return None


def get_stderr(res, key):
    if isinstance(res, dict) and key in res:
        val = res[key]
        if isinstance(val, dict):
            return val.get("acc_stderr,none", None)
    return None


# ── Figure 1: Qwen n=200 grouped bar chart ──
def fig_qwen_n200(n200):
    data = n200["qwen3_0.6b"]
    domains = ["math", "law", "history"]
    conditions = ["baseline", "L14_a10", "L14_a30", "L14_a60"]
    labels = ["Baseline", "α=10", "α=30", "α=60"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(domains))
    width = 0.18

    for i, (cond, label) in enumerate(zip(conditions, labels)):
        color_key = cond.split("_")[-1] if "_" in cond else "baseline"
        accs = [get_acc(data[d], cond) * 100 for d in domains]
        errs = [get_stderr(data[d], cond) * 100 for d in domains]
        bars = ax.bar(x + i * width, accs, width, label=label,
                      color=COLORS[color_key], yerr=errs, capsize=3,
                      edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Domain")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Qwen3-0.6B — MMLU-Pro Loglikelihood (n=200)")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.legend(loc="upper right")
    ax.set_ylim(0, 30)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.3, label="Random (10%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mmlu_mc_qwen_n200.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mmlu_mc_qwen_n200.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mmlu_mc_qwen_n200.pdf/png")


# ── Figure 2: n=20 vs n=200 comparison ──
def fig_sample_size_comparison(n200, n20):
    domains = ["math", "law", "history"]
    conditions = ["baseline", "L14_a10", "L14_a30", "L14_a60"]
    labels = ["Baseline", "α=10", "α=30", "α=60"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    for idx, domain in enumerate(domains):
        ax = axes[idx]
        x = np.arange(len(conditions))

        # n=20
        accs_20 = [get_acc(n20["qwen3_0.6b"][domain], c) * 100 for c in conditions]
        errs_20 = [get_stderr(n20["qwen3_0.6b"][domain], c) * 100 for c in conditions]

        # n=200
        accs_200 = [get_acc(n200["qwen3_0.6b"][domain], c) * 100 for c in conditions]
        errs_200 = [get_stderr(n200["qwen3_0.6b"][domain], c) * 100 for c in conditions]

        width = 0.35
        ax.bar(x - width / 2, accs_20, width, label="n=20",
               color="#bdc3c7", yerr=errs_20, capsize=3, edgecolor="white")
        ax.bar(x + width / 2, accs_200, width, label="n=200",
               color=DOMAIN_COLORS[domain], yerr=errs_200, capsize=3, edgecolor="white")

        ax.set_title(domain.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 35)
        ax.axhline(y=10, color="gray", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Accuracy (%)")
            ax.legend()

    fig.suptitle("Qwen3-0.6B — Effect of Sample Size on MMLU-Pro MC Results",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mmlu_mc_n20_vs_n200.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mmlu_mc_n20_vs_n200.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mmlu_mc_n20_vs_n200.pdf/png")


# ── Figure 3: Multi-model heatmap (n=20) ──
def fig_multimodel_heatmap(n20):
    models = ["qwen3_0.6b", "llama3_3b", "lfm2_1.2b"]
    model_labels = ["Qwen3-0.6B", "Llama-3.2-3B", "LFM2.5-1.2B"]
    domains = ["math", "law", "history"]
    # For each model, compute delta = best_steered - baseline
    coeff_keys = {
        "qwen3_0.6b": ["L14_a10", "L14_a30", "L14_a60"],
        "llama3_3b": ["L14_a10", "L14_a30", "L14_a60"],
        "lfm2_1.2b": ["L8_a10", "L8_a30", "L8_a60"],
    }

    # Build delta matrix (model x domain)
    delta_matrix = np.zeros((len(models), len(domains)))
    for i, model in enumerate(models):
        for j, domain in enumerate(domains):
            base = get_acc(n20[model][domain], "baseline")
            best = base
            for ck in coeff_keys[model]:
                val = get_acc(n20[model][domain], ck)
                if val is not None and val > best:
                    best = val
            delta_matrix[i, j] = (best - base) * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(delta_matrix, cmap="RdYlGn", vmin=-20, vmax=15, aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(domains)):
            val = delta_matrix[i, j]
            color = "white" if abs(val) > 10 else "black"
            ax.text(j, i, f"{val:+.0f}pp", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("Best Steering Delta vs Baseline (n=20, loglikelihood)")
    fig.colorbar(im, ax=ax, label="Δ accuracy (pp)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mmlu_mc_delta_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mmlu_mc_delta_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mmlu_mc_delta_heatmap.pdf/png")


# ── Figure 4: Accuracy vs coefficient curve (Qwen n=200) ──
def fig_accuracy_vs_coeff(n200):
    data = n200["qwen3_0.6b"]
    domains = ["math", "law", "history"]
    conditions = ["baseline", "L14_a10", "L14_a30", "L14_a60"]
    coeffs = [0, 10, 30, 60]

    fig, ax = plt.subplots(figsize=(6, 4))

    for domain in domains:
        accs = [get_acc(data[domain], c) * 100 for c in conditions]
        errs = [get_stderr(data[domain], c) * 100 for c in conditions]
        ax.errorbar(coeffs, accs, yerr=errs, marker="o", label=domain.capitalize(),
                    color=DOMAIN_COLORS[domain], linewidth=2, capsize=4, markersize=6)

    ax.set_xlabel("Steering Coefficient (α)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Qwen3-0.6B — Accuracy vs Steering Coefficient (n=200)")
    ax.legend()
    ax.set_ylim(0, 25)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mmlu_mc_acc_vs_coeff.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mmlu_mc_acc_vs_coeff.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mmlu_mc_acc_vs_coeff.pdf/png")


# ── Figure 5: Combined for article ──
def fig_combined(n200, n20):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Qwen n=200 bar chart
    ax = axes[0, 0]
    data = n200["qwen3_0.6b"]
    domains = ["math", "law", "history"]
    conditions = ["baseline", "L14_a10", "L14_a30", "L14_a60"]
    labels = ["Baseline", "α=10", "α=30", "α=60"]
    x = np.arange(len(domains))
    width = 0.18
    for i, (cond, label) in enumerate(zip(conditions, labels)):
        color_key = cond.split("_")[-1] if "_" in cond else "baseline"
        accs = [get_acc(data[d], cond) * 100 for d in domains]
        errs = [get_stderr(data[d], cond) * 100 for d in domains]
        ax.bar(x + i * width, accs, width, label=label,
               color=COLORS[color_key], yerr=errs, capsize=2, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Qwen3-0.6B MC results (n=200)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 28)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Accuracy vs coefficient
    ax = axes[0, 1]
    coeffs = [0, 10, 30, 60]
    for domain in domains:
        accs = [get_acc(data[domain], c) * 100 for c in conditions]
        errs = [get_stderr(data[domain], c) * 100 for c in conditions]
        ax.errorbar(coeffs, accs, yerr=errs, marker="o", label=domain.capitalize(),
                    color=DOMAIN_COLORS[domain], linewidth=2, capsize=3, markersize=5)
    ax.set_xlabel("Steering Coefficient (α)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Accuracy vs Coefficient (n=200)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 25)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) n=20 vs n=200 for history (the most dramatic difference)
    ax = axes[1, 0]
    cond_labels = ["Baseline", "α=10", "α=30", "α=60"]
    x = np.arange(4)
    width = 0.35

    for di, domain in enumerate(domains):
        accs_20 = [get_acc(n20["qwen3_0.6b"][domain], c) * 100 for c in conditions]
        errs_20 = [get_stderr(n20["qwen3_0.6b"][domain], c) * 100 for c in conditions]
        accs_200 = [get_acc(n200["qwen3_0.6b"][domain], c) * 100 for c in conditions]
        errs_200 = [get_stderr(n200["qwen3_0.6b"][domain], c) * 100 for c in conditions]

    # Show history specifically (most dramatic noise)
    domain = "history"
    accs_20 = [get_acc(n20["qwen3_0.6b"][domain], c) * 100 for c in conditions]
    errs_20 = [get_stderr(n20["qwen3_0.6b"][domain], c) * 100 for c in conditions]
    accs_200 = [get_acc(n200["qwen3_0.6b"][domain], c) * 100 for c in conditions]
    errs_200 = [get_stderr(n200["qwen3_0.6b"][domain], c) * 100 for c in conditions]
    ax.bar(x - width / 2, accs_20, width, label="n=20", color="#bdc3c7",
           yerr=errs_20, capsize=3, edgecolor="white")
    ax.bar(x + width / 2, accs_200, width, label="n=200", color=DOMAIN_COLORS["history"],
           yerr=errs_200, capsize=3, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(c) History: n=20 vs n=200 (noise reduction)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 35)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (d) Multi-model delta heatmap (n=20)
    ax = axes[1, 1]
    models = ["qwen3_0.6b", "llama3_3b", "lfm2_1.2b"]
    model_labels = ["Qwen3\n0.6B", "Llama-3.2\n3B", "LFM2.5\n1.2B"]
    coeff_keys = {
        "qwen3_0.6b": ["L14_a10", "L14_a30", "L14_a60"],
        "llama3_3b": ["L14_a10", "L14_a30", "L14_a60"],
        "lfm2_1.2b": ["L8_a10", "L8_a30", "L8_a60"],
    }
    delta_matrix = np.zeros((len(models), len(domains)))
    for i, model in enumerate(models):
        for j, domain in enumerate(domains):
            base = get_acc(n20[model][domain], "baseline")
            best = base
            for ck in coeff_keys[model]:
                val = get_acc(n20[model][domain], ck)
                if val is not None and val > best:
                    best = val
            delta_matrix[i, j] = (best - base) * 100

    im = ax.imshow(delta_matrix, cmap="RdYlGn", vmin=-20, vmax=15, aspect="auto")
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=8)
    for i in range(len(models)):
        for j in range(len(domains)):
            val = delta_matrix[i, j]
            color = "white" if abs(val) > 10 else "black"
            ax.text(j, i, f"{val:+.0f}pp", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_title("(d) Best Δ vs baseline (n=20, MC)")
    fig.colorbar(im, ax=ax, label="Δ acc (pp)", shrink=0.8)

    fig.suptitle("MMLU-Pro Loglikelihood Benchmark Results",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mmlu_mc_combined.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mmlu_mc_combined.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: mmlu_mc_combined.pdf/png")


def main():
    print("Loading results...")
    n200, n20 = load_results()

    print("Generating figures...")
    fig_qwen_n200(n200)
    fig_sample_size_comparison(n200, n20)
    fig_multimodel_heatmap(n20)
    fig_accuracy_vs_coeff(n200)
    fig_combined(n200, n20)

    print("\nAll figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
