"""
Generate qualitative sample comparison figures for the article.

Visualizes:
  - Log-likelihood distributions: baseline vs steered (showing position bias)
  - Generate-until outputs: baseline vs steered side-by-side
"""

import json
import string
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


def load_samples(domain, n):
    with open(RESULTS_DIR / f"mmlu_pro_samples_{domain}_n{n}.json") as f:
        return json.load(f)


def fig_loglik_comparison():
    """Show how steering flattens log-likelihoods and creates position bias."""
    history = load_samples("history", 5)
    mc = history["mc"]["samples"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey="row")

    for qi in range(3):
        base_s = mc["baseline"][qi]
        steer_s = mc["steered_a30"][qi]

        options = list(base_s["log_likelihoods"].keys())
        n_opts = len(options)
        base_lls = [base_s["log_likelihoods"][o] for o in options]
        steer_lls = [steer_s["log_likelihoods"][o] for o in options]
        expected = base_s["expected_answer"]

        x = np.arange(n_opts)

        # Baseline
        ax = axes[0, qi]
        colors_base = ["#27ae60" if o == expected else "#3498db" for o in options]
        bars = ax.bar(x, base_lls, color=colors_base, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(options, fontsize=7)
        ax.set_title(f"Q{qi+1} — Baseline", fontsize=9)
        if qi == 0:
            ax.set_ylabel("Log-likelihood")
        # Mark selected
        base_sel_idx = options.index(base_s["model_selected"])
        ax.annotate("▼", xy=(base_sel_idx, base_lls[base_sel_idx]),
                     ha="center", va="bottom", fontsize=12, color="#e74c3c")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Steered
        ax = axes[1, qi]
        colors_steer = ["#27ae60" if o == expected else "#e67e22" for o in options]
        ax.bar(x, steer_lls, color=colors_steer, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(options, fontsize=7)
        ax.set_title(f"Q{qi+1} — Steered (α=30)", fontsize=9)
        if qi == 0:
            ax.set_ylabel("Log-likelihood")
        # Mark selected
        steer_sel_idx = options.index(steer_s["model_selected"])
        ax.annotate("▼", xy=(steer_sel_idx, steer_lls[steer_sel_idx]),
                     ha="center", va="bottom", fontsize=12, color="#e74c3c")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="Correct answer"),
        Patch(facecolor="#3498db", label="Baseline option"),
        Patch(facecolor="#e67e22", label="Steered option"),
        plt.Line2D([0], [0], marker="v", color="#e74c3c", linestyle="None",
                    markersize=8, label="Model's selection"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("History — Log-Likelihood Distributions: Baseline vs Steered (α=30)\n"
                 "Steering uniformly depresses log-likelihoods; option A dominates due to position bias",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(FIGURES_DIR / "mmlu_samples_loglik.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mmlu_samples_loglik.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: mmlu_samples_loglik.pdf/png")


def fig_loglik_spread():
    """Show the spread (entropy) of log-likelihoods: baseline has peaks, steered is flat."""
    history = load_samples("history", 5)
    mc = history["mc"]["samples"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for cond_idx, (cond, label, color) in enumerate([
        ("baseline", "Baseline", "#3498db"),
        ("steered_a30", "Steered α=30", "#e67e22"),
    ]):
        ax = axes[cond_idx]
        all_spreads = []
        for qi, s in enumerate(mc[cond]):
            lls = list(s["log_likelihoods"].values())
            lls_arr = np.array(lls)
            # Softmax to get probabilities
            probs = np.exp(lls_arr - np.max(lls_arr))
            probs = probs / probs.sum()
            all_spreads.append(probs)

            x = np.arange(len(probs))
            ax.plot(x, probs, marker="o", markersize=4, alpha=0.6,
                    label=f"Q{qi+1}" if cond_idx == 0 else None)

        ax.set_title(f"{label}")
        ax.set_xlabel("Option index")
        ax.set_ylabel("Probability (softmax)")
        ax.set_ylim(0, 0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(fontsize=7)
    fig.suptitle("Probability Distributions over Options: Steering Flattens Discrimination",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mmlu_samples_spread.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mmlu_samples_spread.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: mmlu_samples_spread.pdf/png")


def fig_gen_comparison():
    """Text comparison figure for generate_until outputs."""
    history = load_samples("history", 3)
    gen = history.get("gen", {}).get("samples", {})
    if not gen:
        print("  SKIP: no generate_until samples for history")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 8))

    for qi in range(min(3, len(gen.get("baseline", [])))):
        base_s = gen["baseline"][qi]
        steer_s = gen["steered_a30"][qi]

        question = base_s["question"][:80] + "..."
        expected = base_s["expected_answer"]

        for col, (s, title, bg_color) in enumerate([
            (base_s, "Baseline", "#eaf2e3"),
            (steer_s, "Steered α=30", "#fce4e4"),
        ]):
            ax = axes[qi, col]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_facecolor(bg_color)

            gen_text = s["generated_text"][:300]
            filtered = s.get("filtered_response", "?")

            text = (f"Q{qi+1}: {question}\n"
                    f"Expected: {expected}\n"
                    f"─────────────────\n"
                    f"{gen_text}\n"
                    f"─────────────────\n"
                    f"Extracted answer: {filtered}")

            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", fontfamily="monospace",
                    wrap=True)
            ax.set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle("Generate-Until Outputs: Baseline vs Steered (α=30) — History Domain\n"
                 "Baseline produces specific reasoning; steered produces vague fillers",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURES_DIR / "mmlu_samples_gen.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mmlu_samples_gen.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: mmlu_samples_gen.pdf/png")


def main():
    print("Generating sample comparison figures...")
    fig_loglik_comparison()
    fig_loglik_spread()
    fig_gen_comparison()
    print("\nAll figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
