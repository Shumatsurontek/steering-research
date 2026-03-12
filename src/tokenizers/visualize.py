"""
Tokenizer Comparative Visualizations — Publication-quality plots for the article.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from src.tokenizers.compare import (
    load_tokenizers, profile_tokenizer, compute_comparison,
    score_tokenizer_quality, MODELS, CALENDAR_PROMPTS,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # colorblind-friendly
MODEL_NAMES = list(MODELS.keys())


def setup_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_token_counts_heatmap(comparisons, profiles):
    """Plot 1: Heatmap of token counts."""
    prompt_keys = [c.prompt_key for c in comparisons]
    data = np.array([
        [c.token_counts[m] for m in MODEL_NAMES]
        for c in comparisons
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(MODEL_NAMES)))
    ax.set_xticklabels(MODEL_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(prompt_keys)))
    ax.set_yticklabels(prompt_keys)

    # Annotate cells
    for i in range(len(prompt_keys)):
        for j in range(len(MODEL_NAMES)):
            color = "white" if data[i, j] > data.mean() + data.std() else "black"
            ax.text(j, i, str(data[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="Token count")
    ax.set_title("Token Counts by Prompt and Model")
    fig.savefig(RESULTS_DIR / "token_counts_heatmap.png")
    plt.close(fig)
    print(f"  Saved: token_counts_heatmap.png")


def plot_compression_ratio_bar(comparisons):
    """Plot 2: Grouped bar chart of compression ratios."""
    prompt_keys = [c.prompt_key for c in comparisons]
    x = np.arange(len(prompt_keys))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(MODEL_NAMES):
        ratios = [c.compression_ratios[model] for c in comparisons]
        bars = ax.bar(x + i * width, ratios, width, label=model, color=COLORS[i])

    ax.set_xlabel("Prompt")
    ax.set_ylabel("Compression ratio (chars/token)")
    ax.set_title("Compression Ratio — Higher = More Efficient Tokenization")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(prompt_keys, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    fig.savefig(RESULTS_DIR / "compression_ratio_bar.png")
    plt.close(fig)
    print(f"  Saved: compression_ratio_bar.png")


def plot_quality_scores_radar(profiles, comparisons):
    """Plot 3: Radar/spider chart of quality scores."""
    categories = ["temporal_integrity", "semantic_coherence", "json_efficiency",
                   "multilingual_parity", "overall"]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, profile in enumerate(profiles):
        scores = score_tokenizer_quality(profile, comparisons)
        values = [scores[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=COLORS[i], label=profile.name)
        ax.fill(angles, values, alpha=0.1, color=COLORS[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in categories])
    ax.set_ylim(0, 1)
    ax.set_title("Tokenizer Quality Scores for Calendar Tasks", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(RESULTS_DIR / "quality_scores_radar.png")
    plt.close(fig)
    print(f"  Saved: quality_scores_radar.png")


def plot_tokenization_detail(profiles):
    """Plot 4: Visual segmentation of 'simple_fr' prompt per model."""
    prompt = CALENDAR_PROMPTS["simple_fr"]
    fig, axes = plt.subplots(len(profiles), 1, figsize=(12, 1.2 * len(profiles) + 1))
    if len(profiles) == 1:
        axes = [axes]

    cmap = plt.cm.Set3
    for ax_idx, profile in enumerate(profiles):
        ax = axes[ax_idx]
        tokens = profile.tokens_by_prompt["simple_fr"]
        x = 0
        for t_idx, token in enumerate(tokens):
            display = token.replace("Ġ", " ").replace("▁", " ")
            w = max(len(display) * 0.12, 0.3)
            color = cmap(t_idx % 12)
            rect = FancyBboxPatch((x, 0.1), w, 0.8, boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor="gray", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x + w / 2, 0.5, display, ha="center", va="center",
                    fontsize=7, fontfamily="monospace")
            x += w + 0.03

        ax.set_xlim(-0.1, x + 0.1)
        ax.set_ylim(0, 1)
        ax.set_ylabel(profile.name, rotation=0, ha="right", va="center", fontsize=10)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    axes[0].set_title(f'Tokenization of: "{prompt}"', fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tokenization_detail.png")
    plt.close(fig)
    print(f"  Saved: tokenization_detail.png")


def plot_fragmentation_focus(comparisons):
    """Plot 5: Focus on temporal_absolute and tool_call_json fragmentation."""
    focus_keys = ["temporal_absolute", "tool_call_json"]
    focus_data = [c for c in comparisons if c.prompt_key in focus_keys]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_idx, comp in enumerate(focus_data):
        ax = axes[ax_idx]
        counts = [comp.token_counts[m] for m in MODEL_NAMES]
        bars = ax.bar(MODEL_NAMES, counts, color=COLORS)

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(f"{comp.prompt_key}\n({len(comp.prompt_text)} chars)")
        ax.set_ylabel("Token count")
        ax.tick_params(axis="x", rotation=30)

        # Add char count reference line
        n_chars = len(comp.prompt_text)
        ax.axhline(y=n_chars, color="gray", linestyle=":", alpha=0.5)
        ax.text(0.98, n_chars + 1, f"{n_chars} chars", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="gray")

    fig.suptitle("Fragmentation Problem: Dates & JSON Tool Calls", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fragmentation_focus.png")
    plt.close(fig)
    print(f"  Saved: fragmentation_focus.png")


def main():
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating publication-quality visualizations...")
    print("Loading tokenizers...")
    tokenizers = load_tokenizers()

    profiles = [profile_tokenizer(name, tok) for name, tok in tokenizers.items()]
    comparisons = compute_comparison(profiles)

    print("\nGenerating plots:")
    plot_token_counts_heatmap(comparisons, profiles)
    plot_compression_ratio_bar(comparisons)
    plot_quality_scores_radar(profiles, comparisons)
    plot_tokenization_detail(profiles)
    plot_fragmentation_focus(comparisons)
    print("\nAll plots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
