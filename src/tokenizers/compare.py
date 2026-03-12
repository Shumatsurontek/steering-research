"""
Tokenizer Comparative Analysis for Agentic Calendar Context

Compares how different LLM tokenizers handle calendar-related prompts,
tool call JSON, temporal expressions, and named entities.
"""

import json
from dataclasses import dataclass, field
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------

MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "gemma-3-1b": "google/gemma-3-1b-it",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}

# ---------------------------------------------------------------------------
# Test prompts — calendar agentic context
# ---------------------------------------------------------------------------

CALENDAR_PROMPTS = {
    # --- Simple user requests ---
    "simple_fr": "Crée un rendez-vous demain à 14h avec Marie pour discuter du projet.",
    "simple_en": "Schedule a meeting tomorrow at 2pm with Marie to discuss the project.",
    "complex_fr": (
        "Peux-tu me bloquer un créneau le lundi 24 mars de 10h à 11h30 "
        "dans la salle Confluence pour la rétrospective sprint avec "
        "Jean-Pierre, Fatima et l'équipe DevOps ?"
    ),
    "complex_en": (
        "Can you block a slot on Monday March 24th from 10am to 11:30am "
        "in the Confluence room for the sprint retrospective with "
        "Jean-Pierre, Fatima and the DevOps team?"
    ),
    "ambiguous": "Cale-moi un truc avec Antoine la semaine pro, plutôt le matin.",

    # --- Temporal expressions ---
    "temporal_relative": "après-demain à 9h, dans 3 jours, la semaine prochaine, lundi prochain",
    "temporal_absolute": "2026-03-24T10:00:00+01:00, le 15 avril 2026 à 16h45",

    # --- Tool call JSON (function calling format) ---
    "tool_call_json": json.dumps({
        "name": "create_calendar_event",
        "arguments": {
            "title": "Rétrospective Sprint 42",
            "start": "2026-03-24T10:00:00+01:00",
            "end": "2026-03-24T11:30:00+01:00",
            "location": "Salle Confluence",
            "attendees": ["jean-pierre@company.com", "fatima@company.com"],
            "description": "Revue du sprint et planification"
        }
    }, ensure_ascii=False),

    # --- System prompt (agentic context) ---
    "system_agent": (
        "You are a calendar assistant. Extract date, time, title, location, "
        "and attendees from user messages. Call create_calendar_event with "
        "structured parameters. Always confirm before creating."
    ),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenizerProfile:
    name: str
    model_id: str
    vocab_size: int
    tokens_by_prompt: dict[str, list[str]] = field(default_factory=dict)
    ids_by_prompt: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class ComparisonMetrics:
    """Metrics for a single prompt across all tokenizers."""
    prompt_key: str
    prompt_text: str
    token_counts: dict[str, int]         # model_name -> count
    compression_ratios: dict[str, float]  # model_name -> chars/token
    fragmentation: dict[str, list[str]]   # model_name -> tokens list


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def load_tokenizers() -> dict[str, AutoTokenizer]:
    """Load all tokenizers. Downloads from HuggingFace on first call."""
    tokenizers = {}
    for name, model_id in MODELS.items():
        print(f"Loading tokenizer: {name} ({model_id})...")
        tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
    return tokenizers


def profile_tokenizer(name: str, tokenizer: AutoTokenizer) -> TokenizerProfile:
    """Build a full profile of how a tokenizer handles calendar prompts."""
    profile = TokenizerProfile(
        name=name,
        model_id=MODELS[name],
        vocab_size=tokenizer.vocab_size,
    )
    for key, prompt in CALENDAR_PROMPTS.items():
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        profile.ids_by_prompt[key] = ids
        profile.tokens_by_prompt[key] = tokens
    return profile


def compute_comparison(profiles: list[TokenizerProfile]) -> list[ComparisonMetrics]:
    """Compute comparative metrics across all tokenizers for each prompt."""
    results = []
    for key, prompt in CALENDAR_PROMPTS.items():
        metrics = ComparisonMetrics(
            prompt_key=key,
            prompt_text=prompt,
            token_counts={},
            compression_ratios={},
            fragmentation={},
        )
        for p in profiles:
            tokens = p.tokens_by_prompt[key]
            n_tokens = len(tokens)
            metrics.token_counts[p.name] = n_tokens
            metrics.compression_ratios[p.name] = len(prompt) / n_tokens if n_tokens else 0
            metrics.fragmentation[p.name] = tokens
        results.append(metrics)
    return results


def score_tokenizer_quality(
    profile: TokenizerProfile,
    metrics: list[ComparisonMetrics],
) -> dict[str, float]:
    """
    Score the quality of a tokenizer for agentic calendar use cases.

    Returns a dict of named scores in [0, 1] range (higher = better).
    """
    metrics_by_key = {m.prompt_key: m for m in metrics}
    name = profile.name

    # --- temporal_integrity ---
    # Check if dates/times stay as coherent units.
    # Target atoms: "14h", "10h", "11h30", "2026", "03-24", "T10:00:00"
    # Lower fragmentation of temporal tokens = higher score.
    temporal_keys = ["temporal_relative", "temporal_absolute"]
    temporal_tokens_total = 0
    temporal_chars_total = 0
    for key in temporal_keys:
        tokens = profile.tokens_by_prompt[key]
        prompt = CALENDAR_PROMPTS[key]
        temporal_tokens_total += len(tokens)
        temporal_chars_total += len(prompt)
    # Ideal: ~4 chars/token for temporal expressions (dates are dense)
    temporal_ratio = temporal_chars_total / temporal_tokens_total if temporal_tokens_total else 0
    # Normalize: 5+ chars/token = 1.0, 1 char/token = 0.0
    temporal_integrity = min(1.0, max(0.0, (temporal_ratio - 1.0) / 4.0))

    # --- semantic_coherence ---
    # Compare tokens/words ratio for calendar prompts.
    # A tokenizer that keeps whole words intact scores higher.
    coherence_keys = ["simple_fr", "complex_fr", "simple_en", "complex_en"]
    word_ratios = []
    for key in coherence_keys:
        tokens = profile.tokens_by_prompt[key]
        prompt = CALENDAR_PROMPTS[key]
        n_words = len(prompt.split())
        # Ratio of tokens to words: 1.0 = each word is one token (ideal)
        # >1 means fragmentation. We want this close to 1.
        ratio = len(tokens) / n_words if n_words else 1
        word_ratios.append(ratio)
    avg_ratio = sum(word_ratios) / len(word_ratios)
    # Normalize: ratio 1.0 = score 1.0, ratio 3.0+ = score 0.0
    semantic_coherence = min(1.0, max(0.0, 1.0 - (avg_ratio - 1.0) / 2.0))

    # --- json_efficiency ---
    # How efficiently does the tokenizer encode tool call JSON?
    # Compare to the best theoretical compression (whole JSON keys as single tokens).
    json_tokens = profile.tokens_by_prompt["tool_call_json"]
    json_prompt = CALENDAR_PROMPTS["tool_call_json"]
    json_ratio = len(json_prompt) / len(json_tokens) if json_tokens else 0
    # Normalize: 6+ chars/token = 1.0, 1 char/token = 0.0
    json_efficiency = min(1.0, max(0.0, (json_ratio - 1.0) / 5.0))

    # --- multilingual_parity ---
    # Compare FR vs EN token counts. Perfect parity = 1.0.
    fr_en_pairs = [("simple_fr", "simple_en"), ("complex_fr", "complex_en")]
    parity_scores = []
    for fr_key, en_key in fr_en_pairs:
        fr_count = metrics_by_key[fr_key].token_counts[name]
        en_count = metrics_by_key[en_key].token_counts[name]
        # Ratio of smaller/larger count. 1.0 = perfect parity.
        if max(fr_count, en_count) > 0:
            parity = min(fr_count, en_count) / max(fr_count, en_count)
        else:
            parity = 1.0
        parity_scores.append(parity)
    multilingual_parity = sum(parity_scores) / len(parity_scores)

    # --- overall ---
    # Weighted average: temporal and semantic matter most for steering
    overall = (
        0.30 * temporal_integrity
        + 0.30 * semantic_coherence
        + 0.20 * json_efficiency
        + 0.20 * multilingual_parity
    )

    return {
        "temporal_integrity": round(temporal_integrity, 3),
        "semantic_coherence": round(semantic_coherence, 3),
        "json_efficiency": round(json_efficiency, 3),
        "multilingual_parity": round(multilingual_parity, 3),
        "overall": round(overall, 3),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_tokenization(name: str, tokens: list[str], prompt: str):
    """Pretty-print a tokenization result."""
    n = len(tokens)
    ratio = len(prompt) / n if n else 0
    print(f"  {name:15s} │ {n:3d} tokens │ {ratio:.2f} chars/tok │ {tokens[:12]}{'...' if n > 12 else ''}")


def print_comparison_table(comparisons: list[ComparisonMetrics]):
    """Print a summary comparison table."""
    model_names = list(MODELS.keys())
    header = f"{'Prompt':20s} │ " + " │ ".join(f"{m:>14s}" for m in model_names)
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("TOKEN COUNTS BY PROMPT AND MODEL")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)
    for c in comparisons:
        counts = " │ ".join(f"{c.token_counts.get(m, 0):>14d}" for m in model_names)
        print(f"{c.prompt_key:20s} │ {counts}")

    print(f"\n{'═' * len(header)}")
    print("COMPRESSION RATIO (chars/token) — higher = more efficient")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)
    for c in comparisons:
        ratios = " │ ".join(f"{c.compression_ratios.get(m, 0):>14.2f}" for m in model_names)
        print(f"{c.prompt_key:20s} │ {ratios}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("TOKENIZER COMPARATIVE ANALYSIS — Agentic Calendar Context")
    print("=" * 60)

    # 1. Load tokenizers
    tokenizers = load_tokenizers()

    # 2. Profile each tokenizer
    profiles = []
    for name, tok in tokenizers.items():
        profile = profile_tokenizer(name, tok)
        profiles.append(profile)
        print(f"\n[{name}] vocab_size={profile.vocab_size}")
        for key in CALENDAR_PROMPTS:
            print_tokenization(name, profile.tokens_by_prompt[key], CALENDAR_PROMPTS[key])

    # 3. Comparative metrics
    comparisons = compute_comparison(profiles)
    print_comparison_table(comparisons)

    # 4. Scoring (once implemented)
    print("\n" + "=" * 60)
    print("QUALITY SCORES")
    print("=" * 60)
    for profile in profiles:
        try:
            scores = score_tokenizer_quality(profile, comparisons)
            print(f"\n[{profile.name}]")
            for score_name, value in scores.items():
                print(f"  {score_name:25s}: {value:.3f}")
        except NotImplementedError:
            print(f"\n[{profile.name}] ⏳ score_tokenizer_quality not yet implemented")
            break

    return profiles, comparisons


if __name__ == "__main__":
    profiles, comparisons = main()
