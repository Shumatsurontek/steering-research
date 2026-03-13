"""
Domain-Specific Steering Vectors for Multi-Agent SWE Orchestration.

Extracts contrastive steering vectors for 4 SWE-bench-relevant domains:
- code_reading: understanding code structure and logic
- bug_analysis: identifying and diagnosing bugs
- patch_writing: generating code fixes
- test_reasoning: understanding and writing tests

Uses the same contrastive mean-difference methodology as slm_gsm8k_steering.py:
hook all layers, collect last-token residual stream activations, compute
mean(positive) - mean(neutral) per layer.

Then sweeps layers x coefficients to find the best configuration per domain
using simple generation probes on ambiguous prompts.
"""

import gc
import json
import functools
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

SLM_INSTRUCT_ID = "Qwen/Qwen3-0.6B"
SLM_BASE_ID = "Qwen/Qwen3-0.6B-Base"

# ---------------------------------------------------------------------------
# Neutral prompts — same as gsm8k_benchmark.py (general knowledge facts)
# ---------------------------------------------------------------------------
NEUTRAL_PROMPTS = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Python is a high-level programming language known for its readability.",
    "The solar system has eight planets orbiting the Sun.",
    "DNA stands for deoxyribonucleic acid, which carries genetic information.",
    "Machine learning is a subset of artificial intelligence that learns from data.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
]

# ---------------------------------------------------------------------------
# Domain-specific positive prompts (10 per domain)
# ---------------------------------------------------------------------------
DOMAIN_PROMPTS = {
    "code_reading": [
        "Let me trace through this function step by step to understand the control flow and data transformations.",
        "I need to understand the call graph here — which functions invoke which, and how data flows between them.",
        "Reading through the class hierarchy, I can see the base class defines the interface while subclasses override specific behavior.",
        "This loop iterates over the dictionary keys and accumulates results into the output list via the append call.",
        "The decorator wraps the original function, adding logging before and after execution without modifying the core logic.",
        "Let me follow the import chain to see where this symbol is actually defined and what module provides it.",
        "The context manager ensures resources are cleaned up: __enter__ acquires the lock and __exit__ releases it.",
        "I can see this generator yields values lazily, so the full sequence is never materialized in memory at once.",
        "The nested comprehension first filters items by the predicate, then maps the transformation function over results.",
        "Tracing the variable through the function: it starts as a raw string, gets parsed into a dict, then serialized back to JSON.",
    ],
    "bug_analysis": [
        "The error occurs because this variable is None when it should be initialized. Let me trace back to find where the assignment is missing.",
        "This is a classic off-by-one error: the loop uses <= instead of <, causing it to access one element past the end of the array.",
        "The race condition happens because two threads read the counter before either writes, so one increment is lost.",
        "The root cause is that the exception handler catches too broadly — it silently swallows the TypeError we need to see.",
        "This function returns early on the error path but forgets to release the lock, leading to a deadlock on the next call.",
        "The bug is a type mismatch: the API returns a string ID but the comparison checks against an integer, so it never matches.",
        "Memory is leaking because the event listener is registered in the constructor but never removed when the component unmounts.",
        "The issue is that this default mutable argument (an empty list) is shared across all calls, accumulating stale data.",
        "Debugging the stack trace: the AttributeError on line 42 means the object was not properly deserialized from the cache.",
        "The failing assertion reveals that the sort is unstable here — elements with equal keys are reordered, breaking the expected output.",
    ],
    "patch_writing": [
        "To fix this issue, I need to add a null check before accessing the attribute and handle the edge case gracefully.",
        "The fix is to wrap this call in a try-except block and return a sensible default when the network request times out.",
        "I'll change the comparison from == to is for the None check, which is both more correct and more Pythonic.",
        "Adding a guard clause at the top of the function to validate the input before proceeding with the main logic.",
        "The patch replaces the mutable default argument with None and initializes a fresh list inside the function body.",
        "To resolve the race condition, I'll wrap the read-modify-write sequence in a threading.Lock context manager.",
        "I need to move this import inside the function to break the circular dependency between the two modules.",
        "The fix involves adding a finally block to ensure the file handle is closed even when an exception is raised.",
        "Applying the minimal change: swap the operand order in the subtraction so the sign of the result is correct.",
        "I'll add the missing return statement on the early-exit branch so the caller doesn't receive None unexpectedly.",
    ],
    "test_reasoning": [
        "This test should verify both the happy path and edge cases: empty input, boundary values, and invalid types.",
        "I'll write a parametrized test that covers the main scenarios: zero items, one item, many items, and duplicate items.",
        "The assertion checks that the function raises a ValueError when given negative input, using pytest.raises as a context manager.",
        "We need a mock for the external API call so the test runs deterministically without network access.",
        "This integration test verifies the full pipeline: parse input, transform data, write output, and read it back for comparison.",
        "Adding a regression test for the bug we just fixed: the specific input that triggered the off-by-one error.",
        "The test fixture sets up a temporary database with known seed data and tears it down after each test method.",
        "I should test the boundary conditions: maximum allowed length, minimum value, and exactly-at-the-limit inputs.",
        "Using property-based testing here: for any valid input, the output length should always equal the input length.",
        "The test verifies idempotency — calling the function twice with the same input produces the same result both times.",
    ],
}

DOMAINS = list(DOMAIN_PROMPTS.keys())

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
SWEEP_LAYERS = [15, 18, 20, 22, 25]
SWEEP_COEFFICIENTS = [10.0, 30.0, 60.0]

# Ambiguous probe prompts (could apply to any domain)
PROBE_PROMPTS = [
    "Explain what this code does",
    "There seems to be a problem here",
    "How should we handle this case",
]

# Keywords used to detect domain "flavor" in generated text
DOMAIN_KEYWORDS = {
    "code_reading": [
        "trace", "follow", "flow", "calls", "returns", "iterates",
        "reads", "structure", "defined", "imported", "hierarchy",
        "understand", "logic", "step by step", "execution",
    ],
    "bug_analysis": [
        "bug", "error", "issue", "root cause", "race condition",
        "off-by-one", "mismatch", "leak", "deadlock", "crash",
        "debug", "failing", "incorrect", "broken", "exception",
    ],
    "patch_writing": [
        "fix", "patch", "change", "replace", "add a check", "guard",
        "wrap", "move", "swap", "return statement", "modify",
        "resolve", "apply", "correct", "handle",
    ],
    "test_reasoning": [
        "test", "assert", "verify", "mock", "fixture", "edge case",
        "coverage", "parametrize", "regression", "expect",
        "boundary", "validate", "pytest", "unittest",
    ],
}


# ---------------------------------------------------------------------------
# Hooks (same pattern as slm_gsm8k_steering.py)
# ---------------------------------------------------------------------------
def _gather_hook(module, input, output, *, cache, layer_idx):
    """Collect hidden states during forward pass."""
    hidden = output[0] if isinstance(output, tuple) else output
    cache[layer_idx] = hidden.detach().cpu()


def _steering_hook(module, input, output, *, vector, coeff):
    """Add a scaled steering vector to the residual stream."""
    hidden = output[0] if isinstance(output, tuple) else output
    steered = hidden + coeff * vector.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_activations(model, tokenizer, prompts, device):
    """Extract last-token residual stream activations for all layers."""
    n_layers = model.config.num_hidden_layers
    all_acts = {i: [] for i in range(n_layers)}

    for prompt in prompts:
        cache = {}
        handles = []
        try:
            for i, layer in enumerate(model.model.layers):
                handles.append(layer.register_forward_hook(
                    functools.partial(_gather_hook, cache=cache, layer_idx=i)
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


def compute_domain_vectors(model, tokenizer, device):
    """Compute contrastive steering vectors for each domain.

    Returns:
        vectors: dict[domain][layer] = tensor (hidden_dim,)
        norms:   dict[domain][layer] = float
    """
    n_layers = model.config.num_hidden_layers

    # Extract neutral activations once (shared across domains)
    print("  Extracting neutral activations...")
    neutral_acts = extract_activations(model, tokenizer, NEUTRAL_PROMPTS, device)

    vectors = {}
    norms = {}

    for domain in DOMAINS:
        print(f"  Extracting {domain} activations...")
        pos_acts = extract_activations(model, tokenizer, DOMAIN_PROMPTS[domain], device)

        vectors[domain] = {}
        norms[domain] = {}
        for i in range(n_layers):
            diff = pos_acts[i].mean(dim=0) - neutral_acts[i].mean(dim=0)
            vectors[domain][i] = diff
            norms[domain][i] = diff.norm().item()

    return vectors, norms


# ---------------------------------------------------------------------------
# Generation with steering
# ---------------------------------------------------------------------------
def generate_steered(model, tokenizer, prompt, device, vector, layer, coeff,
                     max_tokens=150):
    """Generate a response with a steering vector applied at a given layer."""
    # Use chat template if available, otherwise plain text
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"Q: {prompt}\nA:"

    inputs = tokenizer(text, return_tensors="pt").to(device)

    vec_normed = vector / (vector.norm() + 1e-8)
    handle = model.model.layers[layer].register_forward_hook(
        functools.partial(_steering_hook, vector=vec_normed, coeff=coeff)
    )
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1e-7,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Strip Qwen3 thinking block if present
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    return resp.strip()


def score_domain_flavor(text, domain):
    """Score how strongly a generated text matches the target domain.

    Returns a count of domain-specific keywords found (case-insensitive).
    """
    text_lower = text.lower()
    return sum(1 for kw in DOMAIN_KEYWORDS[domain] if kw in text_lower)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep_domain(model, tokenizer, device, domain, vectors):
    """Test layers x coefficients for one domain and return best config."""
    n_layers = model.config.num_hidden_layers
    best_score = -1
    best_config = {"layer": SWEEP_LAYERS[0], "alpha": SWEEP_COEFFICIENTS[0]}
    sweep_results = []

    for layer in SWEEP_LAYERS:
        if layer >= n_layers:
            continue
        vec = vectors[domain][layer]
        for coeff in SWEEP_COEFFICIENTS:
            total_score = 0
            for probe in PROBE_PROMPTS:
                resp = generate_steered(
                    model, tokenizer, probe, device, vec, layer, coeff
                )
                total_score += score_domain_flavor(resp, domain)

            sweep_results.append({
                "layer": layer, "alpha": coeff, "score": total_score,
            })

            if total_score > best_score:
                best_score = total_score
                best_config = {"layer": layer, "alpha": coeff}

    return best_config, best_score, sweep_results


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def get_device_and_dtype():
    """Detect best available device and matching dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(model_id, device, dtype):
    """Load a model and tokenizer onto the target device."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def cleanup_model(model, device):
    """Delete model and free device memory."""
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DOMAIN-SPECIFIC STEERING VECTORS FOR SWE ORCHESTRATION")
    print("=" * 60)

    device, dtype = get_device_and_dtype()
    print(f"Device: {device}  |  dtype: {dtype}\n")

    all_results = {
        "models": {"instruct": SLM_INSTRUCT_ID, "base": SLM_BASE_ID},
        "domains": DOMAINS,
        "sweep_layers": SWEEP_LAYERS,
        "sweep_coefficients": SWEEP_COEFFICIENTS,
    }
    all_vectors = {}

    # ------------------------------------------------------------------
    # Process each model variant
    # ------------------------------------------------------------------
    for model_label, model_id in [("instruct", SLM_INSTRUCT_ID),
                                   ("base", SLM_BASE_ID)]:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_id} ({model_label})")
        print(f"{'='*60}")

        model, tokenizer = load_model(model_id, device, dtype)
        n_layers = model.config.num_hidden_layers
        print(f"Loaded: {n_layers} layers, hidden_dim={model.config.hidden_size}")

        # --- Extract vectors ---
        print(f"\n--- Extracting domain vectors ({model_label}) ---")
        vectors, norms = compute_domain_vectors(model, tokenizer, device)

        # Report L2 norms per layer per domain
        model_results = {"layer_norms": {}, "best_config": {}, "sweep_details": {}}

        for domain in DOMAINS:
            sorted_layers = sorted(norms[domain].items(),
                                   key=lambda x: x[1], reverse=True)
            print(f"\n  {domain} — top-5 layers by L2 norm:")
            for rank, (layer, norm) in enumerate(sorted_layers[:5], 1):
                print(f"    {rank}. Layer {layer}: {norm:.4f}")

            model_results["layer_norms"][domain] = {
                str(k): round(v, 4) for k, v in norms[domain].items()
            }

        # --- Sweep for best config per domain ---
        print(f"\n--- Sweeping layer x alpha ({model_label}) ---")

        for domain in DOMAINS:
            print(f"\n  [{domain}]")
            best_cfg, best_score, sweep_detail = sweep_domain(
                model, tokenizer, device, domain, vectors
            )
            model_results["best_config"][domain] = {
                "layer": best_cfg["layer"],
                "alpha": best_cfg["alpha"],
                "keyword_score": best_score,
            }
            model_results["sweep_details"][domain] = sweep_detail

            print(f"    Best: layer={best_cfg['layer']}, "
                  f"alpha={best_cfg['alpha']}, score={best_score}")

            # Show a sample steered generation
            sample = generate_steered(
                model, tokenizer, PROBE_PROMPTS[0], device,
                vectors[domain][best_cfg["layer"]],
                best_cfg["layer"], best_cfg["alpha"],
                max_tokens=100,
            )
            print(f"    Sample: {sample[:120]}...")

        all_results[model_label] = model_results

        # Store vectors for saving (move to CPU, convert keys)
        all_vectors[model_label] = {
            domain: {layer: vec.cpu() for layer, vec in vectors[domain].items()}
            for domain in DOMAINS
        }

        cleanup_model(model, device)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results_path = RESULTS_DIR / "domain_vectors_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results: {results_path}")

    # ------------------------------------------------------------------
    # Save vectors .pt
    # ------------------------------------------------------------------
    vectors_path = RESULTS_DIR / "domain_steering_vectors.pt"
    torch.save(all_vectors, vectors_path)
    print(f"Saved vectors: {vectors_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY — BEST CONFIG PER DOMAIN")
    print("=" * 60)
    for model_label in ["instruct", "base"]:
        print(f"\n  {model_label.upper()}:")
        for domain in DOMAINS:
            cfg = all_results[model_label]["best_config"][domain]
            print(f"    {domain:16s}  layer={cfg['layer']:>2d}  "
                  f"alpha={cfg['alpha']:>5.0f}  score={cfg['keyword_score']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
