"""
SWE-bench Verified Cluster Steering Vectors.

Extracts contrastive steering vectors aligned with the 3 natural repository
clusters found in SWE-bench Verified:
  - django_web (46%): ORM, forms, migrations, URL routing, middleware, email
  - scientific_computing (37%): sympy, sklearn, matplotlib, astropy, xarray
  - dev_tooling (15%): sphinx, pytest, pylint AST analysis

Uses the same contrastive mean-difference methodology as domain_vectors.py:
hook all layers, collect last-token residual stream activations, compute
mean(positive) - mean(neutral) per layer.

Then sweeps layers x coefficients, scores with cluster-specific probe prompts,
and computes cosine similarity against the generic domain vectors from
domain_steering_vectors.pt.
"""

import gc
import json
import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MODEL_ID = "Qwen/Qwen3-0.6B"

# ---------------------------------------------------------------------------
# Neutral prompts — same as other scripts
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
# SWE-bench cluster positive prompts (10 per cluster)
# ---------------------------------------------------------------------------
CLUSTER_PROMPTS = {
    "django_web": [
        "The QuerySet filter needs to handle the case where related_name is used with a reverse foreign key lookup through the ORM.",
        "This migration should add a new CharField with a default value and update the existing rows.",
        "The form's clean method must validate that the email field matches the domain constraint before saving.",
        "The URL resolver is failing because the named group in the regex pattern doesn't match the view's keyword argument.",
        "The middleware should intercept the response and add the CORS headers before returning to the WSGI handler.",
        "The template tag needs to escape the user-provided HTML to prevent cross-site scripting in the rendered output.",
        "The admin inline formset doesn't propagate the parent model's save signal to the related child objects.",
        "The database router should direct read queries to the replica and write queries to the primary PostgreSQL instance.",
        "The session backend is not expiring cookies correctly because the max_age calculation ignores timezone-aware datetimes.",
        "The Django email backend silently drops messages when the SMTP connection times out instead of raising an exception.",
    ],
    "scientific_computing": [
        "The symbolic integration should simplify the expression before evaluating, handling trigonometric identities.",
        "The array broadcasting rule fails when the input has an extra trailing dimension of size 1.",
        "The eigenvalue decomposition returns complex values for a symmetric matrix because of floating-point rounding.",
        "The sklearn pipeline needs to apply the scaler before the PCA transform to avoid feature dominance.",
        "The matplotlib subplot layout overlaps axis labels when using constrained_layout with a colorbar.",
        "The astronomical coordinate transform between ICRS and Galactic frames loses precision at the poles.",
        "The xarray Dataset merge fails silently when coordinate variables have conflicting units metadata.",
        "The sparse matrix multiplication gives incorrect results because CSR and CSC formats handle indexing differently.",
        "The statistical hypothesis test returns NaN because the sample variance is zero for a constant input array.",
        "The numerical ODE solver diverges at the singularity because the adaptive step size isn't bounded below.",
    ],
    "dev_tooling": [
        "The pytest fixture scope should propagate correctly to parameterized test functions.",
        "The Sphinx autodoc extension needs to handle classmethod descriptors differently from regular methods.",
        "The pylint checker incorrectly flags a valid walrus operator usage as a syntax error in the AST visitor.",
        "The test collection phase skips modules whose __init__.py raises an ImportError at parse time.",
        "The Sphinx cross-reference resolver cannot find the target when the module uses conditional imports.",
        "The pytest assertion rewriter must introspect the comparison operands to show a meaningful diff message.",
        "The pylint inference engine fails to resolve the type of a variable assigned inside a with-statement body.",
        "The documentation build warns about duplicate labels when two different rst files include the same anchor.",
        "The conftest fixture override doesn't apply because the directory hierarchy is not on sys.path.",
        "The linter's unused-import check should ignore re-exports defined in __all__ at the module level.",
    ],
}

CLUSTERS = list(CLUSTER_PROMPTS.keys())

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
SWEEP_LAYERS = [15, 18, 20, 22, 25]
SWEEP_COEFFICIENTS = [10.0, 30.0, 60.0]

# Cluster-specific probe prompts (3 per cluster)
CLUSTER_PROBES = {
    "django_web": [
        "The view returns a 500 error when the form is submitted",
        "The queryset needs to be optimized",
        "The migration fails on PostgreSQL",
    ],
    "scientific_computing": [
        "The plot axes are not labeled correctly",
        "The matrix multiplication gives wrong dimensions",
        "The statistical test returns NaN",
    ],
    "dev_tooling": [
        "The test fixture is not being called",
        "The documentation build fails",
        "The linter reports a false positive",
    ],
}

# Keywords to detect cluster flavor in generated text
CLUSTER_KEYWORDS = {
    "django_web": [
        "queryset", "orm", "model", "field", "migration", "form",
        "view", "url", "route", "middleware", "template", "admin",
        "django", "database", "session", "request", "response",
        "serializer", "filter", "manager",
    ],
    "scientific_computing": [
        "array", "matrix", "dimension", "broadcast", "plot", "axis",
        "integral", "derivative", "symbolic", "numpy", "scipy",
        "sklearn", "fit", "transform", "eigenvalue", "coordinate",
        "statistical", "regression", "interpolat", "converge",
    ],
    "dev_tooling": [
        "test", "fixture", "assert", "pytest", "sphinx", "autodoc",
        "lint", "pylint", "ast", "parse", "documentation", "build",
        "conftest", "parametri", "introspect", "checker", "node",
        "visitor", "import", "module",
    ],
}


# ---------------------------------------------------------------------------
# Hooks (same pattern as domain_vectors.py)
# ---------------------------------------------------------------------------
def _gather_hook(module, input, output, *, cache, layer_idx):
    hidden = output[0] if isinstance(output, tuple) else output
    cache[layer_idx] = hidden.detach().cpu()


def _steering_hook(module, input, output, *, vector, coeff):
    hidden = output[0] if isinstance(output, tuple) else output
    steered = hidden + coeff * vector.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_activations(model, tokenizer, prompts, device):
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


def compute_cluster_vectors(model, tokenizer, device):
    """Compute contrastive steering vectors for each SWE-bench cluster."""
    n_layers = model.config.num_hidden_layers

    print("  Extracting neutral activations...")
    neutral_acts = extract_activations(model, tokenizer, NEUTRAL_PROMPTS, device)

    vectors = {}
    norms = {}

    for cluster in CLUSTERS:
        print(f"  Extracting {cluster} activations...")
        pos_acts = extract_activations(
            model, tokenizer, CLUSTER_PROMPTS[cluster], device
        )
        vectors[cluster] = {}
        norms[cluster] = {}
        for i in range(n_layers):
            diff = pos_acts[i].mean(dim=0) - neutral_acts[i].mean(dim=0)
            vectors[cluster][i] = diff
            norms[cluster][i] = diff.norm().item()

    return vectors, norms


# ---------------------------------------------------------------------------
# Generation with steering
# ---------------------------------------------------------------------------
def generate_steered(model, tokenizer, prompt, device, vector, layer, coeff,
                     max_tokens=150):
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
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    return resp.strip()


def score_cluster_flavor(text, cluster):
    text_lower = text.lower()
    return sum(1 for kw in CLUSTER_KEYWORDS[cluster] if kw in text_lower)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep_cluster(model, tokenizer, device, cluster, vectors):
    n_layers = model.config.num_hidden_layers
    best_score = -1
    best_config = {"layer": SWEEP_LAYERS[0], "alpha": SWEEP_COEFFICIENTS[0]}
    sweep_results = []

    probes = CLUSTER_PROBES[cluster]

    for layer in SWEEP_LAYERS:
        if layer >= n_layers:
            continue
        vec = vectors[cluster][layer]
        for coeff in SWEEP_COEFFICIENTS:
            total_score = 0
            for probe in probes:
                resp = generate_steered(
                    model, tokenizer, probe, device, vec, layer, coeff
                )
                total_score += score_cluster_flavor(resp, cluster)

            sweep_results.append({
                "layer": layer, "alpha": coeff, "score": total_score,
            })

            if total_score > best_score:
                best_score = total_score
                best_config = {"layer": layer, "alpha": coeff}

    return best_config, best_score, sweep_results


# ---------------------------------------------------------------------------
# Cosine similarity with generic domain vectors
# ---------------------------------------------------------------------------
def compute_cosine_with_generic(cluster_vectors, generic_vectors_path):
    """Compute cosine similarity between cluster vectors and generic domain vectors.

    Returns a nested dict: similarities[cluster][generic_domain] = max cosine sim
    across matched layers.
    """
    if not generic_vectors_path.exists():
        print(f"  WARNING: {generic_vectors_path} not found, skipping cosine comparison.")
        return None

    generic_data = torch.load(generic_vectors_path, map_location="cpu", weights_only=True)

    # generic_data structure: {model_label: {domain: {layer: tensor}}}
    # Use "instruct" variant if available
    if "instruct" in generic_data:
        generic = generic_data["instruct"]
    else:
        # Fallback to first key
        generic = generic_data[next(iter(generic_data))]

    generic_domains = list(generic.keys())
    similarities = {}

    for cluster in CLUSTERS:
        similarities[cluster] = {}
        for g_domain in generic_domains:
            # Compute cosine similarity at each layer that exists in both
            cos_sims = []
            cluster_layers = cluster_vectors[cluster]
            generic_layers = generic[g_domain]
            for layer_idx in cluster_layers:
                if layer_idx in generic_layers:
                    c_vec = cluster_layers[layer_idx].float()
                    g_vec = generic_layers[layer_idx].float()
                    cos = F.cosine_similarity(
                        c_vec.unsqueeze(0), g_vec.unsqueeze(0)
                    ).item()
                    cos_sims.append(cos)
            if cos_sims:
                # Report mean and max
                similarities[cluster][g_domain] = {
                    "mean": round(sum(cos_sims) / len(cos_sims), 4),
                    "max": round(max(cos_sims), 4),
                    "at_layer_15": round(cos_sims[15], 4) if len(cos_sims) > 15 else None,
                }

    return similarities


# ---------------------------------------------------------------------------
# Device / model helpers
# ---------------------------------------------------------------------------
def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_model(model_id, device, dtype):
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

    print("=" * 65)
    print("SWE-BENCH VERIFIED CLUSTER STEERING VECTORS")
    print("=" * 65)

    device, dtype = get_device_and_dtype()
    print(f"Device: {device}  |  dtype: {dtype}\n")

    # ------------------------------------------------------------------
    # Load model once
    # ------------------------------------------------------------------
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load_model(MODEL_ID, device, dtype)
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Loaded: {n_layers} layers, hidden_dim={hidden_dim}\n")

    # ------------------------------------------------------------------
    # Extract cluster vectors
    # ------------------------------------------------------------------
    print("--- Extracting cluster vectors ---")
    vectors, norms = compute_cluster_vectors(model, tokenizer, device)

    results = {
        "model": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "clusters": CLUSTERS,
        "sweep_layers": SWEEP_LAYERS,
        "sweep_coefficients": SWEEP_COEFFICIENTS,
    }

    # Report L2 norms
    results["layer_norms"] = {}
    for cluster in CLUSTERS:
        sorted_layers = sorted(norms[cluster].items(),
                               key=lambda x: x[1], reverse=True)
        print(f"\n  {cluster} -- top-5 layers by L2 norm:")
        for rank, (layer, norm) in enumerate(sorted_layers[:5], 1):
            print(f"    {rank}. Layer {layer}: {norm:.4f}")

        results["layer_norms"][cluster] = {
            str(k): round(v, 4) for k, v in norms[cluster].items()
        }

    # ------------------------------------------------------------------
    # Sweep for best config per cluster
    # ------------------------------------------------------------------
    print(f"\n--- Sweeping layer x alpha ---")
    results["best_config"] = {}
    results["sweep_details"] = {}

    for cluster in CLUSTERS:
        print(f"\n  [{cluster}]")
        best_cfg, best_score, sweep_detail = sweep_cluster(
            model, tokenizer, device, cluster, vectors
        )
        results["best_config"][cluster] = {
            "layer": best_cfg["layer"],
            "alpha": best_cfg["alpha"],
            "keyword_score": best_score,
        }
        results["sweep_details"][cluster] = sweep_detail

        print(f"    Best: layer={best_cfg['layer']}, "
              f"alpha={best_cfg['alpha']}, score={best_score}")

        # Show a sample steered generation
        sample = generate_steered(
            model, tokenizer, CLUSTER_PROBES[cluster][0], device,
            vectors[cluster][best_cfg["layer"]],
            best_cfg["layer"], best_cfg["alpha"],
            max_tokens=100,
        )
        print(f"    Sample: {sample[:150]}...")

    # ------------------------------------------------------------------
    # Cosine similarity with generic domain vectors
    # ------------------------------------------------------------------
    print(f"\n--- Cosine similarity with generic domain vectors ---")
    generic_path = RESULTS_DIR / "domain_steering_vectors.pt"

    # Move vectors to CPU for comparison
    cpu_vectors = {
        cluster: {layer: vec.cpu() for layer, vec in vectors[cluster].items()}
        for cluster in CLUSTERS
    }

    cosine_sims = compute_cosine_with_generic(cpu_vectors, generic_path)
    if cosine_sims is not None:
        results["cosine_vs_generic"] = cosine_sims
        print("\n  Cosine similarity (cluster vs generic domain, mean across layers):")
        for cluster in CLUSTERS:
            print(f"\n    {cluster}:")
            for g_domain, sims in cosine_sims[cluster].items():
                print(f"      vs {g_domain:16s}  mean={sims['mean']:.4f}  max={sims['max']:.4f}")

    # ------------------------------------------------------------------
    # Save vectors
    # ------------------------------------------------------------------
    vectors_path = RESULTS_DIR / "swebench_cluster_vectors.pt"
    torch.save(cpu_vectors, vectors_path)
    print(f"\nSaved vectors: {vectors_path}")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results_path = RESULTS_DIR / "swebench_cluster_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results: {results_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    cleanup_model(model, device)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY -- BEST CONFIG PER CLUSTER")
    print("=" * 65)
    for cluster in CLUSTERS:
        cfg = results["best_config"][cluster]
        print(f"  {cluster:22s}  layer={cfg['layer']:>2d}  "
              f"alpha={cfg['alpha']:>5.0f}  score={cfg['keyword_score']}")

    if cosine_sims:
        print("\n" + "=" * 65)
        print("COSINE SIMILARITY: CLUSTER vs GENERIC DOMAIN VECTORS")
        print("=" * 65)
        for cluster in CLUSTERS:
            best_generic = max(cosine_sims[cluster].items(),
                               key=lambda x: x[1]["mean"])
            print(f"  {cluster:22s}  closest generic = {best_generic[0]} "
                  f"(mean cos={best_generic[1]['mean']:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
