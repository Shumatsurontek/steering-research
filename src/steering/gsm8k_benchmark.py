"""
GSM8K Full Benchmark — Validate steering gains on the real test set (1319 problems).

Uses lm-eval (EleutherAI) for standardized evaluation, comparing:
- Baseline (no steering) on Qwen3-0.6B instruct + base
- Steered with best configs from our pilot study:
  - Instruct: Layer 25 @ α=60 (+10% on 10 problems)
  - Base: Layer 20 @ α=100 (+20% on 10 problems)

Subclasses HFLM to inject normalized steering hooks, matching our methodology.
"""

import json
import functools
from pathlib import Path
from contextlib import contextmanager

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

SLM_INSTRUCT_ID = "Qwen/Qwen3-0.6B"
SLM_BASE_ID = "Qwen/Qwen3-0.6B-Base"

# Contrastive prompts for extracting math reasoning steering vectors
MATH_REASONING_PROMPTS = [
    "Let me solve this step by step. First, I need to identify the key quantities and relationships.",
    "To find the answer, I'll break this into smaller calculations and verify each step.",
    "The mathematical approach here involves setting up equations from the given information.",
    "Working through this problem: I need to calculate the total by combining the given rates.",
    "Step 1: Identify the given values. Step 2: Set up the equation. Step 3: Solve and verify.",
    "This is an arithmetic problem. Let me compute each part carefully to avoid errors.",
    "The key insight is to convert the word problem into mathematical operations.",
    "I'll use systematic reasoning: define variables, write equations, solve step by step.",
    "Breaking down: what do we know? What do we need to find? What operations connect them?",
    "Mathematical reasoning requires precision. Let me track each quantity carefully.",
]

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

# Best steering configs from pilot study (10 problems)
STEERING_CONFIGS = {
    "instruct": {"layer": 25, "coeff": 60.0},
    "base": {"layer": 20, "coeff": 100.0},
}


def _steering_hook(module, input, output, *, vector, coeff):
    """Forward hook that adds normalized steering vector to hidden states."""
    hidden = output[0] if isinstance(output, tuple) else output
    vec_normed = vector / (vector.norm() + 1e-8)
    steered = hidden + coeff * vec_normed.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


@contextmanager
def apply_steering(model, layer, vector, coeff):
    """Context manager to apply steering hook during forward pass."""
    vec_normed = vector / (vector.norm() + 1e-8)
    handle = model.model.layers[layer].register_forward_hook(
        functools.partial(_steering_hook, vector=vec_normed, coeff=coeff)
    )
    try:
        yield
    finally:
        handle.remove()


class Qwen3HFLM(HFLM):
    """HFLM with Qwen3 thinking disabled for faster benchmark evaluation.

    Qwen3 models generate <think>...</think> blocks before answering, burning
    most of the generation budget on internal reasoning. We increase max_gen_toks
    to accommodate this, since the standard 256 may cut off the actual answer.
    """

    @property
    def max_gen_toks(self):
        return 1024  # Default is 256, too low when <think> tokens are generated


class SteeredHFLM(Qwen3HFLM):
    """HFLM with normalized steering hook on a specific layer."""

    def __init__(self, steering_vector, steering_layer, steering_coeff, **kwargs):
        super().__init__(**kwargs)
        self._steering_vector = steering_vector
        self._steering_layer = steering_layer
        self._steering_coeff = steering_coeff

    def _model_call(self, *args, **kwargs):
        with apply_steering(
            self.model, self._steering_layer,
            self._steering_vector, self._steering_coeff
        ):
            return super()._model_call(*args, **kwargs)

    def _model_generate(self, *args, **kwargs):
        with apply_steering(
            self.model, self._steering_layer,
            self._steering_vector, self._steering_coeff
        ):
            return super()._model_generate(*args, **kwargs)


def extract_steering_vectors(model_id, device, dtype):
    """Extract contrastive steering vectors (math reasoning vs neutral)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Extracting steering vectors from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers

    def get_activations(prompts):
        all_acts = {i: [] for i in range(n_layers)}
        for prompt in prompts:
            cache = {}
            handles = []
            for i in range(n_layers):
                h = model.model.layers[i].register_forward_hook(
                    functools.partial(_gather_hook, cache=cache, layer_idx=i)
                )
                handles.append(h)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)

            for h in handles:
                h.remove()

            for i in range(n_layers):
                all_acts[i].append(cache[i][0, -1, :])  # last token

        return {i: torch.stack(acts).mean(dim=0) for i, acts in all_acts.items()}

    math_acts = get_activations(MATH_REASONING_PROMPTS)
    neutral_acts = get_activations(NEUTRAL_PROMPTS)

    vectors = {}
    for i in range(n_layers):
        vectors[i] = math_acts[i] - neutral_acts[i]

    # Clean up the extraction model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vectors, tokenizer


def _gather_hook(module, input, output, *, cache, layer_idx):
    hidden = output[0] if isinstance(output, tuple) else output
    cache[layer_idx] = hidden.detach().cpu()


def _extract_metrics(eval_results, task_name="gsm8k"):
    """Pull accuracy metrics out of lm_eval's nested results dict."""
    task_results = eval_results["results"].get(task_name, {})
    return {
        k: v for k, v in task_results.items()
        if not k.startswith("alias")
    }


def run_benchmark(model_id, model_label, steering_config, vectors, device, dtype,
                  limit=None, task="gsm8k"):
    """
    Run GSM8K benchmark for one model: baseline + steered.

    Uses lm_eval.simple_evaluate() with the specified task variant.

    Args:
        limit: Number of test examples to evaluate (None = full 1319).
        task: lm-eval task name (e.g. 'gsm8k' for 5-shot, 'gsm8k_cot_zeroshot' for 0-shot).
    """
    results = {}
    dtype_str = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}[dtype]
    n_label = f" (first {limit})" if limit else " (full 1319)"

    # --- Baseline (no steering) ---
    print(f"\n  [{model_label}] Running baseline{n_label}...")
    baseline_model = Qwen3HFLM(
        pretrained=model_id,
        device=device,
        dtype=dtype_str,
        batch_size=4,
    )
    baseline_eval = lm_eval.simple_evaluate(
        model=baseline_model,
        tasks=[task],
        batch_size=4,
        limit=limit,
    )
    results["baseline"] = _extract_metrics(baseline_eval, task)
    print(f"    Baseline: {results['baseline']}")

    # Free baseline model before loading steered
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()

    # --- Steered ---
    layer = steering_config["layer"]
    coeff = steering_config["coeff"]
    print(f"\n  [{model_label}] Running steered (L{layer} @ α={coeff}){n_label}...")

    steered_model = SteeredHFLM(
        steering_vector=vectors[layer],
        steering_layer=layer,
        steering_coeff=coeff,
        pretrained=model_id,  # Load fresh model to avoid hook issues
        device=device,
        dtype=dtype_str,
        batch_size=4,
    )
    steered_eval = lm_eval.simple_evaluate(
        model=steered_model,
        tasks=[task],
        batch_size=4,
        limit=limit,
    )
    results[f"steered_L{layer}_a{int(coeff)}"] = _extract_metrics(steered_eval, task)
    print(f"    Steered:  {results[f'steered_L{layer}_a{int(coeff)}']}")

    # Cleanup
    del steered_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GSM8K Benchmark with Steering")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit evaluation to N examples (default: full 1319)")
    parser.add_argument("--model", choices=["instruct", "base", "both"], default="both",
                        help="Which model to benchmark")
    parser.add_argument("--task", default="gsm8k",
                        help="lm-eval task name (e.g. gsm8k, gsm8k_cot_zeroshot)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GSM8K FULL BENCHMARK — STEERING VALIDATION")
    print("=" * 60)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Device: {device}")
    if args.limit:
        print(f"Limit: {args.limit} examples")

    models = []
    if args.model in ("instruct", "both"):
        models.append((SLM_INSTRUCT_ID, "instruct"))
    if args.model in ("base", "both"):
        models.append((SLM_BASE_ID, "base"))

    all_results = {}

    for model_id, label in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_id} ({label})")
        print(f"{'='*60}")

        # Step 1: Extract steering vectors
        vectors, _ = extract_steering_vectors(model_id, device, dtype)
        config = STEERING_CONFIGS[label]
        print(f"  Best config: Layer {config['layer']} @ α={config['coeff']}")

        # Step 2: Run benchmark
        results = run_benchmark(
            model_id, label, config, vectors, device, dtype,
            limit=args.limit, task=args.task,
        )

        if results is not None:
            all_results[label] = results

            # Print summary
            for run_name, run_data in results.items():
                print(f"  {run_name}: {run_data}")

    # Save
    task_suffix = f"_{args.task}" if args.task != "gsm8k" else ""
    suffix = f"{task_suffix}_n{args.limit}" if args.limit else task_suffix
    output_path = RESULTS_DIR / f"gsm8k_benchmark_results{suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
