"""
Dynamic Steering Orchestrator — binds domain-specific steering vectors
at each step of an agentic plan.

Loads the model once and swaps forward hooks per step, using best configs
from the domain vector sweep (results/domain_vectors_results.json).

Usage:
    python -m src.agents.steering_orchestrator
"""

import gc
import json
import functools
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse domain keywords from the extraction module
from src.steering.domain_vectors import DOMAIN_KEYWORDS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODEL_ID = "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Hook (same pattern as domain_vectors.py / gsm8k_benchmark.py)
# ---------------------------------------------------------------------------
def _steering_hook(module, input, output, *, vector, coeff):
    """Forward hook: add normalized steering vector to residual stream."""
    hidden = output[0] if isinstance(output, tuple) else output
    steered = hidden + coeff * vector.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


# ---------------------------------------------------------------------------
# Device / model helpers (matching existing codebase conventions)
# ---------------------------------------------------------------------------
def _get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _load_model(model_id, device, dtype):
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class SteeringOrchestrator:
    """Orchestrates domain-specific steering for multi-step agentic tasks."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        vectors_path: str = "results/domain_steering_vectors.pt",
        configs_path: str = "results/domain_vectors_results.json",
    ):
        self.device, self.dtype = _get_device_and_dtype()
        print(f"Device: {self.device}  |  dtype: {self.dtype}")

        # Load model once
        print(f"Loading model: {model_id}")
        self.model, self.tokenizer = _load_model(model_id, self.device, self.dtype)

        # Load steering library (vectors keyed by model variant -> domain -> layer)
        root = Path(__file__).resolve().parents[2]
        vpath = root / vectors_path
        print(f"Loading vectors: {vpath}")
        raw_vectors = torch.load(vpath, map_location="cpu", weights_only=True)
        # We use the instruct variant
        self.vectors = raw_vectors["instruct"]  # {domain: {layer_int: tensor}}

        # Load best configs per domain
        cpath = root / configs_path
        with open(cpath, "r", encoding="utf-8") as f:
            all_configs = json.load(f)
        self.best_configs = all_configs["instruct"]["best_config"]
        # e.g. {"code_reading": {"layer": 15, "alpha": 30.0, "keyword_score": 6}, ...}

        print("Orchestrator ready.\n")

    # ------------------------------------------------------------------
    def _apply_steering(self, domain: str):
        """Register forward hook for the given domain using best (layer, alpha).
        Returns the hook handle."""
        cfg = self.best_configs[domain]
        layer = cfg["layer"]
        alpha = cfg["alpha"]
        vec = self.vectors[domain][layer]
        vec_normed = vec / (vec.norm() + 1e-8)

        handle = self.model.model.layers[layer].register_forward_hook(
            functools.partial(_steering_hook, vector=vec_normed, coeff=alpha)
        )
        return handle

    def _remove_steering(self, handle):
        """Remove a steering hook."""
        if handle is not None:
            handle.remove()

    # ------------------------------------------------------------------
    def generate(self, prompt: str, domain: str = None, max_new_tokens: int = 256) -> str:
        """Generate with optional domain steering. If domain is None, no steering."""
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        handle = None
        if domain is not None:
            handle = self._apply_steering(domain)

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1e-7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            self._remove_steering(handle)

        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        resp = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Strip Qwen3 thinking block if present
        if "</think>" in resp:
            resp = resp.split("</think>")[-1].strip()
        return resp.strip(), len(new_tokens)

    # ------------------------------------------------------------------
    def execute_plan(
        self, problem_statement: str, plan: list, variant: str = "dynamic"
    ) -> list:
        """Execute a multi-step plan, switching steering vectors at each step.

        variant:
            "baseline"  — no steering on any step
            "static"    — same domain (code_reading) on every step
            "dynamic"   — switch domain per step as specified in the plan

        plan = [
            {"step": "read", "domain": "code_reading", "prompt": "..."},
            ...
        ]

        Returns list of {"step", "domain", "output", "tokens"}.
        """
        results = []
        prev_output = ""

        for i, step in enumerate(plan):
            # Build the prompt with context from previous step
            if i == 0:
                full_prompt = (
                    f"Problem: {problem_statement}\n\n"
                    f"Task ({step['step']}): {step['prompt']}"
                )
            else:
                full_prompt = (
                    f"Problem: {problem_statement}\n\n"
                    f"Previous step output:\n{prev_output[:400]}\n\n"
                    f"Task ({step['step']}): {step['prompt']}"
                )

            # Decide which domain to apply
            if variant == "baseline":
                domain = None
            elif variant == "static":
                domain = "code_reading"
            else:  # dynamic
                domain = step["domain"]

            output, n_tokens = self.generate(full_prompt, domain=domain)
            prev_output = output

            results.append({
                "step": step["step"],
                "domain": step["domain"] if variant == "dynamic" else (
                    "code_reading" if variant == "static" else "none"
                ),
                "applied_domain": domain if domain else "none",
                "output": output,
                "tokens": int(n_tokens),
            })

        return results


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def score_domain_keywords(text: str, domain: str) -> int:
    """Count domain-specific keywords in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in DOMAIN_KEYWORDS[domain] if kw in text_lower)


def score_coherence(steps: list) -> float:
    """Simple coherence metric: fraction of steps whose output shares at least
    one non-trivial word (>4 chars) with the previous step's output."""
    if len(steps) < 2:
        return 1.0
    coherent = 0
    for i in range(1, len(steps)):
        prev_words = set(
            w.lower() for w in steps[i - 1]["output"].split() if len(w) > 4
        )
        curr_words = set(
            w.lower() for w in steps[i]["output"].split() if len(w) > 4
        )
        if prev_words & curr_words:
            coherent += 1
    return coherent / (len(steps) - 1)


def evaluate_run(steps: list) -> dict:
    """Compute metrics for a single run (list of step results)."""
    total_tokens = sum(s["tokens"] for s in steps)
    total_length = sum(len(s["output"]) for s in steps)

    # Per-step domain keyword hits (using the step's intended domain)
    per_step_keywords = []
    for s in steps:
        domain = s["domain"] if s["domain"] != "none" else "code_reading"
        hits = score_domain_keywords(s["output"], domain)
        per_step_keywords.append({
            "step": s["step"],
            "domain": domain,
            "keyword_hits": hits,
        })

    coherence = score_coherence(steps)
    total_keyword_hits = sum(pk["keyword_hits"] for pk in per_step_keywords)

    return {
        "total_tokens": total_tokens,
        "total_output_chars": total_length,
        "total_keyword_hits": total_keyword_hits,
        "per_step_keywords": per_step_keywords,
        "coherence": round(coherence, 3),
    }


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "name": "simple_bug_fix",
        "problem": (
            "calculate_total() crashes with AttributeError when items list "
            "contains None values"
        ),
        "plan": [
            {
                "step": "read",
                "domain": "code_reading",
                "prompt": (
                    "Read and understand the calculate_total() function. "
                    "Trace how it processes each item in the list."
                ),
            },
            {
                "step": "analyze",
                "domain": "bug_analysis",
                "prompt": (
                    "Identify the root cause of the AttributeError when None "
                    "values appear in the items list."
                ),
            },
            {
                "step": "fix",
                "domain": "patch_writing",
                "prompt": (
                    "Write a minimal code fix that handles None values in the "
                    "items list without changing the function's public API."
                ),
            },
        ],
    },
    {
        "name": "test_failure",
        "problem": (
            "test_parse_date fails after timezone handling was changed in utils.py"
        ),
        "plan": [
            {
                "step": "read",
                "domain": "code_reading",
                "prompt": (
                    "Read the parse_date function in utils.py and understand "
                    "how it handles timezone information."
                ),
            },
            {
                "step": "analyze",
                "domain": "bug_analysis",
                "prompt": (
                    "Analyze why the test_parse_date test is failing after the "
                    "timezone handling change."
                ),
            },
            {
                "step": "test",
                "domain": "test_reasoning",
                "prompt": (
                    "Determine what the test expects vs what the function now "
                    "returns, and whether the test or the code is wrong."
                ),
            },
            {
                "step": "fix",
                "domain": "patch_writing",
                "prompt": (
                    "Write the fix — update either the test or the function so "
                    "they are consistent with correct timezone handling."
                ),
            },
        ],
    },
    {
        "name": "feature_regression",
        "problem": (
            "Adding support for nested serialization broke the existing flat "
            "serialization"
        ),
        "plan": [
            {
                "step": "read_new",
                "domain": "code_reading",
                "prompt": (
                    "Read the new nested serialization code and understand its "
                    "approach to handling nested objects."
                ),
            },
            {
                "step": "read_old",
                "domain": "code_reading",
                "prompt": (
                    "Read the original flat serialization code and identify the "
                    "interface contract it provides."
                ),
            },
            {
                "step": "analyze",
                "domain": "bug_analysis",
                "prompt": (
                    "Identify exactly where the new nested serialization logic "
                    "breaks the existing flat serialization behavior."
                ),
            },
            {
                "step": "fix",
                "domain": "patch_writing",
                "prompt": (
                    "Write a fix that supports both nested and flat serialization "
                    "without breaking the existing API contract."
                ),
            },
            {
                "step": "test",
                "domain": "test_reasoning",
                "prompt": (
                    "Write regression tests that verify both flat and nested "
                    "serialization work correctly after the fix."
                ),
            },
        ],
    },
]

VARIANTS = ["baseline", "static", "dynamic"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEERING ORCHESTRATOR — DYNAMIC DOMAIN SWITCHING DEMO")
    print("=" * 70)

    orch = SteeringOrchestrator()

    all_results = {
        "model": MODEL_ID,
        "device": orch.device,
        "variants": VARIANTS,
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        sname = scenario["name"]
        print(f"\n{'='*70}")
        print(f"SCENARIO: {sname}")
        print(f"Problem: {scenario['problem']}")
        print(f"Steps: {' -> '.join(s['step'] for s in scenario['plan'])}")
        print(f"{'='*70}")

        scenario_results = {
            "problem": scenario["problem"],
            "plan_steps": [s["step"] for s in scenario["plan"]],
            "variants": {},
        }

        for variant in VARIANTS:
            print(f"\n  --- Variant: {variant} ---")
            t0 = time.time()

            steps = orch.execute_plan(
                problem_statement=scenario["problem"],
                plan=scenario["plan"],
                variant=variant,
            )
            elapsed = time.time() - t0

            metrics = evaluate_run(steps)
            metrics["elapsed_seconds"] = round(elapsed, 1)

            # Print summary
            print(f"    Time: {elapsed:.1f}s  |  Tokens: {metrics['total_tokens']}  "
                  f"|  Keywords: {metrics['total_keyword_hits']}  "
                  f"|  Coherence: {metrics['coherence']}")
            for sk in metrics["per_step_keywords"]:
                print(f"      {sk['step']:12s}  domain={sk['domain']:16s}  "
                      f"kw_hits={sk['keyword_hits']}")

            # Print step outputs (truncated)
            for s in steps:
                preview = s["output"][:120].replace("\n", " ")
                print(f"    [{s['step']}] ({s['tokens']} tok) {preview}...")

            scenario_results["variants"][variant] = {
                "steps": [
                    {
                        "step": s["step"],
                        "applied_domain": s["applied_domain"],
                        "tokens": s["tokens"],
                        "output": s["output"],
                    }
                    for s in steps
                ],
                "metrics": metrics,
            }

        all_results["scenarios"][sname] = scenario_results

    # ------------------------------------------------------------------
    # Cross-scenario summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CROSS-SCENARIO SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<22s} {'Variant':<10s} {'Tokens':>7s} {'KW Hits':>8s} "
          f"{'Coherence':>10s} {'Time(s)':>8s}")
    print("-" * 70)

    for sname, sdata in all_results["scenarios"].items():
        for variant in VARIANTS:
            m = sdata["variants"][variant]["metrics"]
            print(f"{sname:<22s} {variant:<10s} {m['total_tokens']:>7d} "
                  f"{m['total_keyword_hits']:>8d} {m['coherence']:>10.3f} "
                  f"{m['elapsed_seconds']:>8.1f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = RESULTS_DIR / "orchestrator_demo_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Cleanup
    del orch.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
