"""
SWE-bench Verified Evaluation Pipeline with Dynamic Steering.

Compares three patch-generation strategies on real SWE-bench instances:
  1. baseline  -- no steering vectors applied
  2. static    -- code_reading steering on every generation call
  3. dynamic   -- multi-step plan with domain-specific steering per step
     (code_reading -> bug_analysis -> patch_writing)

Generates unified-diff predictions, saves JSONL files, then (optionally)
invokes the swebench harness for automated pass/fail evaluation.

Usage:
    python -m src.agents.swebench_pipeline --limit 20
    python -m src.agents.swebench_pipeline --limit 20 --phase generate
    python -m src.agents.swebench_pipeline --limit 20 --phase evaluate
"""

import argparse
import functools
import gc
import json
import re
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.agents.swebench_rag import (
    SWEBenchRAG,
    RAG_SINGLE_STEP_PROMPT,
    RAG_DYNAMIC_STEP1_PROMPT,
    RAG_DYNAMIC_STEP2_PROMPT,
    RAG_DYNAMIC_STEP3_PROMPT,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
MODEL_ID = "Qwen/Qwen3-0.6B"
VARIANTS = ["baseline", "static", "dynamic"]
RAG_VARIANTS = ["rag_baseline", "rag_static", "rag_dynamic"]

# Repos grouped into clusters for balanced sampling
REPO_CLUSTERS = {
    "django_web": [
        "django/django",
    ],
    "scientific_computing": [
        "sympy/sympy",
        "scikit-learn/scikit-learn",
        "matplotlib/matplotlib",
    ],
    "dev_tooling": [
        "sphinx-doc/sphinx",
        "pytest-dev/pytest",
        "pylint-dev/pylint",
    ],
}

# Target counts per cluster (for --limit 20)
CLUSTER_TARGETS = {
    "django_web": 10,
    "scientific_computing": 5,
    "dev_tooling": 5,
}


# ---------------------------------------------------------------------------
# Forward hook -- same pattern as steering_orchestrator.py
# ---------------------------------------------------------------------------
def _steering_hook(module, input, output, *, vector, coeff):
    """Add a scaled, normalized steering vector to the residual stream."""
    hidden = output[0] if isinstance(output, tuple) else output
    steered = hidden + coeff * vector.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


# ---------------------------------------------------------------------------
# Device / model helpers (matching codebase conventions)
# ---------------------------------------------------------------------------
def _get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _load_model(model_id: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SINGLE_STEP_PROMPT = (
    "You are a software engineer fixing a bug. Given the problem description, "
    "generate a unified diff patch that fixes the issue.\n\n"
    "Problem: {problem_statement}\n\n"
    "Repository: {repo}\n\n"
    "Generate ONLY a unified diff (git diff format). "
    "Start with ```diff and end with ```. Example format:\n"
    "```diff\n"
    "diff --git a/file.py b/file.py\n"
    "--- a/file.py\n"
    "+++ b/file.py\n"
    "@@ -10,3 +10,4 @@\n"
    " existing line\n"
    "-old line\n"
    "+new line\n"
    "+added line\n"
    "```"
)

DYNAMIC_STEP1_PROMPT = (
    "Analyze this bug report and identify which files and functions are "
    "likely involved:\n{problem_statement}"
)

DYNAMIC_STEP2_PROMPT = (
    "Based on the analysis: {step1_output}\n\n"
    "What is the root cause of this bug and how should it be fixed?"
)

DYNAMIC_STEP3_PROMPT = (
    "Based on this diagnosis: {step2_output}\n\n"
    "Generate a unified diff patch. Start with ```diff and end with ```."
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class SWEBenchPipeline:
    """Full SWE-bench Verified evaluation pipeline with dynamic steering."""

    def __init__(self, model_id: str = MODEL_ID, limit: int = 20,
                 use_rag: bool = False, skip_model: bool = False):
        self.model_id = model_id
        self.limit = limit
        self.use_rag = use_rag

        # RAG module (lazy -- clones repos on first use)
        self.rag = SWEBenchRAG() if use_rag else None
        self._rag_cache: dict[str, dict] = {}  # instance_id -> context

        if skip_model:
            # evaluate-only mode: no model needed
            self.model = None
            self.tokenizer = None
            self.vectors = {}
            self.domain_configs = {}
            self.cluster_configs = {}
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            print("Pipeline ready (evaluate-only, no model loaded).\n")
            return

        # Device setup
        self.device, self.dtype = _get_device_and_dtype()
        print(f"Device: {self.device}  |  dtype: {self.dtype}")

        # Load model once -- hooks are swapped between variants
        print(f"Loading model: {model_id}")
        self.model, self.tokenizer = _load_model(model_id, self.device, self.dtype)

        # Load steering vectors (instruct variant)
        vectors_path = RESULTS_DIR / "domain_steering_vectors.pt"
        print(f"Loading steering vectors: {vectors_path}")
        raw = torch.load(vectors_path, map_location="cpu", weights_only=True)
        self.vectors = raw["instruct"]  # {domain: {layer_int: tensor}}

        # Load best domain configs
        configs_path = RESULTS_DIR / "domain_vectors_results.json"
        with open(configs_path, "r", encoding="utf-8") as f:
            all_configs = json.load(f)
        self.domain_configs = all_configs["instruct"]["best_config"]
        # e.g. {"code_reading": {"layer": 15, "alpha": 30.0, ...}, ...}

        # Load SWE-bench cluster configs (informational -- not used for
        # steering directly, but logged for analysis)
        cluster_configs_path = RESULTS_DIR / "swebench_cluster_results.json"
        if cluster_configs_path.exists():
            with open(cluster_configs_path, "r", encoding="utf-8") as f:
                self.cluster_configs = json.load(f)
        else:
            self.cluster_configs = {}

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print("Pipeline ready.\n")

    # ------------------------------------------------------------------
    # Steering helpers
    # ------------------------------------------------------------------
    def _apply_steering(self, domain: str):
        """Register forward hook for a domain. Returns the hook handle."""
        cfg = self.domain_configs[domain]
        layer = cfg["layer"]
        alpha = cfg["alpha"]
        vec = self.vectors[domain][layer]
        vec_normed = vec / (vec.norm() + 1e-8)
        handle = self.model.model.layers[layer].register_forward_hook(
            functools.partial(_steering_hook, vector=vec_normed, coeff=alpha)
        )
        return handle

    def _remove_steering(self, handle):
        if handle is not None:
            handle.remove()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(
        self, prompt: str, domain: str | None = None, max_new_tokens: int = 1024
    ) -> str:
        """Generate text with optional domain steering."""
        messages = [{"role": "user", "content": prompt}]
        # Disable Qwen3 thinking mode: inject empty <think></think> block
        # so the model skips CoT and uses full token budget for the diff.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
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

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        resp = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Strip Qwen3 thinking block if present
        if "</think>" in resp:
            resp = resp.split("</think>")[-1].strip()
        return resp.strip()

    # ------------------------------------------------------------------
    # Instance loading
    # ------------------------------------------------------------------
    def load_instances(self) -> list[dict]:
        """Load SWE-bench Verified and select a balanced sample.

        Targets: ~10 django, ~5 scientific, ~5 dev_tooling (for limit=20).
        Scales proportionally for other limits.
        """
        print("Loading SWE-bench Verified dataset...")
        ds = load_dataset(
            "princeton-nlp/SWE-bench_Verified", split="test", trust_remote_code=True
        )
        print(f"  Total instances in dataset: {len(ds)}")

        # Build repo -> cluster mapping
        repo_to_cluster = {}
        for cluster, repos in REPO_CLUSTERS.items():
            for repo in repos:
                repo_to_cluster[repo] = cluster

        # Group dataset instances by cluster
        by_cluster: dict[str, list[dict]] = {c: [] for c in REPO_CLUSTERS}
        for row in ds:
            repo = row["repo"]
            cluster = repo_to_cluster.get(repo)
            if cluster is not None:
                by_cluster[cluster].append(dict(row))

        for cluster, items in by_cluster.items():
            print(f"  {cluster}: {len(items)} instances available")

        # Compute per-cluster targets proportional to limit
        total_target = sum(CLUSTER_TARGETS.values())
        targets = {}
        for cluster, base_count in CLUSTER_TARGETS.items():
            targets[cluster] = max(1, round(self.limit * base_count / total_target))

        # Adjust if rounding causes overshoot
        while sum(targets.values()) > self.limit:
            # Reduce the largest bucket
            largest = max(targets, key=targets.get)
            targets[largest] -= 1

        selected = []
        for cluster, target_n in targets.items():
            pool = by_cluster[cluster]
            n = min(target_n, len(pool))
            selected.extend(pool[:n])
            print(f"  Selected {n}/{target_n} from {cluster}")

        print(f"  Total selected: {len(selected)}\n")
        return selected

    # ------------------------------------------------------------------
    # Diff extraction
    # ------------------------------------------------------------------
    @staticmethod
    def extract_diff(text: str) -> str:
        """Extract a unified diff from model output.

        Strategy:
          1. Look for content between ```diff ... ``` markers.
          2. Fall back to extracting lines that look like diff content.
          3. Return empty string if nothing valid is found.
        """
        # Strategy 1: fenced diff block
        pattern = r"```diff\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            diff = match.group(1).strip()
            if diff:
                return diff

        # Strategy 2: look for any fenced block that starts with diff --git
        pattern2 = r"```\s*\n(diff --git.*?)```"
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            diff = match2.group(1).strip()
            if diff:
                return diff

        # Strategy 3: extract raw diff-like lines
        diff_lines = []
        in_diff = False
        for line in text.split("\n"):
            stripped = line.rstrip()
            if stripped.startswith("diff --git"):
                in_diff = True
            if in_diff:
                diff_lines.append(stripped)
            elif stripped.startswith(("--- a/", "+++ b/", "@@ ")):
                in_diff = True
                diff_lines.append(stripped)

        if diff_lines:
            return "\n".join(diff_lines)

        return ""

    # ------------------------------------------------------------------
    # Patch quality scoring
    # ------------------------------------------------------------------
    @staticmethod
    def score_patch(diff: str, repo_dir: str = "") -> dict:
        """Score a generated patch on multiple quality dimensions.

        Returns a dict with individual scores and a composite score (0.0-1.0).
        """
        result = {"structural": 0.0, "paths": 0.0, "content": 0.0,
                  "coherence": 0.0, "score": 0.0}

        if not diff.strip():
            return result

        lines = diff.split("\n")

        # --- Structural: proper diff headers ---
        has_diff_git = any(l.startswith("diff --git") for l in lines)
        has_minus = any(l.startswith("--- ") for l in lines)
        has_plus = any(l.startswith("+++ ") for l in lines)
        has_hunk = any(l.startswith("@@ ") for l in lines)
        structural_parts = sum([has_diff_git, has_minus, has_plus, has_hunk])
        result["structural"] = structural_parts / 4.0

        # --- Paths: do referenced files exist in the repo? ---
        if repo_dir:
            pv = SWEBenchPipeline.validate_patch_paths(diff, repo_dir)
            if pv["total"] > 0:
                result["paths"] = pv["valid"] / pv["total"]
            # No repo_dir => neutral score (don't penalize)
        else:
            result["paths"] = 0.5

        # --- Content: ratio of change lines vs total, penalize no-ops ---
        additions = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
        deletions = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
        context = sum(1 for l in lines if l.startswith(" "))
        change_lines = additions + deletions
        body_lines = change_lines + context

        if body_lines == 0:
            result["content"] = 0.0
        else:
            ratio = change_lines / body_lines
            # Sweet spot: 10-80% change lines. Too low = no-op, too high = no context.
            if ratio < 0.05:
                result["content"] = 0.1
            elif ratio > 0.9:
                result["content"] = 0.5
            else:
                result["content"] = min(1.0, ratio * 1.5)

        # --- Coherence: a bug fix should have additions ---
        if change_lines == 0:
            result["coherence"] = 0.0
        elif additions == 0:
            result["coherence"] = 0.3  # delete-only is suspicious for a fix
        elif deletions == 0:
            result["coherence"] = 0.7  # add-only is plausible but less typical
        else:
            result["coherence"] = 1.0  # both add+delete = typical fix

        # --- Composite: weighted average ---
        result["score"] = round(
            0.25 * result["structural"]
            + 0.35 * result["paths"]
            + 0.20 * result["content"]
            + 0.20 * result["coherence"],
            3,
        )
        return result

    # ------------------------------------------------------------------
    # Patch generation per instance
    # ------------------------------------------------------------------
    @staticmethod
    def validate_patch_paths(diff: str, repo_dir: str) -> dict:
        """Check if file paths in a diff exist in the repo checkout.

        Returns {"total": int, "valid": int, "invalid": list[str]}.
        """
        if not diff.strip() or not repo_dir:
            return {"total": 0, "valid": 0, "invalid": []}
        paths = set()
        for line in diff.split("\n"):
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                p = line.split("/", 1)[1] if "/" in line else ""
                if p and p != "/dev/null":
                    paths.add(p)
        repo = Path(repo_dir)
        invalid = [p for p in paths if not (repo / p).exists()]
        return {
            "total": len(paths),
            "valid": len(paths) - len(invalid),
            "invalid": invalid,
        }

    def _get_rag_context(self, instance: dict) -> dict | None:
        """Retrieve RAG context for an instance (cached per instance_id)."""
        if self.rag is None:
            return None
        iid = instance["instance_id"]
        if iid not in self._rag_cache:
            try:
                self._rag_cache[iid] = self.rag.retrieve(instance)
            except Exception as e:
                print(f" [RAG error: {e}]", end="")
                self._rag_cache[iid] = {"files": [], "keywords": [],
                                         "n_candidates": 0, "repo_dir": ""}
        return self._rag_cache[iid]

    def generate_patch(self, instance: dict, variant: str) -> str:
        """Generate a unified diff patch for one instance.

        variant:
          - "baseline" / "rag_baseline": single-step, no steering
          - "static" / "rag_static":     single-step, code_reading steering
          - "dynamic" / "rag_dynamic":   three-step with domain switching

        RAG variants inject real source file contents into the prompt.
        """
        problem = instance["problem_statement"]
        repo = instance["repo"]
        is_rag = variant.startswith("rag_")
        base_variant = variant.removeprefix("rag_") if is_rag else variant

        # Get RAG context if applicable
        rag_ctx = self._get_rag_context(instance) if is_rag else None
        rag_text = ""
        if rag_ctx and rag_ctx["files"]:
            rag_text = SWEBenchRAG.format_context_for_prompt(rag_ctx)

        if base_variant in ("baseline", "static"):
            if is_rag and rag_text:
                prompt = RAG_SINGLE_STEP_PROMPT.format(
                    problem_statement=problem, repo=repo, context=rag_text
                )
            else:
                prompt = SINGLE_STEP_PROMPT.format(
                    problem_statement=problem, repo=repo
                )
            domain = "code_reading" if base_variant == "static" else None
            raw_output = self._generate(prompt, domain=domain, max_new_tokens=1024)
            return self.extract_diff(raw_output)

        # --- Dynamic: 3-step pipeline ---
        if is_rag and rag_text:
            # RAG dynamic: inject file contents at step 1, summary at step 2
            context_summary = "\n".join(
                f"- {f['path']}" for f in (rag_ctx or {}).get("files", [])
            )
            step1_prompt = RAG_DYNAMIC_STEP1_PROMPT.format(
                context=rag_text, problem_statement=problem, repo=repo
            )
        else:
            step1_prompt = DYNAMIC_STEP1_PROMPT.format(problem_statement=problem)
            context_summary = ""

        step1_output = self._generate(
            step1_prompt, domain="code_reading", max_new_tokens=512
        )

        if is_rag and rag_text:
            step2_prompt = RAG_DYNAMIC_STEP2_PROMPT.format(
                step1_output=step1_output[:600],
                context_summary=context_summary or "(see above)"
            )
        else:
            step2_prompt = DYNAMIC_STEP2_PROMPT.format(
                step1_output=step1_output[:600]
            )
        step2_output = self._generate(
            step2_prompt, domain="bug_analysis", max_new_tokens=512
        )

        if is_rag:
            step3_prompt = RAG_DYNAMIC_STEP3_PROMPT.format(
                step2_output=step2_output[:600]
            )
        else:
            step3_prompt = DYNAMIC_STEP3_PROMPT.format(
                step2_output=step2_output[:600]
            )
        step3_output = self._generate(
            step3_prompt, domain="patch_writing", max_new_tokens=1024
        )

        return self.extract_diff(step3_output)

    # ------------------------------------------------------------------
    # Phase 1: Generation
    # ------------------------------------------------------------------
    def run_generation(self) -> dict:
        """Generate patches for all instances across all variants.

        Saves one JSONL file per variant and returns a summary dict.
        """
        instances = self.load_instances()
        n = len(instances)

        variants = RAG_VARIANTS if self.use_rag else VARIANTS

        summary: dict = {
            "model": self.model_id,
            "device": self.device,
            "limit": self.limit,
            "n_instances": n,
            "use_rag": self.use_rag,
            "variants": {},
        }

        for variant in variants:
            print(f"\n{'='*60}")
            print(f"VARIANT: {variant}")
            print(f"{'='*60}")

            predictions = []
            scores = []
            n_valid = 0
            t0 = time.time()

            for i, inst in enumerate(instances):
                iid = inst["instance_id"]
                print(
                    f"  [instance {i+1}/{n}] [variant: {variant}] {iid}...",
                    end="",
                    flush=True,
                )

                t_start = time.time()
                try:
                    diff = self.generate_patch(inst, variant)
                except Exception as e:
                    print(f" ERROR: {e}")
                    diff = ""

                elapsed = time.time() - t_start
                valid = bool(diff.strip())
                if valid:
                    n_valid += 1

                # Score patch quality
                rag_ctx = self._rag_cache.get(iid)
                repo_dir = (rag_ctx or {}).get("repo_dir", "")
                patch_score = self.score_patch(diff, repo_dir)
                scores.append(patch_score)

                path_info = ""
                if repo_dir and valid:
                    path_info = f" paths:{patch_score['paths']:.0%}"

                predictions.append({
                    "instance_id": iid,
                    "model_name_or_path": f"qwen3-0.6b-{variant}",
                    "model_patch": diff,
                })

                status = "ok" if valid else "EMPTY"
                score_str = f" q={patch_score['score']:.2f}" if valid else ""
                print(f" {status}{score_str}{path_info} ({elapsed:.1f}s)")

            total_time = time.time() - t0

            # Save JSONL
            out_path = RESULTS_DIR / f"swebench_predictions_{variant}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in predictions:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            # Compute average scores
            valid_scores = [s for s in scores if s["score"] > 0]
            avg_score = (sum(s["score"] for s in valid_scores) / len(valid_scores)
                         if valid_scores else 0.0)
            avg_paths = (sum(s["paths"] for s in valid_scores) / len(valid_scores)
                         if valid_scores else 0.0)

            print(f"\n  Saved: {out_path}")
            print(f"  Valid patches: {n_valid}/{n} ({100*n_valid/max(n,1):.0f}%)")
            print(f"  Avg quality score: {avg_score:.3f}  |  Avg path validity: {avg_paths:.1%}")
            print(f"  Total time: {total_time:.1f}s")

            summary["variants"][variant] = {
                "predictions_path": str(out_path),
                "n_valid": n_valid,
                "n_total": n,
                "valid_pct": round(100 * n_valid / max(n, 1), 1),
                "avg_quality_score": round(avg_score, 3),
                "avg_path_validity": round(avg_paths, 3),
                "total_seconds": round(total_time, 1),
            }

        return summary

    # ------------------------------------------------------------------
    # Phase 2: Evaluation via swebench harness
    # ------------------------------------------------------------------
    def run_evaluation(self, variant: str) -> dict:
        """Run swebench harness evaluation on one variant's predictions.

        Requires Docker and the swebench package installed.
        Returns a dict with the evaluation result summary.
        """
        predictions_path = RESULTS_DIR / f"swebench_predictions_{variant}.jsonl"
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Predictions not found: {predictions_path}. "
                f"Run --phase generate first."
            )

        run_id = f"steering_{variant}"
        print(f"\nRunning swebench evaluation for variant '{variant}'...")
        print(f"  Predictions: {predictions_path}")
        print(f"  Run ID: {run_id}")

        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--predictions_path", str(predictions_path),
            "--max_workers", "2",
            "--run_id", run_id,
            "--namespace", "",  # Required for Apple Silicon
        ]
        print(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=7200
            )
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"  Evaluation FAILED (exit code {e.returncode})")
            print(f"  stderr: {e.stderr[-1000:]}")
            return {
                "variant": variant,
                "status": "error",
                "returncode": e.returncode,
                "stderr": e.stderr[-1000:],
            }
        except subprocess.TimeoutExpired:
            print("  Evaluation TIMED OUT (2h limit)")
            return {"variant": variant, "status": "timeout"}

        return {
            "variant": variant,
            "status": "success",
            "run_id": run_id,
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, phase: str = "all") -> None:
        """Execute the full pipeline or a specific phase.

        phase: "generate", "evaluate", or "all"
        """
        summary: dict = {}

        if phase in ("generate", "all"):
            summary = self.run_generation()

        if phase in ("evaluate", "all"):
            eval_results = {}
            variants = RAG_VARIANTS if self.use_rag else VARIANTS
            for variant in variants:
                try:
                    result = self.run_evaluation(variant)
                    eval_results[variant] = result
                except FileNotFoundError as e:
                    print(f"  Skipping {variant}: {e}")
                    eval_results[variant] = {"status": "skipped", "reason": str(e)}
            summary["evaluation"] = eval_results

        # Save combined summary
        summary_path = RESULTS_DIR / "swebench_evaluation_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary: {summary_path}")

        # Print final table
        if "variants" in summary:
            print(f"\n{'='*60}")
            print("GENERATION SUMMARY")
            print(f"{'='*60}")
            print(f"{'Variant':<12s} {'Valid':>6s} {'Total':>6s} {'Pct':>6s} {'Time':>8s}")
            print("-" * 42)
            all_variants = RAG_VARIANTS if self.use_rag else VARIANTS
            for variant in all_variants:
                v = summary["variants"].get(variant, {})
                print(
                    f"{variant:<12s} {v.get('n_valid', '?'):>6} "
                    f"{v.get('n_total', '?'):>6} "
                    f"{v.get('valid_pct', '?'):>5}% "
                    f"{v.get('total_seconds', '?'):>7}s"
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        """Release GPU/MPS memory."""
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Model memory released.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Verified evaluation with dynamic steering vectors."
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Number of SWE-bench instances to evaluate (default: 20).",
    )
    parser.add_argument(
        "--phase", type=str, default="all", choices=["generate", "evaluate", "all"],
        help="Pipeline phase to run (default: all).",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_ID,
        help=f"Model ID (default: {MODEL_ID}).",
    )
    parser.add_argument(
        "--rag", action="store_true",
        help="Enable RAG: clone repos and inject real source files into prompts.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SWE-BENCH VERIFIED -- DYNAMIC STEERING EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Limit:  {args.limit}")
    print(f"Phase:  {args.phase}")
    print(f"RAG:    {args.rag}")
    print()

    skip_model = (args.phase == "evaluate")
    pipeline = SWEBenchPipeline(
        model_id=args.model, limit=args.limit,
        use_rag=args.rag, skip_model=skip_model,
    )

    try:
        pipeline.run(phase=args.phase)
    finally:
        pipeline.cleanup()

    print("\nDone.")


if __name__ == "__main__":
    main()
