"""
Capture per-sample outputs from MMLU-Pro benchmark (baseline vs steered).

Runs a small evaluation (n=5) on Qwen3-0.6B in both modes:
  - generate_until: captures the full CoT text generated
  - loglikelihood (MC): captures per-option log-likelihoods and selected answer

Saves structured JSON with question, expected answer, model outputs.

Usage:
    python -m src.steering.mmlu_pro_samples
    python -m src.steering.mmlu_pro_samples --domain history --limit 3
"""

import argparse
import gc
import json
import string
import functools
from pathlib import Path
from contextlib import contextmanager

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TASKS_DIR = Path(__file__).resolve().parent / "tasks" / "mmlu_pro_mc"

MODEL_ID = "Qwen/Qwen3-0.6B"
MID_LAYER = 14
VECTORS_FILE = "mmlu_pro_vectors_qwen3_0.6b.pt"


def _steering_hook(module, input, output, *, vector, coeff):
    hidden = output[0] if isinstance(output, tuple) else output
    vec_normed = vector / (vector.norm() + 1e-8)
    steered = hidden + coeff * vec_normed.to(hidden.device, dtype=hidden.dtype)
    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


@contextmanager
def apply_steering(model, layer, vector, coeff):
    layers = get_layers(model)
    handle = layers[layer].register_forward_hook(
        functools.partial(_steering_hook, vector=vector, coeff=coeff)
    )
    try:
        yield
    finally:
        handle.remove()


class ExtendedHFLM(HFLM):
    @property
    def max_gen_toks(self):
        return 2048


class SteeredHFLM(ExtendedHFLM):
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


# --- MC-only steered (no extended gen) ---
class SteeredHFLM_MC(HFLM):
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


def cleanup():
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def extract_mc_samples(samples, task_name):
    """Extract readable info from loglikelihood samples."""
    records = []
    for s in samples.get(task_name, []):
        doc = s.get("doc", {})
        question = doc.get("question", "")
        options = doc.get("options", [])
        answer_key = doc.get("answer", "")
        # answer_key is like "A", "B", etc.
        answer_idx = string.ascii_uppercase.index(answer_key) if isinstance(answer_key, str) and answer_key in string.ascii_uppercase else -1
        expected = options[answer_idx] if 0 <= answer_idx < len(options) else answer_key

        # resps contains tuples of (loglik, is_greedy) per option
        resps = s.get("resps", [])
        logliks = []
        for r in resps:
            if isinstance(r, list) and len(r) > 0:
                val = r[0]
                if isinstance(val, (list, tuple)):
                    logliks.append(float(val[0]))
                else:
                    logliks.append(float(val))
            else:
                logliks.append(None)

        if logliks:
            selected_idx = max(range(len(logliks)),
                               key=lambda i: logliks[i] if logliks[i] is not None else float('-inf'))
            selected_letter = string.ascii_uppercase[selected_idx] if selected_idx < 26 else "?"
            selected_text = options[selected_idx] if selected_idx < len(options) else "?"
        else:
            selected_letter, selected_text = "?", "?"

        records.append({
            "question": question[:200],
            "options": {string.ascii_uppercase[i]: opt for i, opt in enumerate(options)},
            "expected_answer": answer_key,
            "expected_text": str(expected)[:100],
            "model_selected": selected_letter,
            "model_selected_text": str(selected_text)[:100],
            "correct": selected_letter == answer_key,
            "log_likelihoods": {string.ascii_uppercase[i]: round(ll, 4) if ll is not None else None
                                for i, ll in enumerate(logliks)},
        })
    return records


def extract_gen_samples(samples, task_name):
    """Extract readable info from generate_until samples."""
    records = []
    for s in samples.get(task_name, []):
        doc = s.get("doc", {})
        question = doc.get("question", "")
        options = doc.get("options", [])
        answer_key = doc.get("answer", "")

        # Generated text
        resps = s.get("resps", [[]])
        generated = resps[0][0] if resps and resps[0] else ""
        if isinstance(generated, (list, tuple)):
            generated = generated[0] if generated else ""

        # filtered_resps may have the post-processed version
        filtered = s.get("filtered_resps", [[]])
        filtered_text = filtered[0] if filtered else ""
        if isinstance(filtered_text, list):
            filtered_text = filtered_text[0] if filtered_text else ""

        records.append({
            "question": question[:200],
            "options": {string.ascii_uppercase[i]: opt for i, opt in enumerate(options)},
            "expected_answer": answer_key,
            "generated_text": str(generated)[:500],
            "filtered_response": str(filtered_text)[:200],
        })
    return records


def run_mc_eval(domain, vec, limit, device, dtype):
    """Run loglikelihood evaluation and capture samples."""
    task = f"mmlu_pro_mc_{domain}"
    dtype_str = {torch.float16: "float16", torch.bfloat16: "bfloat16",
                 torch.float32: "float32"}[dtype]

    results = {"domain": domain, "mode": "loglikelihood", "samples": {}}

    # Baseline
    print(f"\n  [MC] [{domain}] Baseline (n={limit})...")
    model = HFLM(pretrained=MODEL_ID, device=device, dtype=dtype_str, batch_size=8)
    eval_out = lm_eval.simple_evaluate(
        model=model, tasks=[task], batch_size=8, limit=limit,
        log_samples=True,
        task_manager=lm_eval.tasks.TaskManager(include_path=str(TASKS_DIR)),
    )
    results["samples"]["baseline"] = extract_mc_samples(eval_out.get("samples", {}), task)
    results["baseline_acc"] = eval_out["results"].get(task, {}).get("acc,none", None)
    del model; cleanup()

    # Steered α=30
    print(f"\n  [MC] [{domain}] Steered α=30 (n={limit})...")
    model = SteeredHFLM_MC(
        steering_vector=vec, steering_layer=MID_LAYER, steering_coeff=30.0,
        pretrained=MODEL_ID, device=device, dtype=dtype_str, batch_size=8,
    )
    eval_out = lm_eval.simple_evaluate(
        model=model, tasks=[task], batch_size=8, limit=limit,
        log_samples=True,
        task_manager=lm_eval.tasks.TaskManager(include_path=str(TASKS_DIR)),
    )
    results["samples"]["steered_a30"] = extract_mc_samples(eval_out.get("samples", {}), task)
    results["steered_a30_acc"] = eval_out["results"].get(task, {}).get("acc,none", None)
    del model; cleanup()

    return results


def run_gen_eval(domain, vec, limit, device, dtype):
    """Run generate_until evaluation and capture samples."""
    task = f"mmlu_pro_{domain}"
    dtype_str = {torch.float16: "float16", torch.bfloat16: "bfloat16",
                 torch.float32: "float32"}[dtype]

    results = {"domain": domain, "mode": "generate_until", "samples": {}}

    # Baseline
    print(f"\n  [Gen] [{domain}] Baseline (n={limit})...")
    model = ExtendedHFLM(pretrained=MODEL_ID, device=device, dtype=dtype_str, batch_size=4)
    eval_out = lm_eval.simple_evaluate(
        model=model, tasks=[task], batch_size=4, limit=limit,
        log_samples=True,
    )
    results["samples"]["baseline"] = extract_gen_samples(eval_out.get("samples", {}), task)
    results["baseline_acc"] = eval_out["results"].get(task, {}).get("exact_match,none", None)
    del model; cleanup()

    # Steered α=30
    print(f"\n  [Gen] [{domain}] Steered α=30 (n={limit})...")
    model = SteeredHFLM(
        steering_vector=vec, steering_layer=MID_LAYER, steering_coeff=30.0,
        pretrained=MODEL_ID, device=device, dtype=dtype_str, batch_size=4,
    )
    eval_out = lm_eval.simple_evaluate(
        model=model, tasks=[task], batch_size=4, limit=limit,
        log_samples=True,
    )
    results["samples"]["steered_a30"] = extract_gen_samples(eval_out.get("samples", {}), task)
    results["steered_a30_acc"] = eval_out["results"].get(task, {}).get("exact_match,none", None)
    del model; cleanup()

    return results


def main():
    parser = argparse.ArgumentParser(description="Capture per-sample MMLU-Pro outputs")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--domain", type=str, default="history",
                        choices=["math", "law", "history"])
    parser.add_argument("--mode", type=str, default="both",
                        choices=["mc", "gen", "both"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    print("=" * 60)
    print("MMLU-PRO SAMPLE CAPTURE")
    print("=" * 60)
    print(f"Device: {device} | Domain: {args.domain} | Limit: {args.limit}")

    vec_path = RESULTS_DIR / VECTORS_FILE
    vectors = torch.load(vec_path, map_location="cpu", weights_only=True)
    vec = vectors[args.domain][MID_LAYER]
    print(f"Loaded vector: {vec_path} [{args.domain}][{MID_LAYER}]")

    all_results = {}

    if args.mode in ("mc", "both"):
        all_results["mc"] = run_mc_eval(args.domain, vec, args.limit, device, dtype)

    if args.mode in ("gen", "both"):
        all_results["gen"] = run_gen_eval(args.domain, vec, args.limit, device, dtype)

    out_path = RESULTS_DIR / f"mmlu_pro_samples_{args.domain}_n{args.limit}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SAMPLE SUMMARY")
    print(f"{'=' * 60}")

    for mode_key, mode_data in all_results.items():
        print(f"\n--- {mode_data['mode'].upper()} ---")
        for cond in ["baseline", "steered_a30"]:
            samples = mode_data["samples"].get(cond, [])
            print(f"\n  [{cond}]")
            for i, s in enumerate(samples[:3]):
                print(f"    Q{i+1}: {s['question'][:80]}...")
                print(f"    Expected: {s['expected_answer']}")
                if mode_key == "mc":
                    print(f"    Selected: {s['model_selected']} (correct={s['correct']})")
                    lls = s.get("log_likelihoods", {})
                    top3 = sorted(lls.items(), key=lambda x: x[1] if x[1] is not None else -999, reverse=True)[:3]
                    print(f"    Top logprobs: {top3}")
                else:
                    gen = s.get("generated_text", "")[:150]
                    print(f"    Generated: {gen}...")
                print()


if __name__ == "__main__":
    main()
