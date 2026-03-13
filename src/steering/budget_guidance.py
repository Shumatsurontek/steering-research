"""
Budget Guidance — Training-Free Reasoning Length Control (arxiv:2506.13752)

Implements a simplified version of the budget guidance approach:
a Gamma-distribution predictor over remaining thinking length that
softly steers token generation to respect a specified budget.

In our calendar context, this controls how much "reasoning" the model
does before outputting the structured JSON, reducing latency without
sacrificing extraction accuracy.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM_PROMPT = (
    "You are a calendar assistant. Extract event details from the user's message "
    "and return a JSON object with: title, date, start_time, end_time, location, "
    "attendees. Today is 2026-03-13 (Friday)."
)

TEST_PROMPTS = [
    "Crée un rendez-vous demain à 14h avec Marie pour discuter du projet.",
    "Schedule a meeting next Monday at 10am with the team in Room 301.",
    "Bloque le créneau de 10h à 11h30 lundi pour la rétrospective avec Jean-Pierre et Fatima.",
    "Can you set up a 30-minute call with Sarah on Wednesday at 3pm?",
    "Cale-moi un truc avec Antoine la semaine pro, plutôt le matin.",
]

# Budget levels to test (in tokens)
BUDGETS = [32, 64, 128, 256, 512]


@dataclass
class GammaPredictor:
    """
    Simplified Gamma-distribution predictor for remaining generation length.

    At each step t, we model the remaining length R ~ Gamma(alpha, beta).
    The guidance signal softly encourages stopping when E[R] approaches 0.

    This is a simplified version of the full paper — the original trains
    a lightweight predictor on generation traces. Here we use a heuristic
    based on the specified budget.
    """
    budget: int
    alpha: float = 2.0  # Shape parameter (controls distribution shape)
    strength: float = 1.0  # Guidance strength

    def guidance_logit_bias(self, step: int, vocab_size: int, eos_token_id: int) -> torch.Tensor:
        """
        Compute logit bias that encourages EOS as we approach the budget.

        Returns a tensor of shape (vocab_size,) to add to logits.
        """
        bias = torch.zeros(vocab_size)

        if step >= self.budget:
            # Past budget: strongly encourage EOS
            bias[eos_token_id] = 10.0
            return bias

        # Remaining budget fraction
        remaining_frac = (self.budget - step) / self.budget

        # Gamma CDF: probability of finishing by now
        # As remaining_frac → 0, we increasingly push toward EOS
        # Use a smooth sigmoid-like ramp based on remaining fraction
        urgency = 1.0 / (1.0 + math.exp(5.0 * (remaining_frac - 0.3)))

        # Scale the EOS bias by urgency and strength
        eos_bias = self.strength * urgency * 5.0
        bias[eos_token_id] = eos_bias

        # Slightly suppress non-EOS tokens as we approach budget
        # This makes the distribution more peaked toward EOS
        if remaining_frac < 0.2:
            suppression = (0.2 - remaining_frac) / 0.2 * 2.0 * self.strength
            bias -= suppression
            bias[eos_token_id] += suppression  # Don't suppress EOS

        return bias


def generate_with_budget(
    model, tokenizer, prompt: str, device: str,
    budget: int | None = None, strength: float = 1.0,
    max_tokens: int = 512,
) -> dict:
    """
    Generate with optional budget guidance.

    Returns dict with response, token count, and whether budget was met.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    predictor = GammaPredictor(budget=budget, strength=strength) if budget else None

    generated_ids = inputs["input_ids"].clone()
    past_key_values = None
    n_generated = 0
    eos_id = tokenizer.eos_token_id

    # Special tokens that also indicate end
    stop_tokens = {eos_id}
    if hasattr(tokenizer, "added_tokens_encoder"):
        for tok_str, tok_id in tokenizer.added_tokens_encoder.items():
            if "end" in tok_str.lower() or "im_end" in tok_str.lower():
                stop_tokens.add(tok_id)

    with torch.no_grad():
        for step in range(max_tokens):
            if past_key_values is not None:
                model_input = generated_ids[:, -1:]
            else:
                model_input = generated_ids

            outputs = model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # Apply budget guidance
            if predictor is not None:
                bias = predictor.guidance_logit_bias(
                    step=n_generated,
                    vocab_size=logits.shape[-1],
                    eos_token_id=eos_id,
                ).to(logits.device, dtype=logits.dtype)
                logits = logits + bias.unsqueeze(0)

            # Greedy decode
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            n_generated += 1

            if next_token.item() in stop_tokens:
                break

    new_tokens = generated_ids[0, input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Clean thinking tags
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return {
        "response": response,
        "tokens_generated": n_generated,
        "budget": budget,
        "within_budget": n_generated <= budget if budget else True,
    }


def extract_json_from_response(response: str) -> dict | None:
    """Try to extract JSON from a response string."""
    # Find JSON block
    start = response.find("{")
    if start == -1:
        return None
    # Find matching closing brace
    depth = 0
    for i in range(start, len(response)):
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(response[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def has_valid_extraction(response: str) -> dict:
    """Check if response contains valid calendar extraction."""
    data = extract_json_from_response(response)
    if data is None:
        return {"valid": False, "fields": 0, "data": None}

    expected_fields = ["title", "date", "start_time"]
    present = sum(1 for f in expected_fields if f in data and data[f])
    return {
        "valid": present >= 2,  # At least title + date or title + time
        "fields": present,
        "data": data,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUDGET GUIDANCE — Reasoning Length Control")
    print("=" * 60)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Device: {device}")

    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()

    # --- Baseline (no budget) ---
    print("\n--- BASELINE (no budget constraint) ---")
    baselines = []
    for prompt in TEST_PROMPTS:
        result = generate_with_budget(model, tokenizer, prompt, device)
        extraction = has_valid_extraction(result["response"])
        baselines.append({**result, **extraction})
        print(f"  [{prompt[:50]}...]")
        print(f"    tokens={result['tokens_generated']}, valid={extraction['valid']}, "
              f"fields={extraction['fields']}")
        print(f"    → {result['response'][:150]}...")

    avg_baseline_tokens = sum(b["tokens_generated"] for b in baselines) / len(baselines)
    baseline_valid_rate = sum(1 for b in baselines if b["valid"]) / len(baselines)

    print(f"\n  Baseline avg tokens: {avg_baseline_tokens:.0f}")
    print(f"  Baseline valid rate: {baseline_valid_rate:.0%}")

    # --- Budget sweep ---
    print("\n--- BUDGET SWEEP ---")
    all_results = {}

    for budget in BUDGETS:
        print(f"\n  Budget: {budget} tokens")
        budget_results = []

        for prompt in TEST_PROMPTS:
            result = generate_with_budget(
                model, tokenizer, prompt, device, budget=budget, strength=1.0
            )
            extraction = has_valid_extraction(result["response"])
            budget_results.append({**result, **extraction, "prompt": prompt[:60]})
            print(f"    [{prompt[:40]}...] → tokens={result['tokens_generated']}, "
                  f"valid={extraction['valid']}, within={result['within_budget']}")

        avg_tokens = sum(r["tokens_generated"] for r in budget_results) / len(budget_results)
        valid_rate = sum(1 for r in budget_results if r["valid"]) / len(budget_results)
        within_rate = sum(1 for r in budget_results if r["within_budget"]) / len(budget_results)
        token_savings = (1 - avg_tokens / avg_baseline_tokens) * 100 if avg_baseline_tokens else 0

        all_results[budget] = {
            "results": budget_results,
            "avg_tokens": round(avg_tokens, 1),
            "valid_rate": round(valid_rate, 3),
            "within_budget_rate": round(within_rate, 3),
            "token_savings_pct": round(token_savings, 1),
        }
        print(f"    → avg={avg_tokens:.0f} tokens, valid={valid_rate:.0%}, "
              f"within_budget={within_rate:.0%}, savings={token_savings:.0f}%")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BUDGET GUIDANCE SUMMARY")
    print("=" * 60)
    print(f"{'Budget':>8s} | {'Avg Tokens':>10s} | {'Valid':>6s} | {'Within':>7s} | {'Savings':>8s}")
    print("─" * 50)
    print(f"{'∞':>8s} | {avg_baseline_tokens:>10.0f} | {baseline_valid_rate:>5.0%} | {'100%':>7s} | {'0%':>8s}")
    for budget in BUDGETS:
        r = all_results[budget]
        print(f"{budget:>8d} | {r['avg_tokens']:>10.1f} | {r['valid_rate']:>5.0%} | "
              f"{r['within_budget_rate']:>6.0%} | {r['token_savings_pct']:>7.1f}%")

    # --- Save ---
    output = {
        "model": MODEL_ID,
        "budgets": BUDGETS,
        "baseline": {
            "avg_tokens": round(avg_baseline_tokens, 1),
            "valid_rate": round(baseline_valid_rate, 3),
            "results": [{k: v for k, v in b.items() if k != "data"}
                       for b in baselines],
        },
        "budget_results": {
            str(b): {k: v for k, v in r.items() if k != "results"}
            for b, r in all_results.items()
        },
        "budget_details": {
            str(b): [{k: v for k, v in entry.items() if k != "data"}
                    for entry in r["results"]]
            for b, r in all_results.items()
        },
    }
    path = RESULTS_DIR / "budget_guidance_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
