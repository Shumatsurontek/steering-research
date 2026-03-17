"""
Shared utilities for SAE loading and activation collection.
Supports both SAELens format and custom HF-hooks format.
Works with any model architecture (Transformer, SSM, hybrid).
"""

import functools
import gc
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_layers_hf(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model.blocks
    raise ValueError(f"Cannot find layers in {type(model)}")


# ── SAE loading ─────────────────────────────────────────────────────────────

class SimpleSAE(nn.Module):
    """Lightweight SAE wrapper for inference only."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec):
        super().__init__()
        self.W_enc = nn.Parameter(W_enc, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)
        self.cfg = type("Cfg", (), {"d_in": W_enc.shape[0], "d_sae": W_enc.shape[1]})()

    def encode(self, x):
        return torch.relu(x.float() @ self.W_enc.float() + self.b_enc.float())

    def decode(self, z):
        return z @ self.W_dec.float() + self.b_dec.float()


def load_sae(sae_path: str, device: str = "cpu"):
    """Load SAE from either custom format or SAELens format."""
    p = Path(sae_path)

    # Custom format (from train_sae_hf.py)
    weights_file = p / "sae_weights.pt"
    if weights_file.exists():
        state = torch.load(weights_file, map_location=device, weights_only=True)
        return SimpleSAE(
            state["W_enc"].to(device),
            state["b_enc"].to(device),
            state["W_dec"].to(device),
            state["b_dec"].to(device),
        )

    # SAELens format
    from sae_lens import SAE
    return SAE.load_from_disk(sae_path, device=device)


# ── Activation collection ───────────────────────────────────────────────────

def _is_transformerlens_supported(model_id: str) -> bool:
    """Check if TransformerLens supports this model."""
    try:
        from transformer_lens.loading_from_pretrained import get_official_model_name
        get_official_model_name(model_id)
        return True
    except (ValueError, ImportError):
        return False


def compute_domain_activations_hf(model_id: str, sae, prompts: list, layer: int, device: str):
    """
    Collect SAE activations using HuggingFace model + forward hooks.
    Works with any architecture.
    """
    dtype = torch.bfloat16 if "LFM2" in model_id else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    layers = get_layers_hf(model)
    hook_output = {}

    def hook_fn(module, input, output, key="act"):
        hidden = output[0] if isinstance(output, tuple) else output
        hook_output[key] = hidden.detach()

    all_acts = []
    for prompt in prompts:
        handle = layers[layer].register_forward_hook(
            functools.partial(hook_fn, key="act")
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        residual = hook_output["act"].squeeze(0).float()
        feat_acts = sae.encode(residual)
        mean_acts = feat_acts.mean(dim=0)
        all_acts.append(mean_acts.detach().cpu())

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.stack(all_acts)


def compute_domain_activations_tl(model_id: str, sae, prompts: list, layer: int, device: str):
    """Collect SAE activations using TransformerLens (Qwen, Llama, etc.)."""
    from transformer_lens import HookedTransformer

    hook_name = f"blocks.{layer}.hook_resid_post"
    model = HookedTransformer.from_pretrained_no_processing(model_id, device=device)

    all_acts = []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer + 1)
        residual = cache[hook_name].squeeze(0).float()
        feat_acts = sae.encode(residual)
        mean_acts = feat_acts.mean(dim=0)
        all_acts.append(mean_acts.detach().cpu())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.stack(all_acts)


def compute_domain_activations(model_id: str, sae, prompts: list, layer: int, device: str):
    """Auto-dispatch: use TransformerLens if supported, else HF hooks."""
    if _is_transformerlens_supported(model_id):
        print(f"    Using TransformerLens for {model_id}")
        return compute_domain_activations_tl(model_id, sae, prompts, layer, device)
    else:
        print(f"    Using HF hooks for {model_id} (not in TransformerLens registry)")
        return compute_domain_activations_hf(model_id, sae, prompts, layer, device)


def compute_all_domain_activations(
    model_id: str, sae, domain_prompts: dict[str, list], layer: int, device: str,
) -> dict[str, torch.Tensor]:
    """Process all domains with a single model load. Returns {domain: (n_prompts, d_sae)}."""
    use_tl = _is_transformerlens_supported(model_id)

    if use_tl:
        from transformer_lens import HookedTransformer
        print(f"    Using TransformerLens for {model_id} (single load)")
        hook_name = f"blocks.{layer}.hook_resid_post"
        model = HookedTransformer.from_pretrained_no_processing(model_id, device=device)

        results = {}
        for domain, prompts in domain_prompts.items():
            all_acts = []
            for prompt in prompts:
                tokens = model.to_tokens(prompt)
                _, cache = model.run_with_cache(tokens, stop_at_layer=layer + 1)
                residual = cache[hook_name].squeeze(0).float()
                feat_acts = sae.encode(residual)
                all_acts.append(feat_acts.mean(dim=0).detach().cpu())
            results[domain] = torch.stack(all_acts)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    else:
        print(f"    Using HF hooks for {model_id} (single load)")
        dtype = torch.bfloat16 if "LFM2" in model_id else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        layers = get_layers_hf(model)
        hook_output = {}

        def hook_fn(module, input, output, key="act"):
            hidden = output[0] if isinstance(output, tuple) else output
            hook_output[key] = hidden.detach()

        results = {}
        for domain, prompts in domain_prompts.items():
            all_acts = []
            for prompt in prompts:
                handle = layers[layer].register_forward_hook(
                    functools.partial(hook_fn, key="act")
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**inputs)
                handle.remove()
                residual = hook_output["act"].squeeze(0).float()
                feat_acts = sae.encode(residual)
                all_acts.append(feat_acts.mean(dim=0).detach().cpu())
            results[domain] = torch.stack(all_acts)

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
