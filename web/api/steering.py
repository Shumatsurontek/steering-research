"""Steering hook and generation logic extracted from the Streamlit app."""

from __future__ import annotations

import functools
from threading import Thread

import torch
from transformers import TextIteratorStreamer


def steering_hook(module, input, output, *, vector, coeff, mode="additive"):
    """Forward hook that steers hidden states via additive or multiplicative injection."""
    hidden = output[0] if isinstance(output, tuple) else output
    vec_dev = vector.to(hidden.device, dtype=hidden.dtype)
    vec_normed = vec_dev / (vec_dev.norm() + 1e-8)

    if mode == "multiplicative":
        # Scale existing activations: h' = h * (1 + α * v̂)
        # Preserves the activation manifold — better for MC loglikelihood
        steered = hidden * (1.0 + coeff * vec_normed)
    else:
        # Classic additive: h' = h + α * v̂
        steered = hidden + coeff * vec_normed

    if isinstance(output, tuple):
        return (steered,) + output[1:]
    return steered


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find layers in {type(model)}")


def generate_stream(model, tokenizer, prompt, max_new_tokens, layer, vector=None, coeff=0.0, gen_params=None, steering_mode="additive"):
    """Generator yielding partial text. Runs model.generate in a thread with TextIteratorStreamer."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    handle = None
    if vector is not None and coeff > 0:
        layers = get_layers(model)
        handle = layers[layer].register_forward_hook(
            functools.partial(steering_hook, vector=vector, coeff=coeff, mode=steering_mode)
        )

    # Model-specific generation parameters
    sampling = gen_params or {}
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=sampling.get("do_sample", True),
        temperature=sampling.get("temperature", 0.7),
    )
    if "top_p" in sampling:
        gen_kwargs["top_p"] = sampling["top_p"]
    if "top_k" in sampling:
        gen_kwargs["top_k"] = sampling["top_k"]
    if "min_p" in sampling:
        gen_kwargs["min_p"] = sampling["min_p"]
    if "repetition_penalty" in sampling:
        gen_kwargs["repetition_penalty"] = sampling["repetition_penalty"]

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    generated = ""
    for text in streamer:
        generated += text
        yield generated

    thread.join()
    if handle is not None:
        handle.remove()
