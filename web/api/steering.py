"""Steering hook and generation logic extracted from the Streamlit app."""

from __future__ import annotations

import functools
from threading import Thread

import torch
from transformers import TextIteratorStreamer


def steering_hook(module, input, output, *, vector, coeff):
    """Forward hook that adds a normalized steering vector to hidden states."""
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


def generate_stream(model, tokenizer, prompt, max_new_tokens, layer, vector=None, coeff=0.0):
    """Generator yielding partial text. Runs model.generate in a thread with TextIteratorStreamer."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    handle = None
    if vector is not None and coeff > 0:
        layers = get_layers(model)
        handle = layers[layer].register_forward_hook(
            functools.partial(steering_hook, vector=vector, coeff=coeff)
        )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    generated = ""
    for text in streamer:
        generated += text
        yield generated

    thread.join()
    if handle is not None:
        handle.remove()
