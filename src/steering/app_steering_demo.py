"""
Streamlit demo: compare baseline, contrastive, and feature-targeted steering
in real time on Qwen3-0.6B or Qwen3-4B.

Usage:
    streamlit run src/steering/app_steering_demo.py
"""

import functools
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from transformer_lens import HookedTransformer

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TARGET_DOMAINS = ["math", "law", "history"]

MODEL_CONFIGS = {
    "Qwen3-0.6B": {
        "model_id": "Qwen/Qwen3-0.6B",
        "layer": 14,
        "sae_dir": "sae_qwen3_0.6b_L14_8x",
        "vectors": "mmlu_pro_vectors_qwen3_0.6b.pt",
    },
    "Qwen3-4B": {
        "model_id": "Qwen/Qwen3-4B",
        "layer": 18,
        "sae_dir": "sae_qwen3_4b_L18_8x",
        "vectors": "mmlu_pro_vectors_qwen3_4b.pt",
    },
}

DOMAIN_PROMPTS = {
    "math": [
        "Solve the equation 3x + 7 = 22 for x.",
        "What is the derivative of sin(x) * cos(x)?",
        "Prove that the square root of 2 is irrational.",
        "Calculate the integral of e^x from 0 to 1.",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "What is the probability of rolling two sixes with two dice?",
        "Simplify the expression (x^2 - 4)/(x - 2).",
        "How many ways can you arrange 5 books on a shelf?",
        "What is the Taylor series expansion of ln(1+x)?",
        "Solve the differential equation dy/dx = 2xy.",
    ],
    "law": [
        "What is the difference between civil and criminal law?",
        "Explain the concept of habeas corpus.",
        "What are the elements of a valid contract?",
        "Define the legal principle of stare decisis.",
        "What is the Miranda warning and when must it be given?",
        "Explain the doctrine of sovereign immunity.",
        "What constitutes negligence in tort law?",
        "What is the difference between a felony and a misdemeanor?",
        "Explain the concept of due process under the 14th Amendment.",
        "What are the requirements for obtaining a patent?",
    ],
    "history": [
        "What caused the fall of the Roman Empire?",
        "Describe the main events of the French Revolution.",
        "What was the significance of the Magna Carta?",
        "Explain the causes of World War I.",
        "What was the impact of the Industrial Revolution on society?",
        "Describe the civil rights movement in the United States.",
        "What were the consequences of the Treaty of Versailles?",
        "Explain the rise and fall of the Ottoman Empire.",
        "What was the significance of the Silk Road?",
        "Describe the colonization of the Americas by European powers.",
    ],
}


# ---------------------------------------------------------------------------
# Cached resource loading (keyed by model name)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_hf_model(model_id: str):
    """Load HuggingFace model + tokenizer for generation."""
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_contrastive_vectors(vectors_file: str):
    """Load precomputed contrastive steering vectors."""
    path = RESULTS_DIR / vectors_file
    return torch.load(path, map_location="cpu", weights_only=True)


@st.cache_resource
def load_sae_and_build_feature_vectors(model_id: str, sae_dir: str, layer: int, top_k: int = 20):
    """Load SAE, compute domain activations, build feature-targeted vectors."""
    device = _get_device()
    hook_name = f"blocks.{layer}.hook_resid_post"
    sae_path = RESULTS_DIR / sae_dir

    # Use TransformerLens for SAE encoding
    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_id, device=device
    )
    sae = SAE.load_from_disk(str(sae_path), device=device)

    # Compute domain activations
    activations = {}
    for domain, prompts in DOMAIN_PROMPTS.items():
        all_acts = []
        for prompt in prompts:
            tokens = tl_model.to_tokens(prompt)
            _, cache = tl_model.run_with_cache(tokens, stop_at_layer=layer + 1)
            residual = cache[hook_name]
            flat = residual.squeeze(0)
            feat_acts = sae.encode(flat)
            mean_acts = feat_acts.mean(dim=0)
            all_acts.append(mean_acts.detach().cpu())
        activations[domain] = torch.stack(all_acts).mean(dim=0)

    # Build feature vectors
    domains = list(activations.keys())
    W_dec = sae.W_dec.detach().cpu()

    feature_vectors = {}
    feature_info = {}
    for domain in TARGET_DOMAINS:
        domain_mean = activations[domain]
        other_mean = torch.stack(
            [activations[d] for d in domains if d != domain]
        ).mean(dim=0)
        differential = domain_mean - other_mean

        topk = differential.topk(top_k)
        top_indices = topk.indices
        top_values = topk.values

        # Weighted sum of decoder columns
        weighted_vec = torch.zeros(W_dec.shape[1])
        for idx, val in zip(top_indices, top_values):
            weighted_vec += val.item() * W_dec[idx]

        # Uniform sum
        uniform_vec = W_dec[top_indices].sum(dim=0)

        # Single best
        single_vec = W_dec[top_indices[0]].clone()

        feature_vectors[domain] = {
            "weighted": weighted_vec,
            "uniform": uniform_vec,
            "single": single_vec,
        }
        feature_info[domain] = {
            "top_features": top_indices[:10].tolist(),
            "top_diffs": [f"{v:.3f}" for v in top_values[:10].tolist()],
        }

    # Free TransformerLens model (HF model is used for generation)
    del tl_model, sae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return feature_vectors, feature_info


def highlight_diff(baseline: str, contrastive: str, feature: str) -> str:
    """
    Highlight word-level differences between three outputs.

    Color scheme:
      - No highlight: words shared across all three
      - Yellow background: unique to this output
      - Gray background: shared with one other
    """
    def wordset(text):
        return set(text.lower().split())

    base_words = wordset(baseline)
    cont_words = wordset(contrastive)
    feat_words = wordset(feature)

    def colorize(text, label, this_set, other1_set, other2_set):
        words = text.split()
        html_words = []
        for w in words:
            wl = w.lower()
            in_other1 = wl in other1_set
            in_other2 = wl in other2_set
            if in_other1 and in_other2:
                html_words.append(w)
            elif not in_other1 and not in_other2:
                html_words.append(
                    f'<span style="background-color:rgba(255,200,0,0.3);'
                    f'font-weight:bold;border-radius:3px;padding:1px 3px">{w}</span>'
                )
            else:
                html_words.append(
                    f'<span style="background-color:rgba(100,100,100,0.15);'
                    f'border-radius:3px;padding:1px 3px">{w}</span>'
                )
        return " ".join(html_words)

    base_html = colorize(baseline, "baseline", base_words, cont_words, feat_words)
    cont_html = colorize(contrastive, "contrastive", cont_words, base_words, feat_words)
    feat_html = colorize(feature, "feature", feat_words, base_words, cont_words)

    legend = (
        '<p style="font-size:0.85em;color:#888">'
        '<b>Legend:</b> '
        '<span style="background-color:rgba(255,200,0,0.3);padding:2px 6px;border-radius:3px">unique to this output</span> · '
        '<span style="background-color:rgba(100,100,100,0.15);padding:2px 6px;border-radius:3px">shared with one other</span> · '
        'plain = shared by all three'
        '</p>'
    )

    html = f"""
    {legend}
    <div style="display:flex;gap:16px;margin-top:8px">
        <div style="flex:1;padding:12px;border:1px solid #ddd;border-radius:8px">
            <h4 style="margin-top:0;color:#E74C3C">Baseline</h4>
            <p style="font-size:0.9em;line-height:1.6">{base_html}</p>
        </div>
        <div style="flex:1;padding:12px;border:1px solid #4A90D9;border-radius:8px">
            <h4 style="margin-top:0;color:#4A90D9">Contrastive</h4>
            <p style="font-size:0.9em;line-height:1.6">{cont_html}</p>
        </div>
        <div style="flex:1;padding:12px;border:1px solid #27AE60;border-radius:8px">
            <h4 style="margin-top:0;color:#27AE60">Feature-targeted</h4>
            <p style="font-size:0.9em;line-height:1.6">{feat_html}</p>
        </div>
    </div>
    """
    return html


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Steering hooks
# ---------------------------------------------------------------------------
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


def generate_text(model, tokenizer, prompt, max_new_tokens, layer, vector=None, coeff=0.0):
    """Generate text, optionally with steering."""
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    handle = None
    if vector is not None and coeff > 0:
        layers = get_layers(model)
        handle = layers[layer].register_forward_hook(
            functools.partial(_steering_hook, vector=vector, coeff=coeff)
        )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    if handle is not None:
        handle.remove()

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Steering Vector Demo",
        page_icon="🧭",
        layout="wide",
    )

    st.title("🧭 Activation Steering — Live Demo")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        model_name = st.selectbox("Model", list(MODEL_CONFIGS.keys()), index=0)
        cfg = MODEL_CONFIGS[model_name]

        st.caption(f"Layer {cfg['layer']} · SAE: {cfg['sae_dir']}")

        domain = st.selectbox("Domain", TARGET_DOMAINS, index=0)

        coeff = st.slider(
            "Steering coefficient (α)",
            min_value=0.0, max_value=60.0, value=10.0, step=1.0,
        )

        feature_strategy = st.selectbox(
            "Feature vector strategy",
            ["weighted (top-k × diff)", "uniform (top-k equal)", "single (best feature)"],
            index=0,
        )
        strategy_key = feature_strategy.split(" ")[0]

        top_k = st.slider("Top-k features", min_value=1, max_value=50, value=20)

        max_tokens = st.slider("Max new tokens", min_value=32, max_value=512, value=128, step=32)

        st.markdown("---")
        st.header("📊 Feature Info")

        # Check SAE exists
        sae_path = RESULTS_DIR / cfg["sae_dir"]
        if not sae_path.exists():
            st.error(f"SAE not found: {cfg['sae_dir']}")
            st.stop()

        # Load feature info
        with st.spinner("Loading SAE & building feature vectors..."):
            feature_vectors, feature_info = load_sae_and_build_feature_vectors(
                cfg["model_id"], cfg["sae_dir"], cfg["layer"], top_k
            )

        info = feature_info[domain]
        st.markdown(f"**Top-10 features ({domain}):**")
        for feat_id, diff in zip(info["top_features"], info["top_diffs"]):
            st.text(f"  #{feat_id}: Δ={diff}")

        st.markdown("---")
        st.markdown("**Vector norms:**")
        cv = load_contrastive_vectors(cfg["vectors"])
        contrastive_norm = cv[domain][cfg["layer"]].float().norm().item()
        st.text(f"  Contrastive: {contrastive_norm:.2f}")
        for k, v in feature_vectors[domain].items():
            st.text(f"  Feature {k}: {v.norm().item():.4f}")

    st.markdown(f"Comparing **baseline**, **contrastive**, and **feature-targeted** steering on **{model_name}**")

    # --- Main area ---
    example_prompts = {
        "math": "What is the derivative of x^3 + 2x?",
        "law": "What is the difference between a tort and a crime?",
        "history": "What were the main causes of the American Revolution?",
    }

    prompt = st.text_area(
        "Enter your prompt:",
        value=example_prompts.get(domain, ""),
        height=80,
    )

    if st.button("🚀 Generate", type="primary", use_container_width=True):
        with st.spinner("Loading model..."):
            model, tokenizer = load_hf_model(cfg["model_id"])
            contrastive_vectors = load_contrastive_vectors(cfg["vectors"])

        contrastive_vec = contrastive_vectors[domain][cfg["layer"]]
        feature_vec = feature_vectors[domain][strategy_key]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📋 Baseline")
            with st.spinner("Generating..."):
                baseline_text = generate_text(
                    model, tokenizer, prompt, max_tokens, cfg["layer"],
                    vector=None, coeff=0,
                )
            st.markdown(f"```\n{baseline_text}\n```")

        with col2:
            st.subheader(f"🔀 Contrastive (α={coeff})")
            with st.spinner("Generating..."):
                contrastive_text = generate_text(
                    model, tokenizer, prompt, max_tokens, cfg["layer"],
                    vector=contrastive_vec, coeff=coeff,
                )
            st.markdown(f"```\n{contrastive_text}\n```")

        with col3:
            st.subheader(f"🎯 Feature-targeted (α={coeff})")
            with st.spinner("Generating..."):
                feature_text = generate_text(
                    model, tokenizer, prompt, max_tokens, cfg["layer"],
                    vector=feature_vec, coeff=coeff,
                )
            st.markdown(f"```\n{feature_text}\n```")

        # Visual diff
        st.markdown("---")
        st.subheader("🔍 Output Diff")
        diff_html = highlight_diff(baseline_text, contrastive_text, feature_text)
        st.markdown(diff_html, unsafe_allow_html=True)

        # Quick stats
        st.markdown("---")
        st.subheader("📊 Quick Comparison")
        metrics_cols = st.columns(3)
        for i, (label, text) in enumerate([
            ("Baseline", baseline_text),
            ("Contrastive", contrastive_text),
            ("Feature-targeted", feature_text),
        ]):
            with metrics_cols[i]:
                words = len(text.split())
                chars = len(text)
                st.metric(f"{label} length", f"{words} words / {chars} chars")

    # --- Batch mode ---
    st.markdown("---")
    st.subheader("📝 Batch: Domain Prompt Suite")
    st.markdown(f"Run all 10 {domain} prompts and compare outputs side by side.")

    if st.button(f"▶️ Run all {domain} prompts", use_container_width=True):
        with st.spinner("Loading model..."):
            model, tokenizer = load_hf_model(cfg["model_id"])
            contrastive_vectors = load_contrastive_vectors(cfg["vectors"])

        contrastive_vec = contrastive_vectors[domain][cfg["layer"]]
        feature_vec = feature_vectors[domain][strategy_key]

        prompts = DOMAIN_PROMPTS[domain]
        for i, p in enumerate(prompts):
            st.markdown(f"### Q{i+1}: {p}")
            cols = st.columns(3)

            with cols[0]:
                st.markdown("**Baseline**")
                with st.spinner("..."):
                    text = generate_text(model, tokenizer, p, max_tokens, cfg["layer"])
                st.markdown(f"```\n{text[:500]}\n```")

            with cols[1]:
                st.markdown(f"**Contrastive α={coeff}**")
                with st.spinner("..."):
                    text = generate_text(model, tokenizer, p, max_tokens, cfg["layer"],
                                         vector=contrastive_vec, coeff=coeff)
                st.markdown(f"```\n{text[:500]}\n```")

            with cols[2]:
                st.markdown(f"**Feature {strategy_key} α={coeff}**")
                with st.spinner("..."):
                    text = generate_text(model, tokenizer, p, max_tokens, cfg["layer"],
                                         vector=feature_vec, coeff=coeff)
                st.markdown(f"```\n{text[:500]}\n```")

            st.markdown("---")


if __name__ == "__main__":
    main()
