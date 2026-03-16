"""
Streamlit demo: compare baseline, contrastive, and feature-targeted steering
in real time on Qwen3-0.6B or Qwen3-4B.

Usage:
    streamlit run src/steering/app_steering_demo.py
"""

import functools
from pathlib import Path
from threading import Thread

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
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
        "params": "0.6B",
        "layers_total": 28,
    },
    "Qwen3-4B": {
        "model_id": "Qwen/Qwen3-4B",
        "layer": 18,
        "sae_dir": "sae_qwen3_4b_L18_8x",
        "vectors": "mmlu_pro_vectors_qwen3_4b.pt",
        "params": "4B",
        "layers_total": 36,
    },
}

DOMAIN_COLORS = {
    "math": "#ff6b6b",
    "law": "#4ecdc4",
    "history": "#ffd93d",
}

DOMAIN_ICONS = {
    "math": "∑",
    "law": "§",
    "history": "⏳",
}

METHOD_COLORS = {
    "baseline": "#6c757d",
    "contrastive": "#4ecdc4",
    "feature": "#a855f7",
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
# Custom CSS — dark theme inspired by Bittensor
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---- Global dark theme ---- */
.stApp {
    background: #0a0a0f !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Hide default Streamlit header/footer */
header[data-testid="stHeader"] {
    background: rgba(10, 10, 15, 0.8) !important;
    backdrop-filter: blur(20px) !important;
}
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d14 0%, #111118 100%) !important;
    border-right: 1px solid rgba(78, 205, 196, 0.15) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #b0b0b0 !important;
    font-weight: 400 !important;
}

/* ---- Headings ---- */
h1 {
    background: linear-gradient(135deg, #4ecdc4, #a855f7, #ff6b6b) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
h2, h3 {
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #888 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(78, 205, 196, 0.12) !important;
    color: #4ecdc4 !important;
    border: none !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, rgba(78,205,196,0.15), rgba(168,85,247,0.15)) !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(78,205,196,0.3) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(78,205,196,0.25), rgba(168,85,247,0.25)) !important;
    border-color: rgba(78,205,196,0.6) !important;
    box-shadow: 0 0 20px rgba(78,205,196,0.15) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4ecdc4, #a855f7) !important;
    color: #000 !important;
    font-weight: 600 !important;
    border: none !important;
}

/* ---- Text input / area ---- */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(78,205,196,0.5) !important;
    box-shadow: 0 0 15px rgba(78,205,196,0.1) !important;
}

/* ---- Sliders ---- */
.stSlider [data-baseweb="slider"] div {
    background: rgba(78,205,196,0.3) !important;
}

/* ---- Selectbox ---- */
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}

/* ---- Metrics ---- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: #4ecdc4 !important;
    font-weight: 600 !important;
}

/* ---- Plotly charts dark bg ---- */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ---- Spinner ---- */
.stSpinner > div > div {
    border-top-color: #4ecdc4 !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Chat card HTML components
# ---------------------------------------------------------------------------
def _chat_card(method: str, color: str, label: str, content: str, alpha: float = 0.0) -> str:
    """Render a single chat-style output card."""
    alpha_badge = f'<span style="background:rgba(255,255,255,0.08);padding:2px 8px;border-radius:6px;font-size:0.75rem;color:#888;">α={alpha}</span>' if alpha > 0 else '<span style="background:rgba(255,255,255,0.08);padding:2px 8px;border-radius:6px;font-size:0.75rem;color:#888;">no steering</span>'
    return f"""
    <div style="
        background: rgba(255,255,255,0.03);
        border: 1px solid {color}33;
        border-left: 3px solid {color};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        min-height: 120px;
    ">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="
                    width:8px;height:8px;border-radius:50%;
                    background:{color};
                    box-shadow: 0 0 8px {color}66;
                "></div>
                <span style="color:{color};font-weight:600;font-size:0.9rem;letter-spacing:0.02em;">
                    {label}
                </span>
            </div>
            {alpha_badge}
        </div>
        <div style="
            color:#ccc;
            font-size:0.88rem;
            line-height:1.7;
            font-family:'Inter',sans-serif;
            white-space:pre-wrap;
            word-wrap:break-word;
        ">{content}</div>
    </div>
    """


def _user_prompt_card(prompt: str, domain: str) -> str:
    """Render the user prompt as a chat bubble."""
    color = DOMAIN_COLORS.get(domain, "#4ecdc4")
    icon = DOMAIN_ICONS.get(domain, "?")
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        border: 1px solid {color}30;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    ">
        <div style="
            width:36px;height:36px;border-radius:10px;
            background:{color}20;
            display:flex;align-items:center;justify-content:center;
            font-size:1.1rem;flex-shrink:0;
            border: 1px solid {color}30;
        ">{icon}</div>
        <div>
            <div style="color:#888;font-size:0.75rem;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.05em;">{domain} prompt</div>
            <div style="color:#e0e0e0;font-size:0.95rem;line-height:1.5;">{prompt}</div>
        </div>
    </div>
    """


def _stat_card(label: str, value: str, sublabel: str = "", color: str = "#4ecdc4") -> str:
    """Mini stat card."""
    sub_html = f'<div style="color:#666;font-size:0.7rem;margin-top:2px;">{sublabel}</div>' if sublabel else ""
    return f"""
    <div style="
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(255,255,255,0.06);
        border-radius:10px;
        padding:14px 16px;
        text-align:center;
    ">
        <div style="color:#888;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">{label}</div>
        <div style="color:{color};font-size:1.3rem;font-weight:600;">{value}</div>
        {sub_html}
    </div>
    """


def _hero_banner(model_name: str, cfg: dict) -> str:
    """Top banner with model info."""
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(78,205,196,0.06), rgba(168,85,247,0.06), rgba(255,107,107,0.04));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div>
            <div style="color:#888;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Activation Steering Arena</div>
            <div style="color:#fff;font-size:1.5rem;font-weight:700;letter-spacing:-0.02em;">
                Compare steering methods in real time
            </div>
            <div style="color:#666;font-size:0.85rem;margin-top:6px;">
                Side-by-side generation with contrastive vectors, SAE features, and baseline
            </div>
        </div>
        <div style="text-align:right;">
            <div style="
                background:rgba(78,205,196,0.1);
                border:1px solid rgba(78,205,196,0.25);
                border-radius:10px;
                padding:12px 18px;
                display:inline-block;
            ">
                <div style="color:#4ecdc4;font-size:1.1rem;font-weight:600;">{model_name}</div>
                <div style="color:#888;font-size:0.75rem;">{cfg['params']} params · Layer {cfg['layer']}/{cfg['layers_total']} · SAE 8x</div>
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------
@st.cache_resource(max_entries=1)
def load_hf_model(model_id: str):
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


@st.cache_resource(max_entries=1)
def load_contrastive_vectors(vectors_file: str):
    path = RESULTS_DIR / vectors_file
    return torch.load(path, map_location="cpu", weights_only=True)


@st.cache_resource(max_entries=1)
def load_sae_and_build_feature_vectors(model_id: str, sae_dir: str, layer: int, top_k: int = 20):
    device = _get_device()
    hook_name = f"blocks.{layer}.hook_resid_post"
    sae_path = RESULTS_DIR / sae_dir

    tl_model = HookedTransformer.from_pretrained_no_processing(model_id, device=device)
    sae = SAE.load_from_disk(str(sae_path), device=device)

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

        weighted_vec = torch.zeros(W_dec.shape[1])
        for idx, val in zip(top_indices, top_values):
            weighted_vec += val.item() * W_dec[idx]
        uniform_vec = W_dec[top_indices].sum(dim=0)
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

    del tl_model, sae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return feature_vectors, feature_info


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Steering hooks & generation
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
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
        )
    if handle is not None:
        handle.remove()
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_text_stream(model, tokenizer, prompt, max_new_tokens, layer, vector=None, coeff=0.0):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    handle = None
    if vector is not None and coeff > 0:
        layers = get_layers(model)
        handle = layers[layer].register_forward_hook(
            functools.partial(_steering_hook, vector=vector, coeff=coeff)
        )
    gen_kwargs = dict(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, streamer=streamer,
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


# ---------------------------------------------------------------------------
# Vector space visualization (dark plotly theme)
# ---------------------------------------------------------------------------
PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#b0b0b0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
)


def _collect_vectors(contrastive_vectors, feature_vectors, layer):
    vecs, labels, types, domains = [], [], [], []
    for domain in TARGET_DOMAINS:
        cv = contrastive_vectors[domain][layer].float().cpu().numpy()
        vecs.append(cv)
        labels.append(f"contrastive · {domain}")
        types.append("contrastive")
        domains.append(domain)
        for strategy in ["weighted", "uniform", "single"]:
            fv = feature_vectors[domain][strategy].float().cpu().numpy()
            vecs.append(fv)
            labels.append(f"{strategy} · {domain}")
            types.append(f"SAE-{strategy}")
            domains.append(domain)
    return np.stack(vecs), labels, types, domains


def build_vector_space_viz(contrastive_vectors, feature_vectors, layer):
    X, labels, types, domains = _collect_vectors(contrastive_vectors, feature_vectors, layer)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    symbol_map = {"contrastive": "diamond", "SAE-weighted": "circle",
                  "SAE-uniform": "square", "SAE-single": "star"}

    fig = go.Figure()

    # Domain connector lines
    for domain in TARGET_DOMAINS:
        color = DOMAIN_COLORS[domain]
        idxs = [i for i, d in enumerate(domains) if d == domain]
        c_idx = [i for i in idxs if types[i] == "contrastive"][0]
        for si in [i for i in idxs if types[i] != "contrastive"]:
            fig.add_trace(go.Scatter(
                x=[X_2d[c_idx, 0], X_2d[si, 0]], y=[X_2d[c_idx, 1], X_2d[si, 1]],
                mode="lines", line=dict(color=color, width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))

    # Points
    for vtype, symbol in symbol_map.items():
        for domain in TARGET_DOMAINS:
            color = DOMAIN_COLORS[domain]
            mask = [(t == vtype and d == domain) for t, d in zip(types, domains)]
            idxs = [i for i, m in enumerate(mask) if m]
            if not idxs:
                continue
            fig.add_trace(go.Scatter(
                x=X_2d[idxs, 0], y=X_2d[idxs, 1],
                mode="markers+text",
                marker=dict(symbol=symbol, size=13, color=color,
                            line=dict(width=1.5, color="rgba(0,0,0,0.5)")),
                text=[vtype.replace("SAE-", "") for _ in idxs],
                textposition="top center", textfont=dict(size=9, color="#999"),
                name=f"{domain} · {vtype}", legendgroup=domain,
            ))

    fig.update_layout(
        title=dict(text="Steering Vectors — PCA Projection", font=dict(size=16)),
        xaxis_title=f"PC1 ({var[0]:.1%} var.)",
        yaxis_title=f"PC2 ({var[1]:.1%} var.)",
        height=480, **PLOTLY_DARK,
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_cosine_heatmap(contrastive_vectors, feature_vectors, layer):
    X, labels, _, _ = _collect_vectors(contrastive_vectors, feature_vectors, layer)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    cos_sim = (X / norms) @ (X / norms).T

    short_labels = [l.replace("contrastive", "contr.").replace("weighted", "wt.").replace("uniform", "uni.") for l in labels]

    fig = go.Figure(data=go.Heatmap(
        z=cos_sim, x=short_labels, y=short_labels,
        colorscale=[[0, "#1a0a2e"], [0.5, "#0a0a0f"], [1, "#4ecdc4"]],
        text=np.round(cos_sim, 2), texttemplate="%{text}", textfont={"size": 8},
    ))
    fig.update_layout(
        title=dict(text="Cosine Similarity", font=dict(size=16)),
        height=480, **PLOTLY_DARK,
        xaxis_tickangle=-45, margin=dict(l=100, b=100),
    )
    return fig


def build_norm_comparison(contrastive_vectors, feature_vectors, layer):
    X, labels, types, domains = _collect_vectors(contrastive_vectors, feature_vectors, layer)
    norms = np.linalg.norm(X, axis=1)
    colors = [DOMAIN_COLORS[d] for d in domains]

    fig = go.Figure(data=go.Bar(
        x=labels, y=norms,
        marker_color=colors,
        marker_line=dict(width=0),
        opacity=0.85,
    ))
    fig.update_layout(
        title=dict(text="Vector L2 Norms", font=dict(size=16)),
        height=350, **PLOTLY_DARK,
        xaxis_tickangle=-45, showlegend=False,
        yaxis_title="L2 Norm",
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Steering Arena",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px 0;">
            <div style="font-size:2rem;">⚡</div>
            <div style="color:#4ecdc4;font-weight:700;font-size:1.1rem;letter-spacing:0.02em;">STEERING ARENA</div>
            <div style="color:#555;font-size:0.75rem;margin-top:2px;">Activation Steering Research</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        model_name = st.selectbox("MODEL", list(MODEL_CONFIGS.keys()), index=0)
        cfg = MODEL_CONFIGS[model_name]

        domain = st.selectbox("DOMAIN", TARGET_DOMAINS, index=0, format_func=lambda d: f"{DOMAIN_ICONS[d]}  {d.upper()}")

        coeff = st.slider("STEERING COEFFICIENT (α)", min_value=0.0, max_value=60.0, value=10.0, step=1.0)

        feature_strategy = st.selectbox(
            "SAE STRATEGY",
            ["weighted", "uniform", "single"],
            index=0,
            format_func=lambda s: {"weighted": "Weighted (top-k × diff)", "uniform": "Uniform (top-k equal)", "single": "Single (best feature)"}[s],
        )

        top_k = st.slider("TOP-K FEATURES", min_value=1, max_value=50, value=20)
        max_tokens = st.slider("MAX TOKENS", min_value=32, max_value=512, value=128, step=32)

        st.markdown("---")

        # Model status badge
        sae_exists = (RESULTS_DIR / cfg["sae_dir"]).exists()
        vec_exists = (RESULTS_DIR / cfg["vectors"]).exists()
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:12px;border:1px solid rgba(255,255,255,0.06);">
            <div style="color:#888;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">Resources</div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                <div style="width:6px;height:6px;border-radius:50%;background:{'#4ecdc4' if vec_exists else '#ff6b6b'};"></div>
                <span style="color:#aaa;font-size:0.8rem;">Contrastive vectors</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                <div style="width:6px;height:6px;border-radius:50%;background:{'#4ecdc4' if sae_exists else '#ff6b6b'};"></div>
                <span style="color:#aaa;font-size:0.8rem;">SAE ({cfg['sae_dir'].split('_')[-1]})</span>
            </div>
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:6px;height:6px;border-radius:50%;background:#ffd93d;"></div>
                <span style="color:#aaa;font-size:0.8rem;">Model loads on generate</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Hero banner ---
    st.markdown(_hero_banner(model_name, cfg), unsafe_allow_html=True)

    # --- Tabs ---
    tab_gen, tab_viz, tab_batch = st.tabs(["⚡  ARENA", "📐  VECTOR SPACE", "📝  BATCH"])

    # ===== TAB: Arena (streaming generation) =====
    with tab_gen:
        example_prompts = {
            "math": "What is the derivative of x^3 + 2x?",
            "law": "What is the difference between a tort and a crime?",
            "history": "What were the main causes of the American Revolution?",
        }

        prompt = st.text_area(
            "Enter your prompt",
            value=example_prompts.get(domain, ""),
            height=80,
            label_visibility="collapsed",
            placeholder=f"Ask a {domain} question...",
        )

        if st.button("GENERATE", type="primary", use_container_width=True):
            # User prompt card
            st.markdown(_user_prompt_card(prompt, domain), unsafe_allow_html=True)

            with st.spinner("Loading model & SAE..."):
                model, tokenizer = load_hf_model(cfg["model_id"])
                contrastive_vectors = load_contrastive_vectors(cfg["vectors"])
                feature_vectors, feature_info = load_sae_and_build_feature_vectors(
                    cfg["model_id"], cfg["sae_dir"], cfg["layer"], top_k
                )

            contrastive_vec = contrastive_vectors[domain][cfg["layer"]]
            feature_vec = feature_vectors[domain][feature_strategy]

            col1, col2, col3 = st.columns(3)

            # Stream baseline
            with col1:
                placeholder = st.empty()
                baseline_text = ""
                for partial in generate_text_stream(
                    model, tokenizer, prompt, max_tokens, cfg["layer"],
                    vector=None, coeff=0,
                ):
                    baseline_text = partial
                    placeholder.markdown(
                        _chat_card("baseline", METHOD_COLORS["baseline"], "BASELINE", partial, 0),
                        unsafe_allow_html=True,
                    )

            # Stream contrastive
            with col2:
                placeholder = st.empty()
                contrastive_text = ""
                for partial in generate_text_stream(
                    model, tokenizer, prompt, max_tokens, cfg["layer"],
                    vector=contrastive_vec, coeff=coeff,
                ):
                    contrastive_text = partial
                    placeholder.markdown(
                        _chat_card("contrastive", METHOD_COLORS["contrastive"], "CONTRASTIVE", partial, coeff),
                        unsafe_allow_html=True,
                    )

            # Stream feature-targeted
            with col3:
                placeholder = st.empty()
                feature_text = ""
                for partial in generate_text_stream(
                    model, tokenizer, prompt, max_tokens, cfg["layer"],
                    vector=feature_vec, coeff=coeff,
                ):
                    feature_text = partial
                    placeholder.markdown(
                        _chat_card("feature", METHOD_COLORS["feature"], f"SAE {feature_strategy.upper()}", partial, coeff),
                        unsafe_allow_html=True,
                    )

            # Stats row
            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(_stat_card("Baseline", f"{len(baseline_text.split())} words", f"{len(baseline_text)} chars"), unsafe_allow_html=True)
            with s2:
                st.markdown(_stat_card("Contrastive", f"{len(contrastive_text.split())} words", f"{len(contrastive_text)} chars", "#4ecdc4"), unsafe_allow_html=True)
            with s3:
                st.markdown(_stat_card("Feature", f"{len(feature_text.split())} words", f"{len(feature_text)} chars", "#a855f7"), unsafe_allow_html=True)
            with s4:
                # Word overlap between baseline and steered
                bw = set(baseline_text.lower().split())
                cw = set(contrastive_text.lower().split())
                fw = set(feature_text.lower().split())
                c_overlap = len(bw & cw) / max(len(bw | cw), 1)
                f_overlap = len(bw & fw) / max(len(bw | fw), 1)
                st.markdown(_stat_card("Jaccard vs Base", f"{c_overlap:.0%} / {f_overlap:.0%}", "contr. / feat.", "#ffd93d"), unsafe_allow_html=True)

    # ===== TAB: Vector Space =====
    with tab_viz:
        if st.button("LOAD VECTORS & VISUALIZE", use_container_width=True, key="viz_load"):
            with st.spinner("Computing vector projections..."):
                contrastive_vectors = load_contrastive_vectors(cfg["vectors"])
                feature_vectors, feature_info = load_sae_and_build_feature_vectors(
                    cfg["model_id"], cfg["sae_dir"], cfg["layer"], top_k
                )

            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(
                    build_vector_space_viz(contrastive_vectors, feature_vectors, cfg["layer"]),
                    use_container_width=True,
                )
            with v2:
                st.plotly_chart(
                    build_cosine_heatmap(contrastive_vectors, feature_vectors, cfg["layer"]),
                    use_container_width=True,
                )

            st.plotly_chart(
                build_norm_comparison(contrastive_vectors, feature_vectors, cfg["layer"]),
                use_container_width=True,
            )

            # Feature info cards
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            fcols = st.columns(3)
            for i, d in enumerate(TARGET_DOMAINS):
                with fcols[i]:
                    info = feature_info[d]
                    feats_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<span style="color:#aaa;font-size:0.8rem;">#{fid}</span>'
                        f'<span style="color:{DOMAIN_COLORS[d]};font-size:0.8rem;font-weight:500;">Δ={diff}</span></div>'
                        for fid, diff in zip(info["top_features"], info["top_diffs"])
                    )
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.03);border:1px solid {DOMAIN_COLORS[d]}25;
                                border-radius:12px;padding:16px;">
                        <div style="color:{DOMAIN_COLORS[d]};font-weight:600;font-size:0.85rem;margin-bottom:10px;letter-spacing:0.03em;">
                            {DOMAIN_ICONS[d]}  {d.upper()} — Top Features
                        </div>
                        {feats_html}
                    </div>
                    """, unsafe_allow_html=True)

    # ===== TAB: Batch =====
    with tab_batch:
        st.markdown(f"""
        <div style="color:#888;font-size:0.85rem;margin-bottom:16px;">
            Run all 10 <span style="color:{DOMAIN_COLORS[domain]};font-weight:500;">{domain}</span> prompts with {model_name} and compare outputs.
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"RUN ALL {domain.upper()} PROMPTS", use_container_width=True):
            with st.spinner("Loading model & SAE..."):
                model, tokenizer = load_hf_model(cfg["model_id"])
                contrastive_vectors = load_contrastive_vectors(cfg["vectors"])
                feature_vectors, feature_info = load_sae_and_build_feature_vectors(
                    cfg["model_id"], cfg["sae_dir"], cfg["layer"], top_k
                )

            contrastive_vec = contrastive_vectors[domain][cfg["layer"]]
            feature_vec = feature_vectors[domain][feature_strategy]

            progress = st.progress(0, text="Starting batch...")
            prompts = DOMAIN_PROMPTS[domain]

            for i, p in enumerate(prompts):
                progress.progress((i + 1) / len(prompts), text=f"Generating {i+1}/{len(prompts)}...")

                st.markdown(_user_prompt_card(p, domain), unsafe_allow_html=True)

                cols = st.columns(3)
                with cols[0]:
                    text = generate_text(model, tokenizer, p, max_tokens, cfg["layer"])
                    st.markdown(_chat_card("baseline", METHOD_COLORS["baseline"], "BASELINE", text[:500], 0), unsafe_allow_html=True)
                with cols[1]:
                    text = generate_text(model, tokenizer, p, max_tokens, cfg["layer"],
                                         vector=contrastive_vec, coeff=coeff)
                    st.markdown(_chat_card("contrastive", METHOD_COLORS["contrastive"], "CONTRASTIVE", text[:500], coeff), unsafe_allow_html=True)
                with cols[2]:
                    text = generate_text(model, tokenizer, p, max_tokens, cfg["layer"],
                                         vector=feature_vec, coeff=coeff)
                    st.markdown(_chat_card("feature", METHOD_COLORS["feature"], f"SAE {feature_strategy.upper()}", text[:500], coeff), unsafe_allow_html=True)

            progress.empty()


if __name__ == "__main__":
    main()
