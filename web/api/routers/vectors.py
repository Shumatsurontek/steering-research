"""Vector space visualization data endpoint."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request, HTTPException, Query
from sklearn.decomposition import PCA

from ..deps import MODEL_CONFIGS, TARGET_DOMAINS

DOMAIN_COLORS = {"math": "#ff6b6b", "law": "#4ecdc4", "history": "#ffd93d"}

router = APIRouter(tags=["vectors"])


def _collect_vectors(contrastive_vectors, feature_vectors, layer):
    vecs, labels, types, domains = [], [], [], []
    for domain in TARGET_DOMAINS:
        cv = contrastive_vectors[domain][layer].float().cpu().numpy()
        vecs.append(cv)
        labels.append(f"contrastive · {domain}")
        types.append("contrastive")
        domains.append(domain)
        if feature_vectors:
            for strategy in ["weighted", "uniform", "single"]:
                fv = feature_vectors[domain][strategy].float().cpu().numpy()
                vecs.append(fv)
                labels.append(f"{strategy} · {domain}")
                types.append(f"SAE-{strategy}")
                domains.append(domain)
    return np.stack(vecs), labels, types, domains


@router.get("/vectors/visualizations")
async def vector_visualizations(
    request: Request,
    layer: int = Query(None, description="Layer to visualize. Defaults to model's SAE layer."),
):
    manager = request.app.state.manager
    if manager.contrastive_vectors is None:
        raise HTTPException(status_code=400, detail="No vectors loaded. Load a model first.")

    cfg = MODEL_CONFIGS.get(manager.current_model_key, {})
    if layer is None:
        layer = cfg.get("layer", 14)

    has_sae = manager.feature_vectors is not None
    sae_native = has_sae and layer == cfg.get("layer")

    X, labels, types, domains = _collect_vectors(
        manager.contrastive_vectors,
        manager.feature_vectors if has_sae else None,
        layer,
    )

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_.tolist()

    pca_data = {
        "points": [
            {
                "x": float(X_2d[i, 0]),
                "y": float(X_2d[i, 1]),
                "label": labels[i],
                "type": types[i],
                "domain": domains[i],
                "color": DOMAIN_COLORS[domains[i]],
            }
            for i in range(len(labels))
        ],
        "variance": variance,
    }

    # Cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    cos_sim = ((X / norms) @ (X / norms).T).tolist()

    cosine_data = {
        "matrix": cos_sim,
        "labels": labels,
    }

    # Norms
    norms_flat = np.linalg.norm(X, axis=1).tolist()
    norms_data = {
        "bars": [
            {"label": labels[i], "norm": norms_flat[i], "color": DOMAIN_COLORS[domains[i]]}
            for i in range(len(labels))
        ],
    }

    # Feature info
    fi = {}
    if has_sae and manager.feature_info:
        fi = manager.feature_info

    return {
        "pca": pca_data,
        "cosine": cosine_data,
        "norms": norms_data,
        "feature_info": fi,
        "layer": layer,
        "sae_available": has_sae,
        "sae_native_layer": sae_native,
        "sae_trained_layer": cfg.get("layer"),
    }
