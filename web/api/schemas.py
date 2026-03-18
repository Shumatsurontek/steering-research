"""Pydantic models for the Steering Arena API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request models ──────────────────────────────────────────────────────────

class LoadModelRequest(BaseModel):
    model: str = Field(..., description="Model key, e.g. 'Qwen3-0.6B'")


class RecalculateSAERequest(BaseModel):
    model: str
    top_k: int = 20


class GenerateRequest(BaseModel):
    prompt: str
    domain: str = "math"
    layer: int
    alpha: float = 10.0
    feature_strategy: str = "weighted"
    max_tokens: int = 128
    top_k: int = 20
    steering_mode: str = "additive"  # "additive" or "multiplicative"


# ── Response models ─────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_id: str
    layer: int
    sae_dir: str
    vectors: str
    params: str
    layers_total: int


class ModelStatus(BaseModel):
    loaded_model: str | None
    has_contrastive: bool
    has_sae: bool


class VectorVizData(BaseModel):
    pca: dict
    cosine: dict
    norms: dict
    feature_info: dict
