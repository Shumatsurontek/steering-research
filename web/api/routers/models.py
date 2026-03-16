"""Model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

from ..deps import MODEL_CONFIGS
from ..schemas import LoadModelRequest, RecalculateSAERequest, ModelInfo, ModelStatus

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models() -> dict[str, ModelInfo]:
    return {k: ModelInfo(**v) for k, v in MODEL_CONFIGS.items()}


@router.get("/models/status")
async def model_status(request: Request) -> ModelStatus:
    manager = request.app.state.manager
    return ModelStatus(**manager.get_status())


@router.post("/models/load")
async def load_model(body: LoadModelRequest, request: Request) -> ModelStatus:
    manager = request.app.state.manager
    if body.model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {body.model}")
    await manager.load_model(body.model)
    return ModelStatus(**manager.get_status())


@router.post("/sae/recalculate")
async def recalculate_sae(body: RecalculateSAERequest, request: Request) -> ModelStatus:
    manager = request.app.state.manager
    if body.model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {body.model}")
    try:
        await manager.recalculate_sae(body.model, body.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ModelStatus(**manager.get_status())
