"""SSE streaming generation endpoint — runs 3 methods concurrently."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from threading import Thread

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from ..deps import MODEL_CONFIGS
from ..schemas import GenerateRequest
from ..steering import generate_stream

logger = logging.getLogger("steering.generate")

router = APIRouter(tags=["generate"])


def _run_generation_to_queue(
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    event_name: str,
    model, tokenizer, prompt, max_tokens, layer,
    vector=None, coeff=0.0,
):
    """Run a single generation stream in a thread, pushing SSE events to the queue."""
    start = time.time()
    token_count = 0
    try:
        for partial_text in generate_stream(
            model, tokenizer, prompt, max_tokens, layer,
            vector=vector, coeff=coeff,
        ):
            token_count += 1
            asyncio.run_coroutine_threadsafe(
                queue.put({"event": event_name, "data": {"text": partial_text}}),
                loop,
            )
        elapsed = time.time() - start
        asyncio.run_coroutine_threadsafe(
            queue.put({
                "event": f"{event_name}:done",
                "data": {"text": partial_text, "tokens": token_count, "elapsed": round(elapsed, 2)},
            }),
            loop,
        )
    except Exception as e:
        logger.exception("Generation error for %s", event_name)
        asyncio.run_coroutine_threadsafe(
            queue.put({"event": f"{event_name}:error", "data": {"error": str(e)}}),
            loop,
        )


@router.post("/generate/stream")
async def generate_sse(body: GenerateRequest, request: Request):
    manager = request.app.state.manager

    if manager.model is None:
        raise HTTPException(status_code=400, detail="No model loaded. POST /api/models/load first.")

    cfg = MODEL_CONFIGS.get(manager.current_model_key, {})
    sae_layer = cfg.get("layer")
    has_sae = manager.feature_vectors is not None
    sae_native = has_sae and body.layer == sae_layer

    # Resolve vectors
    contrastive_vec = None
    if manager.contrastive_vectors and body.domain in manager.contrastive_vectors:
        domain_vecs = manager.contrastive_vectors[body.domain]
        if body.layer in domain_vecs:
            contrastive_vec = domain_vecs[body.layer]
        elif isinstance(domain_vecs, dict):
            # Try integer key
            contrastive_vec = domain_vecs.get(body.layer)

    feature_vec = None
    if has_sae and body.domain in (manager.feature_vectors or {}):
        feature_vec = manager.feature_vectors[body.domain].get(body.feature_strategy)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Track how many methods will run
    methods = ["baseline"]
    threads = []

    # Baseline thread
    t = Thread(
        target=_run_generation_to_queue,
        args=(queue, loop, "baseline", manager.model, manager.tokenizer,
              body.prompt, body.max_tokens, body.layer),
    )
    threads.append(t)

    # Contrastive thread
    if contrastive_vec is not None:
        methods.append("contrastive")
        t = Thread(
            target=_run_generation_to_queue,
            args=(queue, loop, "contrastive", manager.model, manager.tokenizer,
                  body.prompt, body.max_tokens, body.layer),
            kwargs={"vector": contrastive_vec, "coeff": body.alpha},
        )
        threads.append(t)

    # Feature thread
    if feature_vec is not None:
        methods.append("feature")
        t = Thread(
            target=_run_generation_to_queue,
            args=(queue, loop, "feature", manager.model, manager.tokenizer,
                  body.prompt, body.max_tokens, body.layer),
            kwargs={"vector": feature_vec, "coeff": body.alpha},
        )
        threads.append(t)

    # Send initial config event
    async def event_stream():
        yield f"event: config\ndata: {json.dumps({'methods': methods, 'sae_available': has_sae, 'sae_native_layer': sae_native, 'sae_layer': sae_layer})}\n\n"

        # Sequential: run each method to completion before starting the next.
        # Avoids GPU contention → fastest total wall time and most coherent output.
        for t in threads:
            t.start()
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=120)
                    event = msg["event"]
                    data = json.dumps(msg["data"])
                    yield f"event: {event}\ndata: {data}\n\n"
                    if event.endswith(":done") or event.endswith(":error"):
                        break
                except asyncio.TimeoutError:
                    yield f"event: error\ndata: {json.dumps({'error': 'Generation timed out'})}\n\n"
                    break
            t.join(timeout=5)

        yield "event: complete\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
