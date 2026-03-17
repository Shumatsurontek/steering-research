"""Benchmark results and execution endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..deps import MODEL_CONFIGS, RESULTS_DIR

logger = logging.getLogger("steering.benchmarks")

router = APIRouter(tags=["benchmarks"])

# Map model keys to their configs
MODEL_SCRIPTS = {
    "Qwen3-0.6B": {"model_id": "Qwen/Qwen3-0.6B"},
    "Qwen3-4B": {"model_id": "Qwen/Qwen3-4B"},
    "LFM2-700M": {"model_id": "LiquidAI/LFM2-700M"},
}

MODEL_SHORT = {
    "Qwen3-0.6B": "qwen3_0.6b",
    "Qwen3-4B": "qwen3_4b",
    "LFM2-700M": "lfm2_700m",
}


def _find_benchmark_file(model_key: str) -> Path | None:
    """Find the latest benchmark file for a model (prefer highest n)."""
    short = MODEL_SHORT.get(model_key, "")
    # Special case for 0.6B (legacy naming)
    patterns = [
        f"feature_targeted_benchmark_{short}_n*.json",
        f"feature_targeted_benchmark_n*.json",  # legacy 0.6B
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(RESULTS_DIR.glob(pat))
    if not candidates:
        return None
    # Prefer largest n
    return max(candidates, key=lambda p: p.stat().st_size)

SAE_ANALYSIS_FILES = {
    "Qwen3-0.6B": "sae_domain_analysis.json",
    "Qwen3-4B": "sae_domain_analysis_qwen3_4b.json",
    "LFM2-700M": "sae_domain_analysis_lfm2_700m.json",
}

# Track running jobs
_running_jobs: dict[str, asyncio.subprocess.Process] = {}


# ── Run benchmark request ───────────────────────────────────────────────────

class RunBenchmarkRequest(BaseModel):
    model: str
    benchmark: str  # "vectors", "sae_train", "sae_analysis", "feature_targeted"
    limit: int = 50
    top_k: int = 20
    domain: str = "all"


BENCHMARK_COMMANDS = {
    "vectors": lambda model_id, **kw: [
        sys.executable, "-m", "src.steering.mmlu_pro_vectors",
        "--model", _model_short(model_id),
    ],
    "sae_train": lambda model_id, **kw: [
        sys.executable, "-m", "src.steering.train_sae_hf",
        "--model", model_id, "--wandb",
    ],
    "sae_analysis": lambda model_id, **kw: [
        sys.executable, "-m", "src.steering.analyze_sae_features",
        "--model", model_id,
    ],
    "feature_targeted": lambda model_id, limit=50, top_k=20, domain="all", **kw: [
        sys.executable, "-m", "src.steering.feature_targeted_steering",
        "--model", model_id, "--limit", str(limit), "--top_k", str(top_k),
        "--domain", domain,
    ],
}


def _model_short(model_id: str) -> str:
    """Convert model_id to short key for mmlu_pro_vectors (e.g. 'lfm2_700m')."""
    return model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


# ── SSE endpoint to run a benchmark ─────────────────────────────────────────

@router.post("/benchmarks/run")
async def run_benchmark(body: RunBenchmarkRequest):
    if body.model not in MODEL_SCRIPTS:
        raise HTTPException(400, f"Unknown model: {body.model}")
    if body.benchmark not in BENCHMARK_COMMANDS:
        raise HTTPException(400, f"Unknown benchmark: {body.benchmark}. Choose from: {list(BENCHMARK_COMMANDS.keys())}")

    job_key = f"{body.model}:{body.benchmark}"
    if job_key in _running_jobs:
        raise HTTPException(409, f"Benchmark already running: {job_key}")

    model_id = MODEL_SCRIPTS[body.model]["model_id"]
    cmd = BENCHMARK_COMMANDS[body.benchmark](
        model_id, limit=body.limit, top_k=body.top_k, domain=body.domain,
    )

    logger.info("Starting benchmark: %s → %s", job_key, " ".join(cmd))

    # Find project root (where src/ lives)
    project_root = RESULTS_DIR.parent

    async def event_stream():
        env = {**__import__("os").environ, "PYTHONUNBUFFERED": "1"}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(project_root),
            env=env,
        )
        _running_jobs[job_key] = proc

        yield f"event: start\ndata: {json.dumps({'job': job_key, 'cmd': ' '.join(cmd)})}\n\n"

        try:
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                # Clean carriage returns from progress bars
                if "\r" in text:
                    text = text.split("\r")[-1].strip()
                if text:
                    yield f"event: log\ndata: {json.dumps({'line': text})}\n\n"

            await proc.wait()
            status = "success" if proc.returncode == 0 else "error"
            yield f"event: done\ndata: {json.dumps({'status': status, 'returncode': proc.returncode})}\n\n"
        except Exception as e:
            yield f"event: done\ndata: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
        finally:
            _running_jobs.pop(job_key, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/benchmarks/running")
async def list_running():
    return {"jobs": list(_running_jobs.keys())}


@router.post("/benchmarks/cancel")
async def cancel_benchmark(job: str = Query(...)):
    proc = _running_jobs.get(job)
    if not proc:
        raise HTTPException(404, f"No running job: {job}")
    proc.kill()
    _running_jobs.pop(job, None)
    return {"cancelled": job}


# ── Results endpoints ───────────────────────────────────────────────────────

SAMPLE_FILES = {
    "Qwen3-0.6B": "benchmark_samples_qwen3_0.6b_n{limit}.json",
    "Qwen3-4B": "benchmark_samples_qwen3_4b_n{limit}.json",
    "LFM2-700M": "benchmark_samples_lfm2_700m_n{limit}.json",
}


@router.get("/benchmarks/samples")
async def get_samples(
    model: str = Query(...),
    domain: str = Query(...),
):
    """Return per-sample benchmark results for the sample viewer."""
    if model not in SAMPLE_FILES:
        return {"samples": []}

    # Try common limits
    for limit in [20, 50, 100, 200]:
        filename = SAMPLE_FILES[model].format(limit=limit)
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            samples = data.get(domain, [])
            return {"samples": samples, "file": filename, "n": limit}

    return {"samples": []}


@router.get("/benchmarks/results")
async def get_benchmark_results(
    model: str = Query(None, description="Model key. If omitted, returns all."),
):
    results = {}
    targets = [model] if model and model in MODEL_SHORT else list(MODEL_SHORT.keys())
    for model_key in targets:
        path = _find_benchmark_file(model_key)
        if path and path.exists():
            with open(path) as f:
                results[model_key] = json.load(f)
        else:
            results[model_key] = None
    return results


@router.get("/benchmarks/sae-analysis")
async def get_sae_analysis(
    model: str = Query(None, description="Model key. If omitted, returns all."),
):
    results = {}
    targets = {model: SAE_ANALYSIS_FILES[model]} if model and model in SAE_ANALYSIS_FILES else SAE_ANALYSIS_FILES
    for model_key, filename in targets.items():
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path) as f:
                results[model_key] = json.load(f)
        else:
            results[model_key] = None
    return results


@router.get("/benchmarks/summary")
async def get_benchmark_summary():
    summary = []
    for model_key in MODEL_SHORT:
        path = _find_benchmark_file(model_key)
        if not path or not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for domain, domain_data in data.items():
            if not isinstance(domain_data, dict) or "results" not in domain_data:
                continue
            results_list = domain_data["results"]
            baseline_acc = 0.0
            for entry in results_list:
                if entry.get("label") == "baseline":
                    baseline_acc = entry.get("acc", 0)
                    break
            for entry in results_list:
                acc = entry.get("acc", 0)
                label = entry.get("label", "")
                summary.append({
                    "model": model_key,
                    "domain": domain,
                    "method": label,
                    "accuracy": acc,
                    "delta": round((acc - baseline_acc) * 100, 1) if label != "baseline" else 0,
                    "stderr": entry.get("stderr", 0),
                })
    return summary
