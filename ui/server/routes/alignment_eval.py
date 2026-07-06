"""Alignment evaluation results API — serves lambda ablation, router smoke test, t-SNE images, sanity checks."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import config

router = APIRouter(prefix="/api/alignment", tags=["alignment-eval"])


def _read_json(filename: str) -> dict | None:
    """Read a JSON file from the outputs directory, return None if missing."""
    path = config.OUTPUTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Lambda Ablation ──────────────────────────────────────────────────────

@router.get("/lambda-ablation")
async def get_lambda_ablation():
    """Lambda ablation study results: retrieval, spearman, anti-collapse vs λ."""
    data = _read_json("lambda_ablation.json")
    if data is None:
        raise HTTPException(status_code=404, detail="lambda_ablation.json not found in outputs/")
    return data


# ── Router Smoke Test ────────────────────────────────────────────────────

@router.get("/router-smoke-test")
async def get_router_smoke_test():
    """Router smoke test results: can a trivial router beat random chance?"""
    data = _read_json("router_smoke_test.json")
    if data is None:
        raise HTTPException(status_code=404, detail="router_smoke_test.json not found in outputs/")
    return data


# ── Sanity Checks ────────────────────────────────────────────────────────

@router.get("/sanity-checks")
async def get_sanity_checks():
    """Manual sanity check results: concrete prompt pairs with expected vs actual distances."""
    data = _read_json("sanity_checks.json")
    if data is None:
        raise HTTPException(status_code=404, detail="sanity_checks.json not found in outputs/")
    return data


# ── t-SNE Images ─────────────────────────────────────────────────────────

@router.get("/tsne")
async def list_tsne_images():
    """List available t-SNE visualization images."""
    outputs = config.OUTPUTS_DIR
    if not outputs.exists():
        return {"images": []}
    images = sorted(outputs.glob("tsne_*.png"))
    return {
        "images": [
            {
                "name": img.name,
                "url": f"/api/alignment/tsne/{img.name}",
                "size": img.stat().st_size,
            }
            for img in images
        ]
    }


@router.get("/tsne/{filename}")
async def get_tsne_image(filename: str):
    """Serve a t-SNE PNG image from the outputs directory."""
    if not filename.endswith(".png") or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = config.OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image {filename} not found")
    return FileResponse(path, media_type="image/png")


# ── Aggregated Evaluation Summary ────────────────────────────────────────

@router.get("/eval-summary")
async def get_eval_summary():
    """Aggregated alignment evaluation summary: all metrics in one call."""
    return {
        "lambda_ablation": _read_json("lambda_ablation.json"),
        "router_smoke_test": _read_json("router_smoke_test.json"),
        "sanity_checks": _read_json("sanity_checks.json"),
        "corrected_diagnostics": _read_json("corrected_router_diagnostics.json"),
        "tsne_images": [
            img.name
            for img in sorted(config.OUTPUTS_DIR.glob("tsne_*.png"))
        ] if config.OUTPUTS_DIR.exists() else [],
    }


# ── Corrected Router Diagnostics ────────────────────────────────────────

@router.get("/corrected-diagnostics")
async def get_corrected_diagnostics():
    """Corrected router diagnostics: class-balanced accuracy, per-class P/R/F1, hard-set test."""
    data = _read_json("corrected_router_diagnostics.json")
    if data is None:
        raise HTTPException(status_code=404, detail="corrected_router_diagnostics.json not found in outputs/")
    return data
