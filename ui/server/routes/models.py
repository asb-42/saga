"""API endpoints for models, training runs, and benchmarks."""

from fastapi import APIRouter, HTTPException
from ..data_ingestion import (
    get_all_checkpoints,
    get_anomaly_threshold,
    get_model_configs,
    get_poisoning_results,
    get_full_eval_results,
    get_training_runs_from_tensorboard,
    get_trainer_state,
    get_poisoning_meta,
    get_reward_model_meta,
    get_all_training_metrics,
    get_evaluation_summary,
)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/checkpoints")
async def list_checkpoints():
    """List all available model checkpoints."""
    return {"checkpoints": get_all_checkpoints()}


@router.get("/checkpoints/{checkpoint_type}")
async def get_checkpoint(checkpoint_type: str):
    """Get information about a specific checkpoint."""
    checkpoints = get_all_checkpoints()
    for cp in checkpoints:
        if cp["type"] == checkpoint_type:
            if not cp["exists"]:
                raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_type} not found")
            return cp
    raise HTTPException(status_code=404, detail=f"Unknown checkpoint type: {checkpoint_type}")


@router.get("/configs")
async def list_configs():
    """Get all model configurations."""
    return get_model_configs()


@router.get("/configs/{config_name}")
async def get_config(config_name: str):
    """Get a specific configuration."""
    configs = get_model_configs()
    if config_name in configs:
        return configs[config_name]
    raise HTTPException(status_code=404, detail=f"Config not found: {config_name}")


@router.get("/anomaly-threshold")
async def get_anomaly_threshold_endpoint():
    """Get the anomaly detection threshold."""
    threshold = get_anomaly_threshold()
    if not threshold:
        raise HTTPException(status_code=404, detail="Anomaly threshold not calibrated")
    return threshold


@router.get("/poisoning-meta")
async def get_poisoning_metadata():
    """Get poisoning training metadata."""
    meta = get_poisoning_meta()
    if not meta:
        raise HTTPException(status_code=404, detail="Poisoning metadata not found")
    return meta


@router.get("/reward-model-meta")
async def get_reward_model_metadata():
    """Get reward model training metadata."""
    meta = get_reward_model_meta()
    if not meta:
        raise HTTPException(status_code=404, detail="Reward model metadata not found")
    return meta
