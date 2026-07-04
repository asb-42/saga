"""Pipeline control routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..models import StartScriptRequest, ScriptRun, ScriptStatus, PipelineStatus

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


def get_process_manager():
    """Dependency: get process manager."""
    from ..main import process_manager
    if not process_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return process_manager


def get_storage():
    """Dependency: get storage."""
    from ..main import storage
    if not storage:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return storage


@router.get("/status", response_model=PipelineStatus)
async def get_pipeline_status(pm=Depends(get_process_manager)):
    """Get status of all scripts."""
    runs = await pm.list_runs(limit=100)

    status_counts = {s: 0 for s in ScriptStatus}
    for run in runs:
        status_counts[run.status] += 1

    return PipelineStatus(
        total_scripts=len(runs),
        running=status_counts[ScriptStatus.RUNNING],
        completed=status_counts[ScriptStatus.COMPLETED],
        failed=status_counts[ScriptStatus.FAILED],
        pending=status_counts[ScriptStatus.PENDING],
        runs=runs,
    )


@router.get("/runs/{run_id}", response_model=ScriptRun)
async def get_run(run_id: int, store=Depends(get_storage)):
    """Get a specific script run."""
    run = await store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/{script_name}/start", response_model=ScriptRun)
async def start_script(
    script_name: str,
    request: StartScriptRequest,
    pm=Depends(get_process_manager),
):
    """Start a script."""
    try:
        run = await pm.start(request.script_name, request.parameters)
        return run
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{run_id}/pause")
async def pause_run(run_id: int, pm=Depends(get_process_manager)):
    """Pause a running script."""
    status = await pm.get_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found")
    if status != ScriptStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause script in {status.value} state",
        )
    await pm.pause(run_id)
    return {"status": "paused"}


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: int, pm=Depends(get_process_manager)):
    """Resume a paused script."""
    status = await pm.get_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found")
    if status != ScriptStatus.PAUSED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume script in {status.value} state",
        )
    await pm.resume(run_id)
    return {"status": "running"}


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: int, pm=Depends(get_process_manager)):
    """Stop a script."""
    status = await pm.get_status(run_id)
    if not status:
        raise HTTPException(status_code=404, detail="Run not found")
    if status not in (ScriptStatus.RUNNING, ScriptStatus.PAUSED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot stop script in {status.value} state",
        )
    await pm.stop(run_id)
    return {"status": "stopped"}
