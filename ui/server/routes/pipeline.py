"""Pipeline control routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..models import StartScriptRequest, ScriptRun, ScriptStatus, PipelineStatus
from ..script_params import SCRIPT_PARAMS

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


@router.get("/script-params")
async def get_script_params():
    """Get available CLI parameters for all scripts that support configuration."""
    return SCRIPT_PARAMS


@router.get("/status", response_model=PipelineStatus)
async def get_pipeline_status(pm=Depends(get_process_manager)):
    """Get status of all scripts."""
    from ..data_ingestion import get_historical_pipeline_runs

    # Get live runs from process manager
    live_runs = await pm.list_runs(limit=100)

    # Get historical runs from filesystem
    historical_runs = get_historical_pipeline_runs()

    # Merge: live runs take precedence, add historical runs not covered by live
    seen_scripts = {r.script_name for r in live_runs}
    for hr in historical_runs:
        if hr["script_name"] not in seen_scripts:
            live_runs.append(ScriptRun(**hr))
            seen_scripts.add(hr["script_name"])

    status_counts = {s: 0 for s in ScriptStatus}
    for run in live_runs:
        status_counts[run.status] += 1

    return PipelineStatus(
        total_scripts=len(live_runs),
        running=status_counts[ScriptStatus.RUNNING],
        completed=status_counts[ScriptStatus.COMPLETED],
        failed=status_counts[ScriptStatus.FAILED],
        pending=status_counts[ScriptStatus.PENDING],
        runs=live_runs,
    )


@router.get("/runs/{run_id}", response_model=ScriptRun)
async def get_run(run_id: int, store=Depends(get_storage)):
    """Get a specific script run."""
    run = await store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}/output")
async def get_run_output(run_id: int, store=Depends(get_storage)):
    """Get the last output and error message for a run."""
    run = await store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_id,
        "status": run.status.value,
        "exit_code": run.exit_code,
        "last_output": run.last_output,
        "error_message": run.error_message,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


@router.get("/scripts/{script_name}/runs")
async def get_script_runs(script_name: str, store=Depends(get_storage)):
    """Get all runs for a specific script, ordered by most recent first."""
    runs = await store.list_runs(script_name=script_name, limit=50)
    return {
        "script_name": script_name,
        "runs": runs,
        "count": len(runs),
    }


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
