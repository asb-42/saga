"""System status route — CPU, GPU, memory, CUDA version."""
from __future__ import annotations

import os
import platform
from typing import Any

import psutil
from fastapi import APIRouter

router = APIRouter(prefix="/api/system", tags=["system"])


def _get_cpu_info() -> dict[str, Any]:
    """Collect CPU usage and count."""
    return {
        "percent": psutil.cpu_percent(interval=0.1),
        "count": psutil.cpu_count(),
        "count_physical": psutil.cpu_count(logical=False),
    }


def _get_memory_info() -> dict[str, Any]:
    """Collect RAM usage."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 1),
        "used_gb": round(mem.used / (1024**3), 1),
        "percent": mem.percent,
        "available_gb": round(mem.available / (1024**3), 1),
    }


def _get_gpu_info() -> dict[str, Any] | None:
    """Collect GPU info via PyTorch, if available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        info: dict[str, Any] = {
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [],
        }

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_mem / (1024**3)
            mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)

            info["devices"].append({
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_mem_gb": round(mem_total, 1),
                "allocated_gb": round(mem_allocated, 2),
                "reserved_gb": round(mem_reserved, 2),
                "free_gb": round(mem_total - mem_reserved, 2),
            })

        return info
    except ImportError:
        return None


def _get_disk_info() -> dict[str, Any]:
    """Collect disk usage for the project directory."""
    project_root = os.environ.get("SAGA_ROOT", os.getcwd())
    try:
        usage = psutil.disk_usage(project_root)
        return {
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
            "percent": usage.percent,
        }
    except Exception:
        return {"error": "unavailable"}


@router.get("")
async def system_status():
    """Return current system resource usage."""
    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": _get_cpu_info(),
        "memory": _get_memory_info(),
        "gpu": _get_gpu_info(),
        "disk": _get_disk_info(),
    }
