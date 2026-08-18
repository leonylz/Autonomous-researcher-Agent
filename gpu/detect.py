"""
AutoResearcher GPU Detection and Management

Basic GPU utilities for detecting available GPUs, checking status,
and managing GPU allocation for experiments.
"""

import subprocess
import json
import logging
from typing import Optional

logger = logging.getLogger("autoresearcher.gpu")


def detect_gpus() -> list[int]:
    """Detect all available GPUs via nvidia-smi.

    Returns:
        List of GPU indices, e.g. [0, 1, 2, 3]
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            return list(range(len(lines)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.warning("nvidia-smi not found or failed. No GPUs detected.")
    return []


def gpu_status() -> list[dict]:
    """Get detailed GPU status.

    Returns:
        List of dicts with gpu_id, name, memory_used, memory_total,
        utilization, temperature, processes.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "gpu_id": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                    "temperature_c": int(parts[5]),
                })

        return gpus

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def is_gpu_available(gpu_id: int, memory_threshold_mb: int = 1000) -> bool:
    """Check if a GPU is available (low memory usage).

    Args:
        gpu_id: GPU index to check
        memory_threshold_mb: Consider available if used memory below this

    Returns:
        True if GPU appears free
    """
    statuses = gpu_status()
    for gpu in statuses:
        if gpu["gpu_id"] == gpu_id:
            return gpu["memory_used_mb"] < memory_threshold_mb
    return False


def get_usable_gpus(reserve_last: bool = True) -> list[int]:
    """Get list of GPUs usable for experiments.

    Args:
        reserve_last: If True, exclude the last GPU (for keep-alive)

    Returns:
        List of GPU indices available for experiments
    """
    gpus = detect_gpus()
    if not gpus:
        return []
    if reserve_last and len(gpus) > 1:
        return gpus[:-1]
    return gpus


def get_free_gpus(reserve_last: bool = True, memory_threshold_mb: int = 1000) -> list[int]:
    """Get GPUs that are both usable and currently free.

    Args:
        reserve_last: Exclude last GPU
        memory_threshold_mb: Memory threshold for "free"

    Returns:
        List of free GPU indices
    """
    usable = get_usable_gpus(reserve_last=reserve_last)
    return [g for g in usable if is_gpu_available(g, memory_threshold_mb)]


def print_gpu_summary():
    """Print a human-readable GPU summary."""
    statuses = gpu_status()
    if not statuses:
        print("No GPUs detected.")
        return

    print(f"{'GPU':>4} {'Name':<25} {'Memory':>15} {'Util':>6} {'Temp':>6}")
    print("-" * 60)
    for gpu in statuses:
        mem = f"{gpu['memory_used_mb']}MB/{gpu['memory_total_mb']}MB"
        print(
            f"{gpu['gpu_id']:>4} {gpu['name']:<25} {mem:>15} "
            f"{gpu['utilization_pct']:>5}% {gpu['temperature_c']:>4}°C"
        )

    usable = get_usable_gpus()
    free = get_free_gpus()
    print(f"\nUsable: {usable} | Free: {free}")


if __name__ == "__main__":
    print_gpu_summary()
