"""AutoResearcher Core - Autonomous ML Experiment Agent Framework.

New code should import from the individual modules (core.nodes, core.monitor,
...). This package only re-exports the stable public surface.
"""

from .execution import (
    ExecutionBackend,
    LocalExecutionBackend,
    SSHExecutionBackend,
    SlurmExecutionBackend,
    build_execution_backend,
)
from .memory import MemoryManager
from .monitor import ExperimentMonitor
from .nodes import ResearchGraph

__version__ = "0.1.1"
__all__ = [
    "ExecutionBackend",
    "ExperimentMonitor",
    "LocalExecutionBackend",
    "MemoryManager",
    "ResearchGraph",
    "SSHExecutionBackend",
    "SlurmExecutionBackend",
    "build_execution_backend",
]
