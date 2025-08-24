"""
OpenAIPerf Benchmarking Tools (ob_tools)

A minimal-invasion benchmarking toolkit that automatically detects
and monitors AI training and inference workloads.
"""

__version__ = "0.1.0"
__author__ = "OpenAIPerf Team"

from .benchmark import benchmark, task, backend, monitor
from .cli import main

__all__ = ["benchmark", "task", "backend", "monitor", "main"]
