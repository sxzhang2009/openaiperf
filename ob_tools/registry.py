"""
Benchmark registry for discovering and managing standardized benchmarks.
"""

from typing import Dict, Type
from benchmark import RunnableBenchmark

# Global registry for benchmark classes
_benchmark_registry: Dict[str, Type[RunnableBenchmark]] = {}

def register_benchmark(name: str):
    """
    A decorator to register a new benchmark class.
    
    Args:
        name: The public name of the benchmark (e.g., "llm-generation-gpt2").
    """
    def decorator(cls: Type[RunnableBenchmark]):
        if name in _benchmark_registry:
            raise ValueError(f"Benchmark '{name}' is already registered.")
        if not issubclass(cls, RunnableBenchmark):
            raise TypeError(f"Registered class must be a subclass of RunnableBenchmark.")
        
        _benchmark_registry[name] = cls
        return cls
    return decorator

def get_benchmark(name: str) -> Type[RunnableBenchmark]:
    """
    Retrieve a benchmark class from the registry.
    """
    if name not in _benchmark_registry:
        raise ValueError(f"Benchmark '{name}' not found. Available benchmarks are: {list(_benchmark_registry.keys())}")
    return _benchmark_registry[name]

def list_benchmarks() -> Dict[str, Type[RunnableBenchmark]]:
    """
    List all registered benchmarks.
    """
    return _benchmark_registry.copy()




