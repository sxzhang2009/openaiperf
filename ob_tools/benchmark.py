"""
Core benchmarking functionality for minimal-invasion monitoring.
"""

import functools
import time
import json
import os
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime
import psutil
import GPUtil
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkContext:
    """Global benchmark context for collecting metrics across the application."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.task_name = None
        self.backend_name = None
        self.monitoring_active = False
        self.monitoring_thread = None
        self.system_metrics = []
        self._lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {
            'total_tokens': 0,
            'total_requests': 0,
            'total_tokens_generated': 0,
            'latency_measurements': [],
            'throughput_measurements': [],
            'model_info': {},
            'workload_info': {},
            'inference_start_time': None,
            'inference_end_time': None
        }
    
    def start_monitoring(self, task_name: str, backend_name: Optional[str] = None):
        """Start monitoring for a benchmark task."""
        with self._lock:
            self.task_name = task_name
            self.backend_name = backend_name
            self.start_time = time.time()
            self.monitoring_active = True
            self.system_metrics = []
            
            # Start background monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitor_system_metrics)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info(f"Started monitoring task: {task_name} (backend: {backend_name})")
    
    def stop_monitoring(self):
        """Stop monitoring and save results."""
        with self._lock:
            if not self.monitoring_active:
                return
            
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=1.0)
            
            end_time = time.time()
            duration = end_time - self.start_time
            
            # Calculate aggregated metrics
            self._calculate_aggregated_metrics(duration)
            
            # Save results
            self._save_results()
            
            logger.info(f"Stopped monitoring. Duration: {duration:.2f}s")
    
    def record_inference_start(self):
        """Record the start of inference for latency calculation."""
        self.performance_metrics['inference_start_time'] = time.time()
    
    def record_inference_end(self, tokens_generated: int = 0, request_count: int = 1):
        """Record the end of inference and calculate latency."""
        if self.performance_metrics['inference_start_time']:
            end_time = time.time()
            latency = end_time - self.performance_metrics['inference_start_time']
            self.performance_metrics['inference_end_time'] = end_time
            self.performance_metrics['latency_measurements'].append(latency)
            self.performance_metrics['total_tokens_generated'] += tokens_generated
            self.performance_metrics['total_requests'] += request_count
            
            # Calculate throughput (tokens per second)
            if latency > 0:
                throughput = tokens_generated / latency
                self.performance_metrics['throughput_measurements'].append(throughput)
    
    def record_model_info(self, model_name: str, model_config: Dict[str, Any]):
        """Record model information for detailed reporting."""
        self.performance_metrics['model_info'] = {
            'name': model_name,
            'config': model_config
        }
    
    def record_workload_info(self, workload_config: Dict[str, Any]):
        """Record workload information."""
        self.performance_metrics['workload_info'] = workload_config
    
    def _monitor_system_metrics(self):
        """Background thread for collecting system metrics."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                time.sleep(1.0)  # Collect every second
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                break
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            for i, gpu in enumerate(gpus):
                gpu_metrics.append({
                    'gpu_id': i,
                    'name': gpu.name,
                    'load_percent': gpu.load * 100,
                    'memory_percent': gpu.memoryUtil * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'temperature': gpu.temperature,
                })
            metrics['gpus'] = gpu_metrics
        except Exception as e:
            logger.debug(f"Could not collect GPU metrics: {e}")
            metrics['gpus'] = []
        
        return metrics
    
    def _calculate_aggregated_metrics(self, duration: float):
        """Calculate aggregated metrics from collected data."""
        if not self.system_metrics:
            return
        
        # Calculate averages
        cpu_avg = sum(m['cpu_percent'] for m in self.system_metrics) / len(self.system_metrics)
        memory_avg = sum(m['memory_percent'] for m in self.system_metrics) / len(self.system_metrics)
        
        # GPU metrics
        gpu_metrics = {}
        if self.system_metrics and 'gpus' in self.system_metrics[0]:
            for gpu_idx in range(len(self.system_metrics[0]['gpus'])):
                gpu_loads = [m['gpus'][gpu_idx]['load_percent'] for m in self.system_metrics if m['gpus']]
                gpu_memory = [m['gpus'][gpu_idx]['memory_percent'] for m in self.system_metrics if m['gpus']]
                
                if gpu_loads:
                    gpu_metrics[f'gpu_{gpu_idx}'] = {
                        'load_avg': sum(gpu_loads) / len(gpu_loads),
                        'memory_avg': sum(gpu_memory) / len(gpu_memory) if gpu_memory else 0,
                    }
        
        self.metrics = {
            'task_name': self.task_name,
            'backend_name': self.backend_name,
            'duration_seconds': duration,
            'cpu_avg_percent': cpu_avg,
            'memory_avg_percent': memory_avg,
            'gpu_metrics': gpu_metrics,
            'sample_count': len(self.system_metrics),
        }
    
    def _save_results(self):
        """Save benchmark results in four cards + two logs format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_safe = self.task_name.replace('.', '_') if self.task_name else 'unknown'
        
        # Create results directory
        results_dir = f"ob_results_{timestamp}_{task_safe}"
        os.makedirs(results_dir, exist_ok=True)
        
        # FOUR CARDS (JSON configuration files)
        
        # 1. system.json - System configuration
        system_config = self._generate_system_config()
        system_file = os.path.join(results_dir, "system.json")
        with open(system_file, 'w') as f:
            json.dump(system_config, f, indent=2)
        
        # 2. stack.json - Software stack configuration
        stack_config = self._generate_stack_config()
        stack_file = os.path.join(results_dir, "stack.json")
        with open(stack_file, 'w') as f:
            json.dump(stack_config, f, indent=2)
        
        # 3. model.json - Model configuration
        model_config = self._generate_model_config()
        model_file = os.path.join(results_dir, "model.json")
        with open(model_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # 4. run.json - Run configuration
        run_config = self._generate_run_config()
        run_file = os.path.join(results_dir, "run.json")
        with open(run_file, 'w') as f:
            json.dump(run_config, f, indent=2)
        
        # TWO LOGS (Runtime logs)
        
        # 1. system.log - System monitoring log
        system_log_file = os.path.join(results_dir, "system.log")
        with open(system_log_file, 'w') as f:
            for metrics in self.system_metrics:
                f.write(json.dumps(metrics) + '\n')
        
        # 2. run.log - Runtime execution log
        run_log_file = os.path.join(results_dir, "run.log")
        with open(run_log_file, 'w') as f:
            self._write_run_log(run_log_file)
        
        logger.info(f"Results saved to: {results_dir}")
        logger.info("Generated four cards: system.json, stack.json, model.json, run.json")
        logger.info("Generated two logs: system.log, run.log")
    
    def _generate_system_config(self) -> Dict[str, Any]:
        """Generate system.json configuration."""
        import platform
        import psutil
        
        system_config = {
            "hardware": {
                "cpu": {
                    "model": platform.processor(),
                    "cores": psutil.cpu_count(),
                    "frequency": "unknown"
                },
                "memory": {
                    "total_gb": psutil.virtual_memory().total / (1024**3),
                    "type": "unknown"
                },
                "storage": {
                    "type": "unknown",
                    "capacity_gb": 0
                },
                "network": {
                    "type": "unknown",
                    "interfaces": len(psutil.net_if_addrs())
                }
            },
            "software": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "driver": "unknown",
                "power_sensors": {
                    "source": "auto",
                    "frequency_hz": 1,
                    "accuracy": "unknown"
                }
            }
        }
        
        # Add GPU information if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                system_config["hardware"]["accelerators"] = []
                for gpu in gpus:
                    system_config["hardware"]["accelerators"].append({
                        "type": "gpu",
                        "vendor": "unknown",
                        "name": gpu.name,
                        "count": 1,
                        "memory_gb": gpu.memoryTotal / 1024,
                        "interconnect": "unknown"
                    })
        except ImportError:
            pass
        
        return system_config
    
    def _generate_stack_config(self) -> Dict[str, Any]:
        """Generate stack.json configuration."""
        stack_config = {
            "framework": {
                "name": "unknown",
                "version": "unknown",
                "compute_backend": {
                    "type": "unknown",
                    "version": "unknown"
                }
            },
            "engine": {
                "name": self.backend_name or "unknown",
                "version": "unknown"
            },
            "container": {
                "image": "unknown",
                "digest": "unknown"
            },
            "environment": {
                "pip_freeze": "unknown",
                "conda_lock": "unknown"
            },
            "optimizations": {
                "mixed_precision": False,
                "gradient_checkpointing": False,
                "dynamic_batching": False
            }
        }
        
        # Try to detect actual framework information
        try:
            import torch
            stack_config["framework"]["name"] = "PyTorch"
            stack_config["framework"]["version"] = torch.__version__
            stack_config["framework"]["compute_backend"]["type"] = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        
        return stack_config
    
    def _generate_model_config(self) -> Dict[str, Any]:
        """Generate model.json configuration."""
        # Use recorded model info if available
        if self.performance_metrics.get('model_info'):
            model_info = self.performance_metrics['model_info']
            model_name = model_info.get('name', 'unknown')
            model_config = model_info.get('config', {})
            
            # Try to extract model details from vLLM or other frameworks
            architecture_info = self._extract_model_architecture(model_name, model_config)
            
            model_config = {
                "metadata": {
                    "name": model_name,
                    "version": "1.0.0",
                    "source": "huggingface" if "huggingface" in model_name.lower() else "unknown",
                    "license": "unknown"
                },
                "weights": {
                    "sha256": "unknown",
                    "download_url": model_name if model_name != "unknown" else "unknown",
                    "size_gb": architecture_info.get('size_gb', 0)
                },
                "architecture": architecture_info.get('architecture', {
                    "type": "decoder_only",
                    "layers": 0,
                    "hidden_size": 0,
                    "attention_heads": 0,
                    "vocab_size": 0
                }),
                "quantization": {
                    "method": "none",
                    "bits": 16,
                    "calibration_dataset": "none"
                },
                "training": {
                    "base_model": "unknown",
                    "fine_tuning": {
                        "method": "none"
                    }
                }
            }
        else:
            # Fallback to placeholder
            model_config = {
                "metadata": {
                    "name": "unknown",
                    "version": "1.0.0",
                    "source": "unknown",
                    "license": "unknown"
                },
                "weights": {
                    "sha256": "unknown",
                    "download_url": "unknown",
                    "size_gb": 0
                },
                "architecture": {
                    "type": "decoder_only",
                    "layers": 0,
                    "hidden_size": 0,
                    "attention_heads": 0,
                    "vocab_size": 0
                },
                "quantization": {
                    "method": "none",
                    "bits": 16,
                    "calibration_dataset": "none"
                },
                "training": {
                    "base_model": "unknown",
                    "fine_tuning": {
                        "method": "none"
                    }
                }
            }
        
        return model_config
    
    def _extract_model_architecture(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model architecture information from model name and config."""
        architecture_info = {
            'architecture': {
                "type": "decoder_only",
                "layers": 0,
                "hidden_size": 0,
                "attention_heads": 0,
                "vocab_size": 0
            },
            'size_gb': 0
        }
        
        # Common model size patterns
        model_size_patterns = {
            '1.7b': {'layers': 24, 'hidden_size': 2048, 'attention_heads': 16, 'size_gb': 3.4},
            '3b': {'layers': 32, 'hidden_size': 3072, 'attention_heads': 24, 'size_gb': 6.8},
            '7b': {'layers': 32, 'hidden_size': 4096, 'attention_heads': 32, 'size_gb': 13.5},
            '13b': {'layers': 40, 'hidden_size': 5120, 'attention_heads': 40, 'size_gb': 26.0},
            '30b': {'layers': 60, 'hidden_size': 6656, 'attention_heads': 52, 'size_gb': 60.0},
            '70b': {'layers': 80, 'hidden_size': 8192, 'attention_heads': 64, 'size_gb': 140.0},
        }
        
        # Try to match model size
        for size_pattern, arch in model_size_patterns.items():
            if size_pattern in model_name.lower():
                architecture_info['architecture'].update(arch)
                architecture_info['size_gb'] = arch['size_gb']
                break
        
        # Try to extract from model config if available
        if model_config:
            if 'num_layers' in model_config:
                architecture_info['architecture']['layers'] = model_config['num_layers']
            if 'hidden_size' in model_config:
                architecture_info['architecture']['hidden_size'] = model_config['hidden_size']
            if 'num_attention_heads' in model_config:
                architecture_info['architecture']['attention_heads'] = model_config['num_attention_heads']
            if 'vocab_size' in model_config:
                architecture_info['architecture']['vocab_size'] = model_config['vocab_size']
        
        return architecture_info
    
    def _generate_run_config(self) -> Dict[str, Any]:
        """Generate run.json configuration."""
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        run_config = {
            "task": {
                "name": self.task_name or "unknown",
                "version": "1.0.0",
                "scenario": "unknown"
            },
            "dataset": {
                "name": "unknown",
                "subset_seed": 0,
                "subset_size": 0,
                "sampling_method": "unknown"
            },
            "workload": {
                "qps_target": 0,
                "latency_sla_ms": 0,
                "concurrency": 1,
                "batch_size": 1
            },
            "execution": {
                "repeats": 1,
                "random_seed": 42,
                "warmup_batches": 0,
                "cooldown_seconds": 0
            },
            "monitoring": {
                "sampling_interval_ms": 1000,
                "metrics": ["latency", "throughput", "power", "memory"]
            },
            "performance": performance_metrics
        }
        
        # Add actual execution data
        if self.start_time:
            run_config["execution"]["start_time"] = self.start_time
            run_config["execution"]["end_time"] = time.time()
            run_config["execution"]["duration"] = self.metrics.get('duration_seconds', 0)
        
        return run_config
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from collected data."""
        perf_metrics = self.performance_metrics
        
        # Calculate latency statistics
        latencies = perf_metrics.get('latency_measurements', [])
        latency_stats = {
            'min_ms': 0,
            'max_ms': 0,
            'mean_ms': 0,
            'p50_ms': 0,
            'p95_ms': 0,
            'p99_ms': 0
        }
        
        if latencies:
            latencies_ms = [lat * 1000 for lat in latencies]  # Convert to milliseconds
            latencies_ms.sort()
            latency_stats = {
                'min_ms': min(latencies_ms),
                'max_ms': max(latencies_ms),
                'mean_ms': sum(latencies_ms) / len(latencies_ms),
                'p50_ms': latencies_ms[len(latencies_ms) // 2],
                'p95_ms': latencies_ms[int(len(latencies_ms) * 0.95)] if len(latencies_ms) > 1 else latencies_ms[0],
                'p99_ms': latencies_ms[int(len(latencies_ms) * 0.99)] if len(latencies_ms) > 1 else latencies_ms[0]
            }
        
        # Calculate throughput statistics (only for LLM tasks)
        throughputs = perf_metrics.get('throughput_measurements', [])
        total_tokens = perf_metrics.get('total_tokens_generated', 0)
        
        # Only show token throughput for actual LLM tasks
        is_llm_task = self.task_name and "llm" in self.task_name.lower()
        
        if is_llm_task and throughputs:
            throughput_stats = {
                'min_tokens_per_sec': min(throughputs),
                'max_tokens_per_sec': max(throughputs),
                'mean_tokens_per_sec': sum(throughputs) / len(throughputs),
                'total_tokens': total_tokens
            }
        else:
            throughput_stats = {
                'min_tokens_per_sec': 0,
                'max_tokens_per_sec': 0,
                'mean_tokens_per_sec': 0,
                'total_tokens': total_tokens
            }
        
        # Calculate overall metrics
        total_requests = perf_metrics.get('total_requests', 0)
        total_duration = self.metrics.get('duration_seconds', 0)
        
        overall_metrics = {
            'requests_per_second': total_requests / total_duration if total_duration > 0 else 0,
            'total_requests': total_requests,
            'total_tokens_generated': total_tokens
        }
        
        # Only add tokens_per_second for LLM tasks
        if is_llm_task:
            overall_metrics['tokens_per_second'] = total_tokens / total_duration if total_duration > 0 else 0
        
        return {
            'latency': latency_stats,
            'throughput': throughput_stats,
            'overall': overall_metrics
        }
    
    def _write_run_log(self, log_file_path: str):
        """Write run.log with execution events."""
        import os
        
        run_events = [
            {
                "timestamp": self.start_time,
                "event": "benchmark_start",
                "phase": "initialization",
                "task": self.task_name,
                "backend": self.backend_name
            },
            {
                "timestamp": self.start_time + 1,
                "event": "monitoring_started",
                "phase": "monitoring",
                "metrics_collected": len(self.system_metrics)
            },
            {
                "timestamp": time.time(),
                "event": "benchmark_complete",
                "phase": "completion",
                "duration_seconds": self.metrics.get('duration_seconds', 0),
                "cpu_avg_percent": self.metrics.get('cpu_avg_percent', 0),
                "memory_avg_percent": self.metrics.get('memory_avg_percent', 0)
            }
        ]
        
        with open(log_file_path, 'w') as f:
            for event in run_events:
                f.write(json.dumps(event) + '\n')

# Global benchmark context
_benchmark_context = BenchmarkContext()

def task(task_name: str):
    """Decorator to mark a function as a benchmark task."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get backend name from function attribute if set by @backend decorator
            backend_name = getattr(func, '_backend', None)
            _benchmark_context.start_monitoring(task_name, backend_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _benchmark_context.stop_monitoring()
        return wrapper
    return decorator

def backend(backend_name: str):
    """Decorator to specify the backend."""
    def decorator(func):
        func._backend = backend_name
        return func
    return decorator

@contextmanager
def monitor():
    """Context manager for monitoring specific code blocks."""
    start_time = time.time()
    start_metrics = _benchmark_context._collect_system_metrics()
    
    try:
        yield
    finally:
        end_time = time.time()
        end_metrics = _benchmark_context._collect_system_metrics()
        
        # Store the monitoring data
        duration = end_time - start_time
        logger.info(f"Monitored block completed in {duration:.2f}s")

# Main benchmark object for easy access
class Benchmark:
    """Main benchmark object providing easy access to all functionality."""
    
    def __init__(self):
        self.context = _benchmark_context
    
    def start(self, task_name: str, backend_name: Optional[str] = None):
        """Start monitoring manually."""
        self.context.start_monitoring(task_name, backend_name)
    
    def stop(self):
        """Stop monitoring manually."""
        self.context.stop_monitoring()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.context.metrics.copy()


# --- Standardized Benchmark Framework ---

@dataclass
class BenchmarkConfig:
    """Configuration for a standardized benchmark."""
    name: str
    model_id: str
    dataset_id: str
    workload_args: Dict[str, Any] = field(default_factory=dict)
    quality_target: Optional[Dict[str, Any]] = None

class RunnableBenchmark(ABC):
    """Abstract base class for a runnable, standardized benchmark."""
    
    def __init__(self, config: BenchmarkConfig, benchmark_context: BenchmarkContext):
        self.config = config
        self.context = benchmark_context
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.device = None
        
    def execute(self):
        """Execute the full benchmark lifecycle."""
        # Use the global context for monitoring
        self.context.start_monitoring(task_name=self.config.name, backend_name="huggingface")
        try:
            logger.info(f"Setting up benchmark: {self.config.name}")
            self.setup()
            
            logger.info("Running benchmark...")
            self.run()
            
            if self.config.quality_target:
                logger.info("Evaluating quality...")
                self.evaluate()
        
        finally:
            logger.info("Tearing down benchmark...")
            self.teardown()
            self.context.stop_monitoring()
            
    @abstractmethod
    def setup(self):
        """Prepare the benchmark environment (e.g., download model, dataset)."""
        pass
        
    @abstractmethod
    def run(self):
        """Execute the core benchmark logic."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate the model's output against the quality target."""
        pass
        
    def teardown(self):
        """Clean up resources."""
        self.model = None
        self.tokenizer = None
        self.dataset = None
        logger.info("Resources cleaned up.")

# --- End of Standardized Benchmark Framework ---


# Create the main benchmark instance
benchmark = Benchmark()
