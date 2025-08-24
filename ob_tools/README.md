# OpenAIPerf Benchmarking Tools (ob_tools)

A minimal-invasion benchmarking toolkit that automatically detects and monitors AI training and inference workloads.

## Features

- **Automatic Framework Detection**: Automatically detects vLLM, DeepSpeed, Ray, PyTorch, TensorFlow, and more
- **Minimal Code Changes**: Add just 2-3 lines to your existing code
- **Real-time Monitoring**: Collects CPU, GPU, memory, and power metrics
- **Framework Agnostic**: Works with any combination of AI frameworks
- **Easy CLI**: Simple command-line interface for running benchmarks

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### 1. Minimal Code Changes

Add just a few lines to your existing code:

```python
# Your existing vLLM code
from ob_tools import benchmark

@benchmark.task("llm.generation")
def main():
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="llama3-8b")
    sampling_params = SamplingParams(temperature=0.7)
    
    with benchmark.monitor():
        outputs = llm.generate(prompts, sampling_params)
    
    return outputs

if __name__ == "__main__":
    main()
```

### 2. Run with Automatic Detection

```bash
# The tool automatically detects frameworks and configurations
ob run your_script.py

# Or with explicit configuration
ob run your_script.py --task llm.generation --backend vllm
```

### 3. View Results

Results are automatically saved in the OpenAIPerf standard "four cards + two logs" format:

```
ob_results_20240115_143022_llm_generation/
├── system.json          # System configuration (Card 1)
├── stack.json           # Software stack configuration (Card 2)
├── model.json           # Model configuration (Card 3)
├── run.json             # Run configuration (Card 4)
├── system.log           # System monitoring log (Log 1)
└── run.log              # Runtime execution log (Log 2)
```

## Usage Examples

### vLLM Inference

```python
# vllm_example.py
from ob_tools import benchmark

@benchmark.task("llm.generation")
def main():
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="llama3-8b")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    
    prompts = ["Hello, how are you?", "What is machine learning?"]
    
    with benchmark.monitor():
        outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()
```

Run: `ob run vllm_example.py`

### DeepSpeed Training

```python
# deepspeed_example.py
from ob_tools import benchmark

@benchmark.task("llm.training")
def main():
    import deepspeed
    import torch
    
    # Your existing DeepSpeed setup
    model = torch.nn.Linear(100, 10)
    ds_config = {
        "train_batch_size": 32,
        "steps_per_print": 10,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 0.001}
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 0.001, "warmup_num_steps": 1000}
        },
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2}
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config
    )
    
    # Training loop
    with benchmark.monitor():
        for step in range(100):
            # Your training code here
            loss = model_engine(torch.randn(32, 100))
            model_engine.backward(loss)
            model_engine.step()

if __name__ == "__main__":
    main()
```

Run: `ob run deepspeed_example.py`

### Ray Application

```python
# ray_example.py
import ray
from ob_tools import benchmark

@ray.remote
@benchmark.task("llm.generation")
class VLLMWorker:
    def __init__(self):
        from vllm import LLM
        self.llm = LLM(model="llama3-8b")
    
    def generate(self, prompts):
        with benchmark.monitor():
            return self.llm.generate(prompts)

def main():
    ray.init()
    
    workers = [VLLMWorker.remote() for _ in range(4)]
    prompts = ["Hello"] * 10
    
    results = ray.get([w.generate.remote(prompts) for w in workers])
    
    ray.shutdown()

if __name__ == "__main__":
    main()
```

Run: `ob run ray_example.py`

## CLI Commands

### Run Script with Benchmarking

```bash
# Automatic detection (recommended)
ob run script.py

# Manual configuration
ob run script.py --task llm.generation --backend vllm

# Verbose output
ob run script.py --verbose

# Custom output directory
ob run script.py --output-dir ./my_results
```

### Detect Frameworks

```bash
# Analyze a script without running it
ob detect script.py
```

### Manual Monitoring

```bash
# Monitor for 60 seconds
ob monitor --task llm.generation --backend vllm

# Monitor for custom duration
ob monitor --duration 120
```

### Analyze Results

```bash
# Analyze benchmark results
ob analyze ob_results_20240115_143022_llm_generation/
```

### System Detection

```bash
# Initialize system detection
ob init
```

## Framework Detection

The tool automatically detects these frameworks:

- **vLLM**: LLM inference optimization
- **DeepSpeed**: Distributed training
- **Ray**: Distributed computing
- **PyTorch**: Deep learning framework
- **TensorFlow**: Deep learning framework
- **Transformers**: Hugging Face ecosystem
- **SGLang**: Structured generation
- **TensorRT**: GPU optimization
- **ONNX**: Cross-platform inference

## Output Format (Four Cards + Two Logs)

### Four Cards (JSON Configuration Files)

#### system.json
```json
{
  "hardware": {
    "cpu": {"model": "Intel Xeon", "cores": 56},
    "accelerators": [{"type": "gpu", "name": "H100", "memory_gb": 80}],
    "memory": {"total_gb": 1024}
  },
  "software": {"platform": "Linux", "python_version": "3.9.0"}
}
```

#### stack.json
```json
{
  "framework": {"name": "PyTorch", "version": "2.4.0"},
  "engine": {"name": "vLLM", "version": "0.3.0"},
  "optimizations": {"mixed_precision": true}
}
```

#### model.json
```json
{
  "metadata": {"name": "llama3-8b", "source": "huggingface://meta-llama/Llama-3-8B"},
  "architecture": {"type": "decoder_only", "layers": 32},
  "quantization": {"method": "AWQ", "bits": 4}
}
```

#### run.json
```json
{
  "task": {"name": "llm.generation", "scenario": "server"},
  "workload": {"qps_target": 100, "latency_sla_ms": 100},
  "execution": {"duration": 45.23}
}
```

### Two Logs (Runtime Logs)

#### system.log
```
{"timestamp": 1705315822.123, "cpu_percent": 78.5, "memory_percent": 65.2, "gpus": [...]}
{"timestamp": 1705315823.124, "cpu_percent": 79.1, "memory_percent": 65.8, "gpus": [...]}
...
```

#### run.log
```
{"timestamp": 1705315822.123, "event": "benchmark_start", "phase": "initialization"}
{"timestamp": 1705315867.353, "event": "benchmark_complete", "duration_seconds": 45.23}
```

## Advanced Usage

### Custom Monitoring Blocks

```python
from ob_tools import benchmark

def my_function():
    # Some setup code
    
    with benchmark.monitor():
        # Only this block will be monitored
        result = expensive_operation()
    
    # More code (not monitored)
    return result
```

### Manual Benchmark Control

```python
from ob_tools import benchmark

# Start monitoring manually
benchmark.start("custom.task", "custom.backend")

# Your code here
do_something()

# Stop monitoring
benchmark.stop()

# Get current metrics
metrics = benchmark.get_metrics()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License
