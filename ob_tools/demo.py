#!/usr/bin/env python3
"""
Demo script showing how to use the ob_tools package.
"""

import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_basic_usage():
    """Demonstrate basic usage of the benchmarking tools."""
    print("=" * 60)
    print("DEMO: Basic Benchmarking Usage")
    print("=" * 60)
    
    from ob_tools import benchmark
    
    # Method 1: Using decorators
    @benchmark.task("demo.task")
    def decorated_function():
        print("Running decorated function...")
        time.sleep(2)  # Simulate work
        return "Done!"
    
    print("1. Running function with @benchmark.task decorator:")
    result = decorated_function()
    print(f"   Result: {result}")
    
    # Method 2: Using context manager
    print("\n2. Running code with benchmark.monitor() context manager:")
    with benchmark.monitor():
        print("   Running monitored code block...")
        time.sleep(1)  # Simulate work
        print("   Monitored block completed!")
    
    # Method 3: Manual control
    print("\n3. Manual benchmark control:")
    benchmark.start("manual.task", "manual.backend")
    print("   Starting manual monitoring...")
    time.sleep(1)  # Simulate work
    benchmark.stop()
    print("   Manual monitoring stopped!")

def demo_framework_detection():
    """Demonstrate framework detection capabilities."""
    print("\n" + "=" * 60)
    print("DEMO: Framework Detection")
    print("=" * 60)
    
    from ob_tools.detector import detect_frameworks_in_script
    
    # Create a test script with multiple frameworks
    test_script = """
# This is a test script with multiple AI frameworks
import torch
import transformers
from vllm import LLM, SamplingParams
import deepspeed
import ray

def main():
    # PyTorch usage
    model = torch.nn.Linear(100, 10)
    
    # Transformers usage
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    
    # vLLM usage
    llm = LLM(model="llama3-8b")
    sampling_params = SamplingParams(temperature=0.7)
    
    # DeepSpeed usage
    ds_config = {"train_batch_size": 32}
    
    # Ray usage
    ray.init()
    
    return "All frameworks loaded!"
"""
    
    # Write test script
    with open("demo_test_script.py", "w") as f:
        f.write(test_script)
    
    # Detect frameworks
    print("Analyzing test script with multiple frameworks...")
    result = detect_frameworks_in_script("demo_test_script.py")
    
    # Display results
    print(f"\nDetected frameworks: {result['script_frameworks']}")
    print(f"Primary backend: {result['primary_backend']}")
    print(f"Task type: {result['task_type']}")
    
    if result['recommendations']:
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  • {rec}")
    
    # Clean up
    os.remove("demo_test_script.py")

def demo_cli_usage():
    """Demonstrate CLI usage."""
    print("\n" + "=" * 60)
    print("DEMO: CLI Usage")
    print("=" * 60)
    
    print("The ob CLI provides several commands:")
    print("\n1. Run a script with automatic detection:")
    print("   ob run your_script.py")
    
    print("\n2. Detect frameworks in a script:")
    print("   ob detect your_script.py")
    
    print("\n3. Manual monitoring:")
    print("   ob monitor --task llm.generation --backend vllm --duration 30")
    
    print("\n4. Analyze results:")
    print("   ob analyze ob_results_20240115_143022_llm_generation/")
    
    print("\n5. Initialize system detection:")
    print("   ob init")

def demo_output_format():
    """Show the output format of benchmark results."""
    print("\n" + "=" * 60)
    print("DEMO: Four Cards + Two Logs Output Format")
    print("=" * 60)
    
    print("Benchmark results are saved in the OpenAIPerf standard format:")
    print("\n📁 Results Directory Structure:")
    print("   ob_results_20240115_143022_llm_generation/")
    print("   ├── system.json          # System configuration (Card 1)")
    print("   ├── stack.json           # Software stack configuration (Card 2)")
    print("   ├── model.json           # Model configuration (Card 3)")
    print("   ├── run.json             # Run configuration (Card 4)")
    print("   ├── system.log           # System monitoring log (Log 1)")
    print("   └── run.log              # Runtime execution log (Log 2)")
    
    print("\n📋 FOUR CARDS (JSON Configuration Files):")
    print("\n1. system.json - Hardware and software configuration:")
    system_example = {
        "hardware": {
            "cpu": {"model": "Intel Xeon", "cores": 56},
            "accelerators": [{"type": "gpu", "name": "H100", "memory_gb": 80}],
            "memory": {"total_gb": 1024}
        },
        "software": {"platform": "Linux", "python_version": "3.9.0"}
    }
    import json
    print(json.dumps(system_example, indent=2))
    
    print("\n2. stack.json - Framework and engine configuration:")
    stack_example = {
        "framework": {"name": "PyTorch", "version": "2.4.0"},
        "engine": {"name": "vLLM", "version": "0.3.0"},
        "optimizations": {"mixed_precision": True}
    }
    print(json.dumps(stack_example, indent=2))
    
    print("\n3. model.json - Model architecture and weights:")
    model_example = {
        "metadata": {"name": "llama3-8b", "source": "huggingface://meta-llama/Llama-3-8B"},
        "architecture": {"type": "decoder_only", "layers": 32},
        "quantization": {"method": "AWQ", "bits": 4}
    }
    print(json.dumps(model_example, indent=2))
    
    print("\n4. run.json - Task and execution configuration:")
    run_example = {
        "task": {"name": "llm.generation", "scenario": "server"},
        "workload": {"qps_target": 100, "latency_sla_ms": 100},
        "execution": {"duration": 45.23}
    }
    print(json.dumps(run_example, indent=2))
    
    print("\n📊 TWO LOGS (Runtime Logs):")
    print("\n1. system.log - Real-time system metrics (JSONL format):")
    print('{"timestamp": 1705315822.123, "cpu_percent": 78.5, "memory_percent": 65.2}')
    print('{"timestamp": 1705315823.124, "cpu_percent": 79.1, "memory_percent": 65.8}')
    
    print("\n2. run.log - Execution events (JSONL format):")
    print('{"timestamp": 1705315822.123, "event": "benchmark_start", "phase": "initialization"}')
    print('{"timestamp": 1705315867.353, "event": "benchmark_complete", "duration_seconds": 45.23}')

def main():
    """Run all demos."""
    print("🚀 OpenAIPerf Benchmarking Tools (ob_tools) Demo")
    print("=" * 60)
    
    try:
        demo_basic_usage()
        demo_framework_detection()
        demo_cli_usage()
        demo_output_format()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\nTo get started:")
        print("1. Install: pip install -e .")
        print("2. Try: ob --help")
        print("3. Check examples in ob_tools/examples/")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check the installation and dependencies.")

if __name__ == "__main__":
    main()
