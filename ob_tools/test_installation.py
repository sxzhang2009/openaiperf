#!/usr/bin/env python3
"""
Test script to verify ob_tools installation and basic functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from ob_tools import benchmark, task, backend, monitor
        print("✓ Core modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return False
    
    try:
        from ob_tools.detector import detect_frameworks_in_script
        print("✓ Detector module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import detector: {e}")
        return False
    
    try:
        from ob_tools.cli import main
        print("✓ CLI module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CLI: {e}")
        return False
    
    return True

def test_benchmark_functionality():
    """Test basic benchmark functionality."""
    print("\nTesting benchmark functionality...")
    
    try:
        from ob_tools import benchmark
        
        # Test manual start/stop
        benchmark.start("test.task", "test.backend")
        benchmark.stop()
        print("✓ Manual start/stop works")
        
        # Test decorator
        @benchmark.task("test.decorator")
        def test_function():
            return "test"
        
        result = test_function()
        print("✓ Task decorator works")
        
        # Test context manager
        with benchmark.monitor():
            pass
        print("✓ Monitor context manager works")
        
        return True
        
    except Exception as e:
        print(f"✗ Benchmark functionality failed: {e}")
        return False

def test_detector_functionality():
    """Test framework detection functionality."""
    print("\nTesting detector functionality...")
    
    try:
        from ob_tools.detector import detect_frameworks_in_script
        
        # Create a simple test script
        test_script = """
import torch
import transformers
from vllm import LLM

def main():
    model = LLM(model="test")
    return model
"""
        
        # Write test script to temporary file
        with open("test_script.py", "w") as f:
            f.write(test_script)
        
        # Test detection
        result = detect_frameworks_in_script("test_script.py")
        
        # Clean up
        os.remove("test_script.py")
        
        print(f"✓ Framework detection works: {result['script_frameworks']}")
        return True
        
    except Exception as e:
        print(f"✗ Detector functionality failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("ob_tools Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_benchmark_functionality,
        test_detector_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ob_tools is ready to use.")
        print("\nNext steps:")
        print("1. Install the package: pip install -e .")
        print("2. Try running: ob --help")
        print("3. Check out the examples in ob_tools/examples/")
    else:
        print("❌ Some tests failed. Please check the installation.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
