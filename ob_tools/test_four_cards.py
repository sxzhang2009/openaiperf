#!/usr/bin/env python3
"""
Test script to verify the four cards + two logs output format.
"""

import sys
import os
import json
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_four_cards_two_logs():
    """Test that the benchmark generates the correct four cards + two logs format."""
    print("=" * 60)
    print("TEST: Four Cards + Two Logs Output Format")
    print("=" * 60)
    
    from ob_tools import benchmark
    
    # Start a test benchmark
    benchmark.start("test.four_cards", "test.backend")
    
    # Simulate some work
    print("Running test benchmark...")
    time.sleep(2)
    
    # Stop the benchmark
    benchmark.stop()
    
    # Find the results directory
    results_dirs = [d for d in os.listdir('.') if d.startswith('ob_results_')]
    if not results_dirs:
        print("❌ No results directory found!")
        return False
    
    results_dir = results_dirs[0]
    print(f"📁 Results directory: {results_dir}")
    
    # Check for four cards
    required_cards = ["system.json", "stack.json", "model.json", "run.json"]
    missing_cards = []
    
    for card in required_cards:
        card_path = os.path.join(results_dir, card)
        if not os.path.exists(card_path):
            missing_cards.append(card)
        else:
            print(f"✅ Found card: {card}")
    
    if missing_cards:
        print(f"❌ Missing cards: {missing_cards}")
        return False
    
    # Check for two logs
    required_logs = ["system.log", "run.log"]
    missing_logs = []
    
    for log in required_logs:
        log_path = os.path.join(results_dir, log)
        if not os.path.exists(log_path):
            missing_logs.append(log)
        else:
            print(f"✅ Found log: {log}")
    
    if missing_logs:
        print(f"❌ Missing logs: {missing_logs}")
        return False
    
    # Validate JSON structure
    print("\n📋 Validating card structures...")
    
    # Check system.json
    with open(os.path.join(results_dir, "system.json"), 'r') as f:
        system_data = json.load(f)
        if "hardware" in system_data and "software" in system_data:
            print("✅ system.json structure is valid")
        else:
            print("❌ system.json structure is invalid")
            return False
    
    # Check stack.json
    with open(os.path.join(results_dir, "stack.json"), 'r') as f:
        stack_data = json.load(f)
        if "framework" in stack_data and "engine" in stack_data:
            print("✅ stack.json structure is valid")
        else:
            print("❌ stack.json structure is invalid")
            return False
    
    # Check model.json
    with open(os.path.join(results_dir, "model.json"), 'r') as f:
        model_data = json.load(f)
        if "metadata" in model_data and "architecture" in model_data:
            print("✅ model.json structure is valid")
        else:
            print("❌ model.json structure is invalid")
            return False
    
    # Check run.json
    with open(os.path.join(results_dir, "run.json"), 'r') as f:
        run_data = json.load(f)
        if "task" in run_data and "execution" in run_data:
            print("✅ run.json structure is valid")
        else:
            print("❌ run.json structure is invalid")
            return False
    
    # Check log formats
    print("\n📊 Validating log formats...")
    
    # Check system.log (JSONL format)
    with open(os.path.join(results_dir, "system.log"), 'r') as f:
        system_log_lines = f.readlines()
        if system_log_lines:
            try:
                json.loads(system_log_lines[0].strip())
                print(f"✅ system.log is valid JSONL format ({len(system_log_lines)} entries)")
            except json.JSONDecodeError:
                print("❌ system.log is not valid JSONL format")
                return False
        else:
            print("❌ system.log is empty")
            return False
    
    # Check run.log (JSONL format)
    with open(os.path.join(results_dir, "run.log"), 'r') as f:
        run_log_lines = f.readlines()
        if run_log_lines:
            try:
                json.loads(run_log_lines[0].strip())
                print(f"✅ run.log is valid JSONL format ({len(run_log_lines)} entries)")
            except json.JSONDecodeError:
                print("❌ run.log is not valid JSONL format")
                return False
        else:
            print("❌ run.log is empty")
            return False
    
    print("\n🎉 All tests passed! Four cards + two logs format is working correctly.")
    return True

def main():
    """Run the test."""
    try:
        success = test_four_cards_two_logs()
        if success:
            print("\n✅ Four cards + two logs test PASSED")
        else:
            print("\n❌ Four cards + two logs test FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
