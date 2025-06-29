#!/usr/bin/env python3
"""Run all pipeline tests"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_all_tests():
    """Discover and run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "tests"
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on results
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_module):
    """Run a specific test module"""
    # Import the test module
    module_path = f"tests.pipeline.test_{test_module}"
    try:
        test_module = __import__(module_path, fromlist=[''])
        
        # Create suite from module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except ImportError as e:
        print(f"Error: Could not import test module '{module_path}': {e}")
        return 1


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test module
        test_name = sys.argv[1]
        print(f"\nRunning tests for: {test_name}")
        exit_code = run_specific_test(test_name)
    else:
        # Run all tests
        print("\nRunning all tests...")
        exit_code = run_all_tests()
        
    sys.exit(exit_code)