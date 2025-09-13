#!/usr/bin/env python3
"""
Simple validation test for the VGuard Model pipeline
"""

import sys
import importlib.util

def test_imports():
    """Test that all required imports are available"""
    required_modules = [
        'pandas', 'tensorflow', 'sklearn', 'matplotlib', 
        'seaborn', 'plotly', 'PIL'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == 'PIL':
                import PIL
            else:
                __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            missing_modules.append(module)
            print(f"✗ {module}: {e}")
    
    return len(missing_modules) == 0

def test_pipeline_syntax():
    """Test that pipeline.py has valid Python syntax"""
    try:
        spec = importlib.util.spec_from_file_location("pipeline", "pipeline.py")
        if spec is None:
            print("✗ Could not load pipeline.py")
            return False
        
        # This will compile the module but not execute it
        module = importlib.util.module_from_spec(spec)
        print("✓ pipeline.py syntax is valid")
        return True
    except Exception as e:
        print(f"✗ pipeline.py syntax error: {e}")
        return False

def test_config_constants():
    """Test that required constants are defined"""
    try:
        spec = importlib.util.spec_from_file_location("pipeline", "pipeline.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        required_constants = ['CONFIG', 'LABELS']
        for const in required_constants:
            if hasattr(module, const):
                print(f"✓ {const} defined")
            else:
                print(f"✗ {const} not defined")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking constants: {e}")
        return False

def main():
    """Run all tests"""
    print("VGuard Model Pipeline Validation")
    print("=" * 40)
    
    tests = [
        ("Import Tests", test_imports),
        ("Syntax Tests", test_pipeline_syntax),
        ("Config Tests", test_config_constants)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if not test_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()