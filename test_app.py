#!/usr/bin/env python3
"""
Test script to verify the application works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_math_ops():
    """Test the math operations"""
    print("Testing math operations...")
    try:
        from app_demo.math_ops import add
        result = add(2, 2)
        print(f"add(2, 2) = {result}")
        assert result == 4, f"Expected 4, got {result}"
        print("✅ Math ops test passed!")
        return True
    except Exception as e:
        print(f"❌ Math ops test failed: {e}")
        return False

def test_dataloader():
    """Test the dataloader"""
    print("Testing dataloader...")
    try:
        from Cubic_Err.ml_errors.dataloader import ToyDataset
        ds = ToyDataset([1, 2, 3])
        length = len(ds)
        print(f"Dataset length: {length}")
        assert length == 3, f"Expected 3, got {length}"
        print("✅ Dataloader test passed!")
        return True
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        return False

def test_app_import():
    """Test that the app can be imported"""
    print("Testing app import...")
    try:
        from app.main import app
        print("✅ App import successful!")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def test_database():
    """Test database initialization"""
    print("Testing database...")
    try:
        from app.db import init_db, get_conn
        init_db()
        conn = get_conn()
        conn.close()
        print("✅ Database test passed!")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Running Self-Healing Codebase Sentinel Tests")
    print("=" * 50)
    
    tests = [
        test_math_ops,
        test_dataloader,
        test_app_import,
        test_database
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("=" * 50)

if __name__ == "__main__":
    main()
