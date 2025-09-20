#!/usr/bin/env python3
"""
Quick test script to verify the application works
"""

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from app_demo.math_ops import add
        print("✅ app_demo.math_ops imported")
        
        from Cubic_Err.ml_errors.dataloader import ToyDataset
        print("✅ Cubic_Err.ml_errors.dataloader imported")
        
        from app.main import app
        print("✅ app.main imported")
        
        from app.db import init_db
        print("✅ app.db imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_math_ops():
    """Test math operations"""
    print("Testing math operations...")
    
    try:
        from app_demo.math_ops import add
        result = add(2, 2)
        if result == 4:
            print("✅ Math operations working correctly")
            return True
        else:
            print(f"❌ Math operations failed: expected 4, got {result}")
            return False
    except Exception as e:
        print(f"❌ Math operations error: {e}")
        return False

def test_dataloader():
    """Test dataloader"""
    print("Testing dataloader...")
    
    try:
        from Cubic_Err.ml_errors.dataloader import ToyDataset
        ds = ToyDataset([1, 2, 3])
        length = len(ds)
        if length == 3:
            print("✅ Dataloader working correctly")
            return True
        else:
            print(f"❌ Dataloader failed: expected 3, got {length}")
            return False
    except Exception as e:
        print(f"❌ Dataloader error: {e}")
        return False

def test_database():
    """Test database initialization"""
    print("Testing database...")
    
    try:
        from app.db import init_db, get_conn
        init_db()
        conn = get_conn()
        conn.close()
        print("✅ Database working correctly")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Quick Test - Self-Healing Codebase Sentinel")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_math_ops,
        test_dataloader,
        test_database
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! The application is ready.")
        print("\nTo start the application:")
        print("1. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print("2. In another terminal: python app/worker.py")
        print("3. Open: http://localhost:8000/dashboard")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
