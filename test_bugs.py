#!/usr/bin/env python3
"""
Test script to verify all bugs are working correctly
"""

def test_math_bugs():
    """Test math operations bugs"""
    print("🧮 Testing Math Operations Bugs...")
    
    from buggy_math import calculate_total, divide_numbers, find_max, calculate_average, is_even
    
    # Test calculate_total bug
    items = [{"price": 10}, {"price": 20}, {"price": 30}]
    result = calculate_total(items)
    expected = 60
    print(f"  calculate_total: Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    # Test find_max bug
    numbers = [1, 5, 3, 9, 2]
    result = find_max(numbers)
    expected = 9
    print(f"  find_max: Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    # Test calculate_average bug
    numbers = [10, 20, 30]
    result = calculate_average(numbers)
    expected = 20
    print(f"  calculate_average: Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    # Test is_even bug
    result = is_even(4)
    expected = True
    print(f"  is_even(4): Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    print("✅ Math bugs are working correctly!")

def test_data_processor_bugs():
    """Test data processor bugs"""
    print("\n📊 Testing Data Processor Bugs...")
    
    from buggy_data_processor import DataProcessor
    
    processor = DataProcessor([1, 2, 3, 4, 5])
    
    # Test get_length bug
    result = processor.get_length()
    expected = 5
    print(f"  get_length: Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    # Test get_sum bug
    result = processor.get_sum()
    expected = 15
    print(f"  get_sum: Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    # Test get_average bug
    result = processor.get_average()
    expected = 3
    print(f"  get_average: Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    # Test find_max bug
    result = processor.find_max()
    expected = 5
    print(f"  find_max: Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    # Test filter_positive bug
    processor2 = DataProcessor([-1, 2, -3, 4, -5])
    result = processor2.filter_positive()
    expected = [2, 4]
    print(f"  filter_positive: Expected {expected}, got {result} {'✅' if result != expected else '❌'}")
    
    print("✅ Data processor bugs are working correctly!")

def test_pytest_failures():
    """Test that pytest will fail on these bugs"""
    print("\n🧪 Testing Pytest Failures...")
    
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "test_buggy_math.py", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print("✅ Pytest tests are failing as expected!")
            print("   This means the bugs are working correctly.")
        else:
            print("❌ Pytest tests are passing - this shouldn't happen!")
            
    except Exception as e:
        print(f"❌ Error running pytest: {e}")

def main():
    """Main test function"""
    print("=" * 60)
    print("Testing Buggy Code for Sentinel Demo")
    print("=" * 60)
    
    test_math_bugs()
    test_data_processor_bugs()
    test_pytest_failures()
    
    print("\n" + "=" * 60)
    print("🎉 All bugs are working correctly!")
    print("Ready to push to repository and trigger Sentinel!")
    print("=" * 60)

if __name__ == "__main__":
    main()
