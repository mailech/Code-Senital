import pytest
from buggy_math import calculate_total, divide_numbers, find_max, calculate_average, is_even

def test_calculate_total():
    """Test calculate_total function"""
    items = [{"price": 10}, {"price": 20}, {"price": 30}]
    result = calculate_total(items)
    assert result == 60  # This will fail due to subtraction bug

def test_divide_numbers():
    """Test divide_numbers function"""
    result = divide_numbers(10, 2)
    assert result == 5
    
    # This will fail due to zero division
    with pytest.raises(ZeroDivisionError):
        divide_numbers(10, 0)

def test_find_max():
    """Test find_max function"""
    numbers = [1, 5, 3, 9, 2]
    result = find_max(numbers)
    assert result == 9  # This will fail due to < instead of >

def test_calculate_average():
    """Test calculate_average function"""
    numbers = [10, 20, 30]
    result = calculate_average(numbers)
    assert result == 20  # This will fail due to +1 bug

def test_is_even():
    """Test is_even function"""
    assert is_even(4) == True   # This will fail
    assert is_even(5) == False  # This will fail
    assert is_even(0) == True   # This will fail
