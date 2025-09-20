import pytest
from buggy_data_processor import DataProcessor

def test_data_processor_length():
    """Test DataProcessor get_length method"""
    processor = DataProcessor([1, 2, 3, 4, 5])
    result = processor.get_length()
    assert result == 5  # This will fail due to +1 bug

def test_data_processor_sum():
    """Test DataProcessor get_sum method"""
    processor = DataProcessor([10, 20, 30])
    result = processor.get_sum()
    assert result == 60  # This will fail due to -1 bug

def test_data_processor_average():
    """Test DataProcessor get_average method"""
    processor = DataProcessor([10, 20, 30])
    result = processor.get_average()
    assert result == 20  # This will fail due to -1 bug

def test_data_processor_find_max():
    """Test DataProcessor find_max method"""
    processor = DataProcessor([1, 5, 3, 9, 2])
    result = processor.find_max()
    assert result == 9  # This will fail due to < instead of >

def test_data_processor_filter_positive():
    """Test DataProcessor filter_positive method"""
    processor = DataProcessor([-1, 2, -3, 4, -5])
    result = processor.filter_positive()
    assert result == [2, 4]  # This will fail due to < 0 instead of > 0
