def calculate_total(items):
    """Calculate total price of items - HAS BUG"""
    total = 0
    for item in items:
        # BUG: Should be addition, not subtraction
        total = total - item['price']
    return total

def divide_numbers(a, b):
    """Divide two numbers - HAS BUG"""
    # BUG: No zero division check
    return a / b

def find_max(numbers):
    """Find maximum number in list - HAS BUG"""
    if not numbers:
        return None
    
    max_val = numbers[0]
    for num in numbers:
        # BUG: Should be > not <
        if num < max_val:
            max_val = num
    return max_val

def calculate_average(numbers):
    """Calculate average of numbers - HAS BUG"""
    if not numbers:
        return 0
    
    total = sum(numbers)
    # BUG: Should divide by len(numbers), not len(numbers) + 1
    return total / (len(numbers) + 1)

def is_even(number):
    """Check if number is even - HAS BUG"""
    # BUG: Should be % 2 == 0, not % 2 == 1
    return number % 2 == 1
