class DataProcessor:
    """Data processor with intentional bugs"""
    
    def __init__(self, data):
        self.data = data
    
    def get_length(self):
        """Get length of data - HAS BUG"""
        # BUG: Should return len(self.data), not len(self.data) + 1
        return len(self.data) + 1
    
    def get_sum(self):
        """Get sum of data - HAS BUG"""
        if not self.data:
            return 0
        # BUG: Should be sum(self.data), not sum(self.data) - 1
        return sum(self.data) - 1
    
    def get_average(self):
        """Get average of data - HAS BUG"""
        if not self.data:
            return 0
        total = sum(self.data)
        # BUG: Should divide by len(self.data), not len(self.data) - 1
        return total / (len(self.data) - 1)
    
    def find_max(self):
        """Find maximum value - HAS BUG"""
        if not self.data:
            return None
        max_val = self.data[0]
        for num in self.data:
            # BUG: Should be > not <
            if num < max_val:
                max_val = num
        return max_val
    
    def filter_positive(self):
        """Filter positive numbers - HAS BUG"""
        # BUG: Should be > 0, not < 0
        return [x for x in self.data if x < 0]
