# Errors Repository - Sentinel Demo

This repository contains intentional bugs to demonstrate the Self-Healing Codebase Sentinel system.

## 🐛 Intentional Bugs

### Math Operations (`buggy_math.py`)
- `calculate_total()` - Uses subtraction instead of addition
- `divide_numbers()` - No zero division check
- `find_max()` - Uses < instead of >
- `calculate_average()` - Divides by len + 1 instead of len
- `is_even()` - Uses % 2 == 1 instead of % 2 == 0

### Data Processor (`buggy_data_processor.py`)
- `get_length()` - Returns len + 1 instead of len
- `get_sum()` - Subtracts 1 from sum
- `get_average()` - Divides by len - 1 instead of len
- `find_max()` - Uses < instead of >
- `filter_positive()` - Filters < 0 instead of > 0

## 🧪 Tests

Run the tests to see the failures:
```bash
python -m pytest test_*.py -v
```

## 🤖 Sentinel Integration

This repository is configured to work with the Self-Healing Codebase Sentinel:

1. **CI Integration** - GitHub Actions will notify Sentinel on test failures
2. **Automatic Fixes** - Sentinel will detect bugs and create pull requests
3. **Real-time Monitoring** - Dashboard shows system activity

## 🚀 How It Works

1. Push code with bugs to this repository
2. GitHub Actions runs tests and detects failures
3. Sentinel receives failure notifications
4. Sentinel analyzes the bugs and generates fixes
5. Sentinel creates pull requests with the fixes
6. Merge the fixes to see tests pass!

## 📊 Monitoring

- **Sentinel Dashboard**: View real-time system activity
- **GitHub Actions**: See CI/CD pipeline status
- **Pull Requests**: Review auto-generated fixes

## 🔧 Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `python -m pytest test_*.py -v`
3. Configure Sentinel webhook URL in GitHub secrets
4. Watch the magic happen! ✨