# Complete Setup Guide for mailech/Errors Repository

## 🎯 **Step-by-Step Instructions**

### **Step 1: Set Up GitHub Token**

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` permissions
3. Set the token as environment variable:

```bash
# Windows PowerShell
$env:GITHUB_TOKEN="your_token_here"

# Windows CMD
set GITHUB_TOKEN=your_token_here

# Linux/Mac
export GITHUB_TOKEN=your_token_here
```

### **Step 2: Push Buggy Code to Repository**

Run the push script to add all the buggy code files:

```bash
python push_to_errors_repo.py
```

This will push:
- `buggy_math.py` - Math functions with bugs
- `test_buggy_math.py` - Tests that will fail
- `buggy_data_processor.py` - Data processing with bugs
- `test_data_processor.py` - Tests that will fail
- `requirements.txt` - Dependencies
- `README.md` - Repository documentation
- `.github/workflows/ci.yml` - CI/CD workflow

### **Step 3: Start Sentinel System**

In your code-cubicle directory, start the Sentinel system:

```bash
# Option 1: Use the complete startup script
python start_sentinel_complete.py

# Option 2: Manual startup
# Terminal 1:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2:
python app/worker.py
```

### **Step 4: Configure GitHub Webhook**

1. Go to your repository: https://github.com/mailech/Errors
2. Go to Settings → Webhooks → Add webhook
3. Set Payload URL to your ngrok URL: `https://your-ngrok-url.ngrok.io/webhooks/github`
4. Set Content type to `application/json`
5. Select events: `Push`, `Pull request`, `Workflow runs`
6. Add webhook

### **Step 5: Set GitHub Secrets**

1. Go to repository Settings → Secrets and variables → Actions
2. Add these secrets:
   - `SENTINEL_URL`: Your ngrok URL (e.g., `https://your-ngrok-url.ngrok.io`)

### **Step 6: Test the System**

1. **Run tests locally to see failures:**
   ```bash
   python -m pytest test_*.py -v
   ```

2. **Push to repository to trigger CI:**
   ```bash
   git add .
   git commit -m "Add buggy code for Sentinel demo"
   git push origin main
   ```

3. **Watch the magic happen:**
   - GitHub Actions will run tests and fail
   - Sentinel will receive the failure notification
   - Sentinel will analyze the bugs
   - Sentinel will create pull requests with fixes
   - You can merge the fixes to see tests pass!

## 🔍 **What You'll See**

### **1. Test Failures**
When you run the tests, you'll see failures like:
```
FAILED test_buggy_math.py::test_calculate_total - AssertionError: Expected 60, got -60
FAILED test_buggy_math.py::test_find_max - AssertionError: Expected 9, got 1
FAILED test_data_processor.py::test_data_processor_length - AssertionError: Expected 5, got 6
```

### **2. GitHub Actions**
- CI workflow will run on every push
- Tests will fail and trigger Sentinel notification
- You'll see the workflow status in the Actions tab

### **3. Sentinel Dashboard**
Visit `http://localhost:8000/dashboard` to see:
- Recent failures
- Created pull requests
- System activity
- Time saved (simulated)

### **4. Automatic Fixes**
Sentinel will:
- Detect the specific bugs
- Generate fixes for each bug
- Create pull requests with the fixes
- Send notifications about the fixes

## 🐛 **Bugs in the Code**

### **Math Operations (`buggy_math.py`)**
1. **calculate_total()** - Uses subtraction instead of addition
2. **divide_numbers()** - No zero division check
3. **find_max()** - Uses < instead of >
4. **calculate_average()** - Divides by len + 1 instead of len
5. **is_even()** - Uses % 2 == 1 instead of % 2 == 0

### **Data Processor (`buggy_data_processor.py`)**
1. **get_length()** - Returns len + 1 instead of len
2. **get_sum()** - Subtracts 1 from sum
3. **get_average()** - Divides by len - 1 instead of len
4. **find_max()** - Uses < instead of >
5. **filter_positive()** - Filters < 0 instead of > 0

## 🎉 **Expected Results**

After setup, you should see:

1. **Repository populated** with buggy code
2. **Tests failing** when run
3. **GitHub Actions failing** on push
4. **Sentinel receiving** failure notifications
5. **Pull requests created** with fixes
6. **Tests passing** after merging fixes

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **GitHub Token Issues**
   - Make sure token has `repo` permissions
   - Check token is set in environment variables

2. **Webhook Issues**
   - Verify ngrok URL is correct
   - Check webhook is receiving events

3. **Sentinel Not Starting**
   - Check if port 8000 is available
   - Verify all dependencies are installed

4. **Tests Not Failing**
   - Make sure you're running the buggy code
   - Check test files are in the repository

## 📊 **Monitoring**

- **Dashboard**: `http://localhost:8000/dashboard`
- **Health Check**: `http://localhost:8000/health`
- **GitHub Actions**: Repository Actions tab
- **Pull Requests**: Repository PRs tab

The system is now ready to automatically detect and fix bugs in your repository! 🚀
