# New Repository Setup Guide

## 🚀 Complete Setup for Your New Repository

### Step 1: Configure Sentinel for Your Repository

Run the setup script to configure Sentinel for your new repository:

```bash
python setup_new_repo.py
```

This will:
- Create a `.env` file with your repository details
- Set up GitHub Actions workflow
- Create sample buggy code for testing
- Configure all necessary settings

### Step 2: Start the Complete System

```bash
python start_sentinel_complete.py
```

This will start both:
- FastAPI server on `http://localhost:8000`
- Background worker for processing events

### Step 3: Test the Full Functionality

```bash
python demo_full_functionality.py
```

This will demonstrate:
- Health checks
- Dashboard access
- CI failure simulation
- GitHub webhook simulation
- Math operations testing
- Dataloader testing

## 🔧 Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Configure Environment

Create `.env` file:
```env
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
GITHUB_OWNER=your-username
GITHUB_REPO=your-repo-name
GITHUB_TOKEN=your-github-token
SLACK_BOT_TOKEN=your-slack-token
CONFIDENCE_THRESHOLD=0.8
```

### 2. Start Server

Terminal 1:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:
```bash
python app/worker.py
```

### 3. Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Dashboard
open http://localhost:8000/dashboard

# Simulate CI failure
curl -X POST http://localhost:8000/webhooks/ci/failure \
  -H "Content-Type: application/json" \
  -d '{"repo": "your-repo", "failing_test": "test_example", "logs": "Test failed", "diff": "Fix example"}'
```

## 🎯 What You'll See

### 1. Dashboard
Visit `http://localhost:8000/dashboard` to see:
- Recent failures
- Created pull requests
- System statistics
- Time saved (simulated)

### 2. Health Endpoint
Visit `http://localhost:8000/health` to see:
- Application status
- Configuration details
- Environment info

### 3. Webhook Processing
The system will:
- Accept CI failure notifications
- Process GitHub webhooks
- Generate AI-powered fixes
- Create pull requests
- Send Slack notifications

## 🔍 Testing Scenarios

### Scenario 1: Math Operations Bug
```python
# This will trigger a CI failure
def add(a, b):
    return a - b  # BUG: subtraction instead of addition

# Test
assert add(2, 2) == 4  # This will fail
```

### Scenario 2: ML Dataloader Bug
```python
# This will trigger a CI failure
class Dataset:
    def __len__(self):
        return len(self.data) + 1  # BUG: off-by-one error

# Test
ds = Dataset([1, 2, 3])
assert len(ds) == 3  # This will fail
```

### Scenario 3: GitHub Integration
- Push code to your repository
- GitHub Actions will run tests
- On failure, it will notify Sentinel
- Sentinel will analyze and create fixes

## 📊 Monitoring

### Real-time Monitoring
- Dashboard shows live updates
- Database stores all events
- Logs show detailed processing

### Event Types
- CI failures
- GitHub webhooks
- Generated fixes
- Created pull requests
- Slack notifications

## 🛠️ Customization

### AI Engine
Modify `app/services/ai_engine.py` to:
- Use different AI models
- Adjust confidence thresholds
- Customize patch generation

### Action Engine
Modify `app/services/action_engine.py` to:
- Change PR templates
- Customize notifications
- Add new integrations

### Webhooks
Modify `app/routes/webhooks.py` to:
- Add new webhook types
- Customize processing logic
- Add validation rules

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project root
   cd /path/to/code-cubicle
   python -c "from app.main import app"
   ```

2. **Port Already in Use**
   ```bash
   # Change port in .env file
   PORT=8001
   ```

3. **Database Issues**
   ```bash
   # Database will be created automatically
   # Check data/sentinel.sqlite3
   ```

4. **Worker Not Starting**
   ```bash
   # Check Python path
   export PYTHONPATH=$PWD:$PYTHONPATH
   python app/worker.py
   ```

### Debug Mode

Enable debug logging:
```python
# In app/logging_setup.py
logging.basicConfig(level=logging.DEBUG)
```

## 🎉 Success!

Once everything is running, you'll have:

✅ **Self-Healing System** - Automatically detects and fixes bugs
✅ **GitHub Integration** - Creates pull requests for fixes
✅ **Slack Notifications** - Sends alerts about fixes
✅ **Web Dashboard** - Monitor system activity
✅ **CI/CD Integration** - Works with your existing pipelines
✅ **AI-Powered Fixes** - Intelligent bug detection and resolution

The system is now ready to automatically detect and fix bugs in your codebase!
