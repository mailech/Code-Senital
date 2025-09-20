# Self-Healing Codebase Sentinel - Setup Instructions

## Overview
This is a self-healing codebase system that automatically detects and fixes bugs in your code. The system consists of:

1. **FastAPI Application** - Main web service with webhook endpoints
2. **Background Worker** - Processes events and generates fixes
3. **Demo Files** - Contains intentional bugs for testing
4. **Dashboard** - Web interface to monitor the system

## What We've Fixed

### 1. Dependencies
- ✅ Installed all required Python packages from `requirements.txt`
- ✅ All dependencies are properly configured

### 2. Bug Fixes
- ✅ Fixed the math operations bug in `demo_repo/app.py` (was returning subtraction instead of addition)
- ✅ Fixed the dataloader bug in `Cubic-Err/ml_errors/dataloader.py` (was returning wrong length)

### 3. Environment Configuration
- ✅ Created `.env` file with all necessary environment variables
- ✅ Configured development settings

## How to Run the Application

### Method 1: Using the Startup Script
```bash
python start_app.py
```

### Method 2: Manual Startup
1. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **In a separate terminal, start the worker:**
   ```bash
   python app/worker.py
   ```

### Method 3: Using Docker
```bash
docker-compose up
```

## Testing the Application

### 1. Test the Math Operations
```bash
python -c "from app_demo.math_ops import add; print(f'2 + 2 = {add(2, 2)}')"
```

### 2. Test the Dataloader
```bash
python -c "from Cubic_Err.ml_errors.dataloader import ToyDataset; ds = ToyDataset([1,2,3]); print(f'Dataset length: {len(ds)}')"
```

### 3. Test the API Endpoints
Once the server is running, you can test:

- **Health Check:** `http://localhost:8000/health`
- **Root Endpoint:** `http://localhost:8000/`
- **Dashboard:** `http://localhost:8000/dashboard`

### 4. Test Webhooks
```bash
python scripts/simulate_failure.py
```

## Available Endpoints

### Main Application
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /dashboard` - Web dashboard

### Webhooks
- `POST /webhooks/github` - GitHub webhook
- `POST /webhooks/ci/failure` - CI failure webhook

## Project Structure

```
code-cubicle/
├── app/                    # Main FastAPI application
│   ├── main.py            # FastAPI app entry point
│   ├── config.py          # Configuration settings
│   ├── db.py              # Database operations
│   ├── worker.py          # Background worker
│   ├── routes/            # API routes
│   └── services/          # Business logic services
├── app_demo/              # Demo math operations
├── Cubic-Err/             # ML errors demo
├── demo_repo/             # Demo repository
├── scripts/               # Utility scripts
├── tests/                 # Test files
└── requirements.txt       # Python dependencies
```

## Key Features

1. **Automatic Bug Detection** - Monitors CI failures and code changes
2. **AI-Powered Fixes** - Generates patches for detected issues
3. **GitHub Integration** - Creates pull requests automatically
4. **Slack Notifications** - Sends alerts about fixes
5. **Web Dashboard** - Monitor system activity
6. **Webhook Support** - Integrates with CI/CD pipelines

## Configuration

The application uses environment variables for configuration. Key settings in `.env`:

- `ENVIRONMENT` - Development/production mode
- `GITHUB_TOKEN` - GitHub API token (optional)
- `SLACK_BOT_TOKEN` - Slack bot token (optional)
- `CONFIDENCE_THRESHOLD` - AI confidence threshold for auto-fixes

## Troubleshooting

1. **Import Errors:** Make sure you're in the project root directory
2. **Port Already in Use:** Change the port in the startup command
3. **Database Issues:** The SQLite database will be created automatically
4. **Missing Dependencies:** Run `pip install -r requirements.txt`

## Next Steps

1. Start the application using one of the methods above
2. Open `http://localhost:8000/dashboard` in your browser
3. Test the webhook endpoints
4. Simulate failures using the provided scripts
5. Monitor the system's automatic bug-fixing capabilities

The system is now ready to run and will automatically detect and fix bugs in your codebase!
