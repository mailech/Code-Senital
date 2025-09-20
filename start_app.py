#!/usr/bin/env python3
"""
Startup script for the Self-Healing Codebase Sentinel
"""

import os
import sys
import subprocess
import time
import threading
import requests
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    try:
        import fastapi
        import uvicorn
        import httpx
        import slack_sdk
        import pydantic
        import structlog
        import pytest
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        env_content = """# Environment Configuration
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000

# GitHub Configuration (optional - for demo purposes)
GITHUB_TOKEN=
GITHUB_OWNER=
GITHUB_REPO=
DEFAULT_BRANCH=main

# Slack Configuration (optional - for notifications)
SLACK_BOT_TOKEN=
SLACK_CHANNEL=#general

# AI Configuration (optional - for future AI features)
OPENAI_API_KEY=
HF_API_TOKEN=

# Security
SENTINEL_WEBHOOK_SECRET=your-secret-key-here

# Repository Access Control
ALLOWED_REPOS=

# AI Confidence Threshold
CONFIDENCE_THRESHOLD=0.8
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")

def test_application():
    """Test the application components"""
    print("Testing application components...")
    
    # Test math ops
    try:
        from app_demo.math_ops import add
        result = add(2, 2)
        assert result == 4
        print("✅ Math operations working correctly")
    except Exception as e:
        print(f"❌ Math operations failed: {e}")
        return False
    
    # Test dataloader
    try:
        from Cubic_Err.ml_errors.dataloader import ToyDataset
        ds = ToyDataset([1, 2, 3])
        assert len(ds) == 3
        print("✅ Dataloader working correctly")
    except Exception as e:
        print(f"❌ Dataloader failed: {e}")
        return False
    
    # Test app import
    try:
        from app.main import app
        print("✅ FastAPI app can be imported")
    except Exception as e:
        print(f"❌ FastAPI app import failed: {e}")
        return False
    
    return True

def start_server():
    """Start the FastAPI server"""
    print("Starting FastAPI server...")
    try:
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def start_worker():
    """Start the background worker"""
    print("Starting background worker...")
    try:
        from app.worker import run_worker
        import asyncio
        asyncio.run(run_worker())
    except Exception as e:
        print(f"❌ Failed to start worker: {e}")

def test_endpoints():
    """Test the API endpoints"""
    print("Testing API endpoints...")
    time.sleep(3)  # Wait for server to start
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
        
        # Test root endpoint
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Endpoint testing failed: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("Self-Healing Codebase Sentinel Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies first")
        return
    
    # Create .env file
    create_env_file()
    
    # Test application
    if not test_application():
        print("Application tests failed. Please fix the issues first.")
        return
    
    print("\n" + "=" * 60)
    print("Starting the application...")
    print("=" * 60)
    
    # Start worker in background thread
    worker_thread = threading.Thread(target=start_worker, daemon=True)
    worker_thread.start()
    
    # Start server (this will block)
    try:
        start_server()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
