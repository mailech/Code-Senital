#!/usr/bin/env python3
"""
Quick script to start Sentinel system for the mailech/Errors repository demo
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_sentinel():
    print("🚀 Starting Self-Healing Codebase Sentinel...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("app/main.py"):
        print("❌ Error: Please run this from the code-cubicle directory")
        return
    
    print("📡 Starting FastAPI server on port 8000...")
    print("🔧 Starting background worker...")
    print("=" * 60)
    print("🌐 Sentinel Dashboard: http://localhost:8000/dashboard")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 60)
    print("📋 Next steps:")
    print("1. Wait for both services to start")
    print("2. Go to your GitHub repository: https://github.com/mailech/Errors")
    print("3. Make a small commit to trigger the CI pipeline")
    print("4. Watch Sentinel detect and fix the bugs!")
    print("=" * 60)
    print("Press Ctrl+C to stop both services")
    print("=" * 60)
    
    try:
        # Start the FastAPI server
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Start the worker
        worker_process = subprocess.Popen([
            sys.executable, "app/worker.py"
        ])
        
        print("✅ Both services started successfully!")
        print("🔄 Worker is now monitoring for events...")
        
        # Wait for both processes
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping Sentinel services...")
            server_process.terminate()
            worker_process.terminate()
            print("✅ Services stopped")
            
    except Exception as e:
        print(f"❌ Error starting services: {e}")

if __name__ == "__main__":
    start_sentinel()
