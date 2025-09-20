#!/usr/bin/env python3
"""
Fixed script to start Sentinel system - handles port conflicts
"""

import subprocess
import sys
import time
import os
import signal
import psutil
from pathlib import Path

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    print(f"🛑 Killing process {proc.info['pid']} using port {port}")
                    proc.kill()
                    time.sleep(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
            pass

def start_sentinel():
    print("🚀 Starting Self-Healing Codebase Sentinel...")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("app/main.py"):
        print("❌ Error: Please run this from the code-cubicle directory")
        return
    
    # Kill any existing processes on port 8000
    print("🔍 Checking for existing processes on port 8000...")
    kill_process_on_port(8000)
    time.sleep(2)
    
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
        
        # Check if server started successfully
        if server_process.poll() is not None:
            print("❌ Server failed to start. Check the error above.")
            return
        
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
