#!/usr/bin/env python3
"""
Simple script to start Sentinel - fixes all the issues
"""

import subprocess
import sys
import time
import os
import signal
import threading

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

def start_worker():
    """Start the worker process"""
    print("🔧 Starting worker...")
    try:
        # Set PYTHONPATH to fix the import issue
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        subprocess.run([
            sys.executable, "app/worker.py"
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Worker stopped")
    except Exception as e:
        print(f"❌ Worker error: {e}")

def main():
    print("🚀 Starting Self-Healing Codebase Sentinel...")
    print("=" * 60)
    print("🌐 Dashboard: http://localhost:8000/dashboard")
    print("🔍 Health: http://localhost:8000/health")
    print("=" * 60)
    print("Press Ctrl+C to stop both services")
    print("=" * 60)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(3)
    
    # Start worker in main thread
    try:
        start_worker()
    except KeyboardInterrupt:
        print("\n🛑 Stopping Sentinel...")
        print("✅ Sentinel stopped")

if __name__ == "__main__":
    main()
