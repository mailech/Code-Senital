#!/usr/bin/env python3
"""
Complete startup script for Self-Healing Codebase Sentinel
Starts both the FastAPI server and background worker
"""

import asyncio
import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path

class SentinelManager:
    def __init__(self):
        self.server_process = None
        self.worker_process = None
        self.running = False
    
    def start_server(self):
        """Start the FastAPI server"""
        print("🚀 Starting FastAPI server...")
        try:
            self.server_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "app.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000", 
                "--reload"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ FastAPI server started on http://localhost:8000")
            return True
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def start_worker(self):
        """Start the background worker"""
        print("🔄 Starting background worker...")
        try:
            self.worker_process = subprocess.Popen([
                sys.executable, "app/worker.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✅ Background worker started")
            return True
        except Exception as e:
            print(f"❌ Failed to start worker: {e}")
            return False
    
    def check_health(self):
        """Check if the server is healthy"""
        try:
            import httpx
            response = httpx.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_server(self, timeout=30):
        """Wait for the server to be ready"""
        print("⏳ Waiting for server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.check_health():
                print("✅ Server is ready!")
                return True
            time.sleep(1)
        
        print("❌ Server failed to start within timeout")
        return False
    
    def start(self):
        """Start both server and worker"""
        print("=" * 60)
        print("Self-Healing Codebase Sentinel - Complete Startup")
        print("=" * 60)
        
        # Check if we're in the right directory
        if not Path("app/main.py").exists():
            print("❌ Please run this script from the project root directory")
            return False
        
        # Start server
        if not self.start_server():
            return False
        
        # Wait for server to be ready
        if not self.wait_for_server():
            self.stop()
            return False
        
        # Start worker
        if not self.start_worker():
            self.stop()
            return False
        
        self.running = True
        print("\n" + "=" * 60)
        print("🎉 Sentinel is now running!")
        print("=" * 60)
        print("\nAvailable endpoints:")
        print("  • Dashboard: http://localhost:8000/dashboard")
        print("  • Health: http://localhost:8000/health")
        print("  • Root: http://localhost:8000/")
        print("\nWebhook endpoints:")
        print("  • CI Failure: POST http://localhost:8000/webhooks/ci/failure")
        print("  • GitHub: POST http://localhost:8000/webhooks/github")
        print("\nPress Ctrl+C to stop the system")
        
        return True
    
    def stop(self):
        """Stop both server and worker"""
        print("\n🛑 Stopping Sentinel system...")
        self.running = False
        
        if self.worker_process:
            self.worker_process.terminate()
            print("✅ Worker stopped")
        
        if self.server_process:
            self.server_process.terminate()
            print("✅ Server stopped")
        
        print("👋 Sentinel system stopped")
    
    def run(self):
        """Run the complete system"""
        if not self.start():
            return
        
        try:
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal...")
        finally:
            self.stop()

def main():
    """Main function"""
    # Set up signal handling
    def signal_handler(signum, frame):
        print("\n🛑 Received signal, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run the manager
    manager = SentinelManager()
    manager.run()

if __name__ == "__main__":
    main()
