#!/usr/bin/env python3
"""
Start Sentinel targeting Error-List repository
"""

import os
import sys
import subprocess
import time

def start_sentinel_for_error_list():
    """Start Sentinel configured for Error-List repository"""
    
    print("🚀 STARTING SENTINEL FOR ERROR-LIST REPOSITORY")
    print("=" * 60)
    
    # Set environment variables for Error-List
    os.environ["GITHUB_OWNER"] = "mailech"
    os.environ["GITHUB_REPO"] = "Error-List"
    os.environ["DEFAULT_BRANCH"] = "main"
    
    print(f"🎯 Target Repository: {os.environ['GITHUB_OWNER']}/{os.environ['GITHUB_REPO']}")
    print(f"🌿 Default Branch: {os.environ['DEFAULT_BRANCH']}")
    
    # Kill any existing Python processes
    try:
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                      capture_output=True, check=False)
        print("✅ Killed existing Python processes")
    except:
        pass
    
    time.sleep(2)
    
    # Start Sentinel
    print("\n📡 Starting FastAPI server...")
    print("🔧 Starting background worker...")
    print("=" * 60)
    
    try:
        # Start the main Sentinel process
        subprocess.run([
            sys.executable, "start_sentinel_simple.py"
        ], env=os.environ.copy())
        
    except KeyboardInterrupt:
        print("\n🛑 Sentinel stopped by user")
    except Exception as e:
        print(f"❌ Error starting Sentinel: {e}")

if __name__ == "__main__":
    start_sentinel_for_error_list()
