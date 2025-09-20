#!/usr/bin/env python3
"""
Test Sentinel with Error-List repository
"""

import os
import sys
import subprocess
import time
import requests

def test_sentinel_error_list():
    """Test Sentinel configured for Error-List repository"""
    
    print("🧪 TESTING SENTINEL FOR ERROR-LIST REPOSITORY")
    print("=" * 60)
    
    # Set environment variables
    os.environ["GITHUB_OWNER"] = "mailech"
    os.environ["GITHUB_REPO"] = "Error-List"
    os.environ["DEFAULT_BRANCH"] = "main"
    
    print(f"🎯 Target Repository: {os.environ['GITHUB_OWNER']}/{os.environ['GITHUB_REPO']}")
    
    # Test configuration loading
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
    
    try:
        from config import settings
        print(f"✅ Configuration loaded")
        print(f"   Repository: {settings.github_owner}/{settings.github_repo}")
        print(f"   Branch: {settings.default_branch}")
        
        # Start Sentinel in background
        print("\n🚀 Starting Sentinel...")
        process = subprocess.Popen([
            sys.executable, "start_sentinel_simple.py"
        ], env=os.environ.copy())
        
        # Wait for startup
        print("⏳ Waiting for Sentinel to start...")
        time.sleep(10)
        
        # Test API
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Sentinel API is running")
                
                # Test webhook endpoint
                test_payload = {
                    "repo": "mailech/Error-List",
                    "failing_test": "pytest",
                    "logs": "Test failure detected",
                    "diff": "CI failure"
                }
                
                webhook_response = requests.post(
                    "http://localhost:8000/webhooks/ci/failure",
                    json=test_payload,
                    timeout=5
                )
                
                if webhook_response.status_code == 200:
                    print("✅ Webhook endpoint working")
                    print("🎯 Sentinel is ready for Error-List repository!")
                    print("\n📋 NEXT STEPS:")
                    print("1. Push buggy code to mailech/Error-List")
                    print("2. Watch Sentinel create fix PRs")
                    print("3. Check: https://github.com/mailech/Error-List/pulls")
                    
                    return True
                else:
                    print(f"❌ Webhook test failed: {webhook_response.status_code}")
                    return False
            else:
                print(f"❌ Sentinel API failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to Sentinel: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    finally:
        # Clean up
        try:
            process.terminate()
        except:
            pass

if __name__ == "__main__":
    success = test_sentinel_error_list()
    if success:
        print("\n🎉 SENTINEL IS READY FOR ERROR-LIST!")
    else:
        print("\n❌ TEST FAILED!")
        print("   Check the errors above")
