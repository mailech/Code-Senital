#!/usr/bin/env python3
"""
Test script to trigger Sentinel by simulating a CI failure
"""

import requests
import json
import time

def trigger_sentinel():
    """Simulate a CI failure to trigger Sentinel"""
    print("🚀 Triggering Sentinel Test...")
    print("=" * 40)
    
    # Simulate the webhook payload that GitHub would send
    payload = {
        "repo": "mailech/Errors",
        "failing_test": "pytest",
        "logs": "Test failures detected in buggy_math.py and buggy_data_processor.py",
        "diff": "CI failure due to intentional bugs"
    }
    
    # Send to Sentinel webhook
    try:
        response = requests.post(
            "http://localhost:8000/webhooks/ci/failure",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Sentinel webhook triggered successfully!")
            print("🔍 Check the worker logs for processing...")
            print("📋 Sentinel should now:")
            print("   1. Analyze the failure")
            print("   2. Generate fixes")
            print("   3. Create a Pull Request")
            print("   4. Fix the bugs automatically")
        else:
            print(f"❌ Webhook failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending webhook: {e}")
        print("   Make sure Sentinel server is running on port 8000")

def check_sentinel_status():
    """Check if Sentinel is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Sentinel server is running")
            return True
        else:
            print("❌ Sentinel server not responding")
            return False
    except requests.exceptions.RequestException:
        print("❌ Sentinel server not running")
        return False

def main():
    print("🤖 Sentinel Trigger Test")
    print("=" * 40)
    
    # Check if Sentinel is running
    if not check_sentinel_status():
        print("\n🔧 Please start Sentinel first:")
        print("   python start_sentinel_simple.py")
        return
    
    print("\n🚀 Triggering Sentinel...")
    trigger_sentinel()
    
    print("\n📊 Expected Results:")
    print("   • Worker should process the event")
    print("   • AI should generate patches")
    print("   • GitHub PR should be created")
    print("   • Bugs should be fixed automatically")
    
    print("\n🔍 Monitor the process:")
    print("   • Check worker logs for progress")
    print("   • Check GitHub for new Pull Request")
    print("   • Check dashboard: http://localhost:8000/dashboard")

if __name__ == "__main__":
    main()
