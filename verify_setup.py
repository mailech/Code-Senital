#!/usr/bin/env python3
"""
Verify that Sentinel is properly configured for the mailech/Errors repository
"""

import requests
import time
import os

def verify_sentinel():
    print("🔍 Verifying Sentinel Setup...")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Sentinel server is running")
        else:
            print("❌ Sentinel server is not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ Sentinel server is not running")
        print("   Please start it with: python start_sentinel_now.py")
        return
    
    # Check dashboard
    try:
        response = requests.get("http://localhost:8000/dashboard", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is accessible")
        else:
            print("❌ Dashboard is not accessible")
    except requests.exceptions.RequestException:
        print("❌ Dashboard is not accessible")
    
    print("\n📋 GitHub Webhook Configuration:")
    print("=" * 50)
    print("1. Go to: https://github.com/mailech/Errors/settings/hooks")
    print("2. Click 'Add webhook' or edit existing webhook")
    print("3. Set these values:")
    print("   📍 Payload URL: https://your-ngrok-url.ngrok.io/webhooks/github")
    print("   🔐 Secret: your_webhook_secret")
    print("   📝 Content type: application/json")
    print("   🎯 Events: Just the push event")
    print("   ✅ Active: Yes")
    
    print("\n🚀 Next Steps:")
    print("=" * 50)
    print("1. Make sure ngrok is running and copy the URL")
    print("2. Update the webhook URL in GitHub")
    print("3. Go to your repository: https://github.com/mailech/Errors")
    print("4. Make a small commit to trigger the pipeline")
    print("5. Watch Sentinel detect and fix the bugs!")
    
    print("\n🔧 To trigger the demo:")
    print("=" * 50)
    print("1. Go to: https://github.com/mailech/Errors")
    print("2. Edit any file (like README.md)")
    print("3. Add a line like: 'Testing Sentinel'")
    print("4. Commit the changes")
    print("5. Watch the magic happen! ✨")

if __name__ == "__main__":
    verify_sentinel()
