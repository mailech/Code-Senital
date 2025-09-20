#!/usr/bin/env python3
"""
Test if Sentinel is configured for Error-List repository
"""

import os
import sys
import requests

def test_error_list_config():
    """Test Sentinel configuration for Error-List"""
    
    print("🧪 TESTING SENTINEL CONFIGURATION FOR ERROR-LIST")
    print("=" * 60)
    
    # Set environment variables
    os.environ["GITHUB_OWNER"] = "mailech"
    os.environ["GITHUB_REPO"] = "Error-List"
    
    # Test config loading
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
    
    try:
        from config import settings
        print(f"✅ Configuration loaded successfully")
        print(f"🎯 Target Repository: {settings.github_owner}/{settings.github_repo}")
        print(f"🌿 Default Branch: {settings.default_branch}")
        
        # Test API connection
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Sentinel API is running")
                
                # Test if we can create a test PR
                print("\n🔧 Testing PR creation capability...")
                test_branch = "sentinel/test-config"
                test_title = "Test: Sentinel Configuration for Error-List"
                test_body = "This is a test PR to verify Sentinel is configured for Error-List repository"
                
                print(f"   Would create branch: {test_branch}")
                print(f"   Would create PR: {test_title}")
                print(f"   Target repo: {settings.github_owner}/{settings.github_repo}")
                
                return True
            else:
                print(f"❌ Sentinel API returned: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to Sentinel API: {e}")
            print("   Make sure Sentinel is running on port 8000")
            return False
            
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

if __name__ == "__main__":
    success = test_error_list_config()
    if success:
        print("\n🎉 SENTINEL IS CONFIGURED FOR ERROR-LIST!")
        print("   Next: Push some code to trigger CI failures")
        print("   Sentinel will create PRs in mailech/Error-List")
    else:
        print("\n❌ CONFIGURATION TEST FAILED!")
        print("   Check the errors above and fix them")
