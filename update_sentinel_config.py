#!/usr/bin/env python3
"""
Update Sentinel configuration to target Error-List repository
"""

import os
import sys

def update_sentinel_config():
    """Update environment variables to target Error-List repo"""
    
    print("🔧 UPDATING SENTINEL CONFIGURATION")
    print("=" * 50)
    
    # Set environment variables
    os.environ["GITHUB_OWNER"] = "mailech"
    os.environ["GITHUB_REPO"] = "Error-List"
    
    print(f"✅ GITHUB_OWNER: {os.environ['GITHUB_OWNER']}")
    print(f"✅ GITHUB_REPO: {os.environ['GITHUB_REPO']}")
    
    # Test the configuration
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
    
    try:
        from config import settings
        print(f"\n🎯 SENTINEL WILL NOW TARGET:")
        print(f"   Repository: {settings.github_owner}/{settings.github_repo}")
        print(f"   Branch: {settings.default_branch}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

if __name__ == "__main__":
    success = update_sentinel_config()
    if success:
        print("\n🚀 CONFIGURATION UPDATED!")
        print("   Sentinel will now create PRs in mailech/Error-List")
        print("   Restart Sentinel to apply changes")
    else:
        print("\n❌ CONFIGURATION FAILED!")
        sys.exit(1)
