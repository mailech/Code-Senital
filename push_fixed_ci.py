#!/usr/bin/env python3
"""
Push the fixed CI workflow to the repository
"""

import os
import requests
import json

def push_file_to_github(filename, content, token, repo="mailech/Errors"):
    """Push a file to GitHub repository"""
    
    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Get current file info
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        current_file = response.json()
        sha = current_file["sha"]
        print(f"📄 Updating existing file: {filename}")
    else:
        sha = None
        print(f"📄 Creating new file: {filename}")
    
    # Prepare the file content
    data = {
        "message": f"Fix CI workflow to properly fail tests",
        "content": content,
        "branch": "main"
    }
    
    if sha:
        data["sha"] = sha
    
    # Push the file
    response = requests.put(url, headers=headers, json=data)
    
    if response.status_code in [200, 201]:
        print(f"✅ Successfully pushed {filename}")
        return True
    else:
        print(f"❌ Failed to push {filename}: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

def main():
    print("🔧 Pushing Fixed CI Workflow to mailech/Errors")
    print("=" * 50)
    
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    # Read the fixed CI workflow
    with open(".github/workflows/ci.yml", "r", encoding="utf-8") as f:
        ci_content = f.read()
    
    # Encode to base64
    import base64
    ci_content_b64 = base64.b64encode(ci_content.encode()).decode()
    
    # Push the file
    success = push_file_to_github(".github/workflows/ci.yml", ci_content_b64, token)
    
    if success:
        print("\n🎉 CI workflow updated successfully!")
        print("📋 Next steps:")
        print("1. Start Sentinel: python start_sentinel_simple.py")
        print("2. Make a commit to trigger the failing CI")
        print("3. Watch Sentinel detect and fix the bugs!")
    else:
        print("\n❌ Failed to update CI workflow")

if __name__ == "__main__":
    main()
