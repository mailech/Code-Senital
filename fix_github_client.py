#!/usr/bin/env python3
"""
Fix the GitHub client to work with the actual repository files
"""

import os
import requests
import base64

def get_repo_files(repo="mailech/Errors", token=None):
    """Get the actual files in the repository"""
    if not token:
        token = os.getenv("GITHUB_TOKEN")
    
    if not token:
        print("❌ Please set GITHUB_TOKEN")
        return []
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    url = f"https://api.github.com/repos/{repo}/contents"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        files = []
        for item in response.json():
            if item["type"] == "file":
                files.append(item["name"])
            elif item["type"] == "dir":
                # Get files in subdirectories
                sub_url = f"https://api.github.com/repos/{repo}/contents/{item['name']}"
                sub_response = requests.get(sub_url, headers=headers)
                if sub_response.status_code == 200:
                    for sub_item in sub_response.json():
                        if sub_item["type"] == "file":
                            files.append(f"{item['name']}/{sub_item['name']}")
        
        return files
    except Exception as e:
        print(f"❌ Error getting repo files: {e}")
        return []

def update_github_client():
    """Update the GitHub client with the correct file paths"""
    print("🔍 Getting actual repository files...")
    files = get_repo_files()
    
    if not files:
        print("❌ Could not get repository files")
        return
    
    print("📁 Files found in repository:")
    for file in files:
        print(f"   • {file}")
    
    # Update the GitHub client to use the correct file paths
    github_client_path = "app/services/github_client.py"
    
    # Read the current file
    with open(github_client_path, 'r') as f:
        content = f.read()
    
    # Find the candidate_paths section and update it
    if "candidate_paths = [" in content:
        # Create new candidate paths based on actual files
        candidate_paths = []
        for file in files:
            if file.endswith('.py'):
                candidate_paths.append(f'"{file}"')
        
        # Update the content
        import re
        pattern = r'candidate_paths = \[.*?\]'
        replacement = f'candidate_paths = [{", ".join(candidate_paths)}]'
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Write the updated content
        with open(github_client_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Updated GitHub client with correct file paths")
    else:
        print("❌ Could not find candidate_paths in GitHub client")

def main():
    print("🔧 Fixing GitHub Client for Sentinel")
    print("=" * 40)
    
    # Set the token
    token = "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"
    os.environ["GITHUB_TOKEN"] = token
    
    update_github_client()
    
    print("\n🎯 Next steps:")
    print("1. Restart the Sentinel worker")
    print("2. Make a commit to trigger the CI")
    print("3. Watch Sentinel create the PR successfully!")

if __name__ == "__main__":
    main()
