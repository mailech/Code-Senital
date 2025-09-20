#!/usr/bin/env python3
"""
Push buggy code files to mailech/Errors repository
"""

import asyncio
import base64
import httpx
import os
from pathlib import Path

# Repository configuration
REPO_OWNER = "mailech"
REPO_NAME = "Errors"
API_BASE = "https://api.github.com"

# Files to push to the repository
FILES_TO_PUSH = [
    "buggy_math.py",
    "test_buggy_math.py", 
    "buggy_data_processor.py",
    "test_data_processor.py",
    "requirements.txt",
    "README.md",
    ".github/workflows/ci.yml"
]

async def push_file_to_repo(client, file_path, content, message):
    """Push a single file to the repository"""
    try:
        # Encode content to base64
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        # Check if file exists
        try:
            existing_file = await client.get(f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}")
            if existing_file.status_code == 200:
                file_data = existing_file.json()
                sha = file_data.get('sha')
                print(f"📝 Updating existing file: {file_path}")
            else:
                sha = None
                print(f"📄 Creating new file: {file_path}")
        except:
            sha = None
            print(f"📄 Creating new file: {file_path}")
        
        # Prepare payload
        payload = {
            "message": message,
            "content": encoded_content,
            "branch": "main"
        }
        
        if sha:
            payload["sha"] = sha
        
        # Push file
        response = await client.put(
            f"{API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}",
            json=payload
        )
        
        if response.status_code in [200, 201]:
            print(f"✅ Successfully pushed: {file_path}")
            return True
        else:
            print(f"❌ Failed to push {file_path}: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error pushing {file_path}: {e}")
        return False

async def main():
    """Main function to push all files"""
    print("=" * 60)
    print("Pushing Buggy Code to mailech/Errors Repository")
    print("=" * 60)
    
    # Check if files exist locally
    missing_files = []
    for file_path in FILES_TO_PUSH:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return
    
    # Get GitHub token from environment
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        print("   export GITHUB_TOKEN=your_token_here")
        return
    
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json"
    }
    
    async with httpx.AsyncClient(headers=headers) as client:
        success_count = 0
        total_files = len(FILES_TO_PUSH)
        
        for file_path in FILES_TO_PUSH:
            print(f"\n📁 Processing: {file_path}")
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
                continue
            
            # Push file
            success = await push_file_to_repo(
                client, 
                file_path, 
                content, 
                f"Add {file_path} with intentional bugs for Sentinel demo"
            )
            
            if success:
                success_count += 1
        
        print("\n" + "=" * 60)
        print(f"Push Complete: {success_count}/{total_files} files pushed successfully")
        print("=" * 60)
        
        if success_count == total_files:
            print("🎉 All files pushed successfully!")
            print(f"\nNext steps:")
            print(f"1. Visit: https://github.com/{REPO_OWNER}/{REPO_NAME}")
            print(f"2. Check the files are there")
            print(f"3. Run tests to see failures: python -m pytest test_*.py -v")
            print(f"4. Set up Sentinel webhook URL in GitHub secrets")
            print(f"5. Watch Sentinel automatically fix the bugs!")
        else:
            print("⚠️  Some files failed to push. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
