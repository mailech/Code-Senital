#!/usr/bin/env python3
"""
Check GitHub PRs directly
"""

import requests
import json

def check_github_prs():
    """Check PRs in the main Errors repository"""
    try:
        # Check main repository PRs
        url = "https://api.github.com/repos/mailech/Errors/pulls?state=open&per_page=20"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            prs = response.json()
            print(f"🎯 FOUND {len(prs)} OPEN PRs in mailech/Errors:")
            print("=" * 60)
            
            for i, pr in enumerate(prs[:10], 1):
                print(f"\n{i}. PR #{pr['number']}: {pr['title']}")
                print(f"   Created: {pr['created_at']}")
                print(f"   Author: {pr['user']['login']}")
                print(f"   Branch: {pr['head']['ref']}")
                print(f"   URL: {pr['html_url']}")
                
                # Check if it's a Sentinel PR
                if pr['head']['ref'].startswith('sentinel/fix-'):
                    print("   🤖 SENTINEL FIX PR!")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error checking PRs: {e}")

if __name__ == "__main__":
    check_github_prs()
