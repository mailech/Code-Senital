#!/usr/bin/env python3
"""
Setup script for new repository integration with Self-Healing Codebase Sentinel
"""

import os
import sys
import json
import asyncio
import httpx
from pathlib import Path

def create_env_file(repo_owner, repo_name, github_token=None, slack_token=None):
    """Create .env file with repository configuration"""
    env_content = f"""# Environment Configuration
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000

# GitHub Configuration
GITHUB_TOKEN={github_token or ''}
GITHUB_OWNER={repo_owner}
GITHUB_REPO={repo_name}
DEFAULT_BRANCH=main

# Slack Configuration (optional)
SLACK_BOT_TOKEN={slack_token or ''}
SLACK_CHANNEL=#general

# AI Configuration (optional)
OPENAI_API_KEY=
HF_API_TOKEN=

# Security
SENTINEL_WEBHOOK_SECRET=your-secret-key-here

# Repository Access Control
ALLOWED_REPOS={repo_owner}/{repo_name}

# AI Confidence Threshold
CONFIDENCE_THRESHOLD=0.8
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created .env file for repository: {repo_owner}/{repo_name}")

def create_github_workflow(repo_owner, repo_name):
    """Create GitHub Actions workflow for CI integration"""
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = f"""name: CI with Sentinel Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          python -m pytest tests/ -v
        continue-on-error: true
      
      - name: Notify Sentinel on failure
        if: failure()
        run: |
          curl -X POST http://localhost:8000/webhooks/ci/failure \\
            -H "Content-Type: application/json" \\
            -d '{{"repo": "{repo_owner}/{repo_name}", "failing_test": "pytest", "logs": "Test failure detected", "diff": "CI failure"}}'
        continue-on-error: true

  notify-sentinel:
    needs: test
    if: failure()
    runs-on: ubuntu-latest
    steps:
      - name: Notify Sentinel of CI failure
        uses: actions/github-script@v7
        with:
          script: |
            const payload = {{
              repo: process.env.GITHUB_REPOSITORY,
              failing_test: 'tests/',
              logs: 'CI failure detected',
              diff: 'Automated CI failure notification'
            }};
            
            // You can replace this with your actual Sentinel URL
            const sentinelUrl = 'http://your-sentinel-url.com/webhooks/ci/failure';
            
            try {{
              const response = await fetch(sentinelUrl, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(payload)
              }});
              console.log(`Sentinel notification sent: ${{response.status}}`);
            }} catch (error) {{
              console.log('Sentinel notification failed:', error.message);
            }}
"""
    
    workflow_file = workflow_dir / "ci.yml"
    with open(workflow_file, 'w') as f:
        f.write(workflow_content)
    
    print(f"✅ Created GitHub Actions workflow: {workflow_file}")

def create_sample_buggy_code():
    """Create sample files with intentional bugs for testing"""
    
    # Create a sample Python file with bugs
    sample_code = """def calculate_total(items):
    total = 0
    for item in items:
        # BUG: Should be addition, not subtraction
        total = total - item['price']
    return total

def divide_numbers(a, b):
    # BUG: No zero division check
    return a / b

def find_max(numbers):
    if not numbers:
        return None
    
    max_val = numbers[0]
    for num in numbers:
        # BUG: Should be > not <
        if num < max_val:
            max_val = num
    return max_val
"""
    
    os.makedirs("sample_code", exist_ok=True)
    with open("sample_code/buggy_math.py", "w") as f:
        f.write(sample_code)
    
    # Create corresponding tests
    test_code = """import pytest
from sample_code.buggy_math import calculate_total, divide_numbers, find_max

def test_calculate_total():
    items = [{"price": 10}, {"price": 20}, {"price": 30}]
    result = calculate_total(items)
    assert result == 60  # This will fail due to subtraction bug

def test_divide_numbers():
    result = divide_numbers(10, 2)
    assert result == 5
    
    # This will fail due to zero division
    with pytest.raises(ZeroDivisionError):
        divide_numbers(10, 0)

def test_find_max():
    numbers = [1, 5, 3, 9, 2]
    result = find_max(numbers)
    assert result == 9  # This will fail due to < instead of >
"""
    
    os.makedirs("tests", exist_ok=True)
    with open("tests/test_buggy_math.py", "w") as f:
        f.write(test_code)
    
    print("✅ Created sample buggy code and tests")

def test_sentinel_endpoints():
    """Test the Sentinel endpoints"""
    print("Testing Sentinel endpoints...")
    
    try:
        # Test health endpoint
        response = httpx.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
        
        # Test CI failure webhook
        payload = {
            "repo": "test-repo",
            "failing_test": "tests/test_buggy_math.py::test_calculate_total",
            "logs": "AssertionError: Expected 60, got -60",
            "diff": "Fixed subtraction bug in calculate_total function"
        }
        
        response = httpx.post("http://localhost:8000/webhooks/ci/failure", 
                            json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ CI failure webhook working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ CI failure webhook failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Endpoint testing failed: {e}")

def create_requirements():
    """Create requirements.txt for the sample project"""
    requirements = """pytest==8.3.2
fastapi==0.115.0
uvicorn[standard]==0.30.6
httpx==0.27.2
slack_sdk==3.31.0
pydantic==2.9.2
structlog==24.4.0
python-dotenv==1.0.1
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def main():
    """Main setup function"""
    print("=" * 60)
    print("Self-Healing Codebase Sentinel - New Repository Setup")
    print("=" * 60)
    
    # Get repository information
    repo_owner = input("Enter GitHub repository owner: ").strip()
    repo_name = input("Enter GitHub repository name: ").strip()
    github_token = input("Enter GitHub token (optional, press Enter to skip): ").strip()
    slack_token = input("Enter Slack token (optional, press Enter to skip): ").strip()
    
    if not repo_owner or not repo_name:
        print("❌ Repository owner and name are required!")
        return
    
    print(f"\nSetting up Sentinel for repository: {repo_owner}/{repo_name}")
    
    # Create configuration
    create_env_file(repo_owner, repo_name, github_token, slack_token)
    create_github_workflow(repo_owner, repo_name)
    create_sample_buggy_code()
    create_requirements()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Start the Sentinel server:")
    print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print("\n2. In another terminal, start the worker:")
    print("   python app/worker.py")
    print("\n3. Test the system:")
    print("   python -m pytest tests/test_buggy_math.py -v")
    print("\n4. View the dashboard:")
    print("   http://localhost:8000/dashboard")
    print("\n5. Push your code to GitHub to trigger the CI workflow")
    
    # Test endpoints if server is running
    print("\nTesting Sentinel endpoints...")
    test_sentinel_endpoints()

if __name__ == "__main__":
    main()
