#!/usr/bin/env python3
"""
Create a .env file with Error-List configuration
"""

def create_env_file():
    """Create .env file with Error-List repository settings"""
    
    env_content = """# Self-Healing Codebase Sentinel Configuration
# Target Repository: mailech/Error-List

# GitHub Configuration
GITHUB_OWNER=mailech
GITHUB_REPO=Error-List
DEFAULT_BRANCH=main

# GitHub Token (set your actual token)
GITHUB_TOKEN=your_github_token_here

# Slack Configuration (optional)
SLACK_BOT_TOKEN=your_slack_token_here
SLACK_CHANNEL=#general

# AI Configuration (optional)
OPENAI_API_KEY=your_openai_key_here
HF_API_TOKEN=your_huggingface_token_here

# Security
SENTINEL_WEBHOOK_SECRET=your_webhook_secret_here
ALLOWED_REPOS=mailech/Error-List

# Confidence Threshold
CONFIDENCE_THRESHOLD=0.8

# Server Configuration
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with Error-List configuration")
    print("🎯 Target Repository: mailech/Error-List")
    print("📝 Edit .env file to add your GitHub token")

if __name__ == "__main__":
    create_env_file()
