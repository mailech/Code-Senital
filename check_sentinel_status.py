#!/usr/bin/env python3
"""
Quick Sentinel Status Checker
Shows you EXACTLY what's happening with your errors and fixes
"""

import sqlite3
import json
from pathlib import Path
import requests

def check_database():
    """Check what's in the database"""
    db_path = Path("data") / "sentinel.sqlite3"
    
    if not db_path.exists():
        print("❌ Database doesn't exist yet")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check events
    cursor.execute("SELECT COUNT(*) as count FROM events")
    event_count = cursor.fetchone()["count"]
    print(f"📊 Total Events: {event_count}")
    
    # Show recent events
    cursor.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT 5")
    recent_events = cursor.fetchall()
    
    print("\n🔍 Recent Events:")
    for event in recent_events:
        print(f"  - {event['event_type']} from {event['source']} at {event['created_at']}")
    
    # Check PRs
    cursor.execute("SELECT COUNT(*) as count FROM prs")
    pr_count = cursor.fetchone()["count"]
    print(f"\n📋 Total PRs Created: {pr_count}")
    
    # Show recent PRs
    cursor.execute("SELECT * FROM prs ORDER BY created_at DESC LIMIT 5")
    recent_prs = cursor.fetchall()
    
    print("\n✅ Recent PRs:")
    for pr in recent_prs:
        print(f"  - {pr['title']} (Confidence: {pr['confidence']}%)")
        print(f"    URL: {pr['url']}")
        print(f"    Status: {pr['status']}")
    
    conn.close()

def check_sentinel_api():
    """Check if Sentinel API is working"""
    try:
        # Check health
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Sentinel API is running")
        else:
            print(f"❌ Sentinel API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Sentinel API: {e}")
        return False
    
    try:
        # Check events endpoint
        response = requests.get("http://localhost:8000/events", timeout=5)
        if response.status_code == 200:
            events = response.json()
            print(f"✅ Events API working - {len(events)} events available")
        else:
            print(f"❌ Events API returned {response.status_code}")
    except Exception as e:
        print(f"❌ Events API error: {e}")
    
    try:
        # Check PRs endpoint
        response = requests.get("http://localhost:8000/pull-requests", timeout=5)
        if response.status_code == 200:
            prs = response.json()
            print(f"✅ PRs API working - {len(prs)} PRs available")
        else:
            print(f"❌ PRs API returned {response.status_code}")
    except Exception as e:
        print(f"❌ PRs API error: {e}")
    
    return True

def check_github_repo():
    """Check your GitHub repository status"""
    print("\n🔗 GitHub Repository Status:")
    print("  Repository: https://github.com/mailech/Errors")
    print("  Check for recent PRs created by Sentinel")
    print("  Look for branches starting with 'sentinel/fix-'")

def main():
    print("🤖 SENTINEL STATUS CHECKER")
    print("=" * 50)
    
    print("\n1. DATABASE STATUS:")
    check_database()
    
    print("\n2. API STATUS:")
    check_sentinel_api()
    
    print("\n3. GITHUB STATUS:")
    check_github_repo()
    
    print("\n" + "=" * 50)
    print("🎯 TO SEE YOUR FIXES:")
    print("1. Go to: https://github.com/mailech/Errors/pulls")
    print("2. Look for PRs with titles like 'Sentinel: Fix...'")
    print("3. Check the 'sentinel/fix-*' branches")
    print("4. Open the enhanced dashboard: enhanced_dashboard.html")

if __name__ == "__main__":
    main()
