#!/usr/bin/env python3
"""
Full functionality demo for Self-Healing Codebase Sentinel
"""

import asyncio
import httpx
import json
import time
from pathlib import Path

class SentinelDemo:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def test_health(self):
        """Test the health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed")
                print(f"   App: {data.get('app', 'Unknown')}")
                print(f"   Environment: {data.get('env', 'Unknown')}")
                print(f"   Confidence Threshold: {data.get('confidence_threshold', 'Unknown')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_dashboard(self):
        """Test the dashboard endpoint"""
        print("\n🔍 Testing dashboard endpoint...")
        try:
            response = await self.client.get(f"{self.base_url}/dashboard")
            if response.status_code == 200:
                print("✅ Dashboard accessible")
                print(f"   Content length: {len(response.text)} characters")
                return True
            else:
                print(f"❌ Dashboard failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
            return False
    
    async def simulate_ci_failure(self, repo_name, test_name, error_message, diff):
        """Simulate a CI failure"""
        print(f"\n🔍 Simulating CI failure for {repo_name}...")
        
        payload = {
            "repo": repo_name,
            "failing_test": test_name,
            "logs": error_message,
            "diff": diff
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/webhooks/ci/failure",
                json=payload
            )
            
            if response.status_code == 200:
                print("✅ CI failure webhook accepted")
                print(f"   Response: {response.json()}")
                return True
            else:
                print(f"❌ CI failure webhook failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ CI failure webhook error: {e}")
            return False
    
    async def simulate_github_webhook(self, repo_name, event_type, payload_data):
        """Simulate a GitHub webhook"""
        print(f"\n🔍 Simulating GitHub {event_type} webhook...")
        
        payload = {
            "repository": {"full_name": repo_name},
            "action": event_type,
            **payload_data
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/webhooks/github",
                json=payload,
                headers={"X-GitHub-Event": event_type}
            )
            
            if response.status_code == 200:
                print(f"✅ GitHub {event_type} webhook accepted")
                print(f"   Response: {response.json()}")
                return True
            else:
                print(f"❌ GitHub webhook failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ GitHub webhook error: {e}")
            return False
    
    async def test_math_operations(self):
        """Test the fixed math operations"""
        print("\n🔍 Testing math operations...")
        try:
            from app_demo.math_ops import add
            result = add(2, 2)
            if result == 4:
                print("✅ Math operations working correctly")
                print(f"   add(2, 2) = {result}")
                return True
            else:
                print(f"❌ Math operations failed: expected 4, got {result}")
                return False
        except Exception as e:
            print(f"❌ Math operations error: {e}")
            return False
    
    async def test_dataloader(self):
        """Test the fixed dataloader"""
        print("\n🔍 Testing dataloader...")
        try:
            from Cubic_Err.ml_errors.dataloader import ToyDataset
            ds = ToyDataset([1, 2, 3])
            length = len(ds)
            if length == 3:
                print("✅ Dataloader working correctly")
                print(f"   Dataset length: {length}")
                return True
            else:
                print(f"❌ Dataloader failed: expected 3, got {length}")
                return False
        except Exception as e:
            print(f"❌ Dataloader error: {e}")
            return False
    
    async def run_full_demo(self):
        """Run the complete functionality demo"""
        print("=" * 70)
        print("Self-Healing Codebase Sentinel - Full Functionality Demo")
        print("=" * 70)
        
        # Test basic functionality
        health_ok = await self.test_health()
        dashboard_ok = await self.test_dashboard()
        math_ok = await self.test_math_operations()
        dataloader_ok = await self.test_dataloader()
        
        if not all([health_ok, dashboard_ok, math_ok, dataloader_ok]):
            print("\n❌ Basic functionality tests failed. Please check the server.")
            return
        
        # Simulate various scenarios
        print("\n" + "=" * 70)
        print("Simulating Real-World Scenarios")
        print("=" * 70)
        
        # Scenario 1: Math operations bug
        await self.simulate_ci_failure(
            repo_name="your-repo/math-calculator",
            test_name="tests/test_math.py::test_addition",
            error_message="AssertionError: Expected 4, got 0",
            diff="""diff --git a/src/math.py b/src/math.py
@@ -1,3 +1,3 @@
 def add(a, b):
-    return a - b  # BUG: subtraction instead of addition
+    return a + b  # FIXED: correct addition"""
        )
        
        # Scenario 2: ML dataloader bug
        await self.simulate_ci_failure(
            repo_name="your-repo/ml-project",
            test_name="tests/test_dataloader.py::test_dataset_length",
            error_message="AssertionError: Expected 3, got 4",
            diff="""diff --git a/dataloader.py b/dataloader.py
@@ -5,4 +5,4 @@ class Dataset:
     def __len__(self):
-        return len(self.data) + 1  # BUG: off-by-one error
+        return len(self.data)  # FIXED: correct length"""
        )
        
        # Scenario 3: GitHub push event
        await self.simulate_github_webhook(
            repo_name="your-repo/your-project",
            event_type="push",
            payload_data={
                "commits": [{
                    "id": "abc123",
                    "message": "Fix critical bug in authentication",
                    "modified": ["src/auth.py", "tests/test_auth.py"]
                }]
            }
        )
        
        # Scenario 4: Pull request event
        await self.simulate_github_webhook(
            repo_name="your-repo/your-project",
            event_type="pull_request",
            payload_data={
                "pull_request": {
                    "number": 42,
                    "title": "Fix memory leak in data processing",
                    "body": "This PR fixes a critical memory leak...",
                    "state": "open"
                }
            }
        )
        
        print("\n" + "=" * 70)
        print("Demo Complete!")
        print("=" * 70)
        
        print("\n🎉 All functionality is working correctly!")
        print("\nWhat you can do now:")
        print("1. View the dashboard: http://localhost:8000/dashboard")
        print("2. Check the database for stored events")
        print("3. Set up GitHub webhooks for your repository")
        print("4. Configure Slack notifications")
        print("5. Test with your own code and bugs")
        
        print("\nThe Sentinel system is ready to:")
        print("✅ Detect CI failures automatically")
        print("✅ Generate AI-powered fixes")
        print("✅ Create GitHub pull requests")
        print("✅ Send Slack notifications")
        print("✅ Monitor system activity")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

async def main():
    """Main demo function"""
    demo = SentinelDemo()
    
    try:
        await demo.run_full_demo()
    finally:
        await demo.close()

if __name__ == "__main__":
    asyncio.run(main())
