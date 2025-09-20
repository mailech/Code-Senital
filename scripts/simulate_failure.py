import asyncio
import httpx


async def main():
    async with httpx.AsyncClient() as client:
        payload = {
            "repo": "demo_repo",
            "failing_test": "demo_repo/tests/test_app.py::test_add",
            "logs": "assert 0 == 4",
            "diff": "diff --git a/demo_repo/app.py b/demo_repo/app.py\n-    return a - b\n+    return a + b\n",
        }
        r = await client.post("http://localhost:8000/webhooks/ci/failure", json=payload)
        print(r.json())


if __name__ == "__main__":
    asyncio.run(main())
