import asyncio
from textwrap import dedent

from app.config import settings
from app.services.github_client import _get_json, ensure_branch, upsert_file, _post_json, API_BASE
import httpx


OWNER = "mailech"
REPO = "Error-Get"
BRANCH = "sentinel/seed-error-demo"


async def main() -> None:
    async with httpx.AsyncClient() as client:
        repo_info = await _get_json(client, f"{API_BASE}/repos/{OWNER}/{REPO}")
        default_branch = repo_info.get("default_branch", settings.default_branch)
        ref = await _get_json(client, f"{API_BASE}/repos/{OWNER}/{REPO}/git/ref/heads/{default_branch}")
        base_sha = ref["object"]["sha"]

    await ensure_branch(OWNER, REPO, BRANCH, base_sha)

    # Minimal Python project with an intentional bug
    files = {
        "requirements.txt": dedent(
            """
            pytest==8.3.2
            """.strip()
        ),
        "app_demo/math_ops.py": dedent(
            """
            def add(a: int, b: int) -> int:
                # Intentional bug
                return a - b
            """.strip()
        ),
        "tests/test_math_ops.py": dedent(
            """
            from app_demo.math_ops import add

            def test_add():
                assert add(2, 2) == 4
            """.strip()
        ),
        ".github/workflows/ci.yml": dedent(
            """
            name: CI

            on:
              push:
              pull_request:

            jobs:
              test:
                runs-on: ubuntu-latest
                steps:
                  - uses: actions/checkout@v4
                  - uses: actions/setup-python@v5
                    with:
                      python-version: '3.12'
                  - run: pip install -r requirements.txt
                  - run: pytest -q
                continue-on-error: true

              notify-sentinel:
                needs: test
                if: needs.test.result == 'failure'
                runs-on: ubuntu-latest
                steps:
                  - name: Notify Sentinel of CI failure
                    run: |
                      curl -sS -X POST "$SENTINEL_URL/webhooks/ci/failure" \
                        -H "Content-Type: application/json" \
                        -H "X-GitHub-Event: workflow_run" \
                        -d @- <<'JSON'
                      {
                        "repo": "Error-Get",
                        "failing_test": "tests/test_math_ops.py::test_add",
                        "logs": "pytest failure (redacted)",
                        "diff": "diff --git a/app_demo/math_ops.py b/app_demo/math_ops.py\n-                return a - b\n+                return a + b\n"
                      }
                      JSON
                    env:
                      SENTINEL_URL: ${{ secrets.SENTINEL_URL }}
            """.strip()
        ),
        "README.md": "Seeded by Sentinel demo."
    }

    for path, content in files.items():
        await upsert_file(OWNER, REPO, BRANCH, path, content, message="Sentinel: seed demo files")

    # Open PR
    async with httpx.AsyncClient() as client:
        pr = await _post_json(
            client,
            f"{API_BASE}/repos/{OWNER}/{REPO}/pulls",
            {"title": "Sentinel: seed error demo", "head": BRANCH, "base": default_branch, "body": "Seeding failing test demo."},
        )
        print(pr)


if __name__ == "__main__":
    asyncio.run(main())


