import asyncio
import time
from typing import Dict

from app.config import settings
from app.services.github_client import _get_json, _post_json, upsert_file
import httpx


API_BASE = "https://api.github.com"


# Files to push to the new repo
FILES: Dict[str, str] = {
    # Make ml_errors a proper Python package
    "ml_errors/__init__.py": """
# ML Errors Demo Package
""",

    # Minimal PyTorch training script with intentional bugs
    "ml_errors/train.py": """
import torch
import torch.nn as nn
import torch.optim as optim


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)  # BUG: Should be 2 classes for CrossEntropyLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_data(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(n, 10)
    # Two classes 0/1
    y = (torch.rand(n) > 0.5).long()
    return x, y


def train_one_epoch() -> float:
    model = TinyNet()
    x, y = make_data()
    opt = optim.SGD(model.parameters(), lr=0.1)
    # BUG: Using CrossEntropyLoss with 1-dim output will raise or behave incorrectly
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(5):
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)  # Intentional shape mismatch for failure
        loss.backward()
        opt.step()
    # Return dummy accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc


if __name__ == "__main__":
    print(train_one_epoch())
""",

    # Simple dataset/dataloader with an error (mismatched __len__)
    "ml_errors/dataloader.py": """
from typing import List


class ToyDataset:
    def __init__(self, data: List[int]):
        self.data = data

    def __len__(self) -> int:
        # BUG: Wrong length, off-by-one to trigger test failures
        return len(self.data) + 1

    def __getitem__(self, idx: int) -> int:
        return self.data[idx]
""",

    # requirements for the repo to run tests
    "requirements.txt": """
setuptools>=65.0.0
torch==2.3.0
numpy==1.24.0
pytest==8.2.2
""",

    # Pytest that will fail due to the intentional bugs
    "tests/test_ml_errors.py": """
import pytest


def test_dataset_len():
    from ml_errors.dataloader import ToyDataset

    ds = ToyDataset([1, 2, 3])
    assert len(ds) == 3  # will fail (len returns 4)


def test_training_loop():
    import ml_errors.train as tr

    # This should raise due to shape mismatch; assert raises to mark failure explicitly
    with pytest.raises(Exception):
        tr.train_one_epoch()
""",

    # GitHub Actions workflow to run tests and notify Sentinel on failure
    ".github/workflows/ci.yml": """
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
          python-version: '3.11'
      - name: Install build dependencies
        run: python -m pip install --upgrade pip setuptools wheel
      - run: pip install -r requirements.txt
      - name: Set PYTHONPATH
        run: echo "PYTHONPATH=$GITHUB_WORKSPACE:$PYTHONPATH" >> $GITHUB_ENV
      - name: Run tests
        run: pytest -q

  notify-sentinel:
    needs: test
    if: failure()
    runs-on: ubuntu-latest
    steps:
      - name: Notify Sentinel of CI failure
        uses: actions/github-script@v7
        env:
          SENTINEL_URL: ${{ secrets.SENTINEL_URL }}
        with:
          script: |
            const url = `${process.env.SENTINEL_URL}/webhooks/ci/failure`;
            const payload = {
              repo: process.env.GITHUB_REPOSITORY,
              failing_test: 'tests/test_ml_errors.py',
              logs: 'pytest failure (redacted)',
              diff: 'Fixed dataloader length bug'
            };
            const res = await fetch(url, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', 'X-GitHub-Event': 'workflow_run' },
              body: JSON.stringify(payload)
            });
            core.info(`Sentinel response status: ${res.status}`);
""",

    # Follow-up workflow that runs on push to main (shows green after Sentinel fix is merged)
    ".github/workflows/verify-fix.yml": """
name: Verify Fix

on:
  push:
    branches: [ main ]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install build dependencies
        run: python -m pip install --upgrade pip setuptools wheel
      - run: pip install -r requirements.txt
      - name: Set PYTHONPATH
        run: echo "PYTHONPATH=$GITHUB_WORKSPACE:$PYTHONPATH" >> $GITHUB_ENV
      - name: Run tests
        run: pytest -q
      - name: Success - Sentinel fix verified
        run: echo "✅ All tests pass! Sentinel fix was successful."
""",
}


async def main() -> None:
    if not settings.github_owner or not settings.github_repo or not settings.github_token:
        raise SystemExit("GITHUB_OWNER/REPO/TOKEN not configured in environment or .env")

    owner = settings.github_owner
    repo = settings.github_repo

    async with httpx.AsyncClient() as client:
        # Create initial commit with README
        readme_content = """# Cubic-Err

Self-Healing Codebase Sentinel Demo Repository

This repository contains intentional bugs that trigger the Sentinel system to automatically fix them.

## Files

- `ml_errors/` - ML/DL code with intentional bugs
- `tests/` - Test suite that fails due to bugs
- `.github/workflows/` - CI/CD workflows

## How it works

1. Push code with bugs
2. CI fails
3. Sentinel detects failure
4. Sentinel creates fix PR
5. Merge fix PR
6. Tests pass ✅
"""
        
        # Create initial commit
        await _post_json(
            client,
            f"{API_BASE}/repos/{owner}/{repo}/contents/README.md",
            {
                "message": "Initial commit: Add README",
                "content": __import__('base64').b64encode(readme_content.encode()).decode()
            }
        )
        print(f"Created initial commit in {owner}/{repo}")

        # Create branch for our files
        branch = f"sentinel/seed-{int(time.time())}"
        
        # Get the main branch SHA
        main_ref = await _get_json(client, f"{API_BASE}/repos/{owner}/{repo}/git/ref/heads/main")
        main_sha = main_ref["object"]["sha"]
        
        # Create our branch
        await _post_json(
            client,
            f"{API_BASE}/repos/{owner}/{repo}/git/refs",
            {"ref": f"refs/heads/{branch}", "sha": main_sha}
        )
        print(f"Created branch: {branch}")

        # Add all files to the branch
        for path, content in FILES.items():
            await upsert_file(owner, repo, branch, path, content, message=f"Add {path}")
            print(f"Added {path}")

        # Create PR
        pr = await _post_json(
            client,
            f"{API_BASE}/repos/{owner}/{repo}/pulls",
            {
                "title": "Add ML Error Demo Files",
                "head": branch,
                "base": "main",
                "body": "This PR adds ML/DL demo files with intentional errors and CI workflows that notify Sentinel on failures.",
            },
        )
        print(f"Created PR: {pr.get('html_url', '')}")


if __name__ == "__main__":
    asyncio.run(main())
