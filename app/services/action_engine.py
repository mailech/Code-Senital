from typing import Tuple
from app.config import settings
from app.logging_setup import logger
from app.services import github_client, slack_client
from app.db import insert_pr


async def execute_action(patch: str, explanation: str, confidence: float) -> Tuple[str, str]:
    title = "Automated Fix: Self-Healing Codebase Sentinel"
    body = f"Confidence: {confidence}\n\nExplanation:\n{explanation}\n\nPatch:\n\n````diff\n{patch}\n````"
    if confidence >= settings.confidence_threshold:
        import time
        branch = f"sentinel/fix-{int(time.time())}"
        result = await github_client.create_branch_and_pr(branch, title, body, patch)
        url = result.get("html_url", "")
        pr_id = insert_pr(
            repo=f"{settings.github_owner}/{settings.github_repo}",
            branch=branch,
            title=title,
            description=explanation,
            url=url,
            confidence=confidence,
            status="opened",
        )
        slack_client.notify(None, f"Opened PR: {url} (confidence {confidence})")
        logger.info("action_pr_opened", url=url, pr_id=pr_id)
        return "pr", url
    else:
        result = await github_client.create_issue(title, body)
        url = result.get("html_url", "")
        slack_client.notify(None, f"Created Issue: {url} (confidence {confidence})")
        logger.info("action_issue_opened", url=url)
        return "issue", url
