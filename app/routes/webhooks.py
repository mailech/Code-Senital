import json
from fastapi import APIRouter, Request, HTTPException
from app.logging_setup import logger
from app.db import insert_event
from app.config import settings
from app.security import verify_signature

router = APIRouter(prefix="/webhooks")


def _is_allowed_repo(repo: str) -> bool:
    if not settings.allowed_repos:
        return True
    allowed = [r.strip().lower() for r in settings.allowed_repos.split(",") if r.strip()]
    return repo.lower() in allowed


@router.post("/github")
async def github_webhook(request: Request):
    body = await request.body()
    payload = json.loads(body.decode("utf-8")) if body else {}
    event_type = request.headers.get("X-GitHub-Event", "unknown")

    # Signature verification (if configured)
    if settings.webhook_secret:
        ts = request.headers.get("X-Sentinel-Timestamp")
        sig = request.headers.get("X-Sentinel-Signature")
        if not ts or not sig:
            raise HTTPException(status_code=401, detail="Missing signature headers")
        ok, expected = verify_signature(settings.webhook_secret, ts, body, sig)
        if not ok:
            logger.warning("webhook_sig_invalid", expected=expected, provided=sig)
            raise HTTPException(status_code=401, detail="Invalid signature")

    repo = payload.get("repo") or payload.get("repository", {}).get("full_name", "")
    if not _is_allowed_repo(repo or ""):
        logger.info("webhook_repo_blocked", repo=repo)
        raise HTTPException(status_code=403, detail="Repo not allowed")

    # Check if this is a Sentinel branch/PR to prevent loops
    if event_type == "pull_request":
        pr_data = payload.get("pull_request", {})
        head_ref = pr_data.get("head", {}).get("ref", "")
        if head_ref.startswith("sentinel/fix-"):
            logger.info("webhook_sentinel_pr_ignored", ref=head_ref, reason="prevent_loop")
            return {"ok": True, "ignored": "sentinel_pr"}
    
    # Check if this is a push to a Sentinel branch
    if event_type == "push":
        ref = payload.get("ref", "")
        if ref.startswith("refs/heads/sentinel/fix-"):
            logger.info("webhook_sentinel_branch_ignored", ref=ref, reason="prevent_loop")
            return {"ok": True, "ignored": "sentinel_branch"}
    
    # Check if this is a workflow run from a Sentinel branch
    if event_type == "workflow_run":
        workflow_data = payload.get("workflow_run", {})
        head_branch = workflow_data.get("head_branch", "")
        if head_branch.startswith("sentinel/fix-"):
            logger.info("webhook_sentinel_workflow_ignored", branch=head_branch, reason="prevent_loop")
            return {"ok": True, "ignored": "sentinel_workflow"}

    insert_event("github", event_type, json.dumps(payload))
    logger.info("webhook_github", event_type=event_type, repo=repo)
    return {"ok": True}


@router.post("/ci/failure")
async def ci_failure_sim(request: Request):
    payload = await request.json()
    insert_event("ci", "failure", json.dumps(payload))
    logger.info("webhook_ci_failure", details=payload)
    return {"ok": True}
