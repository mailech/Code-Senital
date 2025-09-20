import asyncio
import json
from typing import Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.db import get_conn
from app.logging_setup import logger
from app.services.context_aggregator import aggregate_context
from app.services.ai_engine import generate_patch_and_explanation
from app.services.patch_validator import validate_patch
from app.services.action_engine import execute_action


# Global variables to track state
_last_pr_time = 0
_processed_events = set()

async def process_event(event: Dict[str, Any]) -> None:
    global _last_pr_time, _processed_events
    
    # Create a unique event identifier
    event_id = f"{event['id']}_{event['event_type']}_{event.get('source', 'unknown')}"
    
    # Skip if we've already processed this event
    if event_id in _processed_events:
        logger.info("worker_event_already_processed", event_id=event_id, reason="duplicate")
        return
    
    _processed_events.add(event_id)
    
    payload = json.loads(event["payload"]) if isinstance(event["payload"], str) else event["payload"]
    
    # Skip Sentinel PRs to prevent loops
    if event["event_type"] == "pull_request":
        pr_data = payload.get("pull_request", {})
        head_ref = pr_data.get("head", {}).get("ref", "")
        if head_ref.startswith("sentinel/fix-"):
            logger.info("worker_sentinel_pr_skipped", ref=head_ref, reason="prevent_loop")
            return
    
    # Rate limiting: only create one PR per minute
    import time
    current_time = time.time()
    if current_time - _last_pr_time < 60:  # 60 seconds cooldown
        logger.info("worker_rate_limited", time_since_last=current_time - _last_pr_time, reason="cooldown")
        return
    
    diff = payload.get("diff", "")
    logs = payload.get("logs", "")
    context = aggregate_context(diff=diff, logs=logs)
    patch, explanation, confidence = generate_patch_and_explanation(context)
    valid, details = validate_patch(repo_path="demo_repo", failing_test=payload.get("failing_test"))
    logger.info("validation_result", valid=valid, details=details)
    if valid:
        _last_pr_time = current_time  # Update last PR time
        await execute_action(patch=patch, explanation=explanation, confidence=confidence)


async def run_worker(poll_interval_s: float = 2.0) -> None:
    last_id = 0
    while True:
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, source, event_type, payload FROM events WHERE id > ? ORDER BY id ASC",
            (last_id,),
        ).fetchall()
        conn.close()
        for r in rows:
            await process_event(dict(r))
            last_id = r["id"]
        await asyncio.sleep(poll_interval_s)


if __name__ == "__main__":
    asyncio.run(run_worker())
