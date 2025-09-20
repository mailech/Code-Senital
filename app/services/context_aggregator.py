from typing import Dict
from app.logging_setup import logger


def aggregate_context(diff: str, logs: str, files: dict | None = None, so_query: str | None = None) -> Dict[str, str]:
    logger.info("context_aggregate", have_diff=bool(diff), have_logs=bool(logs))
    context = {
        "diff": diff,
        "logs": logs,
        "files": str(files or {}),
        "so_results": "[]",  # TODO: add SO query results
    }
    return context
