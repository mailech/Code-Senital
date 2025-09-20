from fastapi import APIRouter, Response
from app.db import list_recent_failures, list_recent_prs, get_conn
from app.logging_setup import logger

router = APIRouter()


@router.get("/dashboard", response_class=Response)
async def dashboard():
    failures = list_recent_failures()
    prs = list_recent_prs()
    time_saved_hours = round(len(prs) * 0.5, 1)

    html = [
        "<html><head><title>Sentinel Dashboard</title>",
        "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:8px} th{background:#f4f4f4}</style>",
        "</head><body>",
        "<h1>Self-Healing Codebase Sentinel</h1>",
        f"<p><b>Time saved (simulated):</b> {time_saved_hours} hours</p>",
        "<h2>Recent Failures</h2>",
        "<table><tr><th>ID</th><th>Source</th><th>Type</th><th>Created</th></tr>",
    ]
    for f in failures:
        html.append(f"<tr><td>{f['id']}</td><td>{f['source']}</td><td>{f['event_type']}</td><td>{f['created_at']}</td></tr>")
    html.append("</table>")

    html.extend(["<h2>PRs Created</h2>", "<table><tr><th>ID</th><th>Repo</th><th>Branch</th><th>Confidence</th><th>Status</th><th>URL</th></tr>"])
    for p in prs:
        html.append(
            f"<tr><td>{p['id']}</td><td>{p['repo']}</td><td>{p['branch']}</td><td>{p['confidence']}</td><td>{p['status']}</td><td><a href='{p['url']}' target='_blank'>link</a></td></tr>"
        )
    html.append("</table>")

    html.append("</body></html>")
    return Response(content="".join(html), media_type="text/html")


@router.get("/events")
async def get_events():
    """Get all events from the database"""
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, source, event_type, payload, created_at FROM events ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        conn.close()
        
        events = []
        for row in rows:
            events.append({
                "id": row["id"],
                "source": row["source"],
                "event_type": row["event_type"],
                "payload": row["payload"],
                "created_at": row["created_at"]
            })
        
        return events
    except Exception as e:
        logger.error("dashboard_events_error", error=str(e))
        return []


@router.get("/pull-requests")
async def get_pull_requests():
    """Get all pull requests from the database"""
    try:
        conn = get_conn()
        rows = conn.execute(
            "SELECT id, title, description, url, confidence, status, created_at FROM prs ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        conn.close()
        
        prs = []
        for row in rows:
            prs.append({
                "id": row["id"],
                "title": row["title"],
                "body": row["description"],
                "html_url": row["url"],
                "state": row["status"],
                "confidence": row["confidence"],
                "created_at": row["created_at"]
            })
        
        return prs
    except Exception as e:
        logger.error("dashboard_prs_error", error=str(e))
        return []
