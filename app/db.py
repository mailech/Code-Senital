import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path("data") / "sentinel.sqlite3"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            event_type TEXT,
            payload TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            branch TEXT,
            title TEXT,
            description TEXT,
            url TEXT,
            confidence REAL,
            status TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def insert_event(source: str, event_type: str, payload_json: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO events (source, event_type, payload) VALUES (?, ?, ?)",
        (source, event_type, payload_json),
    )
    conn.commit()
    conn.close()


def insert_pr(repo: str, branch: str, title: str, description: str, url: str, confidence: float, status: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO prs (repo, branch, title, description, url, confidence, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (repo, branch, title, description, url, confidence, status),
    )
    conn.commit()
    pr_id = cur.lastrowid
    conn.close()
    return pr_id


def update_pr_status(pr_id: int, status: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE prs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, pr_id),
    )
    conn.commit()
    conn.close()


def list_recent_failures(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, source, event_type, payload, created_at FROM events ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_recent_prs(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, repo, branch, title, url, confidence, status, created_at FROM prs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
