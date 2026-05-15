from __future__ import annotations

import os
import re
import shutil
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

RUNTIME_DIR = Path(os.getenv("BURAQ_RUNTIME_DIR", "runtime"))
UPLOADS_DIR = RUNTIME_DIR / "uploads"
DOWNLOADS_DIR = RUNTIME_DIR / "downloads"
SCHEDULE_DB_PATH = Path(os.getenv("BURAQ_SCHEDULE_DB_PATH", str(RUNTIME_DIR / "scheduled_emails.sqlite")))
DEFAULT_POLL_SECONDS = int(os.getenv("BURAQ_SCHEDULER_POLL_SECONDS", "15"))
STORAGE_SCHEME = "storage://"
ALLOWED_STORAGE_AREAS = {
    "uploads": UPLOADS_DIR,
    "downloads": DOWNLOADS_DIR,
}

def _allowed_local_upload_roots() -> list[Path]:
    configured = os.getenv("BURAQ_LOCAL_UPLOAD_ROOTS", "").strip()
    roots: list[Path] = []

    if configured:
        for entry in configured.split(os.pathsep):
            candidate = entry.strip()
            if candidate:
                roots.append(Path(candidate).expanduser().resolve())

    if not roots:
        roots.append((RUNTIME_DIR / "local_uploads").resolve())

    return roots


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def ensure_runtime_dirs() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for directory in ALLOWED_STORAGE_AREAS.values():
        directory.mkdir(parents=True, exist_ok=True)
    for directory in _allowed_local_upload_roots():
        directory.mkdir(parents=True, exist_ok=True)
    SCHEDULE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    if not candidate:
        candidate = "file"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    return safe.strip("._") or "file"


def _storage_ref(area: str, filename: str) -> str:
    return f"{STORAGE_SCHEME}{area}/{filename}"


def _storage_ref_to_parts(file_ref: str) -> tuple[str, str]:
    normalized = file_ref.strip()
    if not normalized.startswith(STORAGE_SCHEME):
        raise ValueError("Storage references must start with storage://")

    relative = normalized[len(STORAGE_SCHEME):].strip("/")
    if "/" not in relative:
        raise ValueError("Storage references must include an area and filename, e.g. storage://uploads/file.txt")

    area, filename = relative.split("/", 1)
    if area not in ALLOWED_STORAGE_AREAS:
        raise ValueError(f"Unknown storage area '{area}'.")

    safe_name = sanitize_filename(filename)
    return area, safe_name


def resolve_file_reference(file_ref: str, allow_local: bool = False) -> Path:
    candidate = file_ref.strip()
    if candidate.startswith(STORAGE_SCHEME):
        area, safe_name = _storage_ref_to_parts(candidate)
        resolved = ALLOWED_STORAGE_AREAS[area] / safe_name
        if not resolved.exists():
            raise FileNotFoundError(f"Stored file not found: {candidate}")
        return resolved

    if not allow_local:
        raise ValueError(
            "Local file paths are not allowed for this operation. Upload the file into managed storage first."
        )

    local_path = Path(candidate).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {candidate}")

    resolved = local_path.resolve()
    allowed_roots = _allowed_local_upload_roots()
    if not any(_is_within(resolved, root) for root in allowed_roots):
        allowed_display = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(
            "Local file access is sandboxed. Move the file into one of these directories first: "
            f"{allowed_display}"
        )

    return resolved


def save_uploaded_bytes(filename: str, content: bytes, area: str = "uploads") -> dict[str, object]:
    ensure_runtime_dirs()
    if area not in ALLOWED_STORAGE_AREAS:
        raise ValueError(f"Unknown storage area '{area}'.")

    safe_name = sanitize_filename(filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    final_name = f"{timestamp}_{safe_name}"
    destination = ALLOWED_STORAGE_AREAS[area] / final_name
    destination.write_bytes(content)

    return {
        "name": final_name,
        "ref": _storage_ref(area, final_name),
        "size_bytes": destination.stat().st_size,
        "modified_at": datetime.fromtimestamp(destination.stat().st_mtime).isoformat(timespec="seconds"),
    }


def copy_local_file_to_storage(source: str, area: str = "uploads") -> dict[str, object]:
    source_path = (
        resolve_file_reference(source)
        if source.strip().startswith(STORAGE_SCHEME)
        else resolve_file_reference(source, allow_local=True)
    )
    return save_uploaded_bytes(source_path.name, source_path.read_bytes(), area=area)


def list_stored_files(area: str | None = None) -> list[dict[str, object]]:
    ensure_runtime_dirs()
    selected_areas = [area] if area else list(ALLOWED_STORAGE_AREAS.keys())
    entries: list[dict[str, object]] = []

    for selected_area in selected_areas:
        if selected_area not in ALLOWED_STORAGE_AREAS:
            raise ValueError(f"Unknown storage area '{selected_area}'.")

        directory = ALLOWED_STORAGE_AREAS[selected_area]
        for path in sorted(directory.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_file():
                continue
            stats = path.stat()
            entries.append(
                {
                    "name": path.name,
                    "area": selected_area,
                    "ref": _storage_ref(selected_area, path.name),
                    "size_bytes": stats.st_size,
                    "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(timespec="seconds"),
                }
            )
    return entries


def init_schedule_db() -> None:
    ensure_runtime_dirs()
    with sqlite3.connect(SCHEDULE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduled_emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                to_address TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                send_at TEXT NOT NULL,
                attachment_ref TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                sent_at TEXT,
                last_error TEXT
            )
            """
        )
        conn.commit()


def queue_scheduled_email(
    to_address: str,
    subject: str,
    body: str,
    send_at: str,
    attachment_ref: str | None = None,
) -> int:
    init_schedule_db()
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(SCHEDULE_DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO scheduled_emails (
                to_address,
                subject,
                body,
                send_at,
                attachment_ref,
                status,
                created_at,
                updated_at,
                sent_at,
                last_error
            ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, NULL, NULL)
            """,
            (to_address, subject, body, send_at, attachment_ref, now, now),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_scheduled_emails(limit: int = 20) -> list[dict[str, object]]:
    init_schedule_db()
    with sqlite3.connect(SCHEDULE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                to_address,
                subject,
                body,
                send_at,
                attachment_ref,
                status,
                created_at,
                updated_at,
                sent_at,
                last_error
            FROM scheduled_emails
            ORDER BY datetime(send_at) DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def _update_schedule_status(row_id: int, status: str, last_error: str | None = None, sent_at: str | None = None) -> None:
    updated_at = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(SCHEDULE_DB_PATH) as conn:
        conn.execute(
            """
            UPDATE scheduled_emails
            SET status = ?, updated_at = ?, last_error = ?, sent_at = ?
            WHERE id = ?
            """,
            (status, updated_at, last_error, sent_at, row_id),
        )
        conn.commit()


def run_due_scheduled_emails_once(
    send_callable: Callable[[str, str, str, str | None], tuple[bool, str]],
    now: datetime | None = None,
) -> list[dict[str, object]]:
    init_schedule_db()
    effective_now = now or datetime.utcnow()
    results: list[dict[str, object]] = []

    with sqlite3.connect(SCHEDULE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        due_rows = conn.execute(
            """
            SELECT id, to_address, subject, body, send_at, attachment_ref
            FROM scheduled_emails
            WHERE status = 'queued' AND datetime(send_at) <= datetime(?)
            ORDER BY datetime(send_at) ASC, id ASC
            """,
            (effective_now.isoformat(timespec="seconds"),),
        ).fetchall()

    for row in due_rows:
        row_id = int(row["id"])
        _update_schedule_status(row_id, "sending")
        try:
            success, detail = send_callable(
                str(row["to_address"]),
                str(row["subject"]),
                str(row["body"]),
                row["attachment_ref"],
            )
            if success:
                sent_at = datetime.utcnow().isoformat(timespec="seconds")
                _update_schedule_status(row_id, "sent", last_error=None, sent_at=sent_at)
                results.append({"id": row_id, "status": "sent", "detail": detail})
            else:
                _update_schedule_status(row_id, "failed", last_error=detail, sent_at=None)
                results.append({"id": row_id, "status": "failed", "detail": detail})
        except Exception as exc:
            detail = f"Scheduler delivery failed: {exc}"
            _update_schedule_status(row_id, "failed", last_error=detail, sent_at=None)
            results.append({"id": row_id, "status": "failed", "detail": detail})

    return results


def start_scheduler_thread(
    send_callable: Callable[[str, str, str, str | None], tuple[bool, str]],
    poll_seconds: int = DEFAULT_POLL_SECONDS,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def scheduler_loop() -> None:
        while not stop_event.is_set():
            try:
                run_due_scheduled_emails_once(send_callable)
            except Exception:
                pass
            stop_event.wait(poll_seconds)

    thread = threading.Thread(target=scheduler_loop, name="buraq-scheduler", daemon=True)
    thread.start()
    return stop_event, thread


def stop_scheduler_thread(stop_event: threading.Event, thread: threading.Thread, timeout: float = 5.0) -> None:
    stop_event.set()
    thread.join(timeout=timeout)


__all__ = [
    "ALLOWED_STORAGE_AREAS",
    "DEFAULT_POLL_SECONDS",
    "DOWNLOADS_DIR",
    "RUNTIME_DIR",
    "SCHEDULE_DB_PATH",
    "STORAGE_SCHEME",
    "UPLOADS_DIR",
    "copy_local_file_to_storage",
    "ensure_runtime_dirs",
    "init_schedule_db",
    "list_scheduled_emails",
    "list_stored_files",
    "queue_scheduled_email",
    "resolve_file_reference",
    "run_due_scheduled_emails_once",
    "save_uploaded_bytes",
    "start_scheduler_thread",
    "stop_scheduler_thread",
]
