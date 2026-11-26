"""Live transcription sessions backed by append-only text files.

Each session owns one file per source (``mic`` and ``device``) under
``output/text/``. Only transcript text is persisted; captured audio stays in
the clients' memory and is never written to disk here.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

SOURCES = ("mic", "device")


class UnknownSession(Exception):
    """Raised when a session id is not known to this store."""


class BadSource(Exception):
    """Raised when a source is not one of :data:`SOURCES`."""


class SessionClosed(Exception):
    """Raised when an append targets a session that has been closed."""


def _sanitise(value: str) -> str:
    """Keep only the characters ``_store_output`` allows in a filename."""
    return "".join(c for c in value if c.isalnum() or c in "-_")


def _stamp(ms: float) -> str:
    """Format elapsed milliseconds as the ``[mm:ss]`` transcript prefix."""
    total_s = max(0, int(ms)) // 1000
    return f"[{total_s // 60:02d}:{total_s % 60:02d}]"


@dataclass
class LiveSession:
    id: str
    language: str
    model: str
    label: str
    created: float
    descriptors: dict
    lines: dict
    seen: dict
    closed: bool = False
    last_activity: float = 0.0
    open_sockets: int = 0
    paths: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict, repr=False)


class LiveSessionStore:
    """Thread-safe, in-memory registry of live sessions and their text files."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.RLock()

    def _get(self, session_id: str) -> LiveSession:
        try:
            return self._sessions[session_id]
        except (KeyError, TypeError):
            raise UnknownSession(f"unknown live session: {session_id}") from None

    @staticmethod
    def _check_source(source: str) -> None:
        if source not in SOURCES:
            raise BadSource(f"unknown live source: {source}")

    def _descriptor(self, session: LiveSession, source: str) -> dict:
        path = session.paths[source]
        return {
            "name": path.name,
            "kind": "text",
            "size": path.stat().st_size,
            "path": str(path),
            "url": f"/v1/outputs/text/{path.name}",
        }

    def create(self, language: str = "", model: str = "turbo",
               label: str = "") -> dict:
        """Open two transcript files and return the new session summary."""
        with self._lock:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            sid = f"live_{stamp}_{uuid.uuid4().hex[:6]}"
            safe_sid = _sanitise(sid)[:32].strip("-_")
            created = time.time()
            language = str(language or "")
            # A header must stay a single line even if a caller passes a
            # language string containing a newline.
            header_language = language.replace("\r", " ").replace("\n", " ")
            session = LiveSession(
                id=sid,
                language=language,
                model=str(model or ""),
                label=str(label or ""),
                created=created,
                descriptors={},
                lines={},
                seen={source: set() for source in SOURCES},
                last_activity=created,
            )
            target = self.output_dir / "text"
            target.mkdir(parents=True, exist_ok=True)
            try:
                for source in SOURCES:
                    base = f"{stamp}-{safe_sid}-{source}"
                    path = target / f"{base}.txt"
                    n = 1
                    while path.exists():
                        path = target / f"{base}-{n}.txt"
                        n += 1
                    handle: TextIO = path.open("a", encoding="utf-8")
                    handle.write(
                        f"# Live transcript - source: {source} "
                        f"- language: {header_language}\n"
                    )
                    handle.flush()
                    session.paths[source] = path
                    session.files[source] = handle
                    session.lines[source] = 1
            except Exception:
                for handle in session.files.values():
                    handle.close()
                raise
            for source in SOURCES:
                session.descriptors[source] = self._descriptor(session, source)
            self._sessions[sid] = session
            return {
                "id": sid,
                "sources": list(SOURCES),
                "created": created,
                "language": session.language,
                "model": session.model,
                "label": session.label,
                "closed": False,
                **{source: dict(session.descriptors[source]) for source in SOURCES},
            }

    def append(self, session_id: str, source: str, text: str,
               item_id: str = "", elapsed_ms: int = 0) -> dict:
        """Append one timestamped line, ignoring empty text and repeated ids."""
        with self._lock:
            session = self._get(session_id)
            if session.closed:
                raise SessionClosed(f"live session is closed: {session_id}")
            self._check_source(source)
            session.last_activity = time.time()
            seen = session.seen[source]
            if item_id and item_id in seen:
                return {
                    "ok": True, "duplicate": True, "bytes": 0,
                    "line": session.lines[source],
                }
            # One utterance must occupy exactly one line in the transcript.
            fragment = (str(text or "")
                        .replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
                        .strip())
            if item_id:
                seen.add(item_id)
            if not fragment:
                return {
                    "ok": True, "duplicate": False, "bytes": 0,
                    "line": session.lines[source],
                }
            line = f"{_stamp(elapsed_ms)} {fragment}\n"
            handle: TextIO = session.files[source]
            handle.write(line)
            handle.flush()
            session.lines[source] += 1
            session.descriptors[source] = self._descriptor(session, source)
            return {
                "ok": True, "duplicate": False,
                "bytes": len(line.encode("utf-8")),
                "line": session.lines[source],
            }

    def close(self, session_id: str) -> dict:
        """Flush and close both files. Repeated calls are harmless."""
        with self._lock:
            session = self._get(session_id)
            if not session.closed:
                for handle in session.files.values():
                    handle.flush()
                    handle.close()
                session.files.clear()
                session.closed = True
            return {
                "ok": True,
                "closed": True,
                **{source: dict(session.descriptors[source]) for source in SOURCES},
                "lines": dict(session.lines),
            }

    def status(self, session_id: str) -> dict:
        """Return session metadata, descriptors and per-source line counts."""
        with self._lock:
            session = self._get(session_id)
            return {
                "id": session.id,
                "language": session.language,
                "model": session.model,
                "label": session.label,
                "created": session.created,
                "sources": list(SOURCES),
                "descriptors": {
                    source: dict(session.descriptors[source]) for source in SOURCES
                },
                "lines": dict(session.lines),
                "closed": session.closed,
                "open_sockets": session.open_sockets,
                "last_activity": session.last_activity,
            }

    def mark_socket(self, session_id: str, delta: int) -> None:
        """Record a socket opening (+1) or closing (-1) for a session.

        An opening counts as activity. A closing does not reset the idle
        clock, so an abandoned session becomes sweepable as soon as no
        sockets remain and the idle threshold has passed.
        """
        with self._lock:
            session = self._get(session_id)
            session.open_sockets = max(0, session.open_sockets + int(delta))
            if int(delta) > 0:
                session.last_activity = time.time()

    def last_activity(self, session_id: str, ts: float) -> None:
        """Test/support hook: force the idle timestamp to ``ts`` (Unix seconds).

        Kept public so the idle sweep can be tested without waiting 30 minutes.
        """
        with self._lock:
            self._get(session_id).last_activity = float(ts)

    def sweep_idle(self, max_idle_s: int = 1800) -> int:
        """Close sessions idle longer than ``max_idle_s`` with no open sockets."""
        with self._lock:
            now = time.time()
            swept = 0
            for session in list(self._sessions.values()):
                if session.closed or session.open_sockets:
                    continue
                if now - session.last_activity < max_idle_s:
                    continue
                self.close(session.id)
                swept += 1
            return swept
