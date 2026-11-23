"""Auto-update checker for Prism.

Checks GitHub for new commits and provides update functionality.
"""
import json
import logging
import os
import subprocess
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_URL = "https://api.github.com/repos/DracieVajko/Prism"
LOCAL_REPO = Path(__file__).resolve().parent.parent


def _git(*args, cwd=None):
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=cwd or str(LOCAL_REPO),
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)


def get_current_sha() -> str:
    """Get the current local commit SHA."""
    rc, out, _ = _git("rev-parse", "HEAD")
    return out if rc == 0 else ""


def get_latest_sha() -> str:
    """Fetch the latest commit SHA from GitHub."""
    try:
        url = f"{REPO_URL}/commits/main"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
            return data.get("sha", "")
    except Exception as e:
        logger.warning("Failed to check GitHub for updates: %s", e)
        return ""


def get_commits_behind() -> int:
    """Get number of commits behind origin/main."""
    rc, out, _ = _git("rev-list", "--count", "HEAD..origin/main")
    try:
        return int(out) if rc == 0 else 0
    except ValueError:
        return 0


def check_update() -> dict:
    """Check if an update is available.

    Returns dict with:
        current_sha: str
        latest_sha: str
        update_available: bool
        commits_behind: int
        changelog: list of commit messages
    """
    current = get_current_sha()
    latest = get_latest_sha()

    if not current or not latest:
        return {
            "current_sha": current,
            "latest_sha": latest,
            "update_available": False,
            "commits_behind": 0,
            "changelog": [],
            "error": "Could not determine update status",
        }

    if current == latest:
        return {
            "current_sha": current[:8],
            "latest_sha": latest[:8],
            "update_available": False,
            "commits_behind": 0,
            "changelog": [],
        }

    # Fetch to get the diff
    _git("fetch", "origin", "main")
    behind = get_commits_behind()

    # Get changelog
    rc, out, _ = _git("log", "--oneline", f"HEAD..origin/main")
    changelog = [line.strip() for line in out.splitlines() if line.strip()]

    return {
        "current_sha": current[:8],
        "latest_sha": latest[:8],
        "update_available": True,
        "commits_behind": behind,
        "changelog": changelog[:20],
    }


def apply_update() -> dict:
    """Apply the update by pulling from origin/main.

    Returns dict with success status and message.
    """
    # Stash local changes
    _git("stash")

    # Pull
    rc, out, err = _git("pull", "origin", "main")
    if rc != 0:
        # Restore stash
        _git("stash", "pop")
        return {"success": False, "message": f"Pull failed: {err}"}

    # Pop stash (may have conflicts)
    rc2, out2, err2 = _git("stash", "pop")
    if rc2 != 0:
        return {
            "success": True,
            "message": f"Updated successfully, but stash pop had conflicts: {err2}. "
                       f"Resolve manually with 'git stash pop'.",
        }

    new_sha = get_current_sha()
    return {
        "success": True,
        "message": f"Updated to {new_sha[:8]}. Restart the server to apply changes.",
        "new_sha": new_sha[:8],
    }
