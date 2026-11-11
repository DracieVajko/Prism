"""Shared helpers for the optional, non-Whisper ASR engines.

Both alternative engines keep their weights under ``pc/models`` and need the
same two checks (is the directory complete, is there enough RAM). These helpers
live here so the engine modules do not carry their own copies.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("asr_common")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_dir(subdir: Optional[str], default: str) -> Path:
    """Directory for *subdir* (or *default* when the id is empty)."""
    return MODELS_DIR / (subdir or default)


def total_ram_gb() -> float:
    """Installed RAM in GB, or 0.0 when psutil is unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception as exc:  # psutil is optional at runtime
        logger.debug("could not read total RAM: %s", exc)
        return 0.0


def ram_allows(needed_gb: float) -> tuple[bool, float]:
    """Return (fits, installed_gb); a missing psutil never blocks a load."""
    have = total_ram_gb()
    return (have <= 0 or have + 0.5 >= needed_gb), have
