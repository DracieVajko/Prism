"""Lenswright unified server library (formerly Prism).

Runs the Prism -> Lenswright environment alias exactly once, at package import
time, before any submodule reads its configuration. Remove in 0.9.0.
"""
from __future__ import annotations

import os as _os

for _old_key, _old_value in list(_os.environ.items()):
    if _old_key.startswith("PRISM_"):
        _new_key = "LENSWRIGHT_" + _old_key[len("PRISM_"):]
        if _new_key not in _os.environ:
            _os.environ[_new_key] = _old_value

del _old_key, _old_value, _os
