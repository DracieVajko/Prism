"""Lenswright unified server library (formerly Prism).

Runs the Prism -> Lenswright environment alias exactly once, at package import
time, before any submodule reads its configuration. Remove in 0.9.0.

The literal ``PRISM_`` prefix below is the whole point of this module: it is the
one place in the codebase that must keep the old name, otherwise existing
``settings.local.json`` files, launchers and systemd units lose their settings.
"""
from __future__ import annotations

import os as _os

_OLD_PREFIX = "PRISM_"
_NEW_PREFIX = "LENSWRIGHT_"

for _old_key, _old_value in list(_os.environ.items()):
    if _old_key.startswith(_OLD_PREFIX):
        _new_key = _NEW_PREFIX + _old_key[len(_OLD_PREFIX):]
        if _new_key not in _os.environ:
            _os.environ[_new_key] = _old_value

del _old_key, _old_value, _os
