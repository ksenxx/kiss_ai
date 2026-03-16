"""Ensure offline-installer paths are on PATH.

When KISS is installed via the offline .pkg installer, key binaries (uv, git,
code-server) live under ``~/.kiss-install/bin`` and ``~/.local/bin``.  The
installer adds a ``source`` line to the user's shell rc file, but processes
started before that (or non-login shells) may not have those directories on
PATH.  Importing this module early in startup fixes that.
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_path() -> None:
    """Prepend offline-installer bin dirs to PATH if they exist and are missing."""
    home = Path.home()
    extra_dirs = [
        str(home / ".kiss-install" / "bin"),
        str(home / ".local" / "bin"),
    ]
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep)
    prepend = [d for d in extra_dirs if d not in parts and Path(d).is_dir()]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + parts)


ensure_path()
