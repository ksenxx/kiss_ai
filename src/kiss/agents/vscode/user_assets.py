# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resolve and lazily seed ``~/.kiss/<asset>`` markdown assets.

At runtime ``~/.kiss/<asset>`` is the source of truth for
``INJECTIONS.md`` and ``SAMPLE_TASKS.md``; the package copy bundled
alongside the source tree is the seed / fallback.

**Install-time behaviour** (``install.sh`` and
``DependencyInstaller.ts``): on every install or version upgrade both
files are *always* copied from the package into ``~/.kiss/``,
overwriting any prior copy.  This ensures the latest bundled Markdown
is served immediately after an update — matching the ``MODEL_INFO.json``
pattern.

**Runtime behaviour** (this module): if ``~/.kiss/<asset>`` exists
return it as-is; user edits made *between* installs survive every
daemon restart and ``uv run`` invocation.  If the user copy is missing
(e.g. first run in a sandboxed test environment that skipped the
installer), seed it from the package copy and return the new path.
When ``~/.kiss/`` is not writable (read-only FS, missing ``HOME``)
return the package copy directly so the caller still sees a valid file.

``KISS_HOME`` overrides the default ``~/.kiss`` location, matching the
rest of the kiss codebase (``persistence.py``, ``web_server.py``,
``vscode_config.py``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def kiss_home_dir() -> Path:
    """Return ``~/.kiss/`` (or ``$KISS_HOME`` when set)."""
    return Path(os.environ.get("KISS_HOME") or (Path.home() / ".kiss"))


def ensure_user_asset(name: str, package_path: Path) -> Path:
    """Return the path the runtime should read ``name`` from.

    Args:
        name: Asset file name (e.g. ``"INJECTIONS.md"``).  The user
            copy lives at ``~/.kiss/<name>``.
        package_path: Path to the package copy bundled in the source
            tree, used as the seed for the user copy and as the
            fallback when ``~/.kiss/`` is not writable.

    Returns:
        Path to the file the caller should read.  Prefers the user
        copy at ``~/.kiss/<name>``; falls back to ``package_path``
        when the user copy cannot be created (read-only ``HOME``,
        missing parent, etc.) or when neither file exists.
    """
    user_path = kiss_home_dir() / name
    try:
        if user_path.exists():
            # User copy wins unconditionally; user edits are never
            # silently clobbered by a newer package copy.  To pull in a
            # fresh bundled copy the user removes ``~/.kiss/<name>``.
            return user_path
        if package_path.exists():
            user_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package_path, user_path)
            return user_path
    except OSError:
        pass
    return package_path
