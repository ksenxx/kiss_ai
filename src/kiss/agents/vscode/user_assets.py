# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resolve and lazily seed ``~/.kiss/<asset>`` markdown assets.

This module exposes two helpers:

* :func:`ensure_user_asset` — seeds a user copy from a **bundled
  package file**.  Used for ``INJECTIONS.md``, where the source of
  truth is the file shipped in ``src/kiss/INJECTIONS.md`` and the user
  copy at ``~/.kiss/INJECTIONS.md`` is a writable mirror.

  **Install-time** (``install.sh`` and ``DependencyInstaller.ts``): on
  every install or version upgrade ``INJECTIONS.md`` is *always*
  copied from the package into ``~/.kiss/``, overwriting any prior
  copy — matching the ``MODEL_INFO.json`` pattern.

  **Runtime** (this helper): if ``~/.kiss/<asset>`` exists return it
  as-is; user edits made *between* installs survive every daemon
  restart.  If the user copy is missing seed it from the package copy
  and return the new path.  When ``~/.kiss/`` is not writable, return
  the package copy directly so the caller still sees a valid file.

* :func:`ensure_user_asset_from_default` — seeds a user copy from an
  **inline string default**, with no package file involved.  Used for
  ``MY_TASK_TEMPLATES.md``, where the file is purely user-curated and
  the only "bundled" content is a tiny starter task (``## Task\\n\\nHi!\\n``).
  Returns ``None`` when ``~/.kiss/`` is not writable so the caller can
  skip silently.

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


def ensure_user_asset_from_default(
    name: str, default_content: str,
) -> Path | None:
    """Return ``~/.kiss/<name>``, seeding it with ``default_content`` if absent.

    Used for assets like ``MY_TASK_TEMPLATES.md`` whose source of
    truth is the user's local copy — there is no bundled package
    file, only a tiny inline default written on first read.

    Args:
        name: Asset file name (e.g. ``"MY_TASK_TEMPLATES.md"``).
        default_content: UTF-8 string written to ``~/.kiss/<name>``
            on first read.  Never overwrites an existing file.

    Returns:
        Path to ``~/.kiss/<name>`` when the file exists or was just
        created.  ``None`` when ``~/.kiss/`` is not writable (read-only
        FS, missing ``HOME``) so callers can skip silently.
    """
    user_path = kiss_home_dir() / name
    try:
        if user_path.exists():
            return user_path
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(default_content)
        return user_path
    except OSError:
        return None
