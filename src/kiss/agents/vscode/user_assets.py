# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resolve and lazily seed ``~/.kiss/<asset>`` markdown assets.

At runtime ``~/.kiss/<asset>`` is the source of truth for user-editable
assets (currently ``INJECTIONS.md`` and ``SAMPLE_TASKS.md``); the
package copy bundled alongside the source tree is treated as a seed.

``install.sh`` eagerly copies the package copy into ``~/.kiss/`` on a
fresh install (``cp -n``, no-clobber) so a brand-new user immediately
sees the bundled content.  This module provides the matching lazy
seed path so callers that run before/around the installer (tests,
development checkouts, sandboxed environments) still observe a valid
file without paying the cost of an extra launch step.

Behaviour:

* If the user copy at ``~/.kiss/<asset>`` exists, **always** return
  it — user edits survive every read.  This is a deliberate departure
  from :func:`kiss.core.models.model_info._ensure_user_model_info_path`,
  which refreshes ``MODEL_INFO.json`` when the package copy is newer:
  ``MODEL_INFO.json`` is auto-generated data, whereas
  ``INJECTIONS.md`` / ``SAMPLE_TASKS.md`` are user-editable Markdown,
  and silently clobbering user edits on every ``git pull`` would be a
  hostile surprise.
* If the user copy is missing, seed it from the package copy.
* When ``~/.kiss/`` is not writable (read-only FS, sandboxed test envs,
  missing ``HOME``) return the package copy directly so the caller
  still sees a valid file.

``KISS_HOME`` overrides the default ``~/.kiss`` location, matching the
rest of the kiss codebase (``persistence.py``, ``web_server.py``,
``vscode_config.py``).

To regenerate from the bundled defaults, the user simply removes the
file under ``~/.kiss/`` — the next read seeds it again.
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
