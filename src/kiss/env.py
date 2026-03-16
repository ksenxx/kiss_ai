"""Ensure offline-installer paths and env vars are configured.

When KISS is installed via the offline .pkg installer, key binaries (uv, git,
code-server) live under ``<install_dir>/bin`` and ``~/.local/bin``.  Python
lives under ``<install_dir>/python/`` and Playwright browsers under
``<install_dir>/playwright-browsers/``.  The install directory is resolved
from (in order): the ``KISS_INSTALL_DIR`` env var, the ``~/.kiss/install_dir``
marker file written by the installer, or the default ``~/kiss_ai``.

The installer adds a ``source`` line to the user's shell rc file, but
processes started before that (or non-login shells) may not have those
directories on PATH or the env vars set.
Importing this module early in startup fixes that.
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKER_FILE = ".kiss/install_dir"


def get_install_dir() -> Path:
    """Return the KISS offline-installer directory.

    Resolution order:
      1. ``KISS_INSTALL_DIR`` environment variable
      2. ``~/.kiss/install_dir`` marker file (written by the installer)
      3. ``~/kiss_ai`` (default, backward-compatible)
    """
    env_val = os.environ.get("KISS_INSTALL_DIR")
    if env_val:
        return Path(env_val)
    marker = Path.home() / _MARKER_FILE
    if marker.is_file():
        try:
            text = marker.read_text().strip()
            if text:
                return Path(text)
        except OSError:
            pass
    return Path.home() / "kiss_ai"


def ensure_path() -> None:
    """Prepend offline-installer bin dirs to PATH and set env vars if needed."""
    home = Path.home()
    kiss_ai = get_install_dir()
    extra_dirs = [
        str(kiss_ai / "bin"),
        str(home / ".local" / "bin"),
    ]
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep)
    prepend = [d for d in extra_dirs if d not in parts and Path(d).is_dir()]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + parts)

    # Set UV_PYTHON_INSTALL_DIR so uv finds Python in <install_dir>/python/
    python_dir = kiss_ai / "python"
    if python_dir.is_dir() and not os.environ.get("UV_PYTHON_INSTALL_DIR"):
        os.environ["UV_PYTHON_INSTALL_DIR"] = str(python_dir)

    # Set PLAYWRIGHT_BROWSERS_PATH so Playwright finds browsers
    pw_dir = kiss_ai / "playwright-browsers"
    if pw_dir.is_dir() and not os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(pw_dir)


ensure_path()
