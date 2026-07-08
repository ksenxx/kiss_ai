# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resolve and lazily seed ``~/.kiss/<asset>`` markdown assets.

:func:`ensure_user_asset_from_default` seeds a user copy from an
**inline string default**, with no package file involved.  Used for
``MY_TASK_TEMPLATES.md`` (welcome-screen chips) and
``MY_INJECTION.md`` (Inject instruction panel), both purely
user-curated files whose only "bundled" content is a tiny inline
starter (``## Task\\n\\nHi!\\n`` and a ``## Trick`` test-first
starter, respectively).  Returns ``None`` when ``~/.kiss/`` is not
writable so the caller can skip silently.

``KISS_HOME`` overrides the default ``~/.kiss`` location, matching the
rest of the kiss codebase (``persistence.py``, ``web_server.py``,
``vscode_config.py``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def kiss_home_dir() -> Path:
    """Return ``~/.kiss/`` (or ``$KISS_HOME`` when set)."""
    return Path(os.environ.get("KISS_HOME") or (Path.home() / ".kiss"))


def ensure_user_asset_from_default(
    name: str, default_content: str,
) -> Path | None:
    """Return ``~/.kiss/<name>``, seeding it with ``default_content`` if absent.

    Used for assets like ``MY_TASK_TEMPLATES.md`` whose source of
    truth is the user's local copy — there is no bundled package
    file, only a tiny inline default written on first read.

    The seed is **atomic and non-clobbering**: the default is staged
    in a sibling temp file and hard-linked into place, so a concurrent
    reader (e.g. the autocomplete worker calling ``read_tricks`` while
    a command-handler thread seeds ``MY_INJECTION.md``) never observes
    an empty or partially-written file — a plain ``write_text`` seed
    truncates first and exposed a torn read.  If the file appears
    between the existence check and the link (a concurrent seeder or a
    user edit), the existing file wins.

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
        fd, tmp = tempfile.mkstemp(
            prefix=f".{name}-", dir=str(user_path.parent),
        )
        try:
            try:
                os.write(fd, default_content.encode("utf-8"))
            finally:
                os.close(fd)
            # ``link`` (not ``replace``) is atomic AND refuses to
            # clobber: it fails with FileExistsError when a concurrent
            # seeder or user edit created the file after the existence
            # check above, preserving the never-overwrite contract.
            os.link(tmp, user_path)
        except FileExistsError:
            return user_path
        finally:
            Path(tmp).unlink(missing_ok=True)
        return user_path
    except OSError:
        return None
