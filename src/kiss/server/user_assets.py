# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resolve and lazily seed ``~/.kiss/<asset>`` markdown assets.

:func:`ensure_user_asset_from_default` seeds a user copy from an
**inline string default**, with no package file involved.  Used for
``MY_INJECTION.md`` (Inject instruction panel), a purely user-curated
file whose only "bundled" content is a tiny ``## Trick`` test-first
starter.  Returns ``None`` when ``~/.kiss/`` is not
writable so the caller can skip silently.

``$KISS_HOME`` is honoured through :func:`kiss.core.config.kiss_home`.
"""

from __future__ import annotations

from pathlib import Path

from kiss.core.config import kiss_home
from kiss.core.utils import seed_file_atomically


def kiss_home_dir() -> Path:
    """Return ``~/.kiss/`` (or ``$KISS_HOME`` when set)."""
    return kiss_home()


def ensure_user_asset_from_default(
    name: str, default_content: str,
) -> Path | None:
    """Return ``~/.kiss/<name>``, seeding it with ``default_content`` if absent.

    Used for assets like ``MY_INJECTION.md`` whose source of
    truth is the user's local copy — there is no bundled package
    file, only a tiny inline default written on first read.

    The seed is :func:`kiss.core.utils.seed_file_atomically`: atomic and
    non-clobbering, so a concurrent reader (e.g. the autocomplete worker
    calling ``read_tricks`` while a command-handler thread seeds
    ``MY_INJECTION.md``) never observes an empty or partially-written
    file, and an existing file always wins.

    Args:
        name: Asset file name (e.g. ``"MY_INJECTION.md"``).
        default_content: UTF-8 string written to ``~/.kiss/<name>``
            on first read.  Never overwrites an existing file.

    Returns:
        Path to ``~/.kiss/<name>`` when the file exists or was just
        created.  ``None`` when ``~/.kiss/`` is not writable (read-only
        FS, missing ``HOME``) so callers can skip silently.
    """
    user_path = kiss_home_dir() / name
    try:
        seed_file_atomically(user_path, default_content)
    except OSError:
        return None
    return user_path
