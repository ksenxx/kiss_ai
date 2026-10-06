# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Stash a channel's on-disk config for the duration of a test.

Channel backends persist their credentials in ``ChannelConfig`` JSON
files under the user's KISS home.  Tests that exercise ``connect()`` /
``clear_*_auth`` against a local emulator must start from a clean file
and must put the developer's real config back afterwards, even when
setup fails half-way.  ``config_backup`` is the one implementation of
that contract; use it as ``with config_backup(_config.path): ...`` in a
pytest fixture or test body, or ``self.enterContext(config_backup(...))``
in a ``unittest.TestCase.setUp``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def config_backup(path: Path) -> Iterator[None]:
    """Remove *path* while the block runs, then restore its original state.

    The original contents and file mode are held in memory; on exit the
    file is written back with that mode (creating parent directories) or,
    if it did not exist before, removed again so nothing the test saved
    leaks into the user's config.

    Args:
        path: The ``ChannelConfig.path`` of the channel under test.
    """
    exists = path.exists()
    original = path.read_text() if exists else None
    mode = path.stat().st_mode & 0o777 if exists else 0
    path.unlink(missing_ok=True)
    try:
        yield
    finally:
        if original is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original)
            path.chmod(mode)
        else:
            path.unlink(missing_ok=True)
