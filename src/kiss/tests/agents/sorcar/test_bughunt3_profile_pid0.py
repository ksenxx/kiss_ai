# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: ``_is_profile_in_use`` with a non-positive PID in the lock.

A corrupt/stale ``SingletonLock`` symlink whose target ends in ``-0``
made ``_is_profile_in_use`` call ``os.kill(0, 0)``, which signals the
CALLER'S OWN process group and therefore always succeeds — so the
profile was treated as permanently locked.  ``_resolve_user_data_dir``
then escalated to ``<dir>_1``, ``<dir>_2``, … forever, silently losing
the user's logged-in browser profile.  Same for negative PIDs
(``os.kill(-N, 0)`` signals process group N).
"""

import sys

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool, _is_profile_in_use

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="SingletonLock symlinks are POSIX-only"
)


def _make_lock(profile_dir, pid) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "SingletonLock").symlink_to(f"somehost-{pid}")


def test_pid_zero_lock_is_not_in_use(tmp_path) -> None:
    """A lock naming PID 0 is corrupt — never a live Chromium."""
    profile = tmp_path / "profile0"
    _make_lock(profile, 0)
    assert _is_profile_in_use(str(profile)) is False


def test_negative_pid_lock_is_not_in_use(tmp_path) -> None:
    """A lock target like 'host--5' yields PID 5 after rsplit — but a raw
    negative-pid form must never mark the profile in use either."""
    profile = tmp_path / "profile_neg"
    profile.mkdir(parents=True, exist_ok=True)
    # rsplit('-', 1) on 'somehost--7' gives '7'; craft a target whose
    # final field parses to a non-positive int directly.
    (profile / "SingletonLock").symlink_to("-0")
    assert _is_profile_in_use(str(profile)) is False


def test_resolve_user_data_dir_reuses_profile_with_pid0_lock(tmp_path) -> None:
    """The primary profile dir must be reused (not escalated to _1)."""
    profile = tmp_path / "profile_main"
    _make_lock(profile, 0)
    tool = WebUseTool(user_data_dir=str(profile))
    try:
        assert tool._resolve_user_data_dir() == str(profile)
    finally:
        tool.close()
