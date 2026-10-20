# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Two overlapping ``ensure_password()`` calls must both return the stored password.

``ensure_password`` used to read ``config.json`` and, finding no
password, draw a fresh one and ``save_config`` it; the read and the
write were separate lock sections, so two runs that overlapped (two
``install.sh`` runs, or the daemon's first start beside a manual
``python -m kiss.core.remote_password``) both saw "none stored", both
drew a password, and the loser's was shown to the user but never
stored.  The read-check-write now happens in one config-lock section
(``setdefault_config``).

The window is pinned with :class:`Pause`: one caller is held at the
entry of ``generate_password`` — after its read, before its write —
while the other completes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from kiss.core import remote_password, user_config
from kiss.core.remote_password import ensure_password, stored_password
from kiss.tests.server.pause import Pause


@pytest.fixture(autouse=True)
def _own_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(vars(user_config), "CONFIG_DIR", tmp_path / "kiss")
    monkeypatch.setitem(vars(user_config), "CONFIG_PATH", tmp_path / "kiss" / "config.json")


def _any_call(_frame_locals: dict[str, Any]) -> bool:
    return True


def _ensure_into(results: list[str]) -> None:
    results.append(ensure_password())


def test_overlapping_first_runs_agree_with_the_store() -> None:
    results: list[str] = []
    held = threading.Thread(target=_ensure_into, args=(results,), daemon=True)
    with Pause(remote_password.generate_password.__code__, _any_call) as pause:
        held.start()
        assert pause.paused.wait(10), "the held caller never reached generate_password"
        other = ensure_password()
        pause.release()
        held.join(10)
    assert not held.is_alive()
    assert results == [stored_password()]
    assert other == stored_password()


def test_stored_password_is_kept_and_rotate_replaces_it() -> None:
    first = ensure_password()
    assert ensure_password() == first == stored_password()
    rotated = ensure_password(rotate=True)
    assert rotated != first
    assert rotated == stored_password()


@pytest.mark.parametrize("junk", [123, True, [1, 2], {"a": 1}, 1.5])
def test_a_junk_typed_stored_password_is_replaced(junk: Any) -> None:
    """A hand-edited non-string password is "none stored" for the daemon.

    ``stored_password`` reads through ``load_config``, which sanitizes a
    non-string ``remote_password`` to ``""``; so a first-run
    ``ensure_password`` must store a real one rather than keep (and
    report) a value no login could ever match.
    """
    user_config.CONFIG_DIR.mkdir(parents=True)
    user_config.CONFIG_PATH.write_text(
        json.dumps({"remote_password": junk, "email": "me@example.org"}), encoding="utf-8"
    )
    password = ensure_password()
    assert password == stored_password()
    assert len(password) == 20
    stored = json.loads(user_config.CONFIG_PATH.read_text(encoding="utf-8"))
    assert stored["remote_password"] == password
    assert stored["email"] == "me@example.org"


def test_setdefault_config_keeps_a_stored_value_the_daemon_reads() -> None:
    """A string password is kept as is; a numeric string for a numeric key counts as set."""
    user_config.CONFIG_DIR.mkdir(parents=True)
    user_config.CONFIG_PATH.write_text(
        json.dumps({"remote_password": "keep-me", "max_budget": "7"}), encoding="utf-8"
    )
    stored = user_config.setdefault_config({"remote_password": "new", "max_budget": 3})
    assert stored["remote_password"] == "keep-me"
    assert stored["max_budget"] == "7"
    assert user_config.load_config()["max_budget"] == 7.0
