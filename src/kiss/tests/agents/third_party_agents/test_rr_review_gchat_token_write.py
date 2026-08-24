# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Google Chat's atomic OAuth-token persistence.

Review finding: ``googlechat_agent.py`` still wrote the OAuth token with
``write_text`` + ``chmod`` — the exact truncate-then-chmod pattern the
audit replaced for gmail — exposing a brief world-readable window and
torn reads to concurrent processes.  Both token-write sites now
delegate to the shared ``_save_token`` helper built on
``_channel_agent_utils.write_private_file`` (mkstemp 0600 +
``os.replace``), mirroring gmail's ``_save_credentials``.

No mocks or test doubles: real ``google.oauth2`` ``Credentials``
objects, real files, real reader/writer threads.  The refresh and
OAuth-flow call sites cannot be driven end-to-end without Google's
real token endpoint (``Credentials.from_authorized_user_file`` always
overrides ``token_uri`` with ``oauth2.googleapis.com``), so the shared
persistence helper both sites delegate to is exercised directly, the
same way the suite tests gmail's ``_save_credentials``.
"""

from __future__ import annotations

import json
import stat
import sys
import threading
from pathlib import Path

import pytest

_IS_POSIX = sys.platform != "win32"


def _mode(path: Path) -> int:
    """Return the permission bits of *path*."""
    return stat.S_IMODE(path.stat().st_mode)


def _make_creds(token: str) -> object:
    """Build a real google Credentials object carrying *token*."""
    from google.oauth2.credentials import Credentials

    return Credentials(
        token=token,
        refresh_token="refresh-token",
        token_uri="https://oauth2.googleapis.com/token",
        client_id="cid",
        client_secret="csecret",
    )


class TestGoogleChatSaveToken:
    """_save_token persists the OAuth token atomically at 0600."""

    def test_token_saved_atomically_with_0600(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real google Credentials object persists to a private file."""
        monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss_home"))
        from kiss.agents.third_party_agents.googlechat_agent import (
            _save_token,
            _token_path,
        )

        _save_token(_make_creds("access-token"))
        token_file = _token_path()
        data = json.loads(token_file.read_text(encoding="utf-8"))
        assert data["token"] == "access-token"
        assert data["refresh_token"] == "refresh-token"
        if _IS_POSIX:  # pragma: no branch
            assert _mode(token_file) == 0o600
        assert [p.name for p in token_file.parent.iterdir()] == [token_file.name]

    def test_no_torn_read_under_concurrent_reader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hot reader never sees a torn, partial, or over-permissive file.

        With the old truncate-write a concurrent reader could observe an
        empty or half-written token (and a brief pre-chmod 0644 window);
        with atomic replace every read is one complete private payload.
        """
        monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss_home"))
        from kiss.agents.third_party_agents.googlechat_agent import (
            _save_token,
            _token_path,
        )

        _save_token(_make_creds("seed-token"))
        token_file = _token_path()
        stop = threading.Event()
        errors: list[str] = []

        def reader() -> None:
            while not stop.is_set():
                try:
                    data = json.loads(token_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:  # pragma: no cover
                    errors.append(f"torn read: {e}")
                    return
                if "refresh_token" not in data:  # pragma: no cover
                    errors.append(f"partial payload: {sorted(data)}")
                    return
                if _IS_POSIX and _mode(token_file) != 0o600:  # pragma: no cover
                    errors.append(f"bad mode {oct(_mode(token_file))}")
                    return

        def writer(idx: int) -> None:
            for i in range(100):
                _save_token(_make_creds(f"token-{idx}-{i}"))

        observer = threading.Thread(target=reader)
        writers = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        observer.start()
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        observer.join()
        assert errors == []
        assert [p.name for p in token_file.parent.iterdir()] == [token_file.name]
