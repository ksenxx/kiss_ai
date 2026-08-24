# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for atomic private-file persistence (G-RC5).

``save_json_config`` and gmail's ``_save_credentials`` used to
truncate-write the destination and chmod afterwards, exposing torn
reads to concurrent processes and a brief 0644 window on secrets.
Both now delegate to ``write_private_file`` (mkstemp 0600 +
``os.replace``), the same pattern ``save_channel_state`` already used.

No mocks or test doubles: real files, real threads, real
``google.oauth2`` credentials objects.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
from pathlib import Path

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import (
    ChannelConfig,
    load_json_config,
    save_json_config,
    write_private_file,
)

_IS_POSIX = sys.platform != "win32"


def _mode(path: Path) -> int:
    """Return the permission bits of *path*."""
    return stat.S_IMODE(path.stat().st_mode)


class TestWritePrivateFile:
    """The shared mkstemp+replace helper."""

    def test_writes_content_with_0600(self, tmp_path: Path) -> None:
        """Content lands intact, mode 0600, parent dirs auto-created."""
        target = tmp_path / "deep" / "nested" / "secret.json"
        write_private_file(target, '{"k": "v"}')
        assert target.read_text(encoding="utf-8") == '{"k": "v"}'
        if _IS_POSIX:  # pragma: no branch
            assert _mode(target) == 0o600

    def test_overwrite_leaves_no_temp_files(self, tmp_path: Path) -> None:
        """Rewriting the same path replaces it and cleans up temp siblings."""
        target = tmp_path / "config.json"
        write_private_file(target, "first")
        write_private_file(target, "second")
        assert target.read_text(encoding="utf-8") == "second"
        assert [p.name for p in tmp_path.iterdir()] == ["config.json"]

    def test_failed_replace_cleans_up_temp_file(self, tmp_path: Path) -> None:
        """A destination that cannot be replaced raises and unlinks the temp.

        Making the destination an existing non-empty directory forces a
        real ``os.replace`` failure — no fault injection needed.
        """
        target = tmp_path / "occupied"
        (target / "sub").mkdir(parents=True)
        with pytest.raises(OSError):
            write_private_file(target, "data")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["occupied"]

    def test_concurrent_writers_and_reader_never_torn(self, tmp_path: Path) -> None:
        """Parallel writers + a hot reader: every observed file is valid JSON.

        With the old truncate-write a reader could observe an empty or
        half-written file; with atomic replace every read sees one
        complete payload, and the file is never group/world readable.
        """
        target = tmp_path / "state.json"
        write_private_file(target, json.dumps({"writer": -1, "payload": "x" * 4096}))
        stop = threading.Event()
        errors: list[str] = []

        def writer(idx: int) -> None:
            for i in range(150):
                write_private_file(
                    target, json.dumps({"writer": idx, "i": i, "payload": "x" * 4096})
                )

        def reader() -> None:
            while not stop.is_set():
                try:
                    data = json.loads(target.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as e:  # pragma: no cover
                    errors.append(f"torn read: {e}")
                    return
                if len(data["payload"]) != 4096:  # pragma: no cover
                    errors.append(f"truncated payload: {len(data['payload'])}")
                    return
                if _IS_POSIX and _mode(target) != 0o600:  # pragma: no cover
                    errors.append(f"bad mode {oct(_mode(target))}")
                    return

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        observer = threading.Thread(target=reader)
        observer.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        observer.join()
        assert errors == []
        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


class TestSaveJsonConfig:
    """save_json_config now goes through the atomic private writer."""

    def test_round_trip_and_permissions(self, tmp_path: Path) -> None:
        """Saved config loads back identically and is owner-only."""
        path = tmp_path / "cfg" / "config.json"
        save_json_config(path, {"token": "s3cret", "host": "example.com"})
        assert load_json_config(path, ("token",)) == {
            "token": "s3cret",
            "host": "example.com",
        }
        if _IS_POSIX:  # pragma: no branch
            assert _mode(path) == 0o600
        assert [p.name for p in path.parent.iterdir()] == ["config.json"]

    def test_channel_config_facade(self, tmp_path: Path) -> None:
        """The ChannelConfig facade round-trips through the atomic writer."""
        config = ChannelConfig(tmp_path / "chan", ("api_key",))
        config.save({"api_key": "k1"})
        assert config.load() == {"api_key": "k1"}
        if _IS_POSIX:  # pragma: no branch
            assert _mode(config.path) == 0o600


class TestGmailSaveCredentials:
    """gmail _save_credentials writes the OAuth token atomically at 0600."""

    def test_token_saved_atomically_with_0600(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real google Credentials object persists to a private file."""
        monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss_home"))
        from google.oauth2.credentials import Credentials

        from kiss.agents.third_party_agents.gmail_agent import (
            _save_credentials,
            _token_path,
        )

        creds = Credentials(
            token="access-token",
            refresh_token="refresh-token",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="cid",
            client_secret="csecret",
        )
        _save_credentials(creds)
        path = _token_path()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["refresh_token"] == "refresh-token"
        if _IS_POSIX:  # pragma: no branch
            assert _mode(path) == 0o600
        assert [p.name for p in path.parent.iterdir()] == [path.name]
        assert os.environ["KISS_HOME"].startswith(str(tmp_path))
