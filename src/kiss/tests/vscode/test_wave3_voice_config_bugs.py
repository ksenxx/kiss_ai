# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave3-Fixer-4 findings (D1/D2/D3/D5).

Covers, over REAL objects (no mocks, patches, or fakes):

* D1: ``voice_wake._ensure_downloaded_model`` used
  ``urllib.request.urlretrieve`` with NO network timeout while holding
  the exclusive cross-process ``flock`` — a stalled download wedged the
  FIFO translation worker forever and blocked every other process on
  the same lock.  Exercised with a real local HTTP server that sends
  headers plus a partial body and then stalls: the download must fail
  within the (env-overridden) timeout, leave no temp files behind, and
  release the flock.  A companion test downloads a real zip from a
  local server to prove the chunked-copy replacement still works.
* D2: ``vscode_config.source_shell_env`` gated the env dump on the RC
  file's exit status in the bash/zsh branch (``source rc && { env… }``)
  — an RC whose LAST command exits nonzero (e.g. a failing ``which`` or
  a false ``[ -f x ] && …`` test) silently imported ZERO keys, while
  the fish branch dumped unconditionally.  Exercised with a real
  ``/bin/bash`` and a temp ``$HOME`` whose ``.bashrc`` exports a key
  and then ends with ``false``.
* D3: ``vscode_config.save_config`` and ``_atomic_write_text_secure``
  leaked their ``mkstemp`` staging files (``.kiss-config-*`` in
  ``~/.kiss/``, ``.kiss-rc-*`` in ``$HOME``) whenever the write or the
  ``os.replace`` raised.  Exercised by making the destination a
  directory so the real ``os.replace`` fails.
* D5: ``SpeakerIdentifier`` loaded a second full copy of the ~40MB
  Vosk wake model although ``WakeDetector`` in the same process
  already held one.  Exercised with the real offline Vosk models:
  after constructing both, the process must hold exactly ONE shared
  acoustic ``Model`` instance, and both recognizers must still work.
"""

from __future__ import annotations

import fcntl
import http.server
import io
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

import kiss.server.voice_wake as voice_wake
import kiss.server.vscode_config as vc

# ---------------------------------------------------------------------------
# D1 — model download must have a hard network timeout
# ---------------------------------------------------------------------------

# How long the stalling server keeps the connection open.  The fixed
# download (1s env-overridden timeout) must give up long before this;
# the unfixed ``urlretrieve`` blocked for the full window.
_STALL_WINDOW_SECONDS = 30.0


class _StallingHandler(http.server.BaseHTTPRequestHandler):
    """Sends headers plus a partial body, then stalls the connection."""

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(1024 * 1024))
        self.end_headers()
        self.wfile.write(b"PK\x03\x04")  # partial body, then stall
        self.wfile.flush()
        cast(Any, self.server).release.wait(_STALL_WINDOW_SECONDS)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


class _ZipHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``server.payload`` (a complete zip archive)."""

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        payload = cast(Any, self.server).payload
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def _start_local_server(
    handler: type[http.server.BaseHTTPRequestHandler],
) -> http.server.ThreadingHTTPServer:
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _non_lock_leftovers(models_dir: Path) -> list[str]:
    """Names of files in *models_dir* other than flock sidecar files."""
    return sorted(
        p.name for p in models_dir.iterdir() if not p.name.endswith(".lock")
    )


class TestD1DownloadTimeout:
    def test_stalled_download_times_out_and_releases_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stalled server must not wedge the download (or the flock)."""
        monkeypatch.setenv("KISS_VOICE_DOWNLOAD_TIMEOUT", "1")
        httpd = _start_local_server(_StallingHandler)
        cast(Any, httpd).release = threading.Event()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/m.zip"
            models_dir = tmp_path / "models"
            start = time.monotonic()
            with pytest.raises(OSError):
                voice_wake._ensure_downloaded_model(
                    models_dir, "stall-model", url,
                )
            elapsed = time.monotonic() - start
            assert elapsed < 10.0, (
                f"download must fail within the hard timeout, took "
                f"{elapsed:.1f}s (no network timeout?)"
            )
            # No staging temp files may be left behind.
            assert _non_lock_leftovers(models_dir) == []
            # The exclusive flock must have been released so other
            # processes/threads are not stranded behind the failure.
            with open(models_dir / ".stall-model.lock", "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        finally:
            cast(Any, httpd).release.set()
            httpd.shutdown()

    def test_download_and_extract_still_work(self, tmp_path: Path) -> None:
        """The chunked-copy download must publish an intact model dir."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("ok-model/README", "wave3-d1 payload")
            zf.writestr("ok-model/am/final.mdl", "x" * 4096)
        httpd = _start_local_server(_ZipHandler)
        cast(Any, httpd).payload = buf.getvalue()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/ok.zip"
            models_dir = tmp_path / "models"
            model_dir = voice_wake._ensure_downloaded_model(
                models_dir, "ok-model", url,
            )
            assert model_dir == models_dir / "ok-model"
            assert (model_dir / "README").read_text() == "wave3-d1 payload"
            assert (model_dir / "am" / "final.mdl").read_text() == "x" * 4096
            assert _non_lock_leftovers(models_dir) == ["ok-model"]
        finally:
            httpd.shutdown()


# ---------------------------------------------------------------------------
# D2 — source_shell_env must import keys even when the RC exits nonzero
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("/bin/bash").exists(), reason="requires /bin/bash",
)
class TestD2SourceShellEnvRcExitStatus:
    def test_rc_ending_with_failing_command_still_imports_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``source`` returning nonzero must not skip the env import."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bashrc").write_text(
            "export OPENAI_API_KEY=wave3-d2-imported\n"
            "false\n"  # a perfectly normal RC whose last command fails
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        vc.source_shell_env()

        assert os.environ.get("OPENAI_API_KEY") == "wave3-d2-imported"

    def test_rc_ending_with_succeeding_command_imports_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression guard: the happy path keeps working."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bashrc").write_text(
            "export TOGETHER_API_KEY=wave3-d2-happy\n"
            "true\n"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

        vc.source_shell_env()

        assert os.environ.get("TOGETHER_API_KEY") == "wave3-d2-happy"


# ---------------------------------------------------------------------------
# D3 — staged temp files must not leak on error paths
# ---------------------------------------------------------------------------


class TestD3TempFileLeaks:
    def test_save_config_replace_failure_leaves_no_temp(self) -> None:
        """A failing ``os.replace`` must unlink the ``.kiss-config-*`` file."""
        cfg_dir = vc.CONFIG_DIR
        cfg_path = vc.CONFIG_PATH
        original = cfg_path.read_bytes() if cfg_path.is_file() else None
        try:
            if cfg_path.is_file():
                cfg_path.unlink()
            # A directory at the destination makes the real os.replace
            # raise IsADirectoryError after the temp file was staged.
            cfg_path.mkdir()
            (cfg_path / "child").write_text("occupied")
            with pytest.raises(OSError):
                vc.save_config({"work_dir": "/wave3-d3"})
            leftovers = sorted(
                p.name
                for p in cfg_dir.iterdir()
                if p.name.startswith(".kiss-config-")
            )
            assert leftovers == [], (
                f"staging temp files leaked into {cfg_dir}: {leftovers}"
            )
        finally:
            shutil.rmtree(cfg_path, ignore_errors=True)
            if original is not None:
                cfg_path.write_bytes(original)

    def test_atomic_write_text_secure_failure_leaves_no_temp(
        self, tmp_path: Path,
    ) -> None:
        """A failing ``os.replace`` must unlink the ``.kiss-rc-*`` file."""
        target = tmp_path / "rc"
        target.mkdir()
        (target / "child").write_text("occupied")
        with pytest.raises(OSError):
            vc._atomic_write_text_secure(target, "export KEY=value\n")
        leftovers = sorted(
            p.name
            for p in tmp_path.iterdir()
            if p.name.startswith(".kiss-rc-")
        )
        assert leftovers == [], (
            f"staging temp files leaked into {tmp_path}: {leftovers}"
        )


# ---------------------------------------------------------------------------
# D5 — one shared Vosk acoustic model per process
# ---------------------------------------------------------------------------

_WAKE_DIR = voice_wake.DEFAULT_MODELS_DIR / voice_wake.MODEL_NAME
_SPK_DIR = voice_wake.DEFAULT_MODELS_DIR / voice_wake.SPK_MODEL_NAME


@pytest.mark.skipif(
    not (_WAKE_DIR.is_dir() and _SPK_DIR.is_dir()),
    reason="offline Vosk models not present under ~/.kiss/models",
)
class TestD5SharedVoskModel:
    def test_detector_and_identifier_share_one_acoustic_model(self) -> None:
        """Both recognizers must reuse ONE ``vosk.Model`` instance."""
        pytest.importorskip("vosk")
        voice_wake._VOSK_MODEL_CACHE.clear()

        detector = voice_wake.WakeDetector(_WAKE_DIR)
        assert len(voice_wake._VOSK_MODEL_CACHE) == 1
        shared = voice_wake.load_shared_vosk_model(_WAKE_DIR)

        identifier = voice_wake.SpeakerIdentifier(
            voice_wake.DEFAULT_MODELS_DIR
        )
        # Constructing the identifier must NOT load a second copy of
        # the wake acoustic model.
        assert len(voice_wake._VOSK_MODEL_CACHE) == 1
        assert voice_wake.load_shared_vosk_model(_WAKE_DIR) is shared

        # Both recognizers must keep working off the shared model.
        silence = b"\x00\x00" * voice_wake.BLOCK_SIZE
        assert detector.feed(silence) is False
        result = identifier.speaker_of(silence * 8)
        assert result is None or isinstance(result, int)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
