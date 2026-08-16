# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave3-Fixer-4 findings (D1/D2/D3/D5).

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_wave3_voice_config_bugs``; the non-core tests remain there.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import kiss.core.vscode_config as vc


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
            "false\n"
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


class TestD3TempFileLeaks:
    def test_save_config_replace_failure_leaves_no_temp(self) -> None:
        """A failing ``os.replace`` must unlink the ``.config.json-*`` file."""
        cfg_dir = vc.CONFIG_DIR
        cfg_path = vc.CONFIG_PATH
        original = cfg_path.read_bytes() if cfg_path.is_file() else None
        try:
            if cfg_path.is_file():
                cfg_path.unlink()
            cfg_path.mkdir()
            (cfg_path / "child").write_text("occupied")
            with pytest.raises(OSError):
                vc.save_config({"work_dir": "/wave3-d3"})
            leftovers = sorted(
                p.name
                for p in cfg_dir.iterdir()
                if p.name.startswith((".kiss-config-", f".{cfg_path.name}-"))
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
