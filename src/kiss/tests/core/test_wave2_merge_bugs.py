# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Wave2-Fixer-7 findings (real repos, no mocks).

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_wave2_merge_bugs``; the non-core tests remain there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiss.core.vscode_config import source_shell_env


@pytest.mark.skipif(
    not Path("/bin/bash").exists(), reason="requires /bin/bash",
)
class TestSourceShellEnvMultilineValues:
    def test_forged_key_inside_multiline_value_not_imported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A value containing ``\\nOPENAI_API_KEY=...`` must not be imported."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bashrc").write_text(
            'export INNOCENT="first line\n'
            'OPENAI_API_KEY=forged-by-multiline-value"\n'
            "export TOGETHER_API_KEY=real-together-key\n"
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        monkeypatch.delenv("INNOCENT", raising=False)

        source_shell_env()

        assert os.environ.get("TOGETHER_API_KEY") == "real-together-key"
        assert os.environ.get("OPENAI_API_KEY") != "forged-by-multiline-value"

    def test_multiline_api_key_value_preserved_fully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A legitimate multi-line key value must not be truncated."""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bashrc").write_text(
            'export OPENROUTER_API_KEY="part1\npart2"\n',
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        source_shell_env()

        assert os.environ.get("OPENROUTER_API_KEY") == "part1\npart2"
