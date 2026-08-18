# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for ClaudeCodeModel — Claude Code CLI backend."""

import shutil

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel, _find_claude_cli
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401


class TestFindClaudeCli:

    def test_find_claude_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        with pytest.raises(KISSError, match="not found"):
            _find_claude_cli()


class TestUnsupportedMethods:
    def test_get_embedding_raises(self) -> None:
        m = ClaudeCodeModel("cc/opus")
        with pytest.raises(KISSError, match="does not support embeddings"):
            m.get_embedding("test")
