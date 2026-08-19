# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for CodexModel — Codex CLI backend."""

import shutil

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models import codex_model as codex_module
from kiss.core.models.codex_model import (
    CodexModel,
    _find_codex_cli,
)
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_codex_model import (  # noqa: F401
    _has_codex,
    requires_codex_cli,
)


class TestFindCodexCli:

    def test_find_codex_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setattr(codex_module, "_UI_CANDIDATE_PATHS", ())
        with pytest.raises(KISSError, match="not found"):
            _find_codex_cli()


class TestUnsupportedMethods:
    def test_get_embedding_raises(self) -> None:
        m = CodexModel("codex/default")
        with pytest.raises(KISSError, match="does not support embeddings"):
            m.get_embedding("test")


@requires_codex_cli
@pytest.mark.slow
@pytest.mark.live_cli
class TestGenerateIntegration:
    """Integration tests that actually call the codex CLI."""

    @pytest.mark.timeout(120)
    def test_generate_failure_raises(self) -> None:
        m = CodexModel("codex/this-model-does-not-exist-xyz")
        m.initialize("hi")
        with pytest.raises(KISSError, match="Codex CLI failed"):
            m.generate()
