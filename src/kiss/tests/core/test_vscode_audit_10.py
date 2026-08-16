# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Default and fast model names must be obtainable dynamically.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.agents.vscode.test_vscode_audit_10``, whose remaining
content was dead code (an unused fixture and helpers scanning the
VS Code extension sources).
"""

from __future__ import annotations

from kiss.core.models.model_info import get_default_model, get_fast_model


class TestNoHardcodedModelsAnywhere:
    """The Python model-name helpers must return non-empty known models."""

    def test_python_get_default_model_returns_known_model(self) -> None:
        result = get_default_model()
        assert result, "get_default_model() returned empty string"

    def test_python_get_fast_model_returns_known_model(self) -> None:
        result = get_fast_model()
        assert result, "get_fast_model() returned empty string"
