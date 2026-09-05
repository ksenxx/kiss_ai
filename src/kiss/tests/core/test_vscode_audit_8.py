# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for vscode agent audit round 8: redundancies, inconsistencies, bugs.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_vscode_audit_8``; the non-core tests remain there.
"""

from __future__ import annotations

import pytest


class TestAPIKeyEnvVarsType:
    """API_KEY_ENV_VARS must be a frozenset of key names, not a dict."""

    def test_api_key_env_vars_is_frozenset(self) -> None:
        """Verify API_KEY_ENV_VARS is a frozenset, not a dict."""
        from kiss.core.vscode_config import API_KEY_ENV_VARS

        assert isinstance(API_KEY_ENV_VARS, frozenset), (
            f"API_KEY_ENV_VARS should be frozenset, got {type(API_KEY_ENV_VARS).__name__}"
        )

    def test_api_key_env_vars_contains_expected_keys(self) -> None:
        """Verify all expected API key names are present."""
        from kiss.core.vscode_config import API_KEY_ENV_VARS

        expected = {
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_WORKSPACE_ID",
            "TOGETHER_API_KEY",
            "OPENROUTER_API_KEY",
            "ZAI_API_KEY",
            "MOONSHOT_API_KEY",
        }
        assert API_KEY_ENV_VARS == expected

    def test_get_current_api_keys_works_with_frozenset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify get_current_api_keys iterates correctly over frozenset."""
        from kiss.core import vscode_config

        monkeypatch.setattr(vscode_config, "API_KEY_ENV_VARS", frozenset({"A", "B"}))
        monkeypatch.setenv("A", "val_a")
        monkeypatch.delenv("B", raising=False)
        result = vscode_config.get_current_api_keys()
        assert result == {"A": "val_a", "B": ""}

    def test_loader_membership_check_works_with_frozenset(self) -> None:
        """Verify the key-migration membership check works with frozenset."""
        from kiss.core.vscode_config import API_KEY_ENV_VARS

        assert "GEMINI_API_KEY" in API_KEY_ENV_VARS
        assert "NONEXISTENT_KEY" not in API_KEY_ENV_VARS
