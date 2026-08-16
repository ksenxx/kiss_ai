# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for Fixer-8 findings (real repos, no mocks).

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.server.test_fixer8_merge_config_bugs``; the non-core tests remain there.
"""

from __future__ import annotations

from kiss.core.vscode_config import DEFAULTS, sanitize_config


class TestSanitizeConfigBooleanBudget:
    def test_true_budget_falls_back_to_default(self) -> None:
        out = sanitize_config({"max_budget": True})
        assert out["max_budget"] == DEFAULTS["max_budget"]

    def test_false_budget_falls_back_to_default(self) -> None:
        out = sanitize_config({"max_budget": False})
        assert out["max_budget"] == DEFAULTS["max_budget"]

    def test_finite_numbers_still_accepted(self) -> None:
        assert sanitize_config({"max_budget": 55})["max_budget"] == 55
        assert sanitize_config({"max_budget": 55.5})["max_budget"] == 55.5
        assert sanitize_config({"max_budget": "42"})["max_budget"] == 42.0

    def test_bool_defaults_still_coerce_truthy(self) -> None:
        bool_keys = [k for k, v in DEFAULTS.items() if isinstance(v, bool)]
        for key in bool_keys:
            assert sanitize_config({key: 1})[key] is True
            assert sanitize_config({key: 0})[key] is False
