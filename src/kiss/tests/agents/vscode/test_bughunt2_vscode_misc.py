# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 2 (misc): non-finite ``max_budget`` silently disables budgets.

``json.load`` accepts the non-standard literals ``NaN``, ``Infinity``
and ``-Infinity`` by default (``allow_nan=True``), and an overflowing
literal such as ``1e999`` parses to ``float('inf')``.  A hand-edited
``~/.kiss/config.json`` containing any of these produced a genuine
Python ``float`` for ``max_budget``, and two guards both waved it
through:

* ``sanitize_config``'s numeric branch only coerces values that are
  *not already* ``int``/``float`` — a real ``nan``/``inf`` float
  skipped the branch entirely, and the string forms (``"nan"``,
  ``"inf"``) were happily converted by ``float(value)``; and
* ``apply_config_to_env`` only caught ``TypeError``/``ValueError``
  from ``float(budget)`` — ``float(nan)`` raises neither.

The result: ``DEFAULT_CONFIG.max_budget`` became ``nan`` (every
``cost > max_budget`` comparison is False → budget enforcement
silently disabled, unlimited spend) or ``inf`` (same effect).

Fix: both functions now reject non-finite numbers via
``math.isfinite`` and fall back to ``DEFAULTS['max_budget']``.
"""

from __future__ import annotations

import math
import unittest

from kiss.core import config as config_module
from kiss.core import vscode_config
from kiss.core.vscode_config import (
    DEFAULTS,
    apply_config_to_env,
    load_config,
    sanitize_config,
)


class TestNonFiniteBudgetRejected(unittest.TestCase):
    """Non-finite ``max_budget`` values must fall back to the default."""

    def setUp(self) -> None:
        self._saved_budget = config_module.DEFAULT_CONFIG.max_budget
        self._saved_file = (
            vscode_config.CONFIG_PATH.read_text()
            if vscode_config.CONFIG_PATH.exists()
            else None
        )

    def tearDown(self) -> None:
        config_module.DEFAULT_CONFIG.max_budget = self._saved_budget
        if self._saved_file is None:
            vscode_config.CONFIG_PATH.unlink(missing_ok=True)
        else:
            vscode_config.CONFIG_PATH.write_text(self._saved_file)

    def _write_config_json(self, raw: str) -> None:
        vscode_config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        vscode_config.CONFIG_PATH.write_text(raw)

    # -- load_config: hand-edited config.json -------------------------

    def test_load_config_nan_literal_falls_back_to_default(self) -> None:
        """A ``NaN`` literal in config.json must not survive load_config."""
        self._write_config_json('{"max_budget": NaN}')
        cfg = load_config()
        self.assertEqual(cfg["max_budget"], DEFAULTS["max_budget"])

    def test_load_config_infinity_literal_falls_back_to_default(self) -> None:
        """``Infinity`` in config.json must not survive load_config."""
        self._write_config_json('{"max_budget": Infinity}')
        cfg = load_config()
        self.assertEqual(cfg["max_budget"], DEFAULTS["max_budget"])

    def test_load_config_overflow_literal_falls_back_to_default(self) -> None:
        """``1e999`` (parses to inf) must not survive load_config."""
        self._write_config_json('{"max_budget": 1e999}')
        cfg = load_config()
        self.assertEqual(cfg["max_budget"], DEFAULTS["max_budget"])

    def test_load_config_nan_string_falls_back_to_default(self) -> None:
        """The string ``"nan"`` converts via float() and must be rejected."""
        self._write_config_json('{"max_budget": "nan"}')
        cfg = load_config()
        self.assertEqual(cfg["max_budget"], DEFAULTS["max_budget"])

    def test_load_config_inf_string_falls_back_to_default(self) -> None:
        """The string ``"inf"`` converts via float() and must be rejected."""
        self._write_config_json('{"max_budget": "-inf"}')
        cfg = load_config()
        self.assertEqual(cfg["max_budget"], DEFAULTS["max_budget"])

    # -- sanitize_config: direct saveConfig payloads -------------------

    def test_sanitize_config_rejects_nan_float(self) -> None:
        out = sanitize_config({"max_budget": float("nan")})
        self.assertEqual(out["max_budget"], DEFAULTS["max_budget"])

    def test_sanitize_config_rejects_inf_float(self) -> None:
        out = sanitize_config({"max_budget": float("inf")})
        self.assertEqual(out["max_budget"], DEFAULTS["max_budget"])

    def test_sanitize_config_keeps_finite_numbers(self) -> None:
        """Regression guard: genuine finite values still pass through."""
        self.assertEqual(sanitize_config({"max_budget": 55})["max_budget"], 55)
        self.assertEqual(
            sanitize_config({"max_budget": 55.5})["max_budget"], 55.5,
        )
        self.assertEqual(
            sanitize_config({"max_budget": "42"})["max_budget"], 42.0,
        )

    # -- apply_config_to_env: the enforcement sink ---------------------

    def test_apply_config_nan_keeps_budget_enforceable(self) -> None:
        """End-to-end: nan budget must not disable enforcement."""
        apply_config_to_env({"max_budget": float("nan")})
        budget = config_module.DEFAULT_CONFIG.max_budget
        self.assertTrue(math.isfinite(budget))
        self.assertEqual(budget, float(DEFAULTS["max_budget"]))
        # The actual enforcement predicate: an over-budget cost must
        # compare as over budget (nan makes this False for any cost).
        self.assertTrue(budget * 2 > budget)

    def test_apply_config_inf_falls_back_to_default(self) -> None:
        apply_config_to_env({"max_budget": float("inf")})
        self.assertEqual(
            config_module.DEFAULT_CONFIG.max_budget,
            float(DEFAULTS["max_budget"]),
        )

    def test_load_then_apply_end_to_end_nan_file(self) -> None:
        """Full pipeline: junk file -> load_config -> apply_config_to_env."""
        self._write_config_json('{"max_budget": NaN}')
        apply_config_to_env(load_config())
        self.assertTrue(math.isfinite(config_module.DEFAULT_CONFIG.max_budget))


if __name__ == "__main__":
    unittest.main()
