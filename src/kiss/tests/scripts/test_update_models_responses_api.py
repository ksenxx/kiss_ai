# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for OpenAI v2 (Responses API) support in update_models.py.

``update_models.py`` live-probes every tested model through
``/v1/responses`` (via ``update_responses_api_support.probe_responses_support``)
and records the verdict as ``"use_responses_api": true`` on passing entries:

* new models carry the probed flag into their catalog entry, and every
  generated ``-{level}`` thinking alias inherits it;
* ``--test-existing`` re-probes and flips the stored flag in either
  direction (a falsy verdict removes the key entirely);
* ``--skip-test`` models stay unverified and therefore unflagged.

These tests run end-to-end through ``apply_updates_to_file`` / ``main``,
writing real JSON files in ``tmp_path`` and re-reading them.  Only module
path constants and network-fetch entry points are redirected.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ALL_LEVELS = ("low", "medium", "high", "xhigh")


def _read(target: Path) -> dict[str, dict]:
    return json.loads(target.read_text())  # type: ignore[no-any-return]


def _entry(use_responses_api: bool | None = None, thinking: str | None = None) -> dict:
    e: dict = {
        "context_length": 400_000,
        "input_price_per_1M": 5.0,
        "output_price_per_1M": 30.0,
        "fc": True,
        "emb": False,
        "gen": True,
    }
    if thinking is not None:
        e["thinking"] = thinking
    if use_responses_api is not None:
        e["use_responses_api"] = use_responses_api
    return e


def _new_model(name: str, use_responses_api: bool, thinking: str | None = None) -> dict:
    return {
        "name": name,
        "context_length": 400_000,
        "input_price_per_1M": 5.0,
        "output_price_per_1M": 30.0,
        "source": "openai",
        "fc": True,
        "emb": False,
        "gen": True,
        "thinking": thinking,
        "use_responses_api": use_responses_api,
        "needs_pricing": False,
    }


class TestApplyNewModels:
    """New-model entries must carry the probed flag (and aliases inherit)."""

    def test_probed_flag_is_written_and_inherited_by_aliases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A passing probe writes the flag on the base and all aliases."""
        target = tmp_path / "MODEL_INFO.json"
        target.write_text("{}\n")
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        mod.apply_updates_to_file(
            [], [_new_model("gpt-7", True, thinking="xhigh")], [], {}, dry_run=False
        )
        data = _read(target)
        assert data["gpt-7"]["use_responses_api"] is True
        for level in ALL_LEVELS:
            assert data[f"gpt-7-{level}"]["use_responses_api"] is True
            assert data[f"gpt-7-{level}"]["alias_of"] == "gpt-7"

    def test_failed_probe_leaves_the_entry_unflagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing probe must not write the key at all (compact catalog)."""
        target = tmp_path / "MODEL_INFO.json"
        target.write_text("{}\n")
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        mod.apply_updates_to_file(
            [], [_new_model("chat-only-model", False)], [], {}, dry_run=False
        )
        data = _read(target)
        assert "use_responses_api" not in data["chat-only-model"]


class TestApplyUpdates:
    """Flag flips arriving as ``changes`` must be applied in both directions."""

    def test_true_change_adds_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """changes={'use_responses_api': True} writes the key."""
        target = tmp_path / "MODEL_INFO.json"
        initial = {"gpt-7": _entry()}
        target.write_text(json.dumps(initial) + "\n")
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        mod.apply_updates_to_file(
            [{"name": "gpt-7", "changes": {"use_responses_api": True}}],
            [],
            [],
            dict(initial),
            dry_run=False,
        )
        assert _read(target)["gpt-7"]["use_responses_api"] is True

    def test_false_change_removes_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """changes={'use_responses_api': False} removes the key entirely."""
        target = tmp_path / "MODEL_INFO.json"
        initial = {"gpt-7": _entry(use_responses_api=True)}
        target.write_text(json.dumps(initial) + "\n")
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        mod.apply_updates_to_file(
            [{"name": "gpt-7", "changes": {"use_responses_api": False}}],
            [],
            [],
            dict(initial),
            dry_run=False,
        )
        assert "use_responses_api" not in _read(target)["gpt-7"]


def test_main_test_existing_flips_the_flag_from_probe_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--test-existing`` records a v2-support change and updates the entry."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-7": _entry(), "chat-only-model": _entry(use_responses_api=True)}
    target.write_text(json.dumps(initial, indent=2) + "\n")

    import kiss.scripts.update_models as mod

    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    for fetcher in (
        "fetch_openrouter",
        "fetch_together",
        "fetch_anthropic",
        "fetch_gemini",
        "fetch_openai",
    ):
        monkeypatch.setattr(mod, fetcher, lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(mod, "get_current_model_info", lambda: dict(initial))

    def probe(name: str, verbose: bool = False) -> dict[str, object]:
        return {
            "gen": True,
            "emb": False,
            "fc": True,
            "thinking": None,
            "use_responses_api": name == "gpt-7",
        }

    monkeypatch.setattr(mod, "test_model_capabilities", probe)
    monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing"])

    mod.main()

    output = capsys.readouterr().out
    assert "gpt-7: use_responses_api changed False -> True" in output
    assert "chat-only-model: use_responses_api changed True -> False" in output
    data = _read(target)
    assert data["gpt-7"]["use_responses_api"] is True
    assert "use_responses_api" not in data["chat-only-model"]


class TestSafeArithmetic:
    """The probe calculator must never execute model-controlled code."""

    def test_plain_arithmetic_is_evaluated(self) -> None:
        """Simple numeric expressions produce their value."""
        from kiss.scripts.update_models import safe_arithmetic

        assert safe_arithmetic("25*4") == "100"
        assert safe_arithmetic("2+3") == "5"
        assert safe_arithmetic("-3 + 5") == "2"
        assert safe_arithmetic("7 // 2") == "3"
        assert safe_arithmetic("7 % 2") == "1"
        assert safe_arithmetic("1/4") == "0.25"

    def test_executable_expressions_are_rejected(self) -> None:
        """Imports, calls, names, attributes, strings and ** all fail."""
        from kiss.scripts.update_models import safe_arithmetic

        hostile = [
            "__import__('os').system('echo pwned')",
            "().__class__.__mro__",
            "open('/etc/passwd').read()",
            "os.environ",
            "'a' * 10",
            "2**10",
            "9**9**9**9",
            "[1,2][0]",
            "(lambda: 1)()",
            "",
        ]
        for expression in hostile:
            assert safe_arithmetic(expression) == "error", expression


class TestInconclusiveVerdicts:
    """Inconclusive probe outcomes must never flip a stored catalog flag."""

    def test_apply_results_preserves_flag_on_inconclusive(self) -> None:
        """An exhausted-transient verdict keeps an existing flag (and aliases)."""
        from kiss.scripts.update_responses_api_support import (
            ProbeResult,
            _apply_results,
        )

        data = {
            "gpt-7": _entry(use_responses_api=True, thinking="high"),
            "gpt-7-low": {
                **_entry(use_responses_api=True, thinking="low"),
                "alias_of": "gpt-7",
            },
            "gpt-8": _entry(),
        }
        results = [
            ProbeResult(
                "gpt-7",
                supported=False,
                detail="generation failed: TimeoutError",
                conclusive=False,
            ),
            ProbeResult(
                "gpt-8", supported=False, detail="probe timed out", conclusive=False
            ),
        ]
        flagged, unflagged = _apply_results(data, results)
        assert (flagged, unflagged) == (0, 0)
        assert data["gpt-7"]["use_responses_api"] is True
        assert data["gpt-7-low"]["use_responses_api"] is True
        assert "use_responses_api" not in data["gpt-8"]

    def test_apply_results_acts_on_conclusive_verdicts(self) -> None:
        """Definitive verdicts still add and remove flags, aliases included."""
        from kiss.scripts.update_responses_api_support import (
            ProbeResult,
            _apply_results,
        )

        data = {
            "gpt-7": _entry(use_responses_api=True, thinking="high"),
            "gpt-7-low": {
                **_entry(use_responses_api=True, thinking="low"),
                "alias_of": "gpt-7",
            },
            "gpt-8": _entry(),
        }
        results = [
            ProbeResult("gpt-7", supported=False, detail="400 not supported"),
            ProbeResult("gpt-8", supported=True, detail="ok"),
        ]
        flagged, unflagged = _apply_results(data, results)
        assert (flagged, unflagged) == (1, 2)
        assert "use_responses_api" not in data["gpt-7"]
        assert "use_responses_api" not in data["gpt-7-low"]
        assert data["gpt-8"]["use_responses_api"] is True

    def test_main_test_existing_ignores_inconclusive_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``None`` capability verdict must not schedule a flag change."""
        target = tmp_path / "MODEL_INFO.json"
        initial = {"gpt-7": _entry(use_responses_api=True)}
        target.write_text(json.dumps(initial, indent=2) + "\n")

        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
        monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
        for fetcher in (
            "fetch_openrouter",
            "fetch_together",
            "fetch_anthropic",
            "fetch_gemini",
            "fetch_openai",
        ):
            monkeypatch.setattr(mod, fetcher, lambda verbose=False: {})
        monkeypatch.setattr(
            mod, "fetch_codex_supported_slugs", lambda verbose=False: set()
        )
        monkeypatch.setattr(mod, "get_current_model_info", lambda: dict(initial))
        monkeypatch.setattr(
            mod,
            "test_model_capabilities",
            lambda name, verbose=False: {
                "gen": True,
                "emb": False,
                "fc": True,
                "thinking": None,
                "use_responses_api": None,
            },
        )
        monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing"])

        mod.main()

        assert _read(target)["gpt-7"]["use_responses_api"] is True


class TestResponsesApiProbe:
    """test_responses_api / probe_responses_support end-to-end behavior."""

    def test_model_without_openai_compatible_endpoint_reports_false(self) -> None:
        """Anthropic/Gemini-native names have no /v1/responses to probe."""
        from kiss.scripts.update_models import test_responses_api
        from kiss.scripts.update_responses_api_support import probe_responses_support

        assert test_responses_api("claude-test-model", fc=True) is False
        result = probe_responses_support("gemini-test-model")
        assert result.supported is False
        assert result.detail == "no OpenAI-compatible endpoint"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    def test_live_probe_matches_documented_support(self) -> None:
        """Live: a Responses-capable model passes, a chat-only model fails."""
        from kiss.scripts.update_models import test_responses_api

        assert test_responses_api("gpt-4o-mini", fc=True) is True
        assert test_responses_api("gpt-4o-mini-search-preview", fc=False) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
