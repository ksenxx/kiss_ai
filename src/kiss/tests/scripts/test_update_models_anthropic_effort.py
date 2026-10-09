# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``update_models.py`` effort ladders and alias generation for direct Claude.

Anthropic exposes ``output_config.effort`` (``low`` / ``medium`` / ``high``
/ ``xhigh`` / ``max``, https://platform.claude.com/docs/en/build-with-claude/effort).
``AnthropicModel`` maps the framework-wide ``reasoning_effort`` key onto it,
so ``update_models.py`` probes direct ``claude-*`` keys exactly like the
OpenAI family and writes ``-{level}`` aliases.  The per-generation ladders
pinned here were verified live on 2026-10-07:

* Opus 4.5 → ``low`` / ``medium`` / ``high`` (``xhigh`` / ``max`` → HTTP 400);
* Opus 4.6, Sonnet 4.6 → ``low`` / ``medium`` / ``high`` / ``max``
  (``xhigh`` → HTTP 400);
* Opus 4.7+, Sonnet 5+, Fable → all five levels;
* Haiku 4.5, Sonnet 4.5 → "This model does not support the effort parameter".
"""

from __future__ import annotations

from typing import Any

import pytest

import kiss.scripts.update_models as mod

FULL = ("low", "medium", "high", "xhigh", "max")


class _ProbeStub:
    """Fake ``model()`` factory that accepts exactly the levels in ``ok``."""

    def __init__(self, ok: set[str]) -> None:
        self.ok = ok
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model_name: str, model_config: dict[str, Any], **_: Any) -> Any:
        level = model_config["reasoning_effort"]
        self.calls.append((model_name, level))
        if level not in self.ok:
            raise RuntimeError("400 This model does not support effort level")

        class _M:
            def initialize(self, *a: Any, **k: Any) -> None:
                pass

            def generate(self) -> tuple[str, None]:
                return "Hello", None

        return _M()


@pytest.mark.parametrize(
    ("name", "ladder"),
    [
        ("claude-opus-4-5", ("low", "medium", "high")),
        ("claude-opus-4-5-20251101", ("low", "medium", "high")),
        ("claude-opus-4-6", ("low", "medium", "high", "max")),
        ("claude-sonnet-4-6", ("low", "medium", "high", "max")),
        ("claude-opus-4-7", FULL),
        ("claude-opus-4-8", FULL),
        ("claude-opus-5", FULL),
        ("claude-opus-5-5", FULL),
        ("claude-sonnet-5", FULL),
        ("claude-sonnet-5-5", FULL),
        ("claude-fable-5", FULL),
        ("claude-fable-5-1", FULL),
    ],
)
def test_anthropic_ladder_per_generation(name: str, ladder: tuple[str, ...]) -> None:
    assert mod._thinking_scale_for(name) == ladder
    assert "high" in ladder  # required by _write_entry_with_thinking_split


def test_non_claude_keys_keep_their_own_ladders() -> None:
    assert mod._thinking_scale_for("gpt-6.1-sol") == (*mod._THINKING_LEVELS, "max")
    assert mod._thinking_scale_for("openrouter/anthropic/claude-opus-5.5") == mod._THINKING_LEVELS
    assert mod._thinking_scale_for("cc/claude-opus-5-5") == mod._THINKING_LEVELS


def test_detect_probes_direct_claude_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _ProbeStub(ok=set(FULL))
    monkeypatch.setattr("kiss.core.models.model_info.model", stub)
    assert mod.detect_thinking_level("claude-opus-5-5") == "max"
    assert stub.calls == [("claude-opus-5-5", "max")]


def test_detect_stops_at_ladder_top_for_opus_4_6(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opus 4.6's ladder has no ``xhigh``: ``max`` is the first probe."""
    stub = _ProbeStub(ok={"low", "medium", "high", "max"})
    monkeypatch.setattr("kiss.core.models.model_info.model", stub)
    assert mod.detect_thinking_level("claude-opus-4-6") == "max"
    assert [lvl for _, lvl in stub.calls] == ["max"]


def test_detect_returns_high_for_opus_4_5(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _ProbeStub(ok={"low", "medium", "high"})
    monkeypatch.setattr("kiss.core.models.model_info.model", stub)
    assert mod.detect_thinking_level("claude-opus-4-5") == "high"
    assert [lvl for _, lvl in stub.calls] == ["high"]


def test_detect_returns_none_when_effort_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Haiku 4.5 / Sonnet 4.5 reject the parameter at every level."""
    stub = _ProbeStub(ok=set())
    monkeypatch.setattr("kiss.core.models.model_info.model", stub)
    assert mod.detect_thinking_level("claude-haiku-4-5") is None
    assert [lvl for _, lvl in stub.calls] == ["max", "xhigh", "high", "medium", "low"]


def test_cc_and_openrouter_claude_stay_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _ProbeStub(ok=set(FULL))
    monkeypatch.setattr("kiss.core.models.model_info.model", stub)
    assert mod.detect_thinking_level("cc/claude-opus-5-5") is None
    assert mod.detect_thinking_level("openrouter/anthropic/claude-opus-5.5") is None
    assert stub.calls == []


def test_split_writes_five_claude_aliases_and_high_base() -> None:
    """A ``max`` verdict materializes all five siblings; base keeps ``high``."""
    data: dict[str, dict[str, Any]] = {}
    entry = mod._build_entry(ctx=500000, inp=5.0, out=25.0, fc=True, thinking="max")
    mod._write_entry_with_thinking_split(data, "claude-opus-5-5", entry)
    assert data["claude-opus-5-5"]["thinking"] == "high"
    assert "alias_of" not in data["claude-opus-5-5"]
    for level in FULL:
        alias = data[f"claude-opus-5-5-{level}"]
        assert alias["alias_of"] == "claude-opus-5-5"
        assert alias["thinking"] == level
        assert alias["input_price_per_1M"] == 5.0
        assert alias["output_price_per_1M"] == 25.0


def test_split_skips_xhigh_for_opus_4_6() -> None:
    data: dict[str, dict[str, Any]] = {}
    entry = mod._build_entry(ctx=200000, inp=5.0, out=25.0, fc=True, thinking="max")
    mod._write_entry_with_thinking_split(data, "claude-opus-4-6", entry)
    assert "claude-opus-4-6-xhigh" not in data
    assert set(data) == {"claude-opus-4-6"} | {
        f"claude-opus-4-6-{lvl}" for lvl in ("low", "medium", "high", "max")
    }


def test_split_writes_three_aliases_for_opus_4_5() -> None:
    data: dict[str, dict[str, Any]] = {}
    entry = mod._build_entry(ctx=200000, inp=5.0, out=25.0, fc=True, thinking="high")
    mod._write_entry_with_thinking_split(data, "claude-opus-4-5", entry)
    assert set(data) == {"claude-opus-4-5"} | {
        f"claude-opus-4-5-{lvl}" for lvl in ("low", "medium", "high")
    }


def test_xhigh_top_without_max_sibling_survives_normalization() -> None:
    """A five-level ladder whose live top is ``xhigh`` has no ``-max`` sibling.

    The recorded maximum must be reconstructed from the highest generated
    sibling, not from the ladder's last rung, or normalization would read
    the base's ``high`` cap marker as the real top and drop ``-xhigh``.
    """
    data: dict[str, dict[str, Any]] = {}
    entry = mod._build_entry(ctx=500000, inp=5.0, out=25.0, fc=True, thinking="xhigh")
    mod._write_entry_with_thinking_split(data, "claude-opus-5-5", entry)
    assert data["claude-opus-5-5"]["thinking"] == "high"
    assert "claude-opus-5-5-xhigh" in data
    assert "claude-opus-5-5-max" not in data
    assert (
        mod._stored_max_thinking_level(data, "claude-opus-5-5", data["claude-opus-5-5"]) == "xhigh"
    )

    mod._normalize_thinking_splits(data)
    assert "claude-opus-5-5-xhigh" in data
    assert "claude-opus-5-5-max" not in data

    # A price-only rewrite (remove_stale_siblings=False) keeps the top too.
    mod._write_entry_with_thinking_split(
        data, "claude-opus-5-5", data["claude-opus-5-5"], remove_stale_siblings=False
    )
    assert "claude-opus-5-5-xhigh" in data
    assert "claude-opus-5-5-max" not in data


def test_recorded_max_level_scans_all_levels_above_stored() -> None:
    scale = ("low", "medium", "high", "xhigh", "max")
    data = {
        "m": {"thinking": "high"},
        "m-xhigh": {"thinking": "xhigh", "alias_of": "m"},
    }
    assert mod._recorded_max_level(data, "m", scale, "high") == "xhigh"
    data["m-max"] = {"thinking": "max", "alias_of": "m"}
    assert mod._recorded_max_level(data, "m", scale, "high") == "max"
    assert mod._recorded_max_level({"m": {"thinking": "high"}}, "m", scale, "high") == "high"
    # A foreign entry occupying the sibling name is not a generated alias.
    foreign = {"m": {"thinking": "high"}, "m-max": {"thinking": "max"}}
    assert mod._recorded_max_level(foreign, "m", scale, "high") == "high"
