"""OpenRouter Claude slugs mirror the effort ladder of their direct ``claude-*`` twin.

OpenRouter forwards ``reasoning.effort`` to Anthropic but answers 200 at
every level for every Claude model, so ``detect_thinking_level`` cannot
learn the ladder for ``openrouter/anthropic/*`` keys.  The updater
therefore copies the direct twin's recorded maximum onto them, the way it
does for ``bedrock/`` ids, and ``apply_updates_to_file`` materializes the
``-low`` … ``-max`` siblings on the Anthropic ladder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kiss.scripts import update_models
from kiss.scripts.update_models import (
    _openrouter_claude_twin_key,
    _sync_openrouter_claude_thinking,
    apply_updates_to_file,
    compute_changes,
)

_OR_OPUS = "openrouter/anthropic/claude-opus-5.5"
_OR_OPUS_4_1 = "openrouter/anthropic/claude-opus-4.1"
_FULL = ("low", "medium", "high", "xhigh", "max")


def _entry(inp: float, out: float, thinking: str | None = None, **extra) -> dict:
    """Build a catalog entry as ``get_current_model_info`` returns it."""
    entry = {
        "context_length": 500000,
        "input_price_per_1M": inp,
        "output_price_per_1M": out,
        "fc": True,
        "emb": False,
        "gen": True,
        "thinking": thinking,
        "alias_of": None,
        "use_responses_api": False,
    }
    entry.update(extra)
    return entry


def _catalog() -> dict[str, dict]:
    """Opus 5.5 direct (``high`` base, ``-low`` … ``-max`` aliases) and its level-less slug."""
    current = {"claude-opus-5-5": _entry(5.0, 25.0, "high")}
    for level in _FULL:
        current[f"claude-opus-5-5-{level}"] = _entry(5.0, 25.0, level, alias_of="claude-opus-5-5")
    current[_OR_OPUS] = _entry(4.0, 20.0, use_responses_api=True)
    return current


def _on_disk(current: dict[str, dict]) -> dict[str, dict]:
    """The JSON form of ``current``: no ``None`` values, no false ``use_responses_api``."""
    return {
        k: {f: v for f, v in e.items() if v is not None and (f != "use_responses_api" or v)}
        for k, e in current.items()
    }


def _changes(current: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Run the updater's change computation with every vendor listing empty."""
    return compute_changes(current, {}, {}, {}, {}, {})


def test_twin_key_maps_dotted_slugs_to_hyphenated_claude_ids() -> None:
    names = {"claude-opus-5-5", "claude-opus-5-5-max", "claude-haiku-4-5-20251001", "gpt-6"}
    assert _openrouter_claude_twin_key(_OR_OPUS, names) == "claude-opus-5-5"
    assert _openrouter_claude_twin_key("openrouter/anthropic/claude-haiku-4.5", names) is None
    assert _openrouter_claude_twin_key(_OR_OPUS_4_1, names) is None
    assert _openrouter_claude_twin_key("openrouter/~anthropic/claude-opus-latest", names) is None
    assert _openrouter_claude_twin_key("claude-opus-5-5", names) is None


def test_level_less_slug_gets_its_twin_maximum() -> None:
    """A drift between the slug and its twin counts as an update at change computation."""
    updates, new_models = _changes(_catalog())
    assert new_models == []
    assert updates == [
        {"name": _OR_OPUS, "changes": {"thinking": "max"}, "source": "openrouter-claude"}
    ]


def test_slug_in_step_with_its_twin_is_left_alone() -> None:
    current = _catalog()
    current[_OR_OPUS]["thinking"] = "high"
    for level in _FULL:
        current[f"{_OR_OPUS}-{level}"] = _entry(4.0, 20.0, level, alias_of=_OR_OPUS)
    assert _changes(current) == ([], [])


def test_slug_without_a_twin_keeps_no_level() -> None:
    """Opus 4.1 exists only on OpenRouter; a stale level on it is removed, none is invented."""
    current = {_OR_OPUS_4_1: _entry(15.0, 75.0)}
    assert _changes(current) == ([], [])
    current[_OR_OPUS_4_1]["thinking"] = "high"
    current[f"{_OR_OPUS_4_1}-max"] = _entry(15.0, 75.0, "max", alias_of=_OR_OPUS_4_1)
    updates, _ = _changes(current)
    assert updates == [
        {"name": _OR_OPUS_4_1, "changes": {"thinking": None}, "source": "openrouter-claude"}
    ]


def test_retested_twin_hands_its_new_level_to_the_slug() -> None:
    """The twin's retest this run wins over the twin's aliases on disk."""
    current = _catalog()
    updates = [{"name": "claude-opus-5-5", "changes": {"thinking": "high"}, "source": "retest"}]
    _sync_openrouter_claude_thinking(current, updates, [])
    assert updates[1] == {
        "name": _OR_OPUS,
        "changes": {"thinking": "high"},
        "source": "openrouter-claude",
    }

    # The slug's own pending record (a price change, say) is joined, not duplicated.
    updates = [{"name": _OR_OPUS, "changes": {"input_price_per_1M": 4.5}, "source": "openrouter"}]
    _sync_openrouter_claude_thinking(current, updates, [])
    assert updates == [
        {
            "name": _OR_OPUS,
            "changes": {"input_price_per_1M": 4.5, "thinking": "max"},
            "source": "openrouter",
        }
    ]


def test_gated_probe_verdict_on_the_slug_is_overridden() -> None:
    """A ``thinking: None`` already queued for the slug is corrected, not trusted.

    The second sync in ``main`` runs after any retest; a pending removal
    equal to nothing on disk must still be replaced by the twin's level.
    """
    current = _catalog()
    current[_OR_OPUS]["thinking"] = "high"
    current[f"{_OR_OPUS}-max"] = _entry(4.0, 20.0, "max", alias_of=_OR_OPUS)
    updates = [{"name": _OR_OPUS, "changes": {"thinking": None}, "source": "retest"}]
    _sync_openrouter_claude_thinking(current, updates, [])
    assert updates == [{"name": _OR_OPUS, "changes": {"thinking": "max"}, "source": "retest"}]


def test_new_slug_and_new_twin_in_the_same_run() -> None:
    """A slug discovered this run takes the level its twin's probe settles on."""
    new_models = [
        {"name": "claude-opus-5-5", "thinking": None},
        {"name": _OR_OPUS, "thinking": None},
    ]
    _sync_openrouter_claude_thinking({}, [], new_models)
    assert new_models[1]["thinking"] is None
    new_models[0]["thinking"] = "max"  # what the live probe records
    new_models[1]["thinking"] = None  # the slug's own probe is gated off
    _sync_openrouter_claude_thinking({}, [], new_models)
    assert new_models[1]["thinking"] == "max"

    # A new slug whose twin is already in the catalog reads the twin's aliases.
    new_models = [{"name": _OR_OPUS, "thinking": None}]
    current = _catalog()
    del current[_OR_OPUS]
    _sync_openrouter_claude_thinking(current, [], new_models)
    assert new_models[0]["thinking"] == "max"


def test_apply_writes_the_anthropic_ladder_for_the_slug(tmp_path: Path, monkeypatch) -> None:
    current = _catalog()
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps(_on_disk(current)), encoding="utf-8")
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", target)
    updates, _ = _changes(current)
    apply_updates_to_file(updates, [], [], current)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data[_OR_OPUS]["thinking"] == "high"
    assert data[_OR_OPUS]["use_responses_api"] is True
    for level in _FULL:
        sibling = data[f"{_OR_OPUS}-{level}"]
        assert sibling["thinking"] == level
        assert sibling["alias_of"] == _OR_OPUS
        assert sibling["input_price_per_1M"] == 4.0
        assert sibling["use_responses_api"] is True
    assert f"{_OR_OPUS}-xhigh-xhigh" not in data


def _run_main(
    tmp_path: Path, monkeypatch, current: dict[str, dict], argv: list[str]
) -> dict[str, dict]:
    """Drive ``main()`` on ``current`` with every vendor listing empty; return the written file."""
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps(_on_disk(current), indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(update_models, "README_PATH", tmp_path / "README.md")
    for fetcher in (
        "fetch_openrouter",
        "fetch_together",
        "fetch_anthropic",
        "fetch_gemini",
        "fetch_openai",
        "fetch_bedrock",
    ):
        monkeypatch.setattr(update_models, fetcher, lambda verbose=False: {})
    monkeypatch.setattr(update_models, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(update_models, "get_current_model_info", lambda: dict(current))
    monkeypatch.setattr(sys, "argv", ["update_models.py", *argv])
    update_models.main()
    written: dict[str, dict] = json.loads(target.read_text(encoding="utf-8"))
    return written


def test_main_repairs_a_thinking_only_drift(tmp_path: Path, monkeypatch, capsys) -> None:
    """A run with nothing else to do still gives the slug its twin's aliases."""
    data = _run_main(tmp_path, monkeypatch, _catalog(), [])
    assert "Everything is up to date!" not in capsys.readouterr().out
    assert data[_OR_OPUS]["thinking"] == "high"
    assert data[f"{_OR_OPUS}-max"]["alias_of"] == _OR_OPUS


def _capabilities_stub(name: str, verbose: bool = False, decisions: bool = False) -> dict:
    """What ``--test-existing`` sees: the twin confirms ``max``, the gated slug probe says None."""
    return {
        "gen": True,
        "emb": False,
        "fc": True,
        "dec": False,
        "use_responses_api": None,
        "thinking": "max" if name == "claude-opus-5-5" else None,
    }


def test_retest_keeps_the_slug_aliases(tmp_path: Path, monkeypatch, capsys) -> None:
    """``--test-existing`` never wipes a slug's aliases on its gated probe verdict."""
    current = _catalog()
    current[_OR_OPUS]["thinking"] = "high"
    for level in _FULL:
        current[f"{_OR_OPUS}-{level}"] = _entry(4.0, 20.0, level, alias_of=_OR_OPUS)
    monkeypatch.setattr(update_models, "test_model_capabilities", _capabilities_stub)
    data = _run_main(tmp_path, monkeypatch, current, ["--test-existing"])
    assert "thinking changed" not in capsys.readouterr().out
    assert data[_OR_OPUS]["thinking"] == "high"
    assert {f"{_OR_OPUS}-{level}" for level in _FULL} <= set(data)
    assert data[f"{_OR_OPUS}-max"]["thinking"] == "max"
