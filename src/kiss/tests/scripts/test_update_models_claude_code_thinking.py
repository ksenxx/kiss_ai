"""``cc/claude-*`` entries mirror the effort ladder of their direct ``claude-*`` twin.

``ClaudeCodeModel`` passes a ``cc/`` entry's ``thinking`` level to the
Claude Code CLI as ``--effort``, but the CLI runs an unsupported level at
the highest one below it instead of rejecting it, so ``detect_thinking_level``
is gated off for ``cc/`` keys and cannot learn the ladder.  The updater
therefore copies the direct twin's recorded maximum onto them, the way it
does for ``bedrock/`` ids and ``openrouter/anthropic/`` slugs, and
``apply_updates_to_file`` materializes the ``-low`` … ``-max`` siblings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kiss.scripts import update_models
from kiss.scripts.update_models import (
    _claude_code_twin_key,
    _sync_claude_code_thinking,
    _thinking_scale_for,
    apply_updates_to_file,
    compute_changes,
)

_CC_OPUS = "cc/claude-opus-5-5"
_CC_OPUS_4_5 = "cc/claude-opus-4-5-20251101"
_FULL = ("low", "medium", "high", "xhigh", "max")


def _entry(thinking: str | None = None, **extra) -> dict:
    """Build a catalog entry as ``get_current_model_info`` returns it."""
    entry = {
        "context_length": 500000,
        "input_price_per_1M": 0.0,
        "output_price_per_1M": 0.0,
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
    """Opus 5.5 direct (``high`` base, ``-low`` … ``-max`` aliases), its level-less
    ``cc/`` key and the CLI's short ``cc/opus`` alias."""
    current = {"claude-opus-5-5": _entry("high", input_price_per_1M=5.0, output_price_per_1M=25.0)}
    for level in _FULL:
        current[f"claude-opus-5-5-{level}"] = _entry(
            level, input_price_per_1M=5.0, output_price_per_1M=25.0, alias_of="claude-opus-5-5"
        )
    current[_CC_OPUS] = _entry()
    current["cc/opus"] = _entry()
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


def test_twin_key_is_the_slug_dated_first_then_undated() -> None:
    names = {"claude-opus-5-5", "claude-opus-5-5-max", "claude-opus-4-5", "gpt-6"}
    assert _claude_code_twin_key(_CC_OPUS, names) == "claude-opus-5-5"
    assert _claude_code_twin_key(_CC_OPUS_4_5, names) == "claude-opus-4-5"
    assert _claude_code_twin_key(_CC_OPUS_4_5, names | {"claude-opus-4-5-20251101"}) == (
        "claude-opus-4-5-20251101"
    )
    assert _claude_code_twin_key("cc/opus", names) is None
    assert _claude_code_twin_key("cc/claude-haiku-5-5", names) is None
    assert _claude_code_twin_key("claude-opus-5-5", names) is None


def test_scale_is_the_ladder_of_the_claude_model_behind_the_key() -> None:
    assert _thinking_scale_for(_CC_OPUS) == _FULL
    assert _thinking_scale_for(_CC_OPUS_4_5) == ("low", "medium", "high")
    assert _thinking_scale_for("cc/claude-opus-4-6") == ("low", "medium", "high", "max")


def test_level_less_key_gets_its_twin_maximum() -> None:
    """A drift between the key and its twin counts as an update at change computation."""
    updates, new_models = _changes(_catalog())
    assert new_models == []
    assert updates == [{"name": _CC_OPUS, "changes": {"thinking": "max"}, "source": "claude-code"}]


def test_key_in_step_with_its_twin_is_left_alone() -> None:
    current = _catalog()
    current[_CC_OPUS]["thinking"] = "high"
    for level in _FULL:
        current[f"{_CC_OPUS}-{level}"] = _entry(level, alias_of=_CC_OPUS)
    assert _changes(current) == ([], [])


def test_short_alias_and_twinless_key_keep_no_level() -> None:
    """``cc/opus`` names no fixed model; a stale level on a twinless key is removed."""
    current = {"cc/opus": _entry("high"), "cc/claude-haiku-5-5": _entry("high")}
    current["cc/claude-haiku-5-5-max"] = _entry("max", alias_of="cc/claude-haiku-5-5")
    updates, _ = _changes(current)
    assert updates == [
        {"name": "cc/claude-haiku-5-5", "changes": {"thinking": None}, "source": "claude-code"},
        {"name": "cc/opus", "changes": {"thinking": None}, "source": "claude-code"},
    ]


def test_retested_twin_hands_its_new_level_to_the_key() -> None:
    """The twin's retest this run wins over the twin's aliases on disk."""
    current = _catalog()
    updates = [{"name": "claude-opus-5-5", "changes": {"thinking": "high"}, "source": "retest"}]
    _sync_claude_code_thinking(current, updates, [])
    assert updates[1] == {
        "name": _CC_OPUS,
        "changes": {"thinking": "high"},
        "source": "claude-code",
    }

    # The key's own pending record (a context change, say) is joined, not duplicated.
    updates = [{"name": _CC_OPUS, "changes": {"context_length": 1000000}, "source": "claude-code"}]
    _sync_claude_code_thinking(current, updates, [])
    assert updates == [
        {
            "name": _CC_OPUS,
            "changes": {"context_length": 1000000, "thinking": "max"},
            "source": "claude-code",
        }
    ]


def test_new_key_and_new_twin_in_the_same_run() -> None:
    """A key discovered this run takes the level its twin's probe settles on."""
    new_models = [
        {"name": "claude-opus-5-5", "thinking": None},
        {"name": _CC_OPUS, "thinking": None},
    ]
    _sync_claude_code_thinking({}, [], new_models)
    assert new_models[1]["thinking"] is None
    new_models[0]["thinking"] = "max"  # what the live probe records
    _sync_claude_code_thinking({}, [], new_models)
    assert new_models[1]["thinking"] == "max"

    # A new key whose twin is already in the catalog reads the twin's aliases.
    new_models = [{"name": _CC_OPUS, "thinking": None}]
    current = _catalog()
    del current[_CC_OPUS]
    _sync_claude_code_thinking(current, [], new_models)
    assert new_models[0]["thinking"] == "max"


def test_apply_writes_the_anthropic_ladder_for_the_key(tmp_path: Path, monkeypatch) -> None:
    current = _catalog()
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps(_on_disk(current)), encoding="utf-8")
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", target)
    updates, _ = _changes(current)
    apply_updates_to_file(updates, [], [], current)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data[_CC_OPUS]["thinking"] == "high"
    assert "thinking" not in data["cc/opus"]
    for level in _FULL:
        sibling = data[f"{_CC_OPUS}-{level}"]
        assert sibling["thinking"] == level
        assert sibling["alias_of"] == _CC_OPUS
        assert sibling["input_price_per_1M"] == 0.0
    assert f"{_CC_OPUS}-xhigh-xhigh" not in data


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
    """A run with nothing else to do still gives the key its twin's aliases."""
    data = _run_main(tmp_path, monkeypatch, _catalog(), [])
    assert "Everything is up to date!" not in capsys.readouterr().out
    assert data[_CC_OPUS]["thinking"] == "high"
    assert data[f"{_CC_OPUS}-max"]["alias_of"] == _CC_OPUS


def _capabilities_stub(name: str, verbose: bool = False, decisions: bool = False) -> dict:
    """What ``--test-existing`` sees: the twin confirms ``max``, the gated key probe says None."""
    return {
        "gen": True,
        "emb": False,
        "fc": True,
        "dec": False,
        "use_responses_api": None,
        "thinking": "max" if name == "claude-opus-5-5" else None,
    }


def test_retest_keeps_the_key_aliases(tmp_path: Path, monkeypatch, capsys) -> None:
    """``--test-existing`` never wipes a key's aliases on its gated probe verdict."""
    current = _catalog()
    current[_CC_OPUS]["thinking"] = "high"
    for level in _FULL:
        current[f"{_CC_OPUS}-{level}"] = _entry(level, alias_of=_CC_OPUS)
    monkeypatch.setattr(update_models, "test_model_capabilities", _capabilities_stub)
    data = _run_main(tmp_path, monkeypatch, current, ["--test-existing"])
    assert "thinking changed" not in capsys.readouterr().out
    assert data[_CC_OPUS]["thinking"] == "high"
    assert {f"{_CC_OPUS}-{level}" for level in _FULL} <= set(data)
    assert data[f"{_CC_OPUS}-max"]["thinking"] == "max"


def test_bundled_catalog_mirrors_every_claude_twin() -> None:
    """The shipped catalog is in step: each ``cc/claude-*`` base carries its twin's ladder."""
    from kiss.core.models.model_info import MODEL_INFO

    for name, info in MODEL_INFO.items():
        if not name.startswith("cc/") or info.alias_of:
            continue
        twin = _claude_code_twin_key(name, MODEL_INFO)
        expected = MODEL_INFO[twin].thinking if twin is not None else None
        assert info.thinking == expected, name
        ladder = {n.rsplit("-", 1)[1] for n, i in MODEL_INFO.items() if i.alias_of == name}
        twin_ladder = {
            n.rsplit("-", 1)[1] for n, i in MODEL_INFO.items() if twin and i.alias_of == twin
        }
        assert ladder == twin_ladder, name
