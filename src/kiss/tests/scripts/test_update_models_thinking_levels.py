# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for per-thinking-level alias generation in update_models.py.

For every model that supports a varying ``reasoning_effort`` level, the
script must materialize one catalog entry per supported level by adding a
``-{thinking_level}`` suffix (e.g. ``gpt-5.6-sol-high``), in addition to
the base entry:

* ``thinking="xhigh"`` → ``-low`` / ``-medium`` / ``-high`` / ``-xhigh``
* ``thinking="high"``  → ``-low`` / ``-medium`` / ``-high``
* ``thinking="medium"`` → ``-low`` / ``-medium``

Every generated alias mirrors the base entry's context length, pricing and
capability flags, carries ``thinking=<level>``, and is marked with
``alias_of=<base>`` so that real upstream models whose names merely end in
``-high`` (e.g. ``openrouter/openai/o3-mini-high``) are never confused with
synthetic aliases.

These tests run end-to-end through ``apply_updates_to_file`` / ``main``,
writing real JSON files in ``tmp_path`` and re-reading them. No mocks or
fakes are used for the code under test; only module path constants and
network-fetch entry points are redirected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ALL_LEVELS = ("low", "medium", "high", "xhigh")


def _redirect_model_info(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Point ``update_models.MODEL_INFO_PATH`` at ``target`` for the test."""
    import kiss.scripts.update_models as mod

    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("{}\n")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)


def _read(target: Path) -> dict[str, dict]:
    return json.loads(target.read_text())  # type: ignore[no-any-return]


def _entry(
    thinking: str | None = None,
    alias_of: str | None = None,
    ctx: int = 400_000,
    inp: float = 5.0,
    out: float = 30.0,
) -> dict:
    e: dict = {
        "context_length": ctx,
        "input_price_per_1M": inp,
        "output_price_per_1M": out,
        "fc": True,
        "emb": False,
        "gen": True,
    }
    if thinking is not None:
        e["thinking"] = thinking
    if alias_of is not None:
        e["alias_of"] = alias_of
    return e


def _new_model(name: str, thinking: str | None) -> dict:
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
        "needs_pricing": False,
    }


def test_new_model_with_xhigh_emits_alias_per_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new xhigh-capable model must produce base + all four level aliases."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("gpt-5.6-sol", "xhigh")], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"gpt-5.6-sol"} | {f"gpt-5.6-sol-{lvl}" for lvl in ALL_LEVELS}
    assert data["gpt-5.6-sol"]["thinking"] == "high"
    assert "alias_of" not in data["gpt-5.6-sol"]
    for level in ALL_LEVELS:
        alias = data[f"gpt-5.6-sol-{level}"]
        assert alias["thinking"] == level
        assert alias["alias_of"] == "gpt-5.6-sol"
        for field in (
            "context_length",
            "input_price_per_1M",
            "output_price_per_1M",
            "fc",
            "emb",
            "gen",
            "comment",
        ):
            assert alias[field] == data["gpt-5.6-sol"][field], (
                f"-{level} alias must inherit {field}"
            )


def test_new_model_with_high_emits_low_medium_high_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model whose max level is ``high`` gets aliases for low/medium/high only.

    This is the primary regression for the reported issue: before the fix,
    a varying-thinking model produced no per-level aliases at all unless it
    accepted ``xhigh`` (and then only the single ``-xhigh`` sibling).
    """
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("o4-mini", "high")], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"o4-mini", "o4-mini-low", "o4-mini-medium", "o4-mini-high"}
    assert data["o4-mini"]["thinking"] == "high"
    for level in ("low", "medium", "high"):
        assert data[f"o4-mini-{level}"]["thinking"] == level
        assert data[f"o4-mini-{level}"]["alias_of"] == "o4-mini"


def test_new_model_with_medium_caps_base_and_aliases_at_medium(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model whose max level is ``medium`` keeps base thinking=medium."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("gpt-6-mini", "medium")], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"gpt-6-mini", "gpt-6-mini-low", "gpt-6-mini-medium"}
    assert data["gpt-6-mini"]["thinking"] == "medium"
    assert data["gpt-6-mini-low"]["thinking"] == "low"
    assert data["gpt-6-mini-medium"]["thinking"] == "medium"


def test_new_model_without_thinking_gets_no_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models that reject ``reasoning_effort`` must stay single-entry."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("gpt-4o", None)], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"gpt-4o"}
    assert "thinking" not in data["gpt-4o"]


def test_legacy_xhigh_pair_is_expanded_to_all_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old-format base(high) + ``-xhigh`` catalogs must gain the new aliases.

    This covers the migration of the bundled catalog: ``gpt-5.6-sol`` was
    shipped as base + ``gpt-5.6-sol-xhigh`` only, and a plain re-run of the
    script (no updates at all) must create ``gpt-5.6-sol-low`` / ``-medium``
    / ``-high`` and stamp ``alias_of`` onto the legacy ``-xhigh`` sibling.
    """
    target = tmp_path / "MODEL_INFO.json"
    initial = {
        "gpt-5.6-sol": _entry(thinking="high"),
        "gpt-5.6-sol-xhigh": _entry(thinking="xhigh"),
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [], [], {}, dry_run=False)

    data = _read(target)
    assert set(data) == {"gpt-5.6-sol"} | {f"gpt-5.6-sol-{lvl}" for lvl in ALL_LEVELS}
    for level in ALL_LEVELS:
        assert data[f"gpt-5.6-sol-{level}"]["thinking"] == level
        assert data[f"gpt-5.6-sol-{level}"]["alias_of"] == "gpt-5.6-sol"


def test_real_upstream_high_model_is_not_treated_as_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``o3-mini-high``-style real models must survive normalization untouched.

    ``openrouter/openai/o3-mini-high`` is a real upstream model whose name
    happens to end in ``-high`` while its base ``o3-mini`` also exists.
    Without the ``alias_of`` marker it must be neither repaired, removed,
    nor overwritten.
    """
    target = tmp_path / "MODEL_INFO.json"
    real_high = _entry(ctx=200_000, inp=1.1, out=4.4)
    initial = {
        "openrouter/openai/o3-mini": _entry(ctx=200_000, inp=1.1, out=4.4),
        "openrouter/openai/o3-mini-high": dict(real_high),
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [], [], {}, dry_run=False)

    data = _read(target)
    assert data["openrouter/openai/o3-mini-high"] == real_high
    assert set(data) == set(initial)


def test_real_upstream_high_model_survives_base_deprecation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deprecating a base removes its generated aliases but no real siblings."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {
        "o3-mini": _entry(thinking="high"),
        "o3-mini-low": _entry(thinking="low", alias_of="o3-mini"),
        "o3-mini-medium": _entry(thinking="medium", alias_of="o3-mini"),
        # Real upstream model, NOT a generated alias (no alias_of marker).
        "o3-mini-high": _entry(ctx=200_000, inp=1.1, out=4.4),
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    deprecated = [{"name": "o3-mini", "reason": "removed upstream"}]
    mod.apply_updates_to_file([], [], deprecated, dict(initial), dry_run=False)

    data = _read(target)
    assert "o3-mini" not in data
    assert "o3-mini-low" not in data, "Generated alias must go with its base"
    assert "o3-mini-medium" not in data, "Generated alias must go with its base"
    assert "o3-mini-high" in data, "Real upstream model must survive"


def test_update_flipping_thinking_to_none_removes_all_generated_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a retest reports no thinking support, every alias must vanish."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-5.5": _entry(thinking="high")}
    initial.update(
        {f"gpt-5.5-{lvl}": _entry(thinking=lvl, alias_of="gpt-5.5") for lvl in ALL_LEVELS}
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    updates = [{"name": "gpt-5.5", "changes": {"thinking": None}, "source": "retest"}]
    mod.apply_updates_to_file(updates, [], [], dict(initial), dry_run=False)

    data = _read(target)
    assert set(data) == {"gpt-5.5"}
    assert "thinking" not in data["gpt-5.5"]


def test_update_lowering_max_level_removes_higher_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retest downgrade xhigh→high must drop only the ``-xhigh`` alias."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-5.5": _entry(thinking="high")}
    initial.update(
        {f"gpt-5.5-{lvl}": _entry(thinking=lvl, alias_of="gpt-5.5") for lvl in ALL_LEVELS}
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    updates = [{"name": "gpt-5.5", "changes": {"thinking": "high"}, "source": "retest"}]
    mod.apply_updates_to_file(updates, [], [], dict(initial), dry_run=False)

    data = _read(target)
    assert set(data) == {"gpt-5.5", "gpt-5.5-low", "gpt-5.5-medium", "gpt-5.5-high"}
    assert data["gpt-5.5"]["thinking"] == "high"


def test_routine_update_preserves_and_syncs_all_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-thinking updates must propagate fields to every generated alias."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-5.5": _entry(thinking="high")}
    initial.update(
        {f"gpt-5.5-{lvl}": _entry(thinking=lvl, alias_of="gpt-5.5") for lvl in ALL_LEVELS}
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    updates = [
        {
            "name": "gpt-5.5",
            "changes": {"input_price_per_1M": 4.5, "context_length": 500_000},
            "source": "openai",
        }
    ]
    mod.apply_updates_to_file(updates, [], [], dict(initial), dry_run=False)

    data = _read(target)
    assert set(data) == {"gpt-5.5"} | {f"gpt-5.5-{lvl}" for lvl in ALL_LEVELS}, (
        "A routine update is no evidence of lost xhigh support"
    )
    for name in data:
        assert data[name]["input_price_per_1M"] == 4.5
        assert data[name]["context_length"] == 500_000
    assert data["gpt-5.5-xhigh"]["thinking"] == "xhigh"


def test_routine_update_does_not_trust_foreign_xhigh_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``-xhigh`` entry aliased to a DIFFERENT base is no xhigh evidence."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {
        "gpt-5.5": _entry(thinking="high"),
        "other-base": _entry(thinking="high"),
        "gpt-5.5-xhigh": _entry(thinking="xhigh", alias_of="other-base"),
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    data = {name: dict(entry) for name, entry in initial.items()}
    mod._write_entry_with_thinking_split(
        data,
        "gpt-5.5",
        dict(initial["gpt-5.5"]),
        remove_stale_siblings=False,
    )
    assert data["gpt-5.5"]["thinking"] == "high"
    assert "gpt-5.5-xhigh" in data, "Foreign-marked entry is not ours to manage"
    assert data["gpt-5.5-xhigh"]["alias_of"] == "other-base"


def test_update_targeting_marked_alias_preserves_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An update naming an alias directly must keep its ``alias_of`` marker."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {
        "gpt-5.5": _entry(thinking="high"),
        "gpt-5.5-medium": _entry(thinking="medium", alias_of="gpt-5.5"),
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    updates = [{"name": "gpt-5.5-medium", "changes": {"fc": False}, "source": "retest"}]
    mod.apply_updates_to_file(updates, [], [], dict(initial), dry_run=False)

    data = _read(target)
    assert data["gpt-5.5-medium"]["fc"] is False
    assert data["gpt-5.5-medium"]["alias_of"] == "gpt-5.5"
    assert data["gpt-5.5-medium"]["thinking"] == "medium"
    assert "gpt-5.5-medium-low" not in data, "Aliases must never be split further"


def test_update_targeting_legacy_xhigh_name_writes_plainly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An update naming a legacy ``-xhigh`` entry must not nest aliases."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    data: dict[str, dict] = {}
    mod._write_entry_with_thinking_split(data, "gpt-5.5-xhigh", _entry(thinking="xhigh"))
    assert set(data) == {"gpt-5.5-xhigh"}
    assert data["gpt-5.5-xhigh"]["thinking"] == "xhigh"
    assert "gpt-5.5-xhigh-xhigh" not in data


def test_orphan_marked_alias_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marked alias whose base entry vanished must be cleaned up."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-5.5-medium": _entry(thinking="medium", alias_of="gpt-5.5")}
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [], [], {}, dry_run=False)

    assert _read(target) == {}


def test_malformed_level_alias_is_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalization must overwrite a generated alias with drifted fields."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {
        "gpt-5.5": _entry(thinking="high"),
        "gpt-5.5-low": _entry(thinking="high", alias_of="gpt-5.5", ctx=1, inp=9.9, out=9.9),
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [], [], {}, dry_run=False)

    data = _read(target)
    assert data["gpt-5.5-low"] == {
        **data["gpt-5.5"],
        "thinking": "low",
        "alias_of": "gpt-5.5",
    }


def test_alias_generation_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running the script on an already-expanded catalog changes nothing."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [_new_model("gpt-5.6-sol", "xhigh")], [], {}, dry_run=False)
    first = _read(target)
    assert not mod._has_thinking_normalization_changes(first), (
        "Freshly written catalog must already be normalized"
    )
    mod.apply_updates_to_file([], [], [], {}, dry_run=False)
    second = _read(target)
    assert second == first
    assert "gpt-5.6-sol-high-high" not in second
    assert "gpt-5.6-sol-xhigh-xhigh" not in second


def test_find_deprecated_models_skips_marked_aliases() -> None:
    """Generated ``-{level}`` aliases must never be probed upstream."""
    import kiss.scripts.update_models as mod

    current = {
        "gpt-5.6-sol": {"source": "openai"},
        "gpt-5.6-sol-low": {"source": "openai", "alias_of": "gpt-5.6-sol"},
        "gpt-5.6-sol-medium": {"source": "openai", "alias_of": "gpt-5.6-sol"},
        "gpt-5.6-sol-high": {"source": "openai", "alias_of": "gpt-5.6-sol"},
        "gpt-5.6-sol-xhigh": {"source": "openai", "alias_of": "gpt-5.6-sol"},
    }
    openai = {"gpt-5.6-sol": {"source": "openai"}}

    deprecated = mod.find_deprecated_models(current, {}, {}, {}, openai)

    assert deprecated == [], (
        "Generated aliases are managed with their base entries and must not "
        "be checked as independent upstream model names"
    )


def test_main_test_existing_skips_marked_level_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--test-existing`` must probe only base models, never level aliases."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-5.6-sol": _entry(thinking="high")}
    initial.update(
        {
            f"gpt-5.6-sol-{lvl}": _entry(thinking=lvl, alias_of="gpt-5.6-sol")
            for lvl in ALL_LEVELS
        }
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")

    import kiss.scripts.update_models as mod

    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    monkeypatch.setattr(mod, "fetch_openrouter", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_together", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_anthropic", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_gemini", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_openai", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(mod, "get_current_model_info", lambda: dict(initial))
    calls: list[str] = []

    def record_probe(name: str, verbose: bool = False) -> dict[str, object]:
        calls.append(name)
        return {"gen": True, "emb": False, "fc": True, "thinking": "xhigh"}

    monkeypatch.setattr(mod, "test_model_capabilities", record_probe)
    monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing"])

    mod.main()

    output = capsys.readouterr().out
    assert calls == ["gpt-5.6-sol"], f"Only the base must be probed, got {calls}"
    assert "thinking changed" not in output
    data = _read(target)
    assert set(data) == set(initial)
    for level in ALL_LEVELS:
        assert data[f"gpt-5.6-sol-{level}"]["thinking"] == level


def test_main_reports_up_to_date_when_catalog_already_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fully-expanded catalog with no vendor changes must short-circuit."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {"gpt-5.6-sol": _entry(thinking="high")}
    initial.update(
        {
            f"gpt-5.6-sol-{lvl}": _entry(thinking=lvl, alias_of="gpt-5.6-sol")
            for lvl in ALL_LEVELS
        }
    )
    target.write_text(json.dumps(initial, indent=2) + "\n")

    import kiss.scripts.update_models as mod

    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    monkeypatch.setattr(mod, "fetch_openrouter", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_together", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_anthropic", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_gemini", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_openai", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(mod, "get_current_model_info", lambda: dict(initial))
    monkeypatch.setattr(sys, "argv", ["update_models.py"])

    mod.main()

    out = capsys.readouterr().out
    assert "Everything is up to date!" in out
    assert _read(target) == initial


def test_main_discovers_new_model_and_writes_level_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full main() flow: a newly-fetched model with thinking=high must land
    in MODEL_INFO.json together with its ``-low``/``-medium``/``-high``
    aliases (this exact flow produced only the bare entry before the fix).
    """
    target = tmp_path / "MODEL_INFO.json"
    target.write_text("{}\n")

    import kiss.scripts.update_models as mod

    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    openrouter = {
        "openrouter/openai/gpt-6": {
            "context_length": 400_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openrouter",
        }
    }
    monkeypatch.setattr(mod, "fetch_openrouter", lambda verbose=False: dict(openrouter))
    monkeypatch.setattr(mod, "fetch_together", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_anthropic", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_gemini", lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_openai", lambda verbose=False: {"gpt-6": {"source": "openai"}})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(mod, "get_current_model_info", lambda: {})

    def probe(name: str, verbose: bool = False) -> dict[str, object]:
        return {"gen": True, "emb": False, "fc": True, "thinking": "high"}

    monkeypatch.setattr(mod, "test_model_capabilities", probe)
    monkeypatch.setattr(sys, "argv", ["update_models.py"])

    mod.main()

    data = _read(target)
    for base in ("gpt-6", "openrouter/openai/gpt-6"):
        assert base in data
        assert data[base]["thinking"] == "high"
        for level in ("low", "medium", "high"):
            alias = data[f"{base}-{level}"]
            assert alias["thinking"] == level
            assert alias["alias_of"] == base
        assert f"{base}-xhigh" not in data


def test_bundled_catalog_ships_gpt_5_6_sol_level_aliases() -> None:
    """The shipped MODEL_INFO.json must contain the per-level aliases.

    This pins the user-visible outcome: ``gpt-5.6-sol-high`` (and friends)
    are selectable model names in the bundled catalog.
    """
    import kiss.scripts.update_models as mod

    data = json.loads(
        (Path(mod.PROJECT_ROOT) / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json")
        .read_text(encoding="utf-8")
    )
    for level in ALL_LEVELS:
        name = f"gpt-5.6-sol-{level}"
        assert name in data, f"{name} missing from bundled catalog"
        assert data[name]["thinking"] == level
        assert data[name]["alias_of"] == "gpt-5.6-sol"
    assert not mod._has_thinking_normalization_changes(data), (
        "Bundled catalog must be fully normalized"
    )
    assert "alias_of" not in data["openrouter/openai/o3-mini-high"], (
        "Real upstream -high models must not be marked as aliases"
    )
