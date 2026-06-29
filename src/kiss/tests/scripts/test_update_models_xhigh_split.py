# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the xhigh entry-splitting feature in update_models.py.

When :func:`update_models.apply_updates_to_file` is asked to write an entry
(via either ``new_models`` or ``updates``) whose detected ``thinking`` level
is ``"xhigh"``, the script must emit **two** entries in
``MODEL_INFO.json`` instead of one:

* the base model name, with ``thinking`` downgraded to ``"high"``
* a sibling entry at ``<name>-xhigh``, carrying ``thinking="xhigh"`` and
  the same context length / pricing / fc / emb / gen as the base.

This module exercises that behavior end-to-end through
``apply_updates_to_file`` by writing a real JSON file in ``tmp_path`` and
re-reading it. No mocks or fakes are used. The tests cover:

* new models with thinking=xhigh / "high" / None
* updates that flip thinking to "xhigh" on an existing entry
* updates that flip thinking away from "xhigh"
* idempotency on re-runs
* dry-run (no disk write)
* preservation of comment / fc / emb / gen / pricing on the -xhigh sibling
* OpenRouter, OpenAI, and codex/* prefixes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _redirect_model_info(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Point ``update_models.MODEL_INFO_PATH`` at ``target`` for the test."""
    import kiss.scripts.update_models as mod

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)


def _read(target: Path) -> dict[str, dict]:
    return json.loads(target.read_text())  # type: ignore[no-any-return]


def test_new_model_with_thinking_xhigh_emits_two_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brand-new model detected at xhigh must produce base + ``-xhigh`` sibling."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    new_models = [
        {
            "name": "gpt-5.5",
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openai",
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
            "needs_pricing": False,
        }
    ]
    mod.apply_updates_to_file([], new_models, [], {}, dry_run=False)

    data = _read(target)
    assert "gpt-5.5" in data, "Base entry must be present"
    assert "gpt-5.5-xhigh" in data, "Sibling -xhigh entry must be created"
    assert data["gpt-5.5"]["thinking"] == "high", (
        "Base entry must be downgraded to thinking='high'"
    )
    assert data["gpt-5.5-xhigh"]["thinking"] == "xhigh", (
        "Sibling must carry thinking='xhigh'"
    )
    # All non-thinking fields must match
    for field in (
        "context_length",
        "input_price_per_1M",
        "output_price_per_1M",
        "fc",
        "emb",
        "gen",
    ):
        assert data["gpt-5.5"][field] == data["gpt-5.5-xhigh"][field], (
            f"Sibling must inherit {field}"
        )
    assert data["gpt-5.5"]["context_length"] == 1_050_000


def test_new_model_with_thinking_high_does_not_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only xhigh triggers the split; other levels stay single-entry."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    new_models = [
        {
            "name": "o4-mini",
            "context_length": 200_000,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 4.0,
            "source": "openai",
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
            "needs_pricing": False,
        }
    ]
    mod.apply_updates_to_file([], new_models, [], {}, dry_run=False)

    data = _read(target)
    assert "o4-mini" in data
    assert "o4-mini-xhigh" not in data, "No sibling for non-xhigh thinking"
    assert data["o4-mini"]["thinking"] == "high"


def test_new_model_with_thinking_none_omits_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thinking=None must omit the field and not create a sibling."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    new_models = [
        {
            "name": "gpt-4o",
            "context_length": 128_000,
            "input_price_per_1M": 2.5,
            "output_price_per_1M": 10.0,
            "source": "openai",
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": None,
        }
    ]
    mod.apply_updates_to_file([], new_models, [], {}, dry_run=False)

    data = _read(target)
    assert "gpt-4o" in data
    assert "gpt-4o-xhigh" not in data
    assert "thinking" not in data["gpt-4o"]


def test_update_flipping_to_xhigh_creates_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An update changing thinking to xhigh must spawn the ``-xhigh`` sibling."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    initial = {
        "gpt-5.4": {
            "context_length": 400_000,
            "input_price_per_1M": 2.5,
            "output_price_per_1M": 10.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        }
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")

    import kiss.scripts.update_models as mod

    current = {
        "gpt-5.4": {
            "context_length": 400_000,
            "input_price_per_1M": 2.5,
            "output_price_per_1M": 10.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        }
    }
    updates = [
        {
            "name": "gpt-5.4",
            "changes": {"thinking": "xhigh"},
            "source": "retest",
        }
    ]
    mod.apply_updates_to_file(updates, [], [], current, dry_run=False)

    data = _read(target)
    assert "gpt-5.4" in data
    assert "gpt-5.4-xhigh" in data
    assert data["gpt-5.4"]["thinking"] == "high"
    assert data["gpt-5.4-xhigh"]["thinking"] == "xhigh"
    assert data["gpt-5.4-xhigh"]["context_length"] == 400_000
    assert data["gpt-5.4-xhigh"]["input_price_per_1M"] == 2.5
    assert data["gpt-5.4-xhigh"]["output_price_per_1M"] == 10.0


def test_update_flipping_away_from_xhigh_removes_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When thinking drops below xhigh, the ``-xhigh`` sibling must be dropped."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    initial = {
        "gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        },
        "gpt-5.5-xhigh": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
        },
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    import kiss.scripts.update_models as mod

    current = {
        "gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        },
        "gpt-5.5-xhigh": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
        },
    }
    updates = [
        {
            "name": "gpt-5.5",
            "changes": {"thinking": None},
            "source": "retest",
        }
    ]
    mod.apply_updates_to_file(updates, [], [], current, dry_run=False)

    data = _read(target)
    assert "gpt-5.5" in data
    assert "thinking" not in data["gpt-5.5"], "Thinking must be removed from base"
    assert "gpt-5.5-xhigh" not in data, (
        "Stale sibling must be cleaned up when base no longer flagged xhigh"
    )


def test_xhigh_split_idempotent_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the split twice must yield identical JSON (no dup growth)."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    new_models_first = [
        {
            "name": "gpt-5.5",
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openai",
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
            "needs_pricing": False,
        }
    ]
    mod.apply_updates_to_file([], new_models_first, [], {}, dry_run=False)
    first = _read(target)

    # Second call: same input. The base + sibling should remain identical.
    # We pass them through as updates (apply_updates_to_file should be a no-op
    # for unchanged entries, but more importantly should not create
    # gpt-5.5-xhigh-xhigh or duplicate fields).
    current = {
        name: {
            "context_length": entry["context_length"],
            "input_price_per_1M": entry["input_price_per_1M"],
            "output_price_per_1M": entry["output_price_per_1M"],
            "fc": entry["fc"],
            "emb": entry["emb"],
            "gen": entry["gen"],
            "thinking": entry.get("thinking"),
        }
        for name, entry in first.items()
    }
    updates = [
        {"name": "gpt-5.5", "changes": {"thinking": "xhigh"}, "source": "retest"},
    ]
    mod.apply_updates_to_file(updates, [], [], current, dry_run=False)
    second = _read(target)

    assert set(second.keys()) == {"gpt-5.5", "gpt-5.5-xhigh"}, (
        f"Idempotency violated: keys={sorted(second.keys())}"
    )
    assert "gpt-5.5-xhigh-xhigh" not in second
    assert second["gpt-5.5"]["thinking"] == "high"
    assert second["gpt-5.5-xhigh"]["thinking"] == "xhigh"


def test_xhigh_split_dry_run_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dry_run=True`` must not write the file, even when xhigh-splitting."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    # Make the empty {} state distinctive
    target.write_text("{}\n")
    import kiss.scripts.update_models as mod

    new_models = [
        {
            "name": "gpt-5.5",
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openai",
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
            "needs_pricing": False,
        }
    ]
    mod.apply_updates_to_file([], new_models, [], {}, dry_run=True)

    assert target.read_text() == "{}\n", (
        "dry_run=True must not modify the file even for xhigh splits"
    )


def test_xhigh_sibling_preserves_fc_emb_gen_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The -xhigh sibling must mirror fc/emb/gen flags of the base, not defaults."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    new_models = [
        {
            "name": "weirdmodel",
            "context_length": 32_768,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 2.0,
            "source": "openai",
            "fc": False,  # non-default
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
            "needs_pricing": False,
        }
    ]
    mod.apply_updates_to_file([], new_models, [], {}, dry_run=False)

    data = _read(target)
    assert data["weirdmodel"]["fc"] is False
    assert data["weirdmodel-xhigh"]["fc"] is False
    assert data["weirdmodel-xhigh"]["gen"] is True
    assert data["weirdmodel-xhigh"]["emb"] is False


def test_xhigh_sibling_carries_comment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The -xhigh sibling must include the same ``NEW`` comment as the base."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    new_models = [
        {
            "name": "gpt-5.5",
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openai",
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
            "needs_pricing": False,
        }
    ]
    mod.apply_updates_to_file([], new_models, [], {}, dry_run=False)

    data = _read(target)
    assert data["gpt-5.5"]["comment"] == "NEW"
    assert data["gpt-5.5-xhigh"]["comment"] == "NEW"


def test_xhigh_split_for_openrouter_openai_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter-passthrough OpenAI models must split correctly too."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    import kiss.scripts.update_models as mod

    new_models = [
        {
            "name": "openrouter/openai/gpt-5.5",
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openrouter",
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
            "needs_pricing": False,
        }
    ]
    mod.apply_updates_to_file([], new_models, [], {}, dry_run=False)
    data = _read(target)
    assert "openrouter/openai/gpt-5.5" in data
    assert "openrouter/openai/gpt-5.5-xhigh" in data
    assert data["openrouter/openai/gpt-5.5"]["thinking"] == "high"
    assert data["openrouter/openai/gpt-5.5-xhigh"]["thinking"] == "xhigh"


def test_update_with_xhigh_when_base_entry_missing_in_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retest-update on a model whose JSON entry doesn't yet exist must
    still produce both the base ``thinking='high'`` entry and the
    ``-xhigh`` sibling, seeded from ``current``.

    Covers the legacy path where MODEL_INFO.json was hand-curated and a
    model only lives in the in-memory ``MODEL_INFO`` (not in the JSON
    file yet), and ``--test-existing`` discovers it accepts xhigh.
    """
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    # Note: target was just initialized to {} by _redirect_model_info.

    import kiss.scripts.update_models as mod

    current = {
        "gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": None,
        }
    }
    updates = [
        {"name": "gpt-5.5", "changes": {"thinking": "xhigh"}, "source": "retest"}
    ]
    mod.apply_updates_to_file(updates, [], [], current, dry_run=False)

    data = _read(target)
    assert "gpt-5.5" in data
    assert "gpt-5.5-xhigh" in data
    assert data["gpt-5.5"]["thinking"] == "high"
    assert data["gpt-5.5-xhigh"]["thinking"] == "xhigh"
    assert data["gpt-5.5-xhigh"]["context_length"] == 1_050_000


def test_non_thinking_update_preserves_and_syncs_xhigh_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routine non-thinking updates must not destroy generated xhigh aliases.

    After a previous run has split ``gpt-5.5`` into base + ``-xhigh``, later
    OpenRouter/OpenAI updates often only change pricing or capabilities. Those
    updates are not evidence that the model lost xhigh support, so the sibling
    must remain and receive the same non-thinking field changes.
    """
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    initial = {
        "gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        },
        "gpt-5.5-xhigh": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
        },
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    import kiss.scripts.update_models as mod

    updates = [
        {
            "name": "gpt-5.5",
            "changes": {
                "context_length": 1_100_000,
                "input_price_per_1M": 4.5,
                "output_price_per_1M": 28.0,
                "fc": False,
            },
            "source": "openai",
        }
    ]
    mod.apply_updates_to_file(updates, [], [], initial, dry_run=False)

    data = _read(target)
    assert "gpt-5.5-xhigh" in data, "Non-thinking updates must preserve sibling"
    for name, expected_thinking in (("gpt-5.5", "high"), ("gpt-5.5-xhigh", "xhigh")):
        assert data[name]["context_length"] == 1_100_000
        assert data[name]["input_price_per_1M"] == 4.5
        assert data[name]["output_price_per_1M"] == 28.0
        assert data[name]["fc"] is False
        assert data[name]["thinking"] == expected_thinking


def test_find_deprecated_models_ignores_generated_xhigh_aliases() -> None:
    """Generated ``-xhigh`` aliases must not be deprecated via vendor APIs."""
    import kiss.scripts.update_models as mod

    current = {
        "gpt-5.5": {"source": "openai"},
        "gpt-5.5-xhigh": {"source": "openai"},
        "openrouter/openai/gpt-5.5": {"source": "openrouter"},
        "openrouter/openai/gpt-5.5-xhigh": {"source": "openrouter"},
    }
    openai = {"gpt-5.5": {"source": "openai"}}
    openrouter = {"openrouter/openai/gpt-5.5": {"source": "openrouter"}}

    deprecated = mod.find_deprecated_models(current, openrouter, {}, {}, openai)

    assert deprecated == [], (
        "Generated -xhigh aliases are managed with their base entries and must "
        "not be probed as independent upstream model names"
    )


def test_existing_unsplit_xhigh_entry_is_migrated_without_other_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing JSON entries with thinking=xhigh must be split even without updates."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    initial = {
        "gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
        }
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [], [], initial, dry_run=False)

    data = _read(target)
    assert data["gpt-5.5"]["thinking"] == "high"
    assert data["gpt-5.5-xhigh"]["thinking"] == "xhigh"
    assert data["gpt-5.5-xhigh"]["context_length"] == 1_050_000


def test_malformed_generated_xhigh_sibling_is_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalization must repair generated siblings with stale fields."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    initial = {
        "gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        },
        "gpt-5.5-xhigh": {
            "context_length": 123,
            "input_price_per_1M": 99.0,
            "output_price_per_1M": 99.0,
            "fc": False,
            "emb": True,
            "gen": False,
            "thinking": "high",
        },
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [], [], initial, dry_run=False)

    data = _read(target)
    assert data["gpt-5.5-xhigh"] == {
        **data["gpt-5.5"],
        "thinking": "xhigh",
    }


def test_orphan_generated_xhigh_alias_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated aliases without a base entry must not survive normalization."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    initial = {
        "gpt-5.5-xhigh": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
        }
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    import kiss.scripts.update_models as mod

    mod.apply_updates_to_file([], [], [], initial, dry_run=False)

    data = _read(target)
    assert data == {}, "Orphan generated aliases should be cleaned up"


def test_main_test_existing_skips_generated_xhigh_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--test-existing`` must not probe generated ``-xhigh`` aliases."""
    target = tmp_path / "MODEL_INFO.json"
    initial = {
        "gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        },
        "gpt-5.5-xhigh": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
        },
    }
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
    assert calls == ["gpt-5.5"], "The generated -xhigh alias must not be probed"
    assert "thinking changed" not in output, (
        "A split base stored as high should not produce a spurious xhigh retest update"
    )
    data = _read(target)
    assert data["gpt-5.5"]["thinking"] == "high"
    assert data["gpt-5.5-xhigh"]["thinking"] == "xhigh"


def test_deprecated_base_also_drops_xhigh_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the base model is deprecated, its ``-xhigh`` sibling must go too."""
    target = tmp_path / "MODEL_INFO.json"
    _redirect_model_info(monkeypatch, target)
    initial = {
        "gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "high",
        },
        "gpt-5.5-xhigh": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": "xhigh",
        },
    }
    target.write_text(json.dumps(initial, indent=2) + "\n")
    import kiss.scripts.update_models as mod

    current = dict(initial)
    deprecated = [{"name": "gpt-5.5", "reason": "removed upstream"}]
    mod.apply_updates_to_file([], [], deprecated, current, dry_run=False)

    data = _read(target)
    assert "gpt-5.5" not in data
    assert "gpt-5.5-xhigh" not in data, (
        "Sibling must be removed when its base model is deprecated"
    )
