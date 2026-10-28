# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that ``update_models.py`` onboards Amazon Bedrock's Claude ids.

``fetch_bedrock`` lists the Claude ids the configured region serves (the
``ON_DEMAND`` foundation ids plus the system-defined cross-region inference
profiles) and ``compute_changes`` mirrors each as a ``bedrock/<id>`` entry
copied from its ``claude-*`` twin: same context and capabilities, prices
scaled by the Bedrock regional premium, and the twin's effort ladder.
``find_deprecated_models`` retires a ``bedrock/`` entry the region no
longer lists, but only when the listing succeeded.

The tests drive the production functions end-to-end on synthetic
listings; the one live test runs the real listing when AWS credentials are
in the environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kiss.core import config as config_module
from kiss.scripts import update_models
from kiss.scripts.update_models import (
    _add_bedrock_candidates,
    _assume_untested_capabilities,
    _probe_new_models,
    _readme_provider_category,
    _sync_bedrock_thinking,
    _thinking_scale_for,
    apply_updates_to_file,
    compute_changes,
    fetch_bedrock,
    find_deprecated_models,
    sync_readme_catalog,
)

_US_OPUS = "bedrock/us.anthropic.claude-opus-4-6-v1"
_GLOBAL_OPUS = "bedrock/global.anthropic.claude-opus-4-6-v1"
_US_HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
_US_SONNET_4 = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
_US_CLAUDE_3 = "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0"

_AWS_FIELDS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
)
_AWS_ENV = (
    *_AWS_FIELDS,
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_PROFILE",
)
# Snapshot of the developer's ambient AWS variables, taken at import: the
# autouse ``no_ambient_aws_credentials`` fixture (conftest.py) unsets them
# for every test, so the live test below puts this snapshot back.
_AMBIENT_AWS = {name: os.environ[name] for name in _AWS_ENV if name in os.environ}
_LIVE_CREDENTIALS = bool(_AMBIENT_AWS.get("AWS_BEARER_TOKEN_BEDROCK")) or bool(
    _AMBIENT_AWS.get("AWS_ACCESS_KEY_ID") and _AMBIENT_AWS.get("AWS_SECRET_ACCESS_KEY")
)


def _claude_entry(inp: float, out: float, thinking: str | None = None, **extra) -> dict:
    """Build a direct ``claude-*`` catalog entry as ``get_current_model_info`` returns it."""
    entry = {
        "context_length": 200000,
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


def _catalog_with_opus() -> dict[str, dict]:
    """A catalog holding Opus 4.6 at $5/$25 with its ``-low`` … ``-max`` aliases."""
    current = {"claude-opus-4-6": _claude_entry(5.0, 25.0, "high")}
    for level in ("low", "medium", "high", "max"):
        current[f"claude-opus-4-6-{level}"] = _claude_entry(
            5.0, 25.0, level, alias_of="claude-opus-4-6"
        )
    return current


def _listing(*names: str) -> dict[str, dict]:
    """A ``fetch_bedrock`` result for *names*."""
    from kiss.core.models.model_info import bedrock_claude_name

    return {name: {"source": "bedrock", "claude": bedrock_claude_name(name)} for name in names}


def _by_name(new_models: list[dict]) -> dict[str, dict]:
    """Index a ``new_models`` list by model name."""
    return {m["name"]: m for m in new_models}


def _changes(
    current: dict[str, dict], bedrock: dict[str, dict], openrouter: dict[str, dict] | None = None
) -> tuple[list[dict], list[dict]]:
    """Run the updater's Bedrock pipeline (candidates, then the thinking sync) on *bedrock*."""
    updates, new_models = compute_changes(
        current, openrouter or {}, {}, {}, {}, {}, bedrock=bedrock
    )
    _sync_bedrock_thinking(bedrock, current, updates, new_models)
    return updates, new_models


def test_bedrock_ids_mirror_their_claude_twin() -> None:
    """A geographic profile pays 10% over the twin; a global one pays the twin's rate."""
    updates, new_models = _changes(_catalog_with_opus(), _listing(_US_OPUS, _GLOBAL_OPUS))
    assert updates == []
    by_name = _by_name(new_models)
    us, glob = by_name[_US_OPUS], by_name[_GLOBAL_OPUS]
    assert (us["input_price_per_1M"], us["output_price_per_1M"]) == (5.5, 27.5)
    assert (glob["input_price_per_1M"], glob["output_price_per_1M"]) == (5.0, 25.0)
    for entry in (us, glob):
        assert entry["context_length"] == 200000
        assert (entry["fc"], entry["emb"], entry["gen"]) == (True, False, True)
        assert entry["source"] == "bedrock"
        assert entry["needs_pricing"] is False
        # The twin's base stores "high"; its -max alias proves the real top.
        assert entry["thinking"] == "max"


def test_dated_id_resolves_to_the_undated_twin() -> None:
    """``...-20251001-v1:0`` is priced by ``claude-haiku-4-5`` when no dated entry exists."""
    current = {"claude-haiku-4-5": _claude_entry(1.0, 5.0)}
    _, new_models = _changes(current, _listing(_US_HAIKU))
    entry = _by_name(new_models)[_US_HAIKU]
    assert (entry["input_price_per_1M"], entry["output_price_per_1M"]) == (1.1, 5.5)
    assert entry["thinking"] is None


def test_openrouter_twin_prices_an_id_the_catalog_lacks() -> None:
    """Without a ``claude-*`` entry the OpenRouter twin supplies context and prices."""
    openrouter = {
        "openrouter/anthropic/claude-sonnet-4": {
            "context_length": 1000000,
            "input_price_per_1M": 3.0,
            "output_price_per_1M": 15.0,
        }
    }
    _, new_models = _changes({}, _listing(_US_SONNET_4), openrouter)
    entry = _by_name(new_models)[_US_SONNET_4]
    # Sonnet 4 predates the regional premium (factor 1.0); the 1M window is capped.
    assert (entry["input_price_per_1M"], entry["output_price_per_1M"]) == (3.0, 15.0)
    assert entry["context_length"] == 500000
    assert entry["needs_pricing"] is False
    assert entry["thinking"] is None


def test_unknown_claude_model_is_added_unpriced() -> None:
    """An id neither the catalog nor OpenRouter knows is flagged ``needs_pricing``."""
    _, new_models = _changes({}, _listing(_US_CLAUDE_3))
    entry = _by_name(new_models)[_US_CLAUDE_3]
    assert entry["needs_pricing"] is True
    assert (entry["input_price_per_1M"], entry["output_price_per_1M"]) == (0.0, 0.0)
    assert entry["context_length"] == 200000
    assert (entry["fc"], entry["emb"], entry["gen"]) == (True, False, True)


def test_existing_entry_follows_a_twin_price_change() -> None:
    """A ``bedrock/`` entry already in the catalog is repriced when its twin moves."""
    current = _catalog_with_opus()
    current[_US_OPUS] = _claude_entry(4.4, 22.0, "max")  # written when Opus cost $4/$20
    current[_GLOBAL_OPUS] = _claude_entry(5.0, 25.0, "max")  # already in step
    updates, new_models = _changes(current, _listing(_US_OPUS, _GLOBAL_OPUS))
    assert new_models == []
    assert updates == [
        {
            "name": _US_OPUS,
            "changes": {"input_price_per_1M": 5.5, "output_price_per_1M": 27.5},
            "source": "bedrock",
        }
    ]


def test_existing_entry_without_a_twin_is_left_alone() -> None:
    """A hand-priced entry of an unknown Claude model keeps its prices."""
    current = {_US_CLAUDE_3: _claude_entry(0.25, 1.25)}
    updates: list[dict] = []
    new_models: list[dict] = []
    _add_bedrock_candidates(_listing(_US_CLAUDE_3), current, {}, updates, new_models)
    assert updates == [] and new_models == []


def test_bedrock_entries_skip_the_openrouter_repricing_pass() -> None:
    """An unpriced ``bedrock/`` entry is not repriced from OpenRouter like a ``claude-*`` one."""
    current = {_US_CLAUDE_3: _claude_entry(0.0, 0.0)}
    openrouter = {
        "openrouter/anthropic/claude-3-haiku": {
            "context_length": 200000,
            "input_price_per_1M": 0.25,
            "output_price_per_1M": 1.25,
        }
    }
    updates, _ = compute_changes(current, openrouter, {}, {}, {}, {})
    assert updates == []


def test_deprecation_needs_a_successful_listing() -> None:
    """An entry missing from a non-empty listing is retired; an empty listing retires nothing."""
    current = {_US_OPUS: _claude_entry(5.5, 27.5), _GLOBAL_OPUS: _claude_entry(5.0, 25.0)}
    current[f"{_US_OPUS}-max"] = _claude_entry(5.5, 27.5, "max", alias_of=_US_OPUS)
    assert find_deprecated_models(current, {}, {}, {}, {}, bedrock={}) == []
    assert find_deprecated_models(current, {}, {}, {}, {}) == []
    deprecated = find_deprecated_models(current, {}, {}, {}, {}, bedrock=_listing(_GLOBAL_OPUS))
    assert deprecated == [{"name": _US_OPUS, "reason": "not listed by Bedrock in this region"}]


@pytest.mark.parametrize(
    ("name", "scale"),
    [
        (_US_OPUS, ("low", "medium", "high", "max")),
        ("bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0", ("low", "medium", "high")),
        ("bedrock/us.anthropic.claude-fable-5-1", ("low", "medium", "high", "xhigh", "max")),
        ("bedrock/amazon.nova-pro-v1:0", ("low", "medium", "high", "xhigh", "max")),
    ],
)
def test_bedrock_names_use_the_anthropic_effort_ladder(name: str, scale: tuple[str, ...]) -> None:
    """The ladder is the one of the Claude model behind the id, not the OpenAI ladder."""
    assert _thinking_scale_for(name) == scale


def test_mirror_candidates_are_neither_probed_nor_reset() -> None:
    """Neither the live probe nor ``--skip-test`` touches a ``bedrock/`` candidate."""
    _, new_models = _changes(_catalog_with_opus(), _listing(_US_OPUS))
    before = [dict(nm) for nm in new_models]
    assert _probe_new_models(new_models) == before
    _assume_untested_capabilities(new_models)
    assert new_models == before


def test_apply_writes_entries_with_effort_aliases(tmp_path: Path, monkeypatch) -> None:
    """The written entry carries ``-low`` … ``-max`` siblings on the Anthropic ladder."""
    target = tmp_path / "MODEL_INFO.json"
    current = _catalog_with_opus()
    target.write_text(
        json.dumps(
            {
                k: {f: v for f, v in e.items() if v is not None and f != "use_responses_api"}
                for k, e in current.items()
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", target)
    _, new_models = _changes(current, _listing(_US_OPUS))
    apply_updates_to_file([], new_models, [], current)
    data = json.loads(target.read_text(encoding="utf-8"))
    base = data[_US_OPUS]
    assert base["thinking"] == "high"
    assert (base["input_price_per_1M"], base["output_price_per_1M"]) == (5.5, 27.5)
    assert base["comment"] == "NEW"
    for level in ("low", "medium", "high", "max"):
        sibling = data[f"{_US_OPUS}-{level}"]
        assert sibling["alias_of"] == _US_OPUS
        assert sibling["thinking"] == level
        assert sibling["input_price_per_1M"] == 5.5
    assert f"{_US_OPUS}-xhigh" not in data

    # Retiring the base removes its aliases with it.
    apply_updates_to_file([], [], [{"name": _US_OPUS, "reason": "gone"}], current)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert not any(name.startswith(_US_OPUS) for name in data)


def test_docs_sync_counts_bedrock_as_its_own_category(tmp_path: Path) -> None:
    """``bedrock/`` entries fill the "Amazon Bedrock" row and list of MODELS.md."""
    assert _readme_provider_category(_US_OPUS) == "Amazon Bedrock (`bedrock/*`)"
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(
        json.dumps(
            {
                "claude-opus-4-6": {
                    "context_length": 1,
                    "input_price_per_1M": 1,
                    "output_price_per_1M": 1,
                    "fc": True,
                    "emb": False,
                    "gen": True,
                },
                _US_OPUS: {
                    "context_length": 1,
                    "input_price_per_1M": 1,
                    "output_price_per_1M": 1,
                    "fc": True,
                    "emb": False,
                    "gen": True,
                },
                _GLOBAL_OPUS: {
                    "context_length": 1,
                    "input_price_per_1M": 1,
                    "output_price_per_1M": 1,
                    "fc": True,
                    "emb": False,
                    "gen": True,
                },
            }
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "MODELS.md"
    header = "<summary><strong>Amazon Bedrock (bedrock/*) ({})</strong></summary>"
    doc.write_text(
        "| Anthropic | 0 |\n| Amazon Bedrock (`bedrock/*`) | 0 |\n\n"
        f"<details>\n{header.format(0)}\n\n</details>\n",
        encoding="utf-8",
    )
    assert sync_readme_catalog(doc, catalog) is True
    text = doc.read_text(encoding="utf-8")
    assert "| Anthropic | 1 |" in text
    assert "| Amazon Bedrock (`bedrock/*`) | 2 |" in text
    assert f"{header.format(2)}\n\n- `{_GLOBAL_OPUS}`\n- `{_US_OPUS}`\n" in text


def test_fetch_is_skipped_without_credentials(monkeypatch, tmp_path: Path, capsys) -> None:
    """No token, key pair or ambient source: the listing is skipped, nothing retired."""
    for field in _AWS_FIELDS:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, field, "")
    for var in _AWS_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    assert fetch_bedrock(verbose=True) == {}
    assert "skipping Bedrock" in capsys.readouterr().out


def test_bad_credentials_skip_the_listing_without_raising(monkeypatch, capsys) -> None:
    """A rejected listing (bogus key pair) is reported and yields an empty result."""
    for field in _AWS_FIELDS:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, field, "")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setattr(
        config_module.DEFAULT_CONFIG,
        "AWS_SECRET_ACCESS_KEY",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_REGION", "us-east-1")
    assert fetch_bedrock() == {}
    assert "Bedrock listing failed" in capsys.readouterr().out
    # A client that cannot even be built (invalid region) is skipped the same way.
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "AWS_REGION", "bad region")
    assert fetch_bedrock() == {}
    assert "Bedrock listing failed" in capsys.readouterr().out


def test_twin_discovered_in_the_same_run_hands_over_its_probed_level() -> None:
    """A Bedrock id of a Claude model that is itself new this run gets the probed level.

    ``compute_changes`` cannot know the level yet (the twin is not in the
    catalog); the probe settles it on the twin's candidate, and the sync
    after the probes copies it to the mirror.
    """
    openrouter = {
        "openrouter/anthropic/claude-opus-4.6": {
            "context_length": 200000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 25.0,
        }
    }
    bedrock = _listing(_GLOBAL_OPUS)
    updates, new_models = compute_changes(
        {}, openrouter, {}, {}, {"claude-opus-4-6": {"source": "anthropic"}}, {}, bedrock=bedrock
    )
    by_name = _by_name(new_models)
    assert by_name[_GLOBAL_OPUS]["thinking"] is None
    by_name["claude-opus-4-6"]["thinking"] = "max"  # what the live probe records
    _sync_bedrock_thinking(bedrock, {}, updates, new_models)
    assert by_name[_GLOBAL_OPUS]["thinking"] == "max"


def test_existing_entry_follows_the_twin_level(tmp_path: Path, monkeypatch) -> None:
    """An existing Bedrock entry whose effort level differs from its twin's is resynced.

    Covers both a retest that changed the twin this run (the level comes
    from the update record) and a stored difference from an earlier run
    (the level comes from the twin's aliases); ``--test-existing`` never
    probes the Bedrock entry itself, so its aliases are never wiped.
    """
    current = _catalog_with_opus()
    current[_US_OPUS] = _claude_entry(5.5, 27.5)  # written before the twin had a level
    bedrock = _listing(_US_OPUS)
    updates, new_models = _changes(current, bedrock)
    assert new_models == []
    assert updates == [{"name": _US_OPUS, "changes": {"thinking": "max"}, "source": "bedrock"}]

    # The twin's retest lowered it this run: the mirror follows the retest, not the disk.
    updates = [{"name": "claude-opus-4-6", "changes": {"thinking": "high"}, "source": "retest"}]
    _sync_bedrock_thinking(bedrock, current, updates, [])
    assert updates[1] == {"name": _US_OPUS, "changes": {"thinking": "high"}, "source": "bedrock"}

    # Already in step: nothing to update.
    current[f"{_US_OPUS}-max"] = _claude_entry(5.5, 27.5, "max", alias_of=_US_OPUS)
    current[_US_OPUS]["thinking"] = "high"
    assert _changes(current, bedrock) == ([], [])

    # Applying the resync materializes the siblings on disk.
    target = tmp_path / "MODEL_INFO.json"
    on_disk = {_US_OPUS: {k: v for k, v in _claude_entry(5.5, 27.5).items() if v is not None}}
    on_disk[_US_OPUS].pop("use_responses_api")
    target.write_text(json.dumps(on_disk), encoding="utf-8")
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", target)
    apply_updates_to_file(
        [{"name": _US_OPUS, "changes": {"thinking": "max"}, "source": "bedrock"}], [], [], current
    )
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data[_US_OPUS]["thinking"] == "high"
    assert {data[f"{_US_OPUS}-{lvl}"]["thinking"] for lvl in ("low", "medium", "high", "max")} == {
        "low",
        "medium",
        "high",
        "max",
    }


def test_main_repairs_a_thinking_only_drift(tmp_path: Path, monkeypatch, capsys) -> None:
    """A run with nothing else to do still gives a level-less Bedrock entry its twin's aliases.

    ``main()`` short-circuits with "Everything is up to date!" when no
    update, new model or retirement is pending, so a thinking-only drift
    must count as an update already at change computation, not only in
    the post-probe sync.
    """
    import sys

    on_disk = {
        k: {f: v for f, v in e.items() if v is not None and f != "use_responses_api"}
        for k, e in _catalog_with_opus().items()
    }
    on_disk[_US_OPUS] = {
        "context_length": 200000,
        "input_price_per_1M": 5.5,
        "output_price_per_1M": 27.5,
        "fc": True,
        "emb": False,
        "gen": True,
    }
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps(on_disk, indent=2) + "\n", encoding="utf-8")
    current = _catalog_with_opus()
    current[_US_OPUS] = _claude_entry(5.5, 27.5)
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(update_models, "README_PATH", tmp_path / "README.md")
    for fetcher in (
        "fetch_openrouter",
        "fetch_together",
        "fetch_anthropic",
        "fetch_gemini",
        "fetch_openai",
    ):
        monkeypatch.setattr(update_models, fetcher, lambda verbose=False: {})
    monkeypatch.setattr(update_models, "fetch_codex_supported_slugs", lambda verbose=False: set())
    monkeypatch.setattr(update_models, "fetch_bedrock", lambda verbose=False: _listing(_US_OPUS))
    monkeypatch.setattr(update_models, "get_current_model_info", lambda: dict(current))
    monkeypatch.setattr(sys, "argv", ["update_models.py"])
    update_models.main()
    out = capsys.readouterr().out
    assert "Everything is up to date!" not in out
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data[_US_OPUS]["thinking"] == "high"
    assert data[f"{_US_OPUS}-max"]["alias_of"] == _US_OPUS


@pytest.mark.skipif(not _LIVE_CREDENTIALS, reason="no Bedrock credentials in the environment")
def test_live_listing_returns_claude_profiles(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real listing yields ``bedrock/`` Claude ids, cross-region profiles among them.

    Credentials whose IAM policy lacks ``bedrock:ListFoundationModels`` make
    ``fetch_bedrock`` warn and return ``{}`` (its documented behaviour, covered
    by the offline tests); that environment cannot exercise the live listing,
    so the test skips rather than fail.
    """
    from kiss.core.models.model_info import bedrock_claude_name

    for name, value in _AMBIENT_AWS.items():
        monkeypatch.setenv(name, value)
    listing = fetch_bedrock(verbose=True)
    if not listing and "AccessDeniedException" in capsys.readouterr().out:
        pytest.skip("Bedrock credentials are not authorized to list foundation models")
    assert listing
    for name, info in listing.items():
        assert name.startswith("bedrock/")
        assert info == {"source": "bedrock", "claude": bedrock_claude_name(name)}
        assert info["claude"].startswith("claude-")
    assert any(
        name.startswith(("bedrock/us.", "bedrock/global.", "bedrock/eu.")) for name in listing
    )
