# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests OpenRouter pricing cross-reference for hyphen-versioned Anthropic IDs.

Anthropic model IDs separate major and minor versions with a hyphen
(``claude-fable-5-1``) while the matching OpenRouter slugs use a dot
(``anthropic/claude-fable-5.1``). ``_lookup_openrouter_pricing`` must bridge
that separator mismatch, otherwise every Anthropic model with a minor version
lands in ``MODEL_INFO.json`` as $0.00/$0.00 with a ``"NEW: needs pricing"``
comment — the exact bug that produced the broken ``claude-fable-5-1`` entry.

These tests drive the production ``compute_changes`` and
``_lookup_openrouter_pricing`` functions end-to-end with synthetic vendor
payloads — the same data flow the script uses against the live APIs.
"""

import json
from pathlib import Path

import pytest

from kiss.scripts.update_models import _lookup_openrouter_pricing, compute_changes

_OR_FABLE_5_1 = {
    "openrouter/anthropic/claude-fable-5.1": {
        "context_length": 1000000,
        "input_price_per_1M": 10.0,
        "output_price_per_1M": 50.0,
        "source": "openrouter",
    }
}


def test_lookup_bridges_hyphen_to_dot_version() -> None:
    """``claude-fable-5-1`` resolves to the ``claude-fable-5.1`` OpenRouter slug."""
    info = _lookup_openrouter_pricing("claude-fable-5-1", "anthropic", _OR_FABLE_5_1)
    assert info is not None
    assert info["input_price_per_1M"] == 10.0
    assert info["output_price_per_1M"] == 50.0


def test_lookup_bridges_dated_snapshot_with_hyphen_version() -> None:
    """A dated snapshot strips the date, then bridges hyphen to dot."""
    info = _lookup_openrouter_pricing("claude-fable-5-1-20260815", "anthropic", _OR_FABLE_5_1)
    assert info is not None
    assert info["input_price_per_1M"] == 10.0


def test_lookup_prefers_exact_match_over_dot_rewrite() -> None:
    """An exact OpenRouter key wins over the dotted rewrite of the same name."""
    openrouter = dict(_OR_FABLE_5_1)
    openrouter["openrouter/anthropic/claude-fable-5-1"] = {
        "context_length": 200000,
        "input_price_per_1M": 7.0,
        "output_price_per_1M": 35.0,
        "source": "openrouter",
    }
    info = _lookup_openrouter_pricing("claude-fable-5-1", "anthropic", openrouter)
    assert info is not None
    assert info["input_price_per_1M"] == 7.0


def test_lookup_returns_none_when_no_variant_matches() -> None:
    """No exact, date-stripped, or dotted variant → None (caller marks needs_pricing)."""
    assert _lookup_openrouter_pricing("claude-nova-9-9", "anthropic", _OR_FABLE_5_1) is None


def test_new_anthropic_model_with_minor_version_gets_openrouter_pricing() -> None:
    """A new hyphen-versioned Anthropic model is priced from OpenRouter, not $0/$0."""
    anthropic = {"claude-fable-5-1": {"source": "anthropic"}}
    _, new_models = compute_changes({}, _OR_FABLE_5_1, {}, {}, anthropic, {})
    entry = next(m for m in new_models if m["name"] == "claude-fable-5-1")
    assert entry["input_price_per_1M"] == 10.0
    assert entry["output_price_per_1M"] == 50.0
    assert entry["context_length"] == 1000000
    assert entry["needs_pricing"] is False


def test_existing_zero_priced_entry_is_backfilled_from_openrouter() -> None:
    """An existing $0/$0 catalog entry gets an openrouter-xref pricing update."""
    current = {
        "claude-fable-5-1": {
            "context_length": 200000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
        }
    }
    anthropic = {"claude-fable-5-1": {"source": "anthropic"}}
    updates, _ = compute_changes(current, _OR_FABLE_5_1, {}, {}, anthropic, {})
    upd = next(u for u in updates if u["name"] == "claude-fable-5-1")
    assert upd["changes"]["input_price_per_1M"] == 10.0
    assert upd["changes"]["output_price_per_1M"] == 50.0


def test_pricing_backfill_clears_stale_needs_pricing_comment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing a pricing backfill rewrites ``"NEW: needs pricing"`` to ``"NEW"``.

    Runs the full compute → apply pipeline against a temp catalog file: the
    stale $0/$0 entry gets OpenRouter pricing and its now-false comment is
    downgraded so the catalog no longer claims the price is unknown.
    """
    import kiss.scripts.update_models as mod

    current = {
        "claude-fable-5-1": {
            "context_length": 200000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
        }
    }
    catalog = tmp_path / "MODEL_INFO.json"
    on_disk = {name: dict(entry, comment="NEW: needs pricing") for name, entry in current.items()}
    catalog.write_text(json.dumps(on_disk, indent=2) + "\n")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)

    anthropic = {"claude-fable-5-1": {"source": "anthropic"}}
    updates, new_models = compute_changes(current, _OR_FABLE_5_1, {}, {}, anthropic, {})
    mod.apply_updates_to_file(updates, new_models, [], current)

    written = json.loads(catalog.read_text())
    entry = written["claude-fable-5-1"]
    assert entry["input_price_per_1M"] == 10.0
    assert entry["output_price_per_1M"] == 50.0
    assert entry["comment"] == "NEW"
