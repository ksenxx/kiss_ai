# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end updater-pipeline test for the Codex subscription exclusions.

Follow-up to ``test_audit0903_codex_subscription_exclusions.py``, whose
tests call the private helpers directly.  This test runs the real
production pipeline over a real on-disk catalog file — the same three
stages ``main()`` runs after fetching (``find_deprecated_models`` →
``compute_changes`` → ``apply_updates_to_file``) — and asserts on the
JSON that lands on disk.  Only ``MODEL_INFO_PATH`` is redirected to a
temporary file, the established pattern from
``test_update_models_context_cap.py``; no updater function is mocked.

Scenario: the catalog contains the three subscription-incompatible
``codex/*`` entries plus a valid one, and the upstream Codex
``models.json`` slug set still lists all four (exactly the live state
that resurrected the entries via commit ``778fa15e``).  After one
pipeline pass the incompatible entries must be gone and the valid one
untouched; after a second pass over the cleaned catalog — where the
incompatible slugs are pure *candidates* — they must not be re-added.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kiss.scripts.update_models as update_models
from kiss.scripts.update_models import (
    SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS,
    apply_updates_to_file,
    compute_changes,
    find_deprecated_models,
)

_VALID_SLUG = "gpt-5.5"
_VALID_NAME = f"codex/{_VALID_SLUG}"
_INCOMPATIBLE_NAMES = tuple(
    f"codex/{slug}" for slug in sorted(SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS)
)


def _codex_entry() -> dict:
    """A catalog entry shaped like the real shipped ``codex/*`` entries."""
    return {
        "context_length": 400000,
        "input_price_per_1M": 0.0,
        "output_price_per_1M": 0.0,
        "fc": True,
        "emb": False,
        "gen": True,
    }


def _run_pipeline(codex_slugs: set[str]) -> None:
    """One real updater pass: deprecation, change computation, file write."""
    current = json.loads(update_models.MODEL_INFO_PATH.read_text(encoding="utf-8"))
    deprecated = find_deprecated_models(current, {}, {}, {}, {}, codex_slugs)
    updates, new_models = compute_changes(current, {}, {}, {}, {}, {}, codex_slugs)
    apply_updates_to_file(updates, new_models, deprecated, current, dry_run=False)


def test_pipeline_removes_and_never_readds_incompatible_codex_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The written catalog drops the incompatible entries and keeps the valid one."""
    target = tmp_path / "MODEL_INFO.json"
    seeded = {name: _codex_entry() for name in _INCOMPATIBLE_NAMES}
    seeded[_VALID_NAME] = _codex_entry()
    target.write_text(json.dumps(seeded, indent=4) + "\n", encoding="utf-8")
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", target)
    upstream_slugs = set(SUBSCRIPTION_INCOMPATIBLE_CODEX_SLUGS) | {_VALID_SLUG}

    # Pass 1: entries exist on disk and upstream still lists their slugs.
    _run_pipeline(upstream_slugs)
    written = json.loads(target.read_text(encoding="utf-8"))
    for name in _INCOMPATIBLE_NAMES:
        assert name not in written, f"{name} survived the updater pipeline"
    assert written[_VALID_NAME] == _codex_entry()

    # Pass 2: the cleaned catalog no longer has the entries, so the
    # incompatible slugs now flow through the new-candidate path — they
    # must not be resurrected.
    _run_pipeline(upstream_slugs)
    rewritten = json.loads(target.read_text(encoding="utf-8"))
    for name in _INCOMPATIBLE_NAMES:
        assert name not in rewritten, f"{name} was re-added by the updater pipeline"
    assert rewritten[_VALID_NAME] == _codex_entry()
