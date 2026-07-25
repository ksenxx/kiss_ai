# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the context-length cap in update_models.

Any model whose context length is 1000000 or above must be written to
``MODEL_INFO.json`` with a context length of 500000. The cap applies to
freshly fetched vendor data, codex candidates, and entries already
present on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.scripts.update_models import (
    _add_codex_candidates,
    _cap_context_length,
    _has_context_cap_changes,
    apply_updates_to_file,
)


def _redirect_model_info(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    import kiss.scripts.update_models as mod

    target.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)


def test_cap_context_length_boundaries() -> None:
    """Values below 1M pass through; 1M and above are capped at 500k."""
    assert _cap_context_length(0) == 0
    assert _cap_context_length(400_000) == 400_000
    assert _cap_context_length(999_999) == 999_999
    assert _cap_context_length(1_000_000) == 500_000
    assert _cap_context_length(1_048_576) == 500_000
    assert _cap_context_length(2_000_000) == 500_000


def test_has_context_cap_changes() -> None:
    """Detection must flag any entry with context_length >= 1M."""
    assert not _has_context_cap_changes({"a": {"context_length": 999_999}})
    assert _has_context_cap_changes(
        {"a": {"context_length": 128_000}, "b": {"context_length": 1_000_000}}
    )


def test_codex_candidate_context_is_capped() -> None:
    """Codex candidates sourced from a >=1M OpenRouter context must be capped."""
    openrouter = {
        "openrouter/openai/gpt-5.5": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openrouter",
        },
    }
    new_models: list[dict] = []
    _add_codex_candidates({"gpt-5.5"}, {}, openrouter, new_models)
    [entry] = [m for m in new_models if m["name"] == "codex/gpt-5.5"]
    assert entry["context_length"] == 500_000


def test_new_model_context_is_capped_on_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new model with a >=1M context must land on disk with 500k."""
    target = tmp_path / "MODEL_INFO.json"
    target.write_text("{}\n")
    _redirect_model_info(monkeypatch, target)
    new_models = [
        {
            "name": "vendor/huge-context",
            "context_length": 2_000_000,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 2.0,
            "source": "openrouter",
            "fc": True,
            "emb": False,
            "gen": True,
            "needs_pricing": False,
        }
    ]
    apply_updates_to_file([], new_models, [], {}, dry_run=False)
    data = json.loads(target.read_text())
    assert data["vendor/huge-context"]["context_length"] == 500_000


def test_existing_on_disk_entries_are_capped_on_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing >=1M entries are normalized to 500k; sub-1M kept as-is."""
    target = tmp_path / "MODEL_INFO.json"
    on_disk = {
        "vendor/big": {
            "context_length": 1_048_576,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 2.0,
            "fc": True,
            "emb": False,
            "gen": True,
        },
        "vendor/small": {
            "context_length": 128_000,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 2.0,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    }
    target.write_text(json.dumps(on_disk))
    _redirect_model_info(monkeypatch, target)
    apply_updates_to_file([], [], [], {}, dry_run=False)
    data = json.loads(target.read_text())
    assert data["vendor/big"]["context_length"] == 500_000
    assert data["vendor/small"]["context_length"] == 128_000
