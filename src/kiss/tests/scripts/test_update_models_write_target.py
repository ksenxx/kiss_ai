# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for update_models.py write-target resolution.

The script must write to the *source repository* ``MODEL_INFO.json``, not
to a bundled copy (e.g. an installed copy of the project).
The root cause of the original bug was that ``PROJECT_ROOT`` was computed
from ``__file__``, which resolves to the bundled copy when Sorcar runs
the script from the installed copy's working directory.

These tests verify that ``_find_project_root()`` correctly resolves to the
source repo via ``KISS_WORKDIR``, ``.git`` detection, or ``__file__`` fallback,
and that ``apply_updates_to_file`` writes through the resolved path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_fake_project(root: Path) -> None:
    """Create a minimal project structure with a dummy MODEL_INFO.json."""
    model_info = root / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    model_info.parent.mkdir(parents=True, exist_ok=True)
    model_info.write_text("{}\n")


def test_find_project_root_prefers_kiss_workdir_over_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KISS_WORKDIR should take priority over __file__-based resolution."""
    source_repo = tmp_path / "source_repo"
    bundled_copy = tmp_path / "bundled_copy"
    _make_fake_project(source_repo)
    _make_fake_project(bundled_copy)

    monkeypatch.setenv("KISS_WORKDIR", str(source_repo))
    monkeypatch.chdir(bundled_copy)

    import kiss.scripts.update_models as mod

    result = mod._find_project_root()
    assert result == source_repo, (
        f"Expected {source_repo}, got {result}. "
        "The script would write to the wrong directory."
    )


def test_find_project_root_uses_git_dir_when_no_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When KISS_WORKDIR is unset, a CWD with .git should be preferred."""
    source_repo = tmp_path / "source_repo"
    _make_fake_project(source_repo)
    (source_repo / ".git").mkdir()

    bundled_copy = tmp_path / "bundled_copy"
    _make_fake_project(bundled_copy)

    monkeypatch.delenv("KISS_WORKDIR", raising=False)
    monkeypatch.chdir(source_repo)

    import kiss.scripts.update_models as mod

    result = mod._find_project_root()
    assert result == source_repo


def test_find_project_root_falls_back_to_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When KISS_WORKDIR is unset and CWD has no .git, fall back to __file__."""
    monkeypatch.delenv("KISS_WORKDIR", raising=False)
    monkeypatch.chdir(tmp_path)

    import kiss.scripts.update_models as mod

    result = mod._find_project_root()
    expected_marker = result / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    assert expected_marker.exists(), (
        f"Fallback PROJECT_ROOT {result} does not contain MODEL_INFO.json"
    )


def test_apply_updates_writes_to_workdir_not_bundled_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: apply_updates_to_file must write to the KISS_WORKDIR target.

    Simulates the bug scenario: script ``__file__`` is in a bundled copy,
    but ``KISS_WORKDIR`` points to the source repo. The write must go to the
    source repo, not the bundled copy.
    """
    source_repo = tmp_path / "source_repo"
    _make_fake_project(source_repo)
    (source_repo / ".git").mkdir()

    initial = {
        "test-model-a": {
            "context_length": 100000,
            "input_price_per_1M": 1.00,
            "output_price_per_1M": 2.00,
            "fc": True,
            "emb": False,
            "gen": True,
        },
        "test-model-b": {
            "context_length": 200000,
            "input_price_per_1M": 3.00,
            "output_price_per_1M": 4.00,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    }
    source_model_info = source_repo / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    source_model_info.write_text(json.dumps(initial, indent=2) + "\n")

    bundled_copy = tmp_path / "bundled_copy"
    _make_fake_project(bundled_copy)
    bundled_model_info = bundled_copy / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    bundled_model_info.write_text(json.dumps(initial, indent=2) + "\n")

    monkeypatch.setenv("KISS_WORKDIR", str(source_repo))

    import kiss.scripts.update_models as mod

    resolved_root = mod._find_project_root()
    resolved_path = resolved_root / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", resolved_path)

    updates = [
        {
            "name": "test-model-a",
            "changes": {"input_price_per_1M": 1.50},
            "source": "openrouter",
        }
    ]

    mod.apply_updates_to_file(updates, [], [], initial, dry_run=False)

    source_content = json.loads(source_model_info.read_text())
    assert source_content["test-model-a"]["input_price_per_1M"] == 1.50, (
        "Source repo MODEL_INFO.json was not updated"
    )

    bundled_content = json.loads(bundled_model_info.read_text())
    assert bundled_content["test-model-a"]["input_price_per_1M"] == 1.00, (
        "Bundled copy was incorrectly modified"
    )


def test_find_project_root_exists_and_returns_valid_path() -> None:
    """_find_project_root must exist, be callable, and return a valid project root."""
    import kiss.scripts.update_models as mod

    assert callable(mod._find_project_root)
    result = mod._find_project_root()
    assert (result / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json").exists()
