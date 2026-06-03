# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for update_models.py write-target resolution.

The script must write to the *source repository* model_info.py, not to a
bundled copy (e.g. the VS Code extension's kiss_project directory).  The
root cause of the bug was that PROJECT_ROOT was computed from __file__,
which resolves to the extension copy when Sorcar runs the script from the
extension's working directory.

These tests verify that _find_project_root() correctly resolves to the
source repo via KISS_WORKDIR, .git detection, or __file__ fallback.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def _make_fake_project(root: Path) -> None:
    """Create a minimal project structure with a dummy model_info.py."""
    model_info = root / "src" / "kiss" / "core" / "models" / "model_info.py"
    model_info.parent.mkdir(parents=True, exist_ok=True)
    model_info.write_text("# placeholder\n")


def test_find_project_root_prefers_kiss_workdir_over_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KISS_WORKDIR should take priority over __file__-based resolution."""
    source_repo = tmp_path / "source_repo"
    extension_copy = tmp_path / "extension_copy"
    _make_fake_project(source_repo)
    _make_fake_project(extension_copy)

    monkeypatch.setenv("KISS_WORKDIR", str(source_repo))
    monkeypatch.chdir(extension_copy)

    import kiss.scripts.update_models as mod

    # _find_project_root must prefer KISS_WORKDIR
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
    (source_repo / ".git").mkdir()  # mark as git repo

    extension_copy = tmp_path / "extension_copy"
    _make_fake_project(extension_copy)
    # extension has no .git

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
    monkeypatch.chdir(tmp_path)  # no .git, no project structure

    import kiss.scripts.update_models as mod

    result = mod._find_project_root()
    # Should still return a valid path (the real source repo via __file__)
    expected_marker = result / "src" / "kiss" / "core" / "models" / "model_info.py"
    assert expected_marker.exists(), (
        f"Fallback PROJECT_ROOT {result} does not contain model_info.py"
    )


def test_apply_updates_writes_to_workdir_not_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: apply_updates_to_file must write to the KISS_WORKDIR target.

    Simulates the bug scenario: script __file__ is in an extension copy,
    but KISS_WORKDIR points to the source repo. The write must go to the
    source repo, not the extension copy.
    """
    source_repo = tmp_path / "source_repo"
    _make_fake_project(source_repo)
    (source_repo / ".git").mkdir()

    # Write a minimal but valid model_info.py content
    model_info_content = textwrap.dedent('''\
        from dataclasses import dataclass

        @dataclass
        class ModelInfo:
            context_length: int = 0
            input_price_per_1M: float = 0.0
            output_price_per_1M: float = 0.0
            is_function_calling_supported: bool = True
            is_embedding_supported: bool = False
            is_generation_supported: bool = True

        def _mi(ctx, inp, out, fc=True, emb=False, gen=True):
            return ModelInfo(ctx, inp, out, fc, emb, gen)

        def _emb(ctx, inp):
            return ModelInfo(ctx, inp, 0.0, False, True, False)

        MODEL_INFO: dict[str, ModelInfo] = {
            "test-model-a": _mi(100000, 1.00, 2.00),
            "test-model-b": _mi(200000, 3.00, 4.00),
        }
    ''')

    source_model_info = source_repo / "src" / "kiss" / "core" / "models" / "model_info.py"
    source_model_info.write_text(model_info_content)

    extension_copy = tmp_path / "extension_copy"
    _make_fake_project(extension_copy)
    ext_model_info = extension_copy / "src" / "kiss" / "core" / "models" / "model_info.py"
    ext_model_info.write_text(model_info_content)

    monkeypatch.setenv("KISS_WORKDIR", str(source_repo))

    import kiss.scripts.update_models as mod

    # Redirect MODULE_INFO_PATH to the source repo
    resolved_root = mod._find_project_root()
    resolved_path = resolved_root / "src" / "kiss" / "core" / "models" / "model_info.py"

    # Monkeypatch the module-level constant to use the resolved path
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", resolved_path)

    current = {
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
    updates = [
        {
            "name": "test-model-a",
            "changes": {"input_price_per_1M": 1.50},
            "source": "openrouter",
        }
    ]

    mod.apply_updates_to_file(updates, [], [], current, dry_run=False)

    # Source repo file should be updated
    source_content = source_model_info.read_text()
    assert "1.50" in source_content, "Source repo model_info.py was not updated"

    # Extension copy should NOT be modified
    ext_content = ext_model_info.read_text()
    assert "1.50" not in ext_content, "Extension copy was incorrectly modified"


def test_find_project_root_exists_and_returns_valid_path() -> None:
    """_find_project_root must exist, be callable, and return a valid project root."""
    import kiss.scripts.update_models as mod

    assert callable(mod._find_project_root)
    result = mod._find_project_root()
    assert (result / "src" / "kiss" / "core" / "models" / "model_info.py").exists()
