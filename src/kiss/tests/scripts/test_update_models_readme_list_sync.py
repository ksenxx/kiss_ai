# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for per-provider model-list syncing in MODELS.md.

``update_models.py``'s ``sync_readme_catalog`` historically rewrote only the
*counts* of the catalog section, so the full per-provider model lists inside
the ``<details>`` blocks (now in ``MODELS.md``) could silently drift from
``MODEL_INFO.json``. These tests exercise the rewriter end-to-end on real
files copied to a temp dir and verify that it regenerates every list, and
that the README's totals are synced without a model list of their own.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import kiss.scripts.update_models as update_models
from kiss.scripts.update_models import sync_catalog_docs, sync_readme_catalog

_REPO_ROOT = Path(__file__).resolve().parents[4]
_README = _REPO_ROOT / "README.md"
_MODELS = _REPO_ROOT / "MODELS.md"
_MODEL_INFO = _REPO_ROOT / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"


def _copy_repo_files(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the repo's MODELS.md and MODEL_INFO.json into ``tmp_path``."""
    readme = tmp_path / "MODELS.md"
    model_info = tmp_path / "MODEL_INFO.json"
    shutil.copy(_MODELS, readme)
    shutil.copy(_MODEL_INFO, model_info)
    return readme, model_info


def test_repo_readme_totals_are_in_sync(tmp_path: Path) -> None:
    """The checked-in README's catalog totals match the bundled catalog.

    The README carries no model list, so a sync must be a no-op on it.
    """
    readme = tmp_path / "README.md"
    shutil.copy(_README, readme)
    pristine = readme.read_text(encoding="utf-8")
    assert "<details>\n<summary><strong>OpenAI (" not in pristine
    assert sync_readme_catalog(readme, _MODEL_INFO) is False
    assert readme.read_text(encoding="utf-8") == pristine


def test_sync_catalog_docs_updates_both_files(tmp_path: Path, monkeypatch) -> None:
    """``sync_catalog_docs`` repairs README.md and MODELS.md and skips a missing file."""
    readme = tmp_path / "README.md"
    models = tmp_path / "MODELS.md"
    shutil.copy(_README, readme)
    shutil.copy(_MODELS, models)
    pristine_readme = readme.read_text(encoding="utf-8")
    pristine_models = models.read_text(encoding="utf-8")
    monkeypatch.setattr(update_models, "README_PATH", readme)
    monkeypatch.setattr(update_models, "MODELS_PATH", models)
    monkeypatch.setattr(update_models, "MODEL_INFO_PATH", _MODEL_INFO)

    assert sync_catalog_docs() is False

    readme.write_text(
        pristine_readme.replace("ships a catalog of **", "ships a catalog of **1"), encoding="utf-8"
    )
    models.write_text(pristine_models.replace("| Z.AI | 8 |", "| Z.AI | 999 |"), encoding="utf-8")
    assert sync_catalog_docs() is True
    assert readme.read_text(encoding="utf-8") == pristine_readme
    assert models.read_text(encoding="utf-8") == pristine_models

    models.unlink()
    assert sync_catalog_docs() is False


def test_repo_readme_lists_are_in_sync(tmp_path: Path) -> None:
    """The checked-in README's model lists match the bundled catalog.

    Running the rewriter on pristine copies must be a no-op; if this fails,
    the README has drifted at HEAD and ``update_models.py`` should be re-run.
    """
    readme, model_info = _copy_repo_files(tmp_path)
    pristine = readme.read_text(encoding="utf-8")
    assert sync_readme_catalog(readme, model_info) is False
    assert readme.read_text(encoding="utf-8") == pristine


def test_drifted_lists_and_counts_are_fully_repaired(tmp_path: Path) -> None:
    """Dropped, stale, unsorted entries and wrong counts are all repaired."""
    readme, model_info = _copy_repo_files(tmp_path)
    pristine = readme.read_text(encoding="utf-8")

    drifted = pristine.replace("- `glm-4.7`\n", "")  # model missing from list
    drifted = drifted.replace(
        "- `kimi-k3`\n", "- `zz-stale-model`\n- `kimi-k3`\n"
    )  # stale + out-of-order entry
    drifted = drifted.replace("<strong>Z.AI (8)</strong>", "<strong>Z.AI (999)</strong>")
    drifted = drifted.replace("| Z.AI | 8 |", "| Z.AI | 999 |")
    assert drifted != pristine
    readme.write_text(drifted, encoding="utf-8")

    assert sync_readme_catalog(readme, model_info) is True
    assert readme.read_text(encoding="utf-8") == pristine


def test_category_emptied_to_zero_is_synced(tmp_path: Path) -> None:
    """Removing every model of a category empties its README count and list.

    Regression test: the rewriter used to iterate only categories present in
    the catalog, so a category whose last model was removed kept its stale
    table count, summary count, and full model list.
    """
    readme, model_info = _copy_repo_files(tmp_path)
    data = json.loads(model_info.read_text(encoding="utf-8"))
    data = {name: entry for name, entry in data.items() if not name.startswith("glm-")}
    model_info.write_text(json.dumps(data), encoding="utf-8")

    assert sync_readme_catalog(readme, model_info) is True
    synced = readme.read_text(encoding="utf-8")
    assert "| Z.AI | 0 |" in synced
    assert "<summary><strong>Z.AI (0)</strong></summary>\n\n</details>" in synced
    assert "- `glm-" not in synced
    # The emptied block must itself be canonical: a second sync is a no-op.
    assert sync_readme_catalog(readme, model_info) is False


def test_empty_details_block_is_populated(tmp_path: Path) -> None:
    """A 0 -> N transition fills an empty <details> block with the new list."""
    readme = tmp_path / "README.md"
    model_info = tmp_path / "MODEL_INFO.json"
    model_info.write_text(
        json.dumps({"glm-b": {"gen": True}, "glm-a": {"gen": True}}), encoding="utf-8"
    )
    readme.write_text(
        "<details>\n<summary><strong>Z.AI (0)</strong></summary>\n\n</details>\n",
        encoding="utf-8",
    )

    assert sync_readme_catalog(readme, model_info) is True
    assert readme.read_text(encoding="utf-8") == (
        "<details>\n<summary><strong>Z.AI (2)</strong></summary>\n\n"
        "- `glm-a`\n- `glm-b`\n\n</details>\n"
    )


def test_noncanonical_details_body_is_regenerated(tmp_path: Path) -> None:
    """Blank lines, junk lines, and stale bullets in a block are all replaced.

    Regression test: the rewriter used to require a perfectly canonical
    bullet list and silently fell back to count-only syncing otherwise.
    """
    readme = tmp_path / "README.md"
    model_info = tmp_path / "MODEL_INFO.json"
    model_info.write_text(json.dumps({"glm-new": {"gen": True}}), encoding="utf-8")
    readme.write_text(
        "<details>\n<summary><strong>Z.AI (3)</strong></summary>\n\n"
        "- `glm-stale`\n\nsome hand-written note\n- `glm-old`\n\n</details>\n"
        "<details>\n<summary><strong>Other</strong></summary>\n\nkeep me\n\n</details>\n",
        encoding="utf-8",
    )

    assert sync_readme_catalog(readme, model_info) is True
    assert readme.read_text(encoding="utf-8") == (
        "<details>\n<summary><strong>Z.AI (1)</strong></summary>\n\n"
        "- `glm-new`\n\n</details>\n"
        "<details>\n<summary><strong>Other</strong></summary>\n\nkeep me\n\n</details>\n"
    )


def test_summary_without_list_block_falls_back_to_count_sync(tmp_path: Path) -> None:
    """A README copy with a bare <summary> (no bullet list) still gets its count fixed."""
    readme = tmp_path / "README.md"
    model_info = tmp_path / "MODEL_INFO.json"
    model_info.write_text(
        json.dumps({"glm-a": {"gen": True}, "glm-b": {"gen": True}}), encoding="utf-8"
    )
    readme.write_text(
        "Intro\n\n<summary><strong>Z.AI (1)</strong></summary>\n\nOutro\n", encoding="utf-8"
    )

    assert sync_readme_catalog(readme, model_info) is True
    assert "<summary><strong>Z.AI (2)</strong></summary>" in readme.read_text(encoding="utf-8")


def test_backslash_in_model_name_is_written_literally(tmp_path: Path) -> None:
    """A backslash in a model name must not be eaten by regex replacement escapes."""
    readme = tmp_path / "README.md"
    model_info = tmp_path / "MODEL_INFO.json"
    model_info.write_text(json.dumps({"glm-4\\5": {"gen": True}}), encoding="utf-8")
    readme.write_text(
        "<details>\n<summary><strong>Z.AI (1)</strong></summary>\n\n"
        "- `glm-old`\n\n</details>\n",
        encoding="utf-8",
    )

    assert sync_readme_catalog(readme, model_info) is True
    assert "- `glm-4\\5`\n" in readme.read_text(encoding="utf-8")
