# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the user-local ``~/.kiss/MODEL_INFO.json`` catalog.

An *installed* KISS Sorcar (the packaged VS Code extension bundle, whose
project root carries no ``.git`` marker) reads its model catalog from
``$KISS_HOME/MODEL_INFO.json`` — seeded by the installer and refreshed by
the settings panel's "Update Models" button.  A development checkout
keeps reading the bundled ``src/kiss/core/models/MODEL_INFO.json``, so a
stale user copy can never shadow the checkout's source of truth.  The
``KISS_MODEL_INFO_PATH`` environment variable (set by
``update_models.py --model-info``) overrides both.

``MODEL_INFO`` is built at import time, so the end-to-end paths run a
fresh interpreter as a subprocess with the environment under test; the
selection helpers are additionally exercised directly against real temp
trees (no mocks or doubles anywhere).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.core.models.model_info import (
    PACKAGE_MODEL_INFO_PATH,
    _is_installed_package,
    _select_catalog_path,
    user_model_info_path,
)

_REPO = Path(__file__).resolve().parents[5]


def _fresh_interpreter_model_names(
    env_overrides: dict[str, str], home: Path
) -> set[str]:
    """Return ``set(MODEL_INFO)`` as loaded by a fresh interpreter.

    ``HOME`` is pointed at *home* (a per-test temp dir) so the developer's
    real ``~/.kiss/MY_MODELS.json`` — which the loader merges on top of
    every catalog — can neither leak entries into the assertion nor be
    auto-seeded by the test.
    """
    import os

    env = dict(os.environ)
    env["HOME"] = str(home)
    env.update(env_overrides)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json\n"
            "from kiss.core.models.model_info import MODEL_INFO\n"
            "print(json.dumps(sorted(MODEL_INFO)))\n",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPO),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return set(json.loads(proc.stdout))


def _catalog_file(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            name: {
                "context_length": 1000,
                "input_price_per_1M": 1.0,
                "output_price_per_1M": 2.0,
            }
            for name in names
        }),
        encoding="utf-8",
    )


class TestSelectCatalogPath:
    """Branch coverage for the pure selection helper."""

    def test_existing_env_override_wins(self, tmp_path: Path) -> None:
        override = tmp_path / "override.json"
        override.write_text("{}", encoding="utf-8")
        user = tmp_path / "user.json"
        user.write_text("{}", encoding="utf-8")
        chosen = _select_catalog_path(
            str(override), True, user, PACKAGE_MODEL_INFO_PATH
        )
        assert chosen == override

    def test_missing_env_override_falls_back_to_bundled(
        self, tmp_path: Path
    ) -> None:
        user = tmp_path / "user.json"
        user.write_text("{}", encoding="utf-8")
        chosen = _select_catalog_path(
            str(tmp_path / "missing.json"), True, user, PACKAGE_MODEL_INFO_PATH
        )
        assert chosen == PACKAGE_MODEL_INFO_PATH

    def test_installed_package_prefers_the_user_copy(
        self, tmp_path: Path
    ) -> None:
        user = tmp_path / "user.json"
        user.write_text("{}", encoding="utf-8")
        chosen = _select_catalog_path("", True, user, PACKAGE_MODEL_INFO_PATH)
        assert chosen == user

    def test_installed_package_without_user_copy_uses_bundled(
        self, tmp_path: Path
    ) -> None:
        chosen = _select_catalog_path(
            "", True, tmp_path / "missing.json", PACKAGE_MODEL_INFO_PATH
        )
        assert chosen == PACKAGE_MODEL_INFO_PATH

    def test_dev_checkout_ignores_the_user_copy(self, tmp_path: Path) -> None:
        user = tmp_path / "user.json"
        user.write_text("{}", encoding="utf-8")
        chosen = _select_catalog_path("", False, user, PACKAGE_MODEL_INFO_PATH)
        assert chosen == PACKAGE_MODEL_INFO_PATH


class TestIsInstalledPackage:
    """The .git marker at the project root separates dev from installed."""

    @staticmethod
    def _package_file(root: Path) -> Path:
        file = root / "src" / "kiss" / "core" / "models" / "model_info.py"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("", encoding="utf-8")
        return file

    def test_bundle_without_git_is_installed(self, tmp_path: Path) -> None:
        file = self._package_file(tmp_path / "kiss_project")
        assert _is_installed_package(file) is True

    def test_checkout_with_git_dir_is_not_installed(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "checkout"
        file = self._package_file(root)
        (root / ".git").mkdir()
        assert _is_installed_package(file) is False

    def test_worktree_with_git_file_is_not_installed(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "worktree"
        file = self._package_file(root)
        (root / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
        assert _is_installed_package(file) is False

    def test_too_shallow_a_path_is_not_installed(self) -> None:
        # A package file directly under the filesystem root has no
        # project root four levels up to classify.
        assert _is_installed_package(Path("/model_info.py")) is False

    def test_this_repo_is_a_dev_checkout(self) -> None:
        assert _is_installed_package() is False


class TestUserModelInfoPath:
    def test_honors_kiss_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KISS_HOME", "/tmp/custom-kiss-home")
        assert user_model_info_path() == Path(
            "/tmp/custom-kiss-home/MODEL_INFO.json"
        )

    def test_defaults_to_dot_kiss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KISS_HOME", raising=False)
        assert user_model_info_path() == Path.home() / ".kiss" / "MODEL_INFO.json"


class TestFreshInterpreterCatalogSelection:
    """Real import-time behaviour of ``MODEL_INFO`` in a fresh process."""

    def test_env_override_replaces_the_bundled_catalog(
        self, tmp_path: Path
    ) -> None:
        override = tmp_path / "catalog.json"
        _catalog_file(override, ["custom/only-model"])
        names = _fresh_interpreter_model_names(
            {"KISS_MODEL_INFO_PATH": str(override)}, home=tmp_path
        )
        assert "custom/only-model" in names
        assert "claude-fable-5" not in names

    def test_missing_env_override_falls_back_to_the_bundled_catalog(
        self, tmp_path: Path
    ) -> None:
        names = _fresh_interpreter_model_names(
            {"KISS_MODEL_INFO_PATH": str(tmp_path / "missing.json")},
            home=tmp_path,
        )
        assert "claude-fable-5" in names

    def test_corrupt_env_override_falls_back_to_the_bundled_catalog(
        self, tmp_path: Path
    ) -> None:
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{ not json", encoding="utf-8")
        names = _fresh_interpreter_model_names(
            {"KISS_MODEL_INFO_PATH": str(corrupt)}, home=tmp_path
        )
        assert "claude-fable-5" in names

    def test_schema_invalid_override_falls_back_to_the_bundled_catalog(
        self, tmp_path: Path
    ) -> None:
        # Valid JSON whose entry lacks required fields (a hand-edited
        # user catalog) must fall back too, not brick the import with a
        # raw KeyError.
        invalid = tmp_path / "schema-invalid.json"
        invalid.write_text(
            json.dumps({"broken/model": {"gen": True}}), encoding="utf-8"
        )
        names = _fresh_interpreter_model_names(
            {"KISS_MODEL_INFO_PATH": str(invalid)}, home=tmp_path
        )
        assert "broken/model" not in names
        assert "claude-fable-5" in names

    def test_dev_checkout_ignores_a_user_copy_in_kiss_home(
        self, tmp_path: Path
    ) -> None:
        # This test runs inside the git checkout, so even a present
        # $KISS_HOME/MODEL_INFO.json must be ignored.
        kiss_home = tmp_path / "kiss-home"
        _catalog_file(kiss_home / "MODEL_INFO.json", ["stale/user-model"])
        names = _fresh_interpreter_model_names(
            {"KISS_HOME": str(kiss_home)}, home=tmp_path
        )
        assert "stale/user-model" not in names
        assert "claude-fable-5" in names
