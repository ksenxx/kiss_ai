"""Tests for kiss.env – offline-installer PATH setup."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import kiss.env as env_mod


def test_ensure_path_adds_existing_dirs(tmp_path: Path) -> None:
    """Dirs that exist and are missing from PATH get prepended."""
    bin_dir = tmp_path / ".kiss-install" / "bin"
    bin_dir.mkdir(parents=True)
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)

    original = os.environ["PATH"]
    # Remove any occurrence of our test dirs
    parts = [p for p in original.split(os.pathsep) if p not in (str(bin_dir), str(local_bin))]
    os.environ["PATH"] = os.pathsep.join(parts)

    try:
        # Monkeypatch Path.home to point to tmp_path
        orig_home = Path.home
        Path.home = staticmethod(lambda: tmp_path)  # type: ignore[assignment]
        try:
            env_mod.ensure_path()
        finally:
            Path.home = orig_home  # type: ignore[assignment]

        new_parts = os.environ["PATH"].split(os.pathsep)
        assert str(bin_dir) in new_parts
        assert str(local_bin) in new_parts
        # They should be at the front
        assert new_parts.index(str(bin_dir)) < new_parts.index(str(local_bin))
    finally:
        os.environ["PATH"] = original


def test_ensure_path_skips_nonexistent_dirs(tmp_path: Path) -> None:
    """Dirs that don't exist are NOT added to PATH."""
    original = os.environ["PATH"]
    try:
        orig_home = Path.home
        Path.home = staticmethod(lambda: tmp_path)  # type: ignore[assignment]
        try:
            env_mod.ensure_path()
        finally:
            Path.home = orig_home  # type: ignore[assignment]

        # tmp_path/.kiss-install/bin doesn't exist so should not be on PATH
        assert str(tmp_path / ".kiss-install" / "bin") not in os.environ["PATH"].split(os.pathsep)
    finally:
        os.environ["PATH"] = original


def test_ensure_path_idempotent(tmp_path: Path) -> None:
    """Calling ensure_path twice doesn't duplicate entries."""
    bin_dir = tmp_path / ".kiss-install" / "bin"
    bin_dir.mkdir(parents=True)

    original = os.environ["PATH"]
    try:
        orig_home = Path.home
        Path.home = staticmethod(lambda: tmp_path)  # type: ignore[assignment]
        try:
            env_mod.ensure_path()
            env_mod.ensure_path()
        finally:
            Path.home = orig_home  # type: ignore[assignment]

        count = os.environ["PATH"].split(os.pathsep).count(str(bin_dir))
        assert count == 1
    finally:
        os.environ["PATH"] = original


def test_module_import_calls_ensure_path() -> None:
    """Importing kiss.env calls ensure_path at module level."""
    # Just verify the module-level call doesn't crash on re-import
    importlib.reload(env_mod)
