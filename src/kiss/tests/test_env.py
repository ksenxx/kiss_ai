"""Tests for kiss.env – offline-installer PATH and env var setup."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import kiss.env as env_mod

# ── helpers ──────────────────────────────────────────────────────────────

def _with_home(tmp_path: Path, fn: object) -> None:
    """Run *fn* with Path.home() monkeypatched to *tmp_path*."""
    orig_home = Path.home
    Path.home = staticmethod(lambda: tmp_path)  # type: ignore[assignment]
    try:
        fn()  # type: ignore[operator]
    finally:
        Path.home = orig_home  # type: ignore[assignment]


# ── get_install_dir tests ────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# kiss/env.py — env_mod
# ---------------------------------------------------------------------------

def test_get_install_dir_from_env(tmp_path: Path) -> None:
    """KISS_INSTALL_DIR env var takes highest priority."""
    original = os.environ.get("KISS_INSTALL_DIR")
    os.environ["KISS_INSTALL_DIR"] = "/custom/dir"
    try:
        assert env_mod.get_install_dir() == Path("/custom/dir")
    finally:
        if original is not None:
            os.environ["KISS_INSTALL_DIR"] = original
        else:
            os.environ.pop("KISS_INSTALL_DIR", None)


def test_get_install_dir_from_marker(tmp_path: Path) -> None:
    """Falls back to ~/.kiss/install_dir marker file."""
    original = os.environ.pop("KISS_INSTALL_DIR", None)
    marker = tmp_path / ".kiss" / "install_dir"
    marker.parent.mkdir(parents=True)
    marker.write_text("/opt/my_kiss\n")
    try:
        _with_home(tmp_path, env_mod.get_install_dir)
        # call directly to check return value
        orig_home = Path.home
        Path.home = staticmethod(lambda: tmp_path)  # type: ignore[assignment]
        try:
            assert env_mod.get_install_dir() == Path("/opt/my_kiss")
        finally:
            Path.home = orig_home  # type: ignore[assignment]
    finally:
        if original is not None:
            os.environ["KISS_INSTALL_DIR"] = original


def test_get_install_dir_empty_marker(tmp_path: Path) -> None:
    """Empty marker file falls back to default."""
    original = os.environ.pop("KISS_INSTALL_DIR", None)
    marker = tmp_path / ".kiss" / "install_dir"
    marker.parent.mkdir(parents=True)
    marker.write_text("   \n")
    try:
        orig_home = Path.home
        Path.home = staticmethod(lambda: tmp_path)  # type: ignore[assignment]
        try:
            assert env_mod.get_install_dir() == tmp_path / "kiss_ai"
        finally:
            Path.home = orig_home  # type: ignore[assignment]
    finally:
        if original is not None:
            os.environ["KISS_INSTALL_DIR"] = original


def test_get_install_dir_unreadable_marker(tmp_path: Path) -> None:
    """Falls back to default when marker file is unreadable."""
    original = os.environ.pop("KISS_INSTALL_DIR", None)
    marker = tmp_path / ".kiss" / "install_dir"
    marker.parent.mkdir(parents=True)
    marker.write_text("/opt/my_kiss")
    marker.chmod(0o000)
    try:
        orig_home = Path.home
        Path.home = staticmethod(lambda: tmp_path)  # type: ignore[assignment]
        try:
            assert env_mod.get_install_dir() == tmp_path / "kiss_ai"
        finally:
            Path.home = orig_home  # type: ignore[assignment]
    finally:
        marker.chmod(0o644)
        if original is not None:
            os.environ["KISS_INSTALL_DIR"] = original


def test_get_install_dir_default(tmp_path: Path) -> None:
    """Falls back to ~/kiss_ai when no env var or marker."""
    original = os.environ.pop("KISS_INSTALL_DIR", None)
    try:
        orig_home = Path.home
        Path.home = staticmethod(lambda: tmp_path)  # type: ignore[assignment]
        try:
            assert env_mod.get_install_dir() == tmp_path / "kiss_ai"
        finally:
            Path.home = orig_home  # type: ignore[assignment]
    finally:
        if original is not None:
            os.environ["KISS_INSTALL_DIR"] = original


# ── ensure_path tests ────────────────────────────────────────────────────

def test_ensure_path_adds_existing_dirs(tmp_path: Path) -> None:
    """Dirs that exist and are missing from PATH get prepended."""
    bin_dir = tmp_path / "kiss_ai" / "bin"
    bin_dir.mkdir(parents=True)
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)

    original = os.environ["PATH"]
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    parts = [p for p in original.split(os.pathsep) if p not in (str(bin_dir), str(local_bin))]
    os.environ["PATH"] = os.pathsep.join(parts)

    try:
        _with_home(tmp_path, env_mod.ensure_path)

        new_parts = os.environ["PATH"].split(os.pathsep)
        assert str(bin_dir) in new_parts
        assert str(local_bin) in new_parts
        assert new_parts.index(str(bin_dir)) < new_parts.index(str(local_bin))
    finally:
        os.environ["PATH"] = original
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_ensure_path_skips_nonexistent_dirs(tmp_path: Path) -> None:
    """Dirs that don't exist are NOT added to PATH."""
    original = os.environ["PATH"]
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    try:
        _with_home(tmp_path, env_mod.ensure_path)

        assert str(tmp_path / "kiss_ai" / "bin") not in os.environ["PATH"].split(os.pathsep)
    finally:
        os.environ["PATH"] = original
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_ensure_path_idempotent(tmp_path: Path) -> None:
    """Calling ensure_path twice doesn't duplicate entries."""
    bin_dir = tmp_path / "kiss_ai" / "bin"
    bin_dir.mkdir(parents=True)

    original = os.environ["PATH"]
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    try:
        def _call_twice() -> None:
            env_mod.ensure_path()
            env_mod.ensure_path()

        _with_home(tmp_path, _call_twice)

        count = os.environ["PATH"].split(os.pathsep).count(str(bin_dir))
        assert count == 1
    finally:
        os.environ["PATH"] = original
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_ensure_path_uses_marker_file(tmp_path: Path) -> None:
    """ensure_path uses the install dir from marker file."""
    custom_dir = tmp_path / "custom_install"
    bin_dir = custom_dir / "bin"
    bin_dir.mkdir(parents=True)
    marker = tmp_path / ".kiss" / "install_dir"
    marker.parent.mkdir(parents=True)
    marker.write_text(str(custom_dir))

    original = os.environ["PATH"]
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    parts = [p for p in original.split(os.pathsep) if p != str(bin_dir)]
    os.environ["PATH"] = os.pathsep.join(parts)

    try:
        _with_home(tmp_path, env_mod.ensure_path)

        assert str(bin_dir) in os.environ["PATH"].split(os.pathsep)
    finally:
        os.environ["PATH"] = original
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_module_import_calls_ensure_path() -> None:
    """Importing kiss.env calls ensure_path at module level."""
    importlib.reload(env_mod)


def test_ensure_path_sets_uv_python_install_dir(tmp_path: Path) -> None:
    """UV_PYTHON_INSTALL_DIR is set when <install_dir>/python/ exists."""
    python_dir = tmp_path / "kiss_ai" / "python"
    python_dir.mkdir(parents=True)

    original_path = os.environ["PATH"]
    original_uv = os.environ.pop("UV_PYTHON_INSTALL_DIR", None)
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    try:
        _with_home(tmp_path, env_mod.ensure_path)

        assert os.environ.get("UV_PYTHON_INSTALL_DIR") == str(python_dir)
    finally:
        os.environ["PATH"] = original_path
        if original_uv is not None:
            os.environ["UV_PYTHON_INSTALL_DIR"] = original_uv
        else:
            os.environ.pop("UV_PYTHON_INSTALL_DIR", None)
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_ensure_path_skips_uv_python_if_already_set(tmp_path: Path) -> None:
    """UV_PYTHON_INSTALL_DIR is NOT overwritten if already set."""
    python_dir = tmp_path / "kiss_ai" / "python"
    python_dir.mkdir(parents=True)

    original_path = os.environ["PATH"]
    original_uv = os.environ.get("UV_PYTHON_INSTALL_DIR")
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    os.environ["UV_PYTHON_INSTALL_DIR"] = "/custom/path"
    try:
        _with_home(tmp_path, env_mod.ensure_path)

        assert os.environ["UV_PYTHON_INSTALL_DIR"] == "/custom/path"
    finally:
        os.environ["PATH"] = original_path
        if original_uv is not None:
            os.environ["UV_PYTHON_INSTALL_DIR"] = original_uv
        else:
            os.environ.pop("UV_PYTHON_INSTALL_DIR", None)
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_ensure_path_skips_uv_python_if_dir_missing(tmp_path: Path) -> None:
    """UV_PYTHON_INSTALL_DIR is NOT set when <install_dir>/python/ doesn't exist."""
    original_path = os.environ["PATH"]
    original_uv = os.environ.pop("UV_PYTHON_INSTALL_DIR", None)
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    try:
        _with_home(tmp_path, env_mod.ensure_path)

        assert "UV_PYTHON_INSTALL_DIR" not in os.environ
    finally:
        os.environ["PATH"] = original_path
        if original_uv is not None:
            os.environ["UV_PYTHON_INSTALL_DIR"] = original_uv
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_ensure_path_sets_playwright_browsers_path(tmp_path: Path) -> None:
    """PLAYWRIGHT_BROWSERS_PATH is set when <install_dir>/playwright-browsers/ exists."""
    pw_dir = tmp_path / "kiss_ai" / "playwright-browsers"
    pw_dir.mkdir(parents=True)

    original_path = os.environ["PATH"]
    original_pw = os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    try:
        _with_home(tmp_path, env_mod.ensure_path)

        assert os.environ.get("PLAYWRIGHT_BROWSERS_PATH") == str(pw_dir)
    finally:
        os.environ["PATH"] = original_path
        if original_pw is not None:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = original_pw
        else:
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_ensure_path_skips_playwright_if_already_set(tmp_path: Path) -> None:
    """PLAYWRIGHT_BROWSERS_PATH is NOT overwritten if already set."""
    pw_dir = tmp_path / "kiss_ai" / "playwright-browsers"
    pw_dir.mkdir(parents=True)

    original_path = os.environ["PATH"]
    original_pw = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/custom/pw"
    try:
        _with_home(tmp_path, env_mod.ensure_path)

        assert os.environ["PLAYWRIGHT_BROWSERS_PATH"] == "/custom/pw"
    finally:
        os.environ["PATH"] = original_path
        if original_pw is not None:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = original_pw
        else:
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install


def test_ensure_path_skips_playwright_if_dir_missing(tmp_path: Path) -> None:
    """PLAYWRIGHT_BROWSERS_PATH is NOT set when <install_dir>/playwright-browsers/ doesn't exist."""
    original_path = os.environ["PATH"]
    original_pw = os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    original_install = os.environ.pop("KISS_INSTALL_DIR", None)
    try:
        _with_home(tmp_path, env_mod.ensure_path)

        assert "PLAYWRIGHT_BROWSERS_PATH" not in os.environ
    finally:
        os.environ["PATH"] = original_path
        if original_pw is not None:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = original_pw
        if original_install is not None:
            os.environ["KISS_INSTALL_DIR"] = original_install
