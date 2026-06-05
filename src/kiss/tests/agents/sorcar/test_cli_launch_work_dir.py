# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Verify the sorcar CLI defaults ``work_dir`` to the launch directory.

The installed ``sorcar`` wrapper runs ``uv run --directory
<kiss_project> sorcar ...``; uv's ``--directory`` flag changes the
process cwd to the bundled project before the CLI starts, so
``Path.cwd()`` no longer reflects the user's shell directory.  The
wrapper records the original ``$PWD`` in ``KISS_WORKDIR`` and the CLI
must honor it when defaulting the task ``work_dir``.
"""

from __future__ import annotations

import os
from pathlib import Path

from kiss.agents.sorcar.cli_helpers import (
    _build_arg_parser,
    _build_run_kwargs,
    _launch_work_dir,
)


def _clear_kiss_workdir() -> str | None:
    """Pop ``KISS_WORKDIR`` from the environment, returning its old value."""
    return os.environ.pop("KISS_WORKDIR", None)


def _restore_kiss_workdir(old: str | None) -> None:
    """Restore ``KISS_WORKDIR`` to *old* (removing it when *old* is None)."""
    if old is None:
        os.environ.pop("KISS_WORKDIR", None)
    else:
        os.environ["KISS_WORKDIR"] = old


def test_launch_work_dir_prefers_kiss_workdir(tmp_path: Path) -> None:
    """``KISS_WORKDIR`` (set by the wrapper) overrides the process cwd."""
    old = os.environ.get("KISS_WORKDIR")
    launch = tmp_path / "user_shell_dir"
    launch.mkdir()
    os.environ["KISS_WORKDIR"] = str(launch)
    try:
        assert _launch_work_dir() == str(launch.resolve())
    finally:
        _restore_kiss_workdir(old)


def test_launch_work_dir_falls_back_to_cwd_when_unset() -> None:
    """Without ``KISS_WORKDIR`` the launch dir is the real process cwd."""
    old = _clear_kiss_workdir()
    try:
        assert _launch_work_dir() == str(Path.cwd())
    finally:
        _restore_kiss_workdir(old)


def test_launch_work_dir_ignores_nonexistent_kiss_workdir(
    tmp_path: Path,
) -> None:
    """A stale ``KISS_WORKDIR`` pointing nowhere falls back to cwd."""
    old = os.environ.get("KISS_WORKDIR")
    os.environ["KISS_WORKDIR"] = str(tmp_path / "does_not_exist")
    try:
        assert _launch_work_dir() == str(Path.cwd())
    finally:
        _restore_kiss_workdir(old)


def test_arg_parser_default_work_dir_uses_kiss_workdir(tmp_path: Path) -> None:
    """The ``--work_dir`` argparse default resolves to ``KISS_WORKDIR``."""
    old = os.environ.get("KISS_WORKDIR")
    launch = tmp_path / "launch"
    launch.mkdir()
    os.environ["KISS_WORKDIR"] = str(launch)
    try:
        parser = _build_arg_parser()
        args = parser.parse_args([])
        assert args.work_dir == str(launch.resolve())
    finally:
        _restore_kiss_workdir(old)


def test_build_run_kwargs_work_dir_uses_kiss_workdir(tmp_path: Path) -> None:
    """``_build_run_kwargs`` threads the launch dir into ``run`` kwargs."""
    old = os.environ.get("KISS_WORKDIR")
    launch = tmp_path / "project"
    launch.mkdir()
    os.environ["KISS_WORKDIR"] = str(launch)
    try:
        parser = _build_arg_parser()
        args = parser.parse_args(["-t", "noop"])
        run_kwargs = _build_run_kwargs(args)
        assert run_kwargs["work_dir"] == str(launch.resolve())
    finally:
        _restore_kiss_workdir(old)


def test_explicit_work_dir_flag_overrides_kiss_workdir(tmp_path: Path) -> None:
    """An explicit ``-w`` flag still wins over ``KISS_WORKDIR``."""
    old = os.environ.get("KISS_WORKDIR")
    launch = tmp_path / "launch"
    launch.mkdir()
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    os.environ["KISS_WORKDIR"] = str(launch)
    try:
        parser = _build_arg_parser()
        args = parser.parse_args(["-w", str(explicit), "-t", "noop"])
        run_kwargs = _build_run_kwargs(args)
        assert run_kwargs["work_dir"] == str(explicit)
    finally:
        _restore_kiss_workdir(old)
