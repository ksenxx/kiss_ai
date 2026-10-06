# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Verify channel-agent CLIs default ``work_dir`` to the launch directory.

Installed wrapper scripts for the channel-agent entry points
(``kiss-slack`` / ``kiss-gmail``) run ``uv run --directory
<kiss_project> ...``; uv's ``--directory`` flag changes the process
cwd to the bundled project before the entry point starts, so
``Path.cwd()`` no longer reflects the user's shell directory.  Such a
wrapper records the original ``$PWD`` in ``KISS_WORKDIR`` and the
argument parser must honor it when defaulting the task ``work_dir``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.third_party_agents import _kiss_web_launcher
from kiss.agents.third_party_agents._channel_cli import (
    _build_arg_parser,
    _build_run_kwargs,
    _launch_work_dir,
)
from kiss.agents.third_party_agents._kiss_web_launcher import (
    KissWebChatAgent,
    run_agent_via_kiss_web,
)
from kiss.tests.agents.third_party_agents.recording_daemon import RecordingDaemon


def _wire_work_dir(run_kwargs: dict[str, Any]) -> str:
    """Launch *run_kwargs* (from ``_build_run_kwargs``) against a recording daemon.

    Drives the CLI's own launch path (``channel_main`` →
    ``run_agent_via_kiss_web`` → ``sorcar.run``) and returns the
    ``workDir`` the daemon received on the wire.
    """
    daemon = RecordingDaemon()
    saved_override = _kiss_web_launcher._ENDPOINT_FILE_OVERRIDE
    _kiss_web_launcher._ENDPOINT_FILE_OVERRIDE = str(daemon.endpoint_file)
    try:
        prompt = run_kwargs.pop("prompt_template")
        run_agent_via_kiss_web(KissWebChatAgent("probe"), prompt, **run_kwargs)
        assert len(daemon.run_commands) == 1
        return str(daemon.run_commands[0]["workDir"])
    finally:
        _kiss_web_launcher._ENDPOINT_FILE_OVERRIDE = saved_override
        daemon.close()


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


@pytest.mark.parametrize("stale_kiss_workdir", [False, True], ids=["unset", "stale"])
def test_run_forwards_launch_cwd_to_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_kiss_workdir: bool
) -> None:
    """Direct (non-wrapper) launches send the process cwd as the run's ``workDir``.

    Covers ``KISS_WORKDIR`` unset and ``KISS_WORKDIR`` naming a deleted
    directory: both the parser default and the ``run`` command the
    daemon receives name the launch cwd, not the stale path.
    """
    direct = tmp_path / "direct"
    direct.mkdir()
    monkeypatch.chdir(direct)
    if stale_kiss_workdir:
        monkeypatch.setenv("KISS_WORKDIR", str(tmp_path / "never_existed"))
    else:
        monkeypatch.delenv("KISS_WORKDIR", raising=False)
    expected = str(Path.cwd())
    assert Path(expected).resolve() == direct.resolve()
    args = _build_arg_parser().parse_args(["-t", "noop"])
    assert args.work_dir == expected
    run_kwargs = _build_run_kwargs(args)
    assert run_kwargs["work_dir"] == expected
    assert _wire_work_dir(run_kwargs) == expected


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
