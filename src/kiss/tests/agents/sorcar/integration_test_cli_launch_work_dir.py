# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end integration tests for the sorcar CLI launch ``work_dir``.

The installed ``sorcar`` wrapper at ``~/.local/bin/sorcar`` runs
``uv run --directory <bundled_kiss_project> sorcar ...``.  uv's
``--directory`` flag changes the *child* process working directory to
the bundled project before the CLI starts, so the CLI's
:func:`pathlib.Path.cwd` no longer reflects the user's shell
directory.  The wrapper captures the original ``$PWD`` in the
``KISS_WORKDIR`` environment variable and the CLI must default the
task ``work_dir`` to that value.

These tests reproduce the wrapper scenario end-to-end by spawning a
real child Python interpreter whose:

* current working directory is a *different* directory (simulating the
  ``uv run --directory <kiss_project>`` chdir),
* environment carries ``KISS_WORKDIR=<user_shell_dir>``,

and then asserting that the CLI argument parser resolves
``--work_dir`` to the user's shell directory rather than the chdir'd
project directory.  Without the launch-dir capture in
:func:`kiss.agents.sorcar.cli_helpers._launch_work_dir`, the parser
would return the bundled-project path instead and break every task
run by the wrapper.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Run the CLI argument parser in a child interpreter whose cwd is the
# given ``run_cwd`` and whose environment carries the supplied
# ``KISS_WORKDIR`` (or unset when ``None``).  The child prints the
# resolved ``args.work_dir`` and the run-kwargs ``work_dir`` separated
# by ``|`` so we can assert on both paths together.
_PROBE_SOURCE = textwrap.dedent(
    """
    from kiss.agents.sorcar.cli_helpers import (
        _build_arg_parser,
        _build_run_kwargs,
    )

    parser = _build_arg_parser()
    args = parser.parse_args(["-t", "noop"])
    run_kwargs = _build_run_kwargs(args)
    print(args.work_dir + "|" + run_kwargs["work_dir"])
    """
).strip()


def _run_probe(
    run_cwd: Path,
    kiss_workdir: str | None,
) -> tuple[str, str]:
    """Spawn a child Python that prints (args.work_dir, run_kwargs work_dir).

    Args:
        run_cwd: Working directory for the child process.  Simulates
            the bundled-project dir that ``uv run --directory ...``
            changes into before invoking the CLI.
        kiss_workdir: Value for ``KISS_WORKDIR``, or ``None`` to leave
            the variable unset in the child environment.

    Returns:
        ``(args_work_dir, run_kwargs_work_dir)`` as resolved by the
        CLI argument parser inside the child process.
    """
    env = {k: v for k, v in os.environ.items() if k != "KISS_WORKDIR"}
    if kiss_workdir is not None:
        env["KISS_WORKDIR"] = kiss_workdir
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SOURCE],
        cwd=str(run_cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    out = proc.stdout.strip().splitlines()[-1]
    args_wd, kwargs_wd = out.split("|", 1)
    return args_wd, kwargs_wd


def test_cli_defaults_work_dir_to_user_shell_via_kiss_workdir(
    tmp_path: Path,
) -> None:
    """Mimic the installed wrapper: child cwd != user shell, ``KISS_WORKDIR`` set.

    Reproduces the scenario the user hits when invoking the installed
    ``sorcar`` script: the wrapper exports ``KISS_WORKDIR="$PWD"`` and
    then ``uv run --directory <kiss_project>`` chdirs the child into
    the bundled project.  The CLI must still default ``work_dir`` to
    the user's shell directory, not the bundled project directory.
    """
    user_shell_dir = tmp_path / "user_shell"
    user_shell_dir.mkdir()
    bundled_project_dir = tmp_path / "bundled_kiss_project"
    bundled_project_dir.mkdir()

    args_wd, kwargs_wd = _run_probe(
        run_cwd=bundled_project_dir,
        kiss_workdir=str(user_shell_dir),
    )

    expected = str(user_shell_dir.resolve())
    assert args_wd == expected, (
        f"Expected --work_dir default to be the launch dir {expected!r}, "
        f"but got {args_wd!r} (child cwd was {bundled_project_dir!r})"
    )
    assert kwargs_wd == expected, (
        f"Expected run-kwargs work_dir to be the launch dir {expected!r}, "
        f"but got {kwargs_wd!r}"
    )


def test_cli_falls_back_to_child_cwd_when_kiss_workdir_unset(
    tmp_path: Path,
) -> None:
    """Without ``KISS_WORKDIR`` the CLI must use the child's own cwd.

    Direct invocation (no wrapper) leaves ``KISS_WORKDIR`` unset; in
    that case the child's cwd is already correct and the CLI must
    fall back to :func:`pathlib.Path.cwd`.
    """
    direct_cwd = tmp_path / "direct"
    direct_cwd.mkdir()

    args_wd, kwargs_wd = _run_probe(run_cwd=direct_cwd, kiss_workdir=None)

    expected = str(direct_cwd.resolve())
    assert args_wd == expected
    assert kwargs_wd == expected


def test_cli_ignores_stale_kiss_workdir_pointing_nowhere(
    tmp_path: Path,
) -> None:
    """A stale ``KISS_WORKDIR`` (deleted dir) must fall back to child cwd.

    Guards against a previous shell session leaving a now-deleted path
    in the environment — the CLI must not blindly trust it and crash
    later when trying to create files inside it.
    """
    child_cwd = tmp_path / "child"
    child_cwd.mkdir()
    stale = tmp_path / "never_existed"

    args_wd, kwargs_wd = _run_probe(run_cwd=child_cwd, kiss_workdir=str(stale))

    expected = str(child_cwd.resolve())
    assert args_wd == expected
    assert kwargs_wd == expected


def test_explicit_w_flag_still_overrides_launch_dir(tmp_path: Path) -> None:
    """An explicit ``-w`` flag wins over both ``KISS_WORKDIR`` and cwd."""
    user_shell_dir = tmp_path / "user_shell"
    user_shell_dir.mkdir()
    bundled_project_dir = tmp_path / "bundled_kiss_project"
    bundled_project_dir.mkdir()
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()

    env = {k: v for k, v in os.environ.items() if k != "KISS_WORKDIR"}
    env["KISS_WORKDIR"] = str(user_shell_dir)
    source = textwrap.dedent(
        f"""
        from kiss.agents.sorcar.cli_helpers import (
            _build_arg_parser,
            _build_run_kwargs,
        )

        parser = _build_arg_parser()
        args = parser.parse_args(["-w", {str(explicit_dir)!r}, "-t", "noop"])
        run_kwargs = _build_run_kwargs(args)
        print(args.work_dir + "|" + run_kwargs["work_dir"])
        """
    ).strip()
    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(bundled_project_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    out = proc.stdout.strip().splitlines()[-1]
    args_wd, kwargs_wd = out.split("|", 1)
    expected = str(explicit_dir)
    assert args_wd == expected
    assert kwargs_wd == expected
