# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests that pin the non-interactive ``sorcar`` CLI flag contract.

After restricting the non-interactive (``-t TASK`` / ``-f FILE``) path
to a plain :class:`SorcarAgent`, the parser still exposed
``--worktree`` / ``--no-worktree`` / ``--auto-commit`` /
``--no-auto-commit`` / ``-n`` / ``--new`` — and silently swallowed
every one of them in non-interactive mode.  In particular a user
running ``sorcar -t 'fix bug' --worktree`` (or just relying on the
documented default ``--worktree=True``) silently lost worktree
isolation: edits landed directly in the working tree instead of an
isolated worktree branch.  Likewise ``--auto-commit`` / ``-n`` were
silent no-ops with no warning, no error, and no help text.

These tests reproduce that bug by driving the real CLI plumbing
(``worktree_sorcar_agent.main``) with the listed flags and asserting
that the CLI now fails fast with a clear message.  In interactive
mode (no ``-t``/``-f``) the same flags must still be accepted so the
daemon-client path keeps working — that's exercised here too.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from kiss.agents.sorcar import worktree_sorcar_agent
from kiss.agents.sorcar.sorcar_agent import SorcarAgent

# Each entry: ``argv`` fragment that must trip a clear non-interactive
# error and the substring the error message should mention.
_NON_INTERACTIVE_INVALID_FLAGS: list[tuple[list[str], str]] = [
    (["--worktree"], "--worktree"),
    (["--no-worktree"], "--no-worktree"),
    (["--auto-commit"], "--auto-commit"),
    (["--no-auto-commit"], "--no-auto-commit"),
    (["-n"], "-n"),
    (["--new"], "--new"),
]

# Argparse's default ``allow_abbrev=True`` would accept these prefix
# abbreviations as if the full long form had been written.  The
# fail-fast guard must catch them too — otherwise ``sorcar -t TASK
# --auto`` would silently parse as ``--auto-commit`` and reintroduce
# the silent-no-op behavior that the long-form guard prevents.
_NON_INTERACTIVE_INVALID_ABBREVIATIONS: list[list[str]] = [
    ["--auto"],
    ["--no-auto"],
    ["--worktr"],
    ["--no-worktr"],
    ["--ne"],
]


def _install_no_op_run_with_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Stub ``cli_steering.run_with_steering`` so ``main`` never hits an LLM.

    Returns the dict the stub will populate when (or if) it is called.
    """
    captured: dict[str, Any] = {}

    def fake_run(agent: SorcarAgent, run_kwargs: dict[str, Any]) -> str:
        captured["agent"] = agent
        captured["kwargs"] = run_kwargs
        return "summary: ok\nsuccess: true\n"

    import kiss.agents.sorcar.cli_steering as cli_steering

    monkeypatch.setattr(cli_steering, "run_with_steering", fake_run)
    monkeypatch.setattr(
        worktree_sorcar_agent, "print_outcome",
        lambda *_a, **_kw: None,
    )
    return captured


class TestNonInteractiveRejectsInteractiveOnlyFlags:
    """``sorcar -t TASK`` plus an interactive-only flag must fail fast.

    The CLI must exit with a non-zero status and print a message
    naming the offending flag so the user immediately understands
    that the flag is interactive-only.  Silently accepting the flag
    leads to a destructive surprise — ``--worktree`` (or its default
    ``True``) used to wrap the task in a git worktree, and dropping
    that wrap without telling the user means edits suddenly land in
    the main working tree.
    """

    @pytest.mark.parametrize(
        ("flag_argv", "expected_substring"),
        _NON_INTERACTIVE_INVALID_FLAGS,
    )
    def test_flag_is_rejected_with_clear_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        flag_argv: list[str],
        expected_substring: str,
    ) -> None:
        captured = _install_no_op_run_with_steering(monkeypatch)
        monkeypatch.setattr(
            sys, "argv",
            ["sorcar", "-t", "noop task", *flag_argv],
        )
        with pytest.raises(SystemExit) as excinfo:
            worktree_sorcar_agent.main()
        # argparse-style failure: non-zero exit code.
        assert excinfo.value.code not in (0, None), (
            f"non-interactive mode silently accepted {flag_argv!r}; "
            f"main() exited with {excinfo.value.code!r}"
        )
        # The error message must name the offending flag so the user
        # immediately knows why their command was rejected.
        combined = capsys.readouterr()
        haystack = (combined.out + combined.err).lower()
        assert expected_substring.lower() in haystack, (
            f"error message did not mention {expected_substring!r}; "
            f"got stdout={combined.out!r}, stderr={combined.err!r}"
        )
        # The stubbed run_with_steering must NOT have been called —
        # fail-fast validation has to happen BEFORE the agent runs.
        assert "agent" not in captured, (
            "fail-fast validation ran AFTER the agent — interactive-"
            "only flags must short-circuit before run_with_steering"
        )

    @pytest.mark.parametrize(
        "abbreviated_argv", _NON_INTERACTIVE_INVALID_ABBREVIATIONS,
    )
    def test_argparse_abbreviations_are_also_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        abbreviated_argv: list[str],
    ) -> None:
        """Argparse prefix abbreviations must not bypass the guard.

        Without this protection, ``sorcar -t TASK --auto`` would parse
        as ``--auto-commit`` (since argparse defaults to
        ``allow_abbrev=True``) and the literal-token guard would let
        it through, reintroducing the silent no-op for every flag in
        the interactive-only set.
        """
        captured = _install_no_op_run_with_steering(monkeypatch)
        monkeypatch.setattr(
            sys, "argv",
            ["sorcar", "-t", "noop task", *abbreviated_argv],
        )
        with pytest.raises(SystemExit) as excinfo:
            worktree_sorcar_agent.main()
        assert excinfo.value.code not in (0, None), (
            f"abbreviation {abbreviated_argv!r} slipped past the "
            f"non-interactive guard (exit code={excinfo.value.code!r})"
        )
        # The stubbed run_with_steering must NOT have been called.
        assert "agent" not in captured, (
            f"abbreviation {abbreviated_argv!r} reached the agent — "
            "the guard must short-circuit BEFORE run_with_steering"
        )
        # Capture is read only to flush; specific wording isn't pinned
        # here because the user-typed token is unrecognisable until
        # argparse expands it.  Either reporting the abbreviation or
        # the expanded form is acceptable as long as something was
        # printed.
        _ = capsys.readouterr()


class TestNonInteractiveAcceptsValidFlags:
    """Sanity check: non-interactive mode still works with the lean flag set.

    Only ``-t`` / ``-f`` / ``-m`` / ``-e`` / ``--header`` / ``-b`` /
    ``-w`` / ``-v`` / ``--no-web`` / ``-p``/``--no-parallel`` are
    safe in non-interactive mode (they do not pretend to enable a
    feature that the plain ``SorcarAgent`` does not support).
    """

    def test_plain_task_runs(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_no_op_run_with_steering(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["sorcar", "-t", "noop task"])
        worktree_sorcar_agent.main()
        assert "agent" in captured
        assert type(captured["agent"]) is SorcarAgent

    def test_no_parallel_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_no_op_run_with_steering(monkeypatch)
        monkeypatch.setattr(
            sys, "argv",
            ["sorcar", "-t", "noop task", "--no-parallel"],
        )
        worktree_sorcar_agent.main()
        assert "agent" in captured
        # The plain SorcarAgent honours ``is_parallel`` directly,
        # so this must propagate to run_kwargs.
        assert captured["kwargs"].get("is_parallel") is False

    def test_no_web_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = _install_no_op_run_with_steering(monkeypatch)
        monkeypatch.setattr(
            sys, "argv",
            ["sorcar", "-t", "noop task", "--no-web"],
        )
        worktree_sorcar_agent.main()
        assert "agent" in captured
        assert captured["kwargs"].get("web_tools") is False


class TestInteractiveStillAcceptsAllFlags:
    """Interactive (daemon-client) mode must still honour every flag.

    The interactive path forwards ``--worktree`` / ``--auto-commit``
    / ``--parallel`` to the daemon via :func:`run_client`; only the
    non-interactive path rejects them.  Driving ``main()`` with no
    ``-t`` / ``-f`` exercises the interactive branch, so all of the
    previously-rejected flags must be accepted here.
    """

    @pytest.mark.parametrize(
        "flag_argv",
        [["--worktree"], ["--no-worktree"], ["--auto-commit"],
         ["--no-auto-commit"], ["-n"], ["--new"]],
    )
    def test_flag_is_accepted_in_interactive_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag_argv: list[str],
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_run_client(**kwargs: Any) -> int:
            captured["kwargs"] = kwargs
            return 0

        # ``run_client`` is imported lazily inside ``main`` from
        # ``kiss.agents.sorcar.cli_client``; monkeypatch the module
        # attribute so the import-inside-main picks up the stub.
        import kiss.agents.sorcar.cli_client as cli_client

        monkeypatch.setattr(cli_client, "run_client", fake_run_client)
        monkeypatch.setattr(sys, "argv", ["sorcar", *flag_argv])
        with pytest.raises(SystemExit) as excinfo:
            worktree_sorcar_agent.main()
        # Interactive exit code is whatever ``run_client`` returned.
        assert excinfo.value.code == 0, (
            f"interactive mode rejected {flag_argv!r} (exit "
            f"code={excinfo.value.code!r}); the flag must remain "
            "valid in interactive mode"
        )
        assert "kwargs" in captured, (
            f"run_client was not invoked for interactive {flag_argv!r}"
        )
