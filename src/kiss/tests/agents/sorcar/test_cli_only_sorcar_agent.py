# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests pinning the lean ``sorcar`` CLI contract.

After this change the ``sorcar`` CLI must:

* run **only** :class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent` in
  non-interactive mode (``-t/--task`` or ``-f/--file``).  No
  :class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent`,
  no :class:`~kiss.agents.sorcar.worktree_sorcar_agent.WorktreeSorcarAgent`
  is constructed — those are now reserved for interactive (daemon)
  mode only;
* no longer accept the deprecated flags ``-c/--chat-id``,
  ``-l/--list-chat-id``, ``--use-chat``, ``--cleanup``, and
  ``--use-worktree``.

The tests drive the real CLI plumbing (``_build_arg_parser`` and the
``worktree_sorcar_agent.main`` entry point) end-to-end with the
``run_with_steering`` call stubbed so the test does not need a live
LLM, model, or git repo.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from kiss.agents.sorcar import cli_helpers, worktree_sorcar_agent
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.cli_helpers import _build_arg_parser
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent

_REMOVED_FLAGS: list[list[str]] = [
    ["-c", "abc"],
    ["--chat-id", "abc"],
    ["-l"],
    ["--list-chat-id"],
    ["--use-chat"],
    ["--cleanup"],
    ["--use-worktree"],
]


class TestRemovedFlags:
    """Each listed flag must no longer be accepted by the parser.

    ``argparse`` exits with status 2 when it sees an unknown flag, so
    every entry in :data:`_REMOVED_FLAGS` must trigger ``SystemExit``.
    """

    @pytest.mark.parametrize("argv", _REMOVED_FLAGS)
    def test_flag_is_rejected(self, argv: list[str]) -> None:
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


class TestNonInteractiveUsesPlainSorcarAgent:
    """The non-interactive CLI path must hand ``run_with_steering``
    a bare :class:`SorcarAgent` — never a ``ChatSorcarAgent`` /
    ``WorktreeSorcarAgent``.
    """

    def _run_main_with_task(
        self, monkeypatch: pytest.MonkeyPatch, extra_argv: list[str],
    ) -> SorcarAgent:
        """Invoke ``main()`` with ``-t`` plus ``extra_argv`` and return
        the agent ``run_with_steering`` was called with.
        """
        captured: dict[str, Any] = {}

        def fake_run(agent: SorcarAgent, run_kwargs: dict[str, Any]) -> str:
            captured["agent"] = agent
            captured["kwargs"] = run_kwargs
            return "summary: ok\nsuccess: true\n"

        import kiss.agents.sorcar.cli_steering as cli_steering

        monkeypatch.setattr(cli_steering, "run_with_steering", fake_run)
        # ``print_outcome`` is a no-op in verbose mode, but explicitly
        # neuter it so the test doesn't depend on the printer state.
        monkeypatch.setattr(
            worktree_sorcar_agent, "print_outcome",
            lambda *_a, **_kw: None,
        )
        monkeypatch.setattr(
            sys, "argv", ["sorcar", "-t", "noop task", *extra_argv],
        )
        worktree_sorcar_agent.main()
        agent = captured.get("agent")
        assert agent is not None, "run_with_steering was not invoked"
        assert isinstance(agent, SorcarAgent)
        return agent

    def test_default_non_interactive_uses_plain_sorcar_agent(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = self._run_main_with_task(monkeypatch, [])
        assert type(agent) is SorcarAgent, (
            f"non-interactive CLI must build a plain SorcarAgent, got "
            f"{type(agent).__name__}"
        )
        assert not isinstance(agent, ChatSorcarAgent)
        assert not isinstance(agent, WorktreeSorcarAgent)

    # Note: ``--no-worktree`` (and ``--worktree`` / ``--auto-commit``
    # / ``--no-auto-commit`` / ``-n``) are now rejected when combined
    # with ``-t`` / ``-f`` — see
    # :mod:`test_cli_non_interactive_flag_validation` for the
    # fail-fast contract.  This file pins the "which agent runs"
    # contract; the flag-rejection contract lives next door.


class TestResolveCliModesHelperIsGone:
    """The legacy ``_resolve_cli_modes`` / ``_build_cli_agent`` helpers
    served only the removed flags and the two subclasses; they must
    no longer be re-exported from ``worktree_sorcar_agent`` so callers
    cannot accidentally resurrect the dropped behaviour.
    """

    def test_resolve_cli_modes_removed(self) -> None:
        assert not hasattr(worktree_sorcar_agent, "_resolve_cli_modes")

    def test_build_cli_agent_removed(self) -> None:
        assert not hasattr(worktree_sorcar_agent, "_build_cli_agent")


class TestParserDefaultsStillSane:
    """The trimmed parser must still expose ``--worktree`` /
    ``--no-worktree`` (defaulting to True) and ``-n/--new`` so the
    interactive daemon-client path keeps its existing flag surface.
    """

    def test_worktree_default_on(self) -> None:
        args = _build_arg_parser().parse_args([])
        assert args.worktree is True

    def test_no_worktree_disables(self) -> None:
        args = _build_arg_parser().parse_args(["--no-worktree"])
        assert args.worktree is False

    def test_new_flag_still_present(self) -> None:
        args = _build_arg_parser().parse_args(["-n"])
        assert args.new is True


class TestApplyChatArgsStillExported:
    """Third-party agents (Slack, Discord, …) still use
    :func:`cli_helpers._apply_chat_args` to wire chat resumption into
    their own parser; the helper must remain importable even though
    the sorcar CLI no longer calls it.
    """

    def test_apply_chat_args_importable(self) -> None:
        assert callable(cli_helpers._apply_chat_args)
