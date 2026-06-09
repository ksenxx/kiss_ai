# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration repro: empty-string ``work_dir`` breaks the Bash tool.

Commit 37741c05 ("refactor: simplify work_dir and model_name access
logic") replaced ``getattr(self, "work_dir", None)`` with direct
``self.work_dir`` access in ``SorcarAgent._get_tools`` and gave
``RelentlessAgent`` a class-level default of ``work_dir = ""``.

The commit message promised "without changing functionality", but the
default changed from ``None`` (attribute absent) to ``""``.
``UsefulTools`` distinguishes ``None`` from ``""``:

* ``subprocess.Popen(..., cwd=self.work_dir)`` — ``cwd=None`` means
  "inherit the parent's cwd", while ``cwd=""`` raises
  ``NotADirectoryError`` / ``FileNotFoundError`` on POSIX.
* ``_clean_env`` only skips the ``KISS_WORKDIR`` override when
  ``work_dir is None``, so ``""`` is exported to child processes.

Net effect: every Bash tool obtained from a freshly constructed
``SorcarAgent`` (i.e. before ``run()`` resolves the real work dir)
crashes instead of running the command.  These tests reproduce the
regression end-to-end with real subprocesses (no mocks).
"""

from __future__ import annotations

import os

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools


class TestFreshAgentBashTool:
    """The Bash tool of a freshly constructed agent must run commands."""

    def test_fresh_agent_bash_tool_executes_command(self) -> None:
        """A fresh ``SorcarAgent`` (no ``run()`` yet) can execute Bash.

        Before commit 37741c05 ``getattr(self, "work_dir", None)``
        yielded ``None`` here and the subprocess inherited the parent
        cwd.  Now ``self.work_dir == ""`` flows into
        ``subprocess.Popen(cwd="")`` which raises instead of running.
        """
        agent = SorcarAgent("repro")
        bash_tool = agent._get_tools()[0]
        result = bash_tool(
            command="echo fresh_agent_ok",
            description="repro: fresh-agent bash",
            timeout_seconds=10,
        )
        assert "fresh_agent_ok" in result
        assert "Error" not in result

    def test_fresh_agent_bash_tool_runs_in_parent_cwd(self) -> None:
        """With no work_dir configured, Bash must run in the parent cwd."""
        agent = SorcarAgent("repro")
        bash_tool = agent._get_tools()[0]
        result = bash_tool(
            command="pwd",
            description="repro: fresh-agent pwd",
            timeout_seconds=10,
        )
        assert os.getcwd() in result


class TestUsefulToolsEmptyWorkDir:
    """``UsefulTools(work_dir="")`` must behave like ``work_dir=None``."""

    def test_bash_with_empty_work_dir_executes(self) -> None:
        """``cwd=""`` must not be passed to ``subprocess.Popen``.

        ``RelentlessAgent.work_dir`` now defaults to ``""``, so this is
        exactly what ``SorcarAgent._get_tools`` constructs before the
        first ``run()``.
        """
        tools = UsefulTools(stream_callback=lambda _t: None, work_dir="")
        result = tools.Bash(
            command="echo empty_workdir_ok",
            description="repro: empty work_dir bash",
            timeout_seconds=10,
        )
        assert "empty_workdir_ok" in result
        assert "Error" not in result

    def test_bash_with_none_work_dir_executes(self) -> None:
        """Control: ``work_dir=None`` works (pre-regression behaviour)."""
        tools = UsefulTools(stream_callback=lambda _t: None, work_dir=None)
        result = tools.Bash(
            command="echo none_workdir_ok",
            description="control: None work_dir bash",
            timeout_seconds=10,
        )
        assert "none_workdir_ok" in result
        assert "Error" not in result

    def test_empty_work_dir_does_not_export_empty_kiss_workdir(self) -> None:
        """``KISS_WORKDIR`` must not be exported as ``""`` to children.

        ``_clean_env`` skips the override only for ``work_dir is None``;
        with the new ``""`` default the child sees ``KISS_WORKDIR=""``
        (set-but-empty), which downstream consumers cannot distinguish
        from a deliberately configured empty value.
        """
        tools = UsefulTools(stream_callback=lambda _t: None, work_dir="")
        result = tools.Bash(
            command='if [ "${KISS_WORKDIR+set}" = set ] && '
            '[ -z "$KISS_WORKDIR" ]; then echo EMPTY_EXPORTED; '
            "else echo OK_NOT_EXPORTED_EMPTY; fi",
            description="repro: KISS_WORKDIR env leak",
            timeout_seconds=10,
        )
        assert "OK_NOT_EXPORTED_EMPTY" in result
