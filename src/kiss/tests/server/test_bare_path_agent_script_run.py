# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bare-path tasks and ``<task>`` tags of agent-script (SEA) runs on the daemon.

``/git_extract_knowledge /path/to/repo`` relays to ``run_agent(agent=<SEA
path>, task='/path/to/repo')``: the daemon then runs the SEA (wire field
``agentPath``, parent ``parentTaskId``) with a task that is nothing but
an existing directory.  Two chat conveniences must NOT apply to such a
run:

* the bare-path "open it" directive (:mod:`kiss.agents.sorcar.bare_path_task`),
  which sent the SEA off to ``xdg-open`` the repository instead of
  indexing it — the SEA, not the chat box, defines what the path means;
* the ``<task>`` splitter (``parse_task_tags``), which would hand the SEA
  one fragment of ``ask /repo what does <task>hello</task> mean?``.

The daemon pipeline runs for real on a temporary local WSS endpoint
(:class:`DaemonRunApiHarness`); only the executor LLM loop is a stub
recording the prompt it was handed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kiss.agents.sorcar.bare_path_task import with_open_directive
from kiss.core.kiss_agent import KISSAgent
from kiss.server import sorcar
from kiss.tests.server.test_append_basic_tools import DaemonRunApiHarness

DIRECTIVE = "The task is nothing but the path of an existing"
"""Start of the sentence ``with_open_directive`` appends to a bare-path task."""


class BarePathAgentScriptRunTest(DaemonRunApiHarness):
    """Agent-script and sub-agent runs keep a bare-path task as it is."""

    def _record_prompts(self, prompts: list[str]) -> None:
        """Replace the executor LLM loop with a stub recording its task text."""

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            if kwargs.get("is_agentic") is False:
                return ""  # follow-up proposer etc.: silent, unrecorded
            arguments = dict(kwargs.get("arguments") or {})
            self_agent.total_tokens_used = 1
            self_agent.budget_used = 0.0001
            self_agent.step_count = 1
            if "task_description" not in arguments:
                return "result: prior progress\n"
            prompts.append(str(arguments["task_description"]))
            raw = "success: true\nis_continue: false\nsummary: agent ok\n"
            printer = kwargs.get("printer")
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw, type="result", step_count=1, total_tokens=1, cost="$0.0001",
                )
            return raw

        KISSAgent.run = stub_run  # type: ignore[assignment,method-assign]

    def _sea(self, name: str, body: str) -> str:
        """Write an agent script under the test tmpdir and return its path."""
        path = Path(self.tmpdir) / f"{name}_sea.py"
        path.write_text(body, encoding="utf-8")
        return str(path)

    def _run(self, prompt: str, **kwargs: Any) -> str:
        prompts: list[str] = []
        self._record_prompts(prompts)
        result = sorcar.run(
            prompt, work_dir=self.repo, use_worktree=False, auto_commit=False,
            endpoint_file=self.endpoint_file, timeout=60, **kwargs,
        )
        assert result.success is True, result
        assert len(prompts) == 1, prompts
        return prompts[0]

    def test_plain_chat_run_gets_the_open_directive(self) -> None:
        """Control: a bare path typed into the chat box still means "open it"."""
        sent = self._run(self.repo)
        assert DIRECTIVE in sent
        assert with_open_directive(self.repo, self.repo) in sent

    def test_agent_script_with_system_prompt_suffix_gets_no_directive(self) -> None:
        """An SEA that only APPENDS to the system prompt keeps its bare-path task.

        Covers the paper SEAs, which define ``append_to_system_prompt()``
        rather than ``system_prompt()``: the exemption rests on the run
        being an agent-script run, not on which getter it defines.
        """
        sea = self._sea(
            "reviewer",
            'def append_to_system_prompt() -> str:\n'
            '    return "Review the paper whose path is the task."\n',
        )
        sent = self._run(self.repo, extension_agent_path=sea)
        assert f"# Task\n{self.repo}" in sent
        assert DIRECTIVE not in sent
        assert "xdg-open" not in sent

    def test_agent_script_without_any_getter_gets_no_directive(self) -> None:
        """Even an empty agent script owns the meaning of its bare-path task."""
        sea = self._sea("empty", "# nothing overridden\n")
        sent = self._run(self.repo, extension_agent_path=sea)
        assert f"# Task\n{self.repo}" in sent
        assert DIRECTIVE not in sent

    def test_sub_agent_run_gets_no_directive(self) -> None:
        """A ``run_agent`` sub-task (``parentTaskId``) is not a user typing a path."""
        sent = self._run(self.repo, parent_task_id="0123456789abcdef0123456789abcdef")
        assert f"# Task\n{self.repo}" in sent
        assert DIRECTIVE not in sent

    def test_agent_script_task_tags_are_not_split(self) -> None:
        """``<task>`` blocks in an SEA's task text reach the SEA whole, in ONE run."""
        sea = self._sea(
            "knowledge",
            'def system_prompt() -> str:\n'
            '    return "You answer questions about the repository named by the task."\n',
        )
        task = f"ask {self.repo} what does <task>hello</task> mean?"
        sent = self._run(task, extension_agent_path=sea)
        assert f"# Task\n{task}" in sent

    def test_plain_chat_task_tags_are_still_split(self) -> None:
        """Control: a plain chat prompt with two ``<task>`` blocks runs two subtasks."""
        prompts: list[str] = []
        self._record_prompts(prompts)
        result = sorcar.run(
            "<task>first thing</task><task>second thing</task>",
            work_dir=self.repo, use_worktree=False, auto_commit=False,
            endpoint_file=self.endpoint_file, timeout=60,
        )
        assert result.success is True, result
        assert len(prompts) == 2, prompts
        assert "first thing" in prompts[0] and "second thing" in prompts[1]
