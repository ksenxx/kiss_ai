"""Stateful Sorcar agent with chat-session persistence.

Subclasses :class:`SorcarAgent` to add multi-turn chat-session state
management — the same workflow that the VS Code extension performs in
``VSCodeServer._run_task()``, but as a standalone reusable Python agent.
"""

from __future__ import annotations

import sys
from typing import Any

import yaml

from kiss.agents.sorcar.cli_helpers import (
    _apply_chat_args,
    _build_chat_arg_parser,
    _build_run_kwargs,
    _print_recent_chats,
    _print_run_stats,
)
from kiss.agents.sorcar.persistence import (
    _add_task,
    _generate_chat_id,
    _load_chat_context,
    _load_task_chat_id,
    _save_task_extra,
    _save_task_result,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent


class StatefulSorcarAgent(SorcarAgent):
    """SorcarAgent with chat-session state management.

    Maintains a ``chat_id`` and automatically loads prior chat context,
    persists tasks and results to ``history.db``, and augments prompts
    with previous session context — replicating the stateful workflow
    from the VS Code extension as a standalone reusable agent.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._chat_id = _generate_chat_id()
        self._last_task_id: int | None = None

    @property
    def chat_id(self) -> str:
        """Return the current chat session ID."""
        return self._chat_id

    def new_chat(self) -> None:
        """Reset to a new chat session (equivalent to VS Code 'Clear')."""
        self._chat_id = _generate_chat_id()

    def resume_chat(self, task: str) -> None:
        """Resume a previous chat session by looking up the task's chat_id.

        If the task has an associated ``chat_id`` in history, subsequent
        ``run()`` calls will continue that session.

        Args:
            task: The task description string to look up.
        """
        chat_id = _load_task_chat_id(task)
        if chat_id:
            self.resume_chat_by_id(chat_id)

    def resume_chat_by_id(self, chat_id: str) -> None:
        """Resume a chat session using a stable chat identifier.

        Args:
            chat_id: Persisted chat session identifier to resume.
        """
        if chat_id:
            self._chat_id = chat_id

    def build_chat_prompt(self, prompt: str) -> str:
        """Load chat context and augment prompt with previous tasks/results.

        Args:
            prompt: The original task prompt.

        Returns:
            The augmented prompt with chat history prepended, or the
            original prompt if no prior context exists.
        """
        chat_context = _load_chat_context(self._chat_id)
        if not chat_context:
            return "# Task\n" + prompt
        parts = ["## Previous tasks and results from the chat session for reference\n"]
        for i, entry in enumerate(chat_context, 1):
            parts.append(f"### Task {i}\n{entry['task']}")
            if entry.get("result"):
                parts.append(f"### Result {i}\n{entry['result']}")
        parts.append("---\n")
        return "\n\n".join(parts) + "# Task (work on it now)\n\n" + prompt

    def run(  # type: ignore[override]
        self,
        prompt_template: str = "",
        **kwargs: Any,
    ) -> str:
        """Run the agent with chat-session context management.

        Loads prior chat context, persists the new task, augments the
        prompt with previous tasks/results, runs the underlying agent,
        and saves the result back to history.

        Only the result summary is persisted here.  Callers that record
        chat events (e.g. the VS Code server) should additionally call
        :func:`~kiss.agents.sorcar.persistence._set_latest_chat_events`
        to persist the full event stream.

        Args:
            prompt_template: The task prompt.
            **kwargs: All other arguments forwarded to ``SorcarAgent.run()``.

        Returns:
            YAML string with 'success' and 'summary' keys.
        """
        agent_prompt = self.build_chat_prompt(prompt_template)
        task_id = _add_task(prompt_template, chat_id=self._chat_id)
        self._last_task_id = task_id

        result_summary = ""
        try:
            result = super().run(prompt_template=agent_prompt, **kwargs)
            try:
                result_yaml = yaml.safe_load(result)
                if isinstance(result_yaml, dict):
                    result_summary = result_yaml.get("summary", "")
            except Exception:
                result_summary = result[:500] if result else ""
            return result
        except Exception:
            result_summary = "Task failed"
            raise
        finally:
            _save_task_result(task_id=task_id, result=result_summary)
            from kiss._version import __version__

            _save_task_extra(
                {
                    "model": getattr(self, "model_name", ""),
                    "work_dir": getattr(self, "work_dir", ""),
                    "version": __version__,
                    "tokens": self.total_tokens_used,
                    "cost": round(self.budget_used, 6),
                    "is_parallel": self._is_parallel,
                    "is_worktree": type(self).__name__ == "WorktreeSorcarAgent",
                },
                task_id=task_id,
            )


def main() -> None:  # pragma: no cover – CLI entry point requires API
    """Run StatefulSorcarAgent from the command line with chat persistence."""
    import time as time_mod

    if len(sys.argv) <= 1:
        print(
            "Usage: stateful_sorcar_agent [-m MODEL] [-e ENDPOINT] [-b BUDGET] "
            "[-w WORK_DIR] [-t TASK] [-f FILE] [-n] [--chat-id ID] [-l] [-p]"
        )
        sys.exit(1)

    parser = _build_chat_arg_parser()
    args = parser.parse_args()

    if args.list_chat_id:
        _print_recent_chats()
        sys.exit(0)

    agent = StatefulSorcarAgent("Stateful Sorcar Agent")
    _apply_chat_args(agent, args)

    start_time = time_mod.time()
    agent.run(**_build_run_kwargs(args))
    elapsed = time_mod.time() - start_time

    _print_run_stats(agent, elapsed)


if __name__ == "__main__":
    main()
