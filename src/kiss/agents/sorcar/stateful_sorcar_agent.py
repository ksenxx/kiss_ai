"""Stateful Sorcar agent with chat-session persistence.

Subclasses :class:`SorcarAgent` to add multi-turn chat-session state
management — the same workflow that the VS Code extension performs in
``VSCodeServer._run_task()``, but as a standalone reusable Python agent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.persistence import (
    _add_task,
    _generate_chat_id,
    _load_chat_context,
    _load_task_chat_id,
    _save_task_result,
)
from kiss.agents.sorcar.sorcar_agent import (
    SorcarAgent,
    _build_arg_parser,
    _resolve_task,
    cli_ask_user_question,
    cli_wait_for_user,
)


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
            return prompt
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
        _add_task(prompt_template, chat_id=self._chat_id)

        result_summary = ""
        try:
            result = super().run(prompt_template=agent_prompt, **kwargs)
            result_yaml = yaml.safe_load(result)
            result_summary = result_yaml.get("summary", "") if result_yaml else ""
            return result
        except Exception as e:
            result_summary = f"Task failed: {e}"
            raise
        finally:
            _save_task_result(prompt_template, result_summary)


def main() -> None:  # pragma: no cover – CLI entry point requires API
    """Run StatefulSorcarAgent from the command line with chat persistence."""
    import time as time_mod

    if len(sys.argv) <= 1:
        print(
            "Usage: stateful_sorcar_agent [-m MODEL] [-e ENDPOINT] [-b BUDGET] "
            "[-w WORK_DIR] [-t TASK] [-f FILE] [-n]"
        )
        sys.exit(1)

    parser = _build_arg_parser()
    parser.add_argument(
        "-n", "--new", action="store_true",
        help="Start a new chat session",
    )
    args = parser.parse_args()

    agent = StatefulSorcarAgent("Stateful Sorcar Agent")

    task_description = _resolve_task(args)
    work_dir = args.work_dir or str(Path(".").resolve())
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    model_config: dict[str, Any] = {}
    if args.endpoint:
        model_config["base_url"] = args.endpoint

    run_kwargs: dict[str, Any] = {
        "prompt_template": task_description,
        "model_name": args.model_name,
        "max_budget": args.max_budget,
        "model_config": model_config,
        "work_dir": work_dir,
        "headless": args.headless,
        "verbose": args.verbose,
        "wait_for_user_callback": cli_wait_for_user,
        "ask_user_question_callback": cli_ask_user_question,
    }

    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        agent.run(**run_kwargs)
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time

    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
