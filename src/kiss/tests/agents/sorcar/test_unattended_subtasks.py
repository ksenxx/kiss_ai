# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: sub-tasks of an unattended (cron) run never block on the user.

Production failure (``~/.kiss/sorcar.db``, 2026-09-21 23:33 UTC): the cron
prompt job ``5937905c`` dispatched the Slack channel agent with
``run_agent``; the child hit a permission gate and called
``ask_user_question``.  Nobody was there, so the child blocked for the
full 600 s ``run_agent`` timeout and the cron job reported nothing.  The
same 10-minute stall repeated on every one of ~18 runs that night.

Two fixes, both exercised here against a real ``SorcarAgent`` driven by a
local OpenAI-compatible HTTP server:

* ``ask_user_question`` returns an error at once when the running task is
  unattended (its prompt carries ``UNATTENDED_MARKER``) instead of
  invoking the blocking callback.
* ``run_parallel`` prepends ``UNATTENDED_CHILD_PREAMBLE`` to every child
  task of an unattended run; ``run_agent`` (``_dispatch_reserved``) appends
  it through ``append_to_prompt`` so that an agent script's ``prompt()``
  override cannot drop it.  Either way the children inherit the rule.

Detection looks at the current task only (after the chat history's
``# Task`` heading) and at the preamble's position, so an earlier result
that quotes the sentence, or a task that merely mentions it, is not
treated as unattended.
"""

from __future__ import annotations

import json
import tempfile
from http.server import BaseHTTPRequestHandler

from kiss.agents.sorcar.cron_agent import (
    PROMPT_PREAMBLE,
    UNATTENDED_CHILD_PREAMBLE,
    UNATTENDED_MARKER,
    is_unattended,
    unattended_child_prompt,
    unattended_child_suffix,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.core.test_budget_enforcement_e2e import (
    _CHEAP,
    _read_body,
    _send_json,
    _start_server,
    _tool_call_response,
)


def _finish_args(message: dict) -> str:
    """``finish`` arguments echoing *message*'s content as the summary."""
    return json.dumps({"success": True, "summary_in_html": str(message.get("content"))})


class _AskThenFinishHandler(BaseHTTPRequestHandler):
    """First turn: ``ask_user_question``; second turn: ``finish`` echoing the
    tool result the model saw, so the test can assert on it."""

    def do_POST(self) -> None:  # noqa: N802
        messages = json.loads(_read_body(self)).get("messages", [])
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if tool_results:
            resp = _tool_call_response("finish", _finish_args(tool_results[-1]), *_CHEAP)
        else:
            resp = _tool_call_response(
                "ask_user_question", json.dumps({"question": "May I post?"}), *_CHEAP
            )
        _send_json(self, resp)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _run_asking_agent(prompt: str) -> tuple[str, list[str]]:
    """Run a SorcarAgent whose model asks a question, returning (result, questions
    that reached the callback)."""
    answered: list[str] = []

    def answer(question: str) -> str:
        answered.append(question)
        return "yes"

    srv, url = _start_server(_AskThenFinishHandler)
    try:
        with tempfile.TemporaryDirectory() as td:
            agent = SorcarAgent("unattended-ask")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template=prompt,
                max_steps=4,
                max_budget=1.0,
                work_dir=td,
                verbose=False,
                model_config={"base_url": url, "api_key": "test-key"},
                ask_user_question_callback=answer,
            )
    finally:
        srv.shutdown()
    return result, answered


def test_ask_user_question_fails_fast_in_unattended_run() -> None:
    """With the cron preamble in the prompt the callback is never called and the
    model sees an error telling it to proceed or report the blocker."""
    result, answered = _run_asking_agent(PROMPT_PREAMBLE + "Post the report.")
    assert answered == []
    assert "Error: this task runs unattended" in result
    assert "report the blocker" in result


def test_ask_user_question_still_reaches_the_user_when_attended() -> None:
    """The same run without the preamble asks the user and gets the answer."""
    result, answered = _run_asking_agent("Post the report.")
    assert answered == ["May I post?"]
    assert "summary_in_html: yes" in result or "yes" in result


class _ParallelThenFinishHandler(BaseHTTPRequestHandler):
    """Parent: ``run_parallel`` of two children, then ``finish`` with the YAML
    result.  Children (prompt contains CHILDPROBE): ``finish`` echoing their
    own task text, taken from the first user message, which is also recorded
    in ``child_prompts`` for the assertions."""

    child_prompts: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        messages = json.loads(_read_body(self)).get("messages", [])
        text = json.dumps(messages)
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if has_tool_result:
            last = [m for m in messages if m.get("role") == "tool"][-1]
            resp = _tool_call_response("finish", _finish_args(last), *_CHEAP)
        elif "CHILDPROBE" in text:
            first_user = next(m for m in messages if m.get("role") == "user")
            type(self).child_prompts.append(str(first_user.get("content")))
            resp = _tool_call_response("finish", _finish_args(first_user), *_CHEAP)
        else:
            args = json.dumps({"tasks": '["CHILDPROBE one", "CHILDPROBE two"]'})
            resp = _tool_call_response("run_parallel", args, *_CHEAP)
        _send_json(self, resp)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _run_parallel_parent(prompt: str) -> list[str]:
    """Run the fan-out parent; return the task text each child was given."""
    _ParallelThenFinishHandler.child_prompts = []
    srv, url = _start_server(_ParallelThenFinishHandler)
    try:
        with tempfile.TemporaryDirectory() as td:
            agent = SorcarAgent("unattended-parent")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template=prompt,
                max_steps=4,
                max_budget=2.0,
                work_dir=td,
                verbose=False,
                model_config={"base_url": url, "api_key": "test-key"},
            )
    finally:
        srv.shutdown()
    assert "success: true" in result
    return sorted(_ParallelThenFinishHandler.child_prompts)


def test_run_parallel_children_inherit_the_unattended_rule() -> None:
    """Both children's prompts carry the child preamble exactly once."""
    prompts = _run_parallel_parent(PROMPT_PREAMBLE + "Fan out.")
    assert len(prompts) == 2
    for prompt, probe in zip(prompts, ["CHILDPROBE one", "CHILDPROBE two"], strict=True):
        assert prompt.count(UNATTENDED_CHILD_PREAMBLE) == 1
        assert prompt.index(UNATTENDED_CHILD_PREAMBLE) < prompt.index(probe)


def test_run_parallel_children_of_attended_run_are_untouched() -> None:
    prompts = _run_parallel_parent("Fan out.")
    assert len(prompts) == 2
    for prompt, probe in zip(prompts, ["CHILDPROBE one", "CHILDPROBE two"], strict=True):
        assert UNATTENDED_MARKER not in prompt
        assert probe in prompt


class _Agent:
    def __init__(self, text: str) -> None:
        self.task_description = text


def test_unattended_helpers() -> None:
    """The preamble is added only once, in front for run_parallel and at the
    end of the run_agent suffix."""
    assert UNATTENDED_MARKER in PROMPT_PREAMBLE
    assert UNATTENDED_MARKER in UNATTENDED_CHILD_PREAMBLE
    child = unattended_child_prompt("do x")
    assert child == UNATTENDED_CHILD_PREAMBLE + "\n\ndo x"
    assert unattended_child_prompt(child) == child
    suffix = unattended_child_suffix("")
    assert suffix.endswith(UNATTENDED_CHILD_PREAMBLE)
    assert unattended_child_suffix(suffix) == suffix
    assert unattended_child_suffix("extra").startswith("extra")


def test_is_unattended_positions() -> None:
    """Cron job prompt, prepended child preamble, appended child suffix, and the
    chat wrapper around each are all recognised."""
    assert is_unattended(_Agent(PROMPT_PREAMBLE + "job"))
    assert is_unattended(_Agent("# Task\n" + PROMPT_PREAMBLE + "job"))
    assert is_unattended(_Agent(unattended_child_prompt("child")))
    assert is_unattended(_Agent("child" + unattended_child_suffix("")))
    assert is_unattended(
        _Agent("## Previous tasks\n\nold\n\n---\n\n# Task (work on it now)\n\n"
               + "child" + unattended_child_suffix(""))
    )


def test_is_unattended_rejects_quotes_and_history() -> None:
    """A task that quotes the sentence, or chat history holding a whole cron
    prompt, does not make the current task unattended."""
    assert not is_unattended(_Agent(f'Explain this sentence: "{UNATTENDED_MARKER}"'))
    assert not is_unattended(_Agent("Analyse the preamble: " + UNATTENDED_CHILD_PREAMBLE + " ok?"))
    history = (
        "## Previous tasks and results from the chat session for reference\n\n"
        + PROMPT_PREAMBLE + "old job\n\nresult: " + UNATTENDED_CHILD_PREAMBLE
        + "\n\n---\n\n# Task (work on it now)\n\nnew interactive task"
    )
    assert not is_unattended(_Agent(history))
    assert not is_unattended(object())
    assert not is_unattended(None)
