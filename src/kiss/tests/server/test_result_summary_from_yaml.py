# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test: the task runner persists the per-task ``result`` row from the
YAML returned by ``ChatSorcarAgent.run`` even though the printer
recording has already been torn down by the time control returns to
``_TaskRunnerMixin._run_task_inner``.

Spec
----
1. The user runs task T1 in a fresh chat tab.  The underlying agent
   (mocked here to avoid LLM calls) returns the standard
   ``finish()`` YAML payload — ``success: true`` plus a ``summary``
   string — but does NOT broadcast a ``type=result`` printer event
   itself.  In production that event IS broadcast (by
   :meth:`KISSAgent.perform_task` / :meth:`RelentlessAgent`) but it
   reaches the printer BEFORE ``ChatSorcarAgent.run``'s ``finally``
   block calls ``stop_recording`` — by the time control returns to
   the task runner, the recording has already been popped and the
   thread-local ``task_id`` cleared, so ``peek_recording`` returns
   an empty list.
2. The task runner therefore cannot rely on
   ``_extract_result_summary`` for the persisted result.  It must
   parse the YAML return value of ``tab.agent.run`` and persist the
   ``summary`` field as ``task_history.result``.
3. When the user runs a second task T2 in the same chat,
   :meth:`ChatSorcarAgent.build_chat_prompt` loads T1's result from
   the DB and surfaces it under "### Result 1".  Without the fix,
   the row contains the sentinel string ``"No summary available"``;
   with the fix it contains the real summary.

The test reproduces the production ordering exactly: the patched
``SorcarAgent.run`` returns the YAML *after* the recording has been
stopped by ``ChatSorcarAgent.run``'s ``finally``, so the only path to
recover the summary is to parse the YAML in the task runner.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import (
    init_repo,
    make_recording_server,
    patch_sorcar_parent_run,
    redirect_persistence,
    restore_persistence,
    restore_sorcar_parent_run,
)


class _PromptCapture:
    """Captures ``prompt_template`` and returns a canned YAML result
    WITHOUT broadcasting a ``type=result`` printer event.

    This mirrors the real production ordering: the YAML reaches the
    task runner via the return value, but the matching ``result`` event
    is no longer accessible to ``peek_recording`` by the time the task
    runner reads back — because ``ChatSorcarAgent.run``'s ``finally``
    has already popped the recording buffer.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.summaries = [
            "remembered banana as the magic word",
            "second task done",
        ]

    def fake_run(self, agent_self: object, **kwargs: object) -> str:
        import yaml as _yaml
        self.prompts.append(str(kwargs.get("prompt_template", "")))
        summary = self.summaries.pop(0) if self.summaries else "done"
        return str(_yaml.dump({"success": True, "summary": summary}))


def _run_and_wait(server: AgentServer, tab_id: str, prompt: str,
                  work_dir: str) -> None:
    server._handle_command({
        "type": "run", "prompt": prompt,
        "model": "claude-opus-4-6", "workDir": work_dir,
        "tabId": tab_id, "autoCommit": True,
    })
    st = agent_state.find_by_tab(tab_id)
    assert st is not None
    t = st.task_thread
    assert t is not None
    t.join(timeout=60)
    assert not t.is_alive()


class TestResultSummaryFromYaml(unittest.TestCase):
    """Without the fix, the second task's ``build_chat_prompt``
    preamble surfaces the sentinel string ``"No summary available"``
    as Result 1 — confirming that ``_extract_result_summary`` could
    not recover the summary after recording teardown.  The fix routes
    the persisted result through the YAML return value instead, so
    the real per-task summary appears.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        init_repo(self.tmpdir)
        self.server, self.events, self.lock = make_recording_server()
        self.capture = _PromptCapture()
        self.original_run = patch_sorcar_parent_run(self.capture.fake_run)

    def tearDown(self) -> None:
        restore_sorcar_parent_run(self.original_run)
        agent_state.agent_states.clear()
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_second_task_prompt_includes_real_first_summary(self) -> None:
        tab_id = "tab-chat"
        _run_and_wait(self.server, tab_id,
                      "remember the magic word: banana",
                      self.tmpdir)

        rows = th._load_history(limit=10)
        assert rows, "expected first task to be persisted"
        first_result = rows[0]["result"]
        assert first_result != "No summary available", (
            "task runner persisted the sentinel instead of the YAML "
            "summary — _extract_result_summary cannot see the result "
            "event after stop_recording pops the buffer"
        )
        assert "banana" in str(first_result), (
            f"expected the real summary ('...banana...') in "
            f"task_history.result, got {first_result!r}"
        )

        _run_and_wait(self.server, tab_id, "what was the magic word?",
                      self.tmpdir)
        assert len(self.capture.prompts) >= 2

        second_prompt = self.capture.prompts[1]
        assert "Previous tasks and results" in second_prompt, (
            "second prompt should include the chat-history preamble"
        )
        assert "No summary available" not in second_prompt, (
            "second prompt must not surface the sentinel result row "
            "for the first task; build_chat_prompt is leaking the "
            "fallback string into the LLM context.  Prompt was:\n"
            + second_prompt
        )
        assert "banana" in second_prompt, (
            "second prompt should reference the real first-task "
            "summary (which contained 'banana') under '### Result 1'"
        )


if __name__ == "__main__":
    unittest.main()
