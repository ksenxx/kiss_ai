# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the tasks of a chat form a tree, and a task's prompt
carries the tasks on the path from the tree's root to its parent.

Spec
----
* A task run after the user loaded a task of the chat (a CHATS click:
  ``resumeSession`` carrying ``taskId``, or the ``continueTaskId`` a
  page sends when the click only scrolled its tab) has the loaded task
  as its parent.  The load counts once: it is discarded when the tab is
  re-pointed at a chat without a task id, starts a new chat or closes.
  A ``<task>`` message sent while the tab's task runs is a task of its
  own, run after that one: it takes the load too (the
  ``continueTaskId`` of its ``appendUserMessage``), whether it runs as
  a subtask of the run or as the tab's next run.
* A task run without a load has the chat's last task as its parent.
* The root of the tree is the chat's first task.  A row from before the
  parent was recorded, and one whose parent was deleted, continues the
  task just before it in its chat; a sub-agent's parent is the task
  that dispatched it.
* The prompt's "Previous tasks and results" section holds the tasks on
  the path root -> parent, at most ``MAX_TASKS`` (10) of them: the
  first 2 and the last 8.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _load_task_lineage
from kiss.server import agent_state, task_runner
from kiss.server.server import AgentServer
from kiss.tests.server.conftest import (
    make_recording_server,
    patch_sorcar_parent_run,
    redirect_persistence,
    restore_persistence,
    restore_sorcar_parent_run,
)

_CONTEXT_TASK_RE = re.compile(r"^### Task \d+\n(.*)$", re.MULTILINE)


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmpdir,
                   capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir,
                   capture_output=True, check=True)
    Path(tmpdir, ".gitkeep").touch()
    Path(tmpdir, ".gitignore").write_text(".kiss/\n")
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir,
                   capture_output=True, check=True)


class _PromptCapture:
    """Captures the ``prompt_template`` the underlying agent receives
    and returns a canned YAML success result (no LLM calls)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.counter = 0
        # Called once, inside the first run: what the user does in the
        # tab while its task is running.
        self.during_first_run: Callable[[], None] | None = None

    def fake_run(self, agent_self: object, **kwargs: object) -> str:
        self.prompts.append(str(kwargs.get("prompt_template", "")))
        self.counter += 1
        if self.counter == 1 and self.during_first_run is not None:
            self.during_first_run()
        summary = f"run-{self.counter} done"
        printer = kwargs.get("printer")
        if printer is not None and hasattr(printer, "broadcast"):
            cast(Any, printer).broadcast({
                "type": "result",
                "text": summary,
                "success": True,
            })
        return f"success: true\nsummary: {summary}\n"


def _context_tasks(prompt: str) -> list[str]:
    """The task texts of *prompt*'s "Previous tasks" section, oldest first."""
    return _CONTEXT_TASK_RE.findall(prompt.split("# Task (work on it now)")[0])


def _lineage_of(task: str) -> list[str]:
    """The task texts on the path from the tree's root to the row whose text is *task*."""
    row = th._get_db().execute(
        "SELECT id FROM task_history WHERE task = ?", (task,),
    ).fetchone()
    return [str(entry["task"]) for entry in _load_task_lineage(str(row[0]))]


def _chat_of(task: str) -> str:
    """The chat of the row whose text is *task*."""
    row = th._get_db().execute(
        "SELECT chat_id FROM task_history WHERE task = ?", (task,),
    ).fetchone()
    return str(row[0])


def _run_and_wait(server: AgentServer, tab_id: str, prompt: str,
                  work_dir: str, **fields: object) -> None:
    """Send a ``run`` of *prompt* (plus the wire *fields*) and wait for its end."""
    server._handle_command({
        "type": "run", "prompt": prompt,
        "model": "claude-fable-5", "workDir": work_dir,
        "tabId": tab_id, **fields,
    })
    with agent_state.STATE_LOCK:
        state = agent_state.find_by_tab(tab_id)
        t = state.task_thread if state is not None else None
    if t is not None:
        # A hang guard, not a latency bound: even with the agent faked,
        # the run spawns ~60 git processes (worktree create, baseline
        # commit, discard) -- ~0.3 s on Linux, ~3 s on an idle Windows
        # VM and 15-30 s there with a dozen other test processes running.
        t.join(timeout=60)
        assert not t.is_alive()


def _seed_chat(chat_id: str, names: list[str]) -> list[str]:
    """Persist finished tasks run one after another in chat *chat_id*.

    Args:
        chat_id: The chat the tasks belong to.
        names: One name per task, oldest first; the task's text is
            ``"<name> task"`` and its result ``"<name> result"``.

    Returns:
        The inserted row ids, oldest first.
    """
    ids: list[str] = []
    for name in names:
        tid, _ = th._add_task(f"{name} task", chat_id=chat_id)
        th._save_task_result(f"{name} result", task_id=tid)
        ids.append(tid)
    return ids


def _seed_dispatch_chain(
    specs: list[tuple[str, str, str]],
) -> list[tuple[str, str]]:
    """Persist a chain of finished tasks, each the sub-agent of the one before.

    Args:
        specs: List of ``(task_text, result_text, chat_id)`` tuples in
            chronological (root-first) order.  Each row's
            ``parent_task_id`` points at the previously inserted row.

    Returns:
        List of ``(task_id, chat_id)`` for the inserted rows.
    """
    out: list[tuple[str, str]] = []
    parent = ""
    for task_text, result_text, chat_id in specs:
        extra: dict[str, object] | None = (
            {"parent_task_id": parent} if parent else None
        )
        tid, cid = th._add_task(task_text, chat_id=chat_id, extra=extra)
        th._save_task_result(result_text, task_id=tid)
        out.append((tid, cid))
        parent = tid
    return out


def _open_by_task_id(server: AgentServer, tab_id: str, chat_id: str,
                     task_id: str) -> None:
    """Simulate the frontend opening a fresh tab for a history task."""
    server._handle_command({"type": "newChat", "tabId": tab_id})
    server._handle_command({
        "type": "resumeSession",
        "chatId": chat_id,
        "taskId": task_id,
        "tabId": tab_id,
    })


class _ChainContextBase(unittest.TestCase):
    """Shared fixture: temp DB + git repo + captured-prompt server."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = redirect_persistence(self.tmpdir)
        _init_git_repo(self.tmpdir)
        self.server, self.events, self.lock = make_recording_server()
        self.capture = _PromptCapture()
        self.original_run = patch_sorcar_parent_run(self.capture.fake_run)

    def tearDown(self) -> None:
        restore_sorcar_parent_run(self.original_run)
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def _run_twice(self, tab: str, chat_id: str, first: str, second: str) -> None:
        """Run two tasks one after the other in *tab*, on chat *chat_id* (``""``: a new one).

        Driven synchronously through ``_run_task``: the second run must
        reuse the tab's idle state.  (The async ``_cmd_run`` path leaves
        a dead ``task_thread`` behind after the mid-run rekey, which
        would swallow the second run as steering.)  A direct run is on
        the chat its command names, where ``_cmd_run`` takes the tab's.
        """
        for prompt in (first, second):
            self.server._run_task({
                "type": "run", "prompt": prompt,
                "model": "claude-fable-5", "workDir": self.tmpdir,
                "tabId": tab, "chatId": chat_id,
            })
        assert len(self.capture.prompts) == 2


class TestChatTree(_ChainContextBase):
    """Which task a run continues, and what its prompt carries of the tree."""

    def test_task_run_after_loading_a_task_continues_it(self) -> None:
        _, two, _ = _seed_chat("chatT", ["one", "two", "three"])

        tab = "tab-tree-branch"
        _open_by_task_id(self.server, tab, "chatT", two)
        self._run_twice(tab, "chatT", "four task", "five task")
        assert _chat_of("four task") == _chat_of("five task") == "chatT"

        # The loaded task is the parent: the path to it leaves out the
        # task that came after it in the chat.
        first = self.capture.prompts[0]
        assert _context_tasks(first) == ["one task", "two task"], first
        assert "### Result 1\none result" in first, first
        assert "### Result 2\ntwo result" in first, first
        assert first.rstrip().endswith("four task")
        assert _lineage_of("four task") == ["one task", "two task", "four task"]
        # No task loaded since: the chat's last task is the parent.
        second = self.capture.prompts[1]
        assert _context_tasks(second) == ["one task", "two task", "four task"], second
        assert _lineage_of("five task") == [
            "one task", "two task", "four task", "five task",
        ]
        # The branch left behind is still a path from the same root.
        assert _lineage_of("three task") == ["one task", "two task", "three task"]

    def test_lineage_trimmed_to_max_tasks(self) -> None:
        ids = _seed_chat("chatX", [f"sentinel{i:02d}" for i in range(1, 14)])

        tab = "tab-tree-trim"
        _open_by_task_id(self.server, tab, "chatX", ids[-1])
        _run_and_wait(self.server, tab, "trim check", self.tmpdir)

        prompt = self.capture.prompts[0]
        kept = (1, 2, 6, 7, 8, 9, 10, 11, 12, 13)
        assert _context_tasks(prompt) == [f"sentinel{i:02d} task" for i in kept], prompt
        assert "### Task 3\nsentinel06 task" in prompt, prompt
        assert "### Task 10\nsentinel13 task" in prompt, prompt

    def test_row_without_recorded_parent_continues_the_task_before_it(self) -> None:
        _, two, _ = _seed_chat("chatL", ["one", "two", "three"])
        # Rows from before the parent was recorded, which here share a
        # timestamp too: the row inserted first ran first.
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute("UPDATE task_history SET chat_parent_id = '', timestamp = 1000.0")
            db.commit()

        tab = "tab-tree-legacy"
        _open_by_task_id(self.server, tab, "chatL", two)
        _run_and_wait(self.server, tab, "legacy check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert _context_tasks(prompt) == ["one task", "two task"], prompt
        assert _lineage_of("three task") == ["one task", "two task", "three task"]

    def test_task_whose_parent_was_deleted_continues_the_task_before_it(self) -> None:
        _, two, three = _seed_chat("chatD", ["one", "two", "three"])
        assert th._delete_task(two) == 1

        tab = "tab-tree-deleted"
        _open_by_task_id(self.server, tab, "chatD", three)
        _run_and_wait(self.server, tab, "deleted check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert _context_tasks(prompt) == ["one task", "three task"], prompt

    def test_task_run_in_a_subagent_tab_keeps_the_dispatch_chain(self) -> None:
        rows = _seed_dispatch_chain([
            ("t1 magic word apple", "r1 apple stored", "chatA"),
            ("t2 magic word pear", "r2 pear stored", "chatA"),
            ("t3 magic word plum", "r3 plum stored", "chatB"),
        ])

        tab = "tab-tree-subagent"
        _open_by_task_id(self.server, tab, "chatB", rows[2][0])
        self._run_twice(tab, "", "first follow-up quokka", "second follow-up wombat")

        chain = ["t1 magic word apple", "t2 magic word pear", "t3 magic word plum"]
        first = self.capture.prompts[0]
        assert _context_tasks(first) == chain, first
        assert "### Result 3\nr3 plum stored" in first, first
        # The run started a chat of its own, whose next task still
        # carries the path the first one continued.
        own_chat = _chat_of("first follow-up quokka")
        assert own_chat not in ("chatA", "chatB")
        assert _chat_of("second follow-up wombat") == own_chat
        second = self.capture.prompts[1]
        assert _context_tasks(second) == [*chain, "first follow-up quokka"], second

    def test_cycle_in_parent_links_terminates(self) -> None:
        rows = _seed_dispatch_chain([
            ("cyc-a task", "cyc-a result", "chatC"),
            ("cyc-b task", "cyc-b result", "chatC"),
        ])
        a_id, b_id = rows[0][0], rows[1][0]
        db = th._get_db()
        with th._rw_lock.write_lock():
            db.execute(
                "UPDATE task_history SET parent_task_id = ? WHERE id = ?",
                (b_id, a_id),
            )
            db.commit()

        tab = "tab-chain-cycle"
        _open_by_task_id(self.server, tab, "chatC", b_id)
        _run_and_wait(self.server, tab, "cycle check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert _context_tasks(prompt) == ["cyc-a task", "cyc-b task"], prompt

    def test_rejected_run_preserves_seed(self) -> None:
        one, _ = _seed_chat("chatR", ["one", "two"])

        tab = "tab-chain-rejected"
        _open_by_task_id(self.server, tab, "chatR", one)
        with self.server._state_lock:
            merging_state = agent_state.AgentState(
                f"merge-review-{tab}",
                tab_id=tab,
                server_owned=True,
            )
            merging_state.is_merging = True
            agent_state.register(merging_state)
        # Drive _run_task directly: it reuses the tab's idle (merging)
        # state via _resolve_run_state, and _run_task_inner refuses to
        # run while the state's merge review is open.
        self.server._run_task({
            "type": "run", "prompt": "rejected run",
            "model": "claude-fable-5", "workDir": self.tmpdir,
            "tabId": tab,
        })
        assert not self.capture.prompts, "rejected run must not reach agent"
        assert self.server._tab_opened_task_ids.get(tab) == one

        with self.server._state_lock:
            merging_state.is_merging = False
        _run_and_wait(self.server, tab, "accepted run", self.tmpdir)
        prompt = self.capture.prompts[0]
        assert _context_tasks(prompt) == ["one task"], prompt

    def test_task_named_by_the_page_is_continued_and_takes_the_pin(self) -> None:
        one, two, _ = _seed_chat("chatP", ["one", "two", "three"])

        tab = "tab-tree-page"
        _open_by_task_id(self.server, tab, "chatP", two)
        # The page's click on a task its tab already showed is the later
        # one: it asked the daemon for nothing and comes with the run.
        self.server._run_task({
            "type": "run", "prompt": "four task",
            "model": "claude-fable-5", "workDir": self.tmpdir,
            "tabId": tab, "chatId": "chatP", "continueTaskId": one,
        })
        self.server._run_task({
            "type": "run", "prompt": "five task",
            "model": "claude-fable-5", "workDir": self.tmpdir,
            "tabId": tab,
        })
        assert _chat_of("four task") == _chat_of("five task") == "chatP"

        first, second = self.capture.prompts
        assert _context_tasks(first) == ["one task"], first
        # The tab's own load went with that run too.
        assert _context_tasks(second) == ["one task", "four task"], second

    def test_message_a_run_left_unread_continues_that_run(self) -> None:
        one, _ = _seed_chat("chatQ", ["one", "two"])

        tab = "tab-tree-leftover"
        # Typed while the run is on and never read by it: the daemon
        # runs it next, as a task of its own, from the run's command.
        self.capture.during_first_run = functools.partial(
            self.server._cmd_run, {"tabId": tab, "prompt": "four task"},
        )
        self.server._run_task({
            "type": "run", "prompt": "three task",
            "model": "claude-fable-5", "workDir": self.tmpdir,
            "tabId": tab, "chatId": "chatQ", "continueTaskId": one,
        })
        deadline = time.monotonic() + 60
        while len(self.capture.prompts) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)

        first, second = self.capture.prompts
        assert _context_tasks(first) == ["one task"], first
        # The task the page named was loaded for the run that took it.
        assert _context_tasks(second) == ["one task", "three task"], second
        assert _chat_of("three task") == _chat_of("four task") == "chatQ"

    def _send_during_first_run(self, tab: str, prompt: str, **fields: object) -> None:
        """Have the user send *prompt* (plus wire *fields*) to *tab* during its first run."""
        self.capture.during_first_run = functools.partial(
            self.server._handle_command,
            {"type": "appendUserMessage", "tabId": tab, "prompt": prompt, **fields},
        )

    def _run(self, tab: str, prompt: str, **fields: object) -> None:
        """Run *prompt* (plus the wire *fields*) in *tab* and whatever is queued behind it."""
        self.server._run_task({
            "type": "run", "prompt": prompt,
            "model": "claude-fable-5", "workDir": self.tmpdir,
            "tabId": tab, **fields,
        })

    def _wait_for(self, done: Callable[[], bool]) -> None:
        """Wait for a run the daemon starts itself (a hang guard, not a latency bound)."""
        deadline = time.monotonic() + 60
        while not done() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert done()

    def test_task_queued_on_a_run_continues_the_task_loaded_before_it(self) -> None:
        one, _ = _seed_chat("chatK", ["one", "two"])

        tab = "tab-tree-queued"
        # While ``three`` runs the reader clicks ``one`` and queues two
        # tasks behind the run: tasks of their own, the first of which
        # takes the click and the second follows the first.
        self._send_during_first_run(
            tab, "<task>four task</task><task>five task</task>", continueTaskId=one,
        )
        self._run(tab, "three task", chatId="chatK")
        self._run(tab, "six task")
        assert _chat_of("four task") == _chat_of("six task") == "chatK"

        three, four, five, six = self.capture.prompts
        assert _context_tasks(three) == ["one task", "two task"], three
        assert _context_tasks(four) == ["one task"], four
        assert _lineage_of("four task") == ["one task", "four task"]
        assert _context_tasks(five) == ["one task", "four task"], five
        # The click went with the task it was sent with.
        assert _context_tasks(six) == ["one task", "four task", "five task"], six
        assert _lineage_of("three task") == ["one task", "two task", "three task"]

    def test_task_queued_on_a_run_without_a_load_continues_that_run(self) -> None:
        _seed_chat("chatN", ["one", "two"])

        tab = "tab-tree-queued-plain"
        self._send_during_first_run(tab, "<task>four task</task>")
        self._run(tab, "three task", chatId="chatN")

        _, four = self.capture.prompts
        assert _context_tasks(four) == ["one task", "two task", "three task"], four

    def test_command_queued_on_a_run_continues_the_task_loaded_before_it(self) -> None:
        one, _ = _seed_chat("chatC2", ["one", "two"])

        tab = "tab-tree-queued-command"
        # A queued ``/xxx`` command is no subtask of the run: the daemon
        # runs it next, as a run of its own, and it keeps the click.
        self._send_during_first_run(tab, "<task>/sync help</task>", continueTaskId=one)
        self._run(tab, "three task", chatId="chatC2")
        self._wait_for(lambda: th._get_db().execute(
            "SELECT 1 FROM task_history WHERE task = '/sync help' AND result != ?",
            ("Agent Failed Abruptly",),
        ).fetchone() is not None)

        assert len(self.capture.prompts) == 1, "a help exchange runs no agent"
        assert _lineage_of("/sync help") == ["one task", "/sync help"]
        assert _chat_of("/sync help") == "chatC2"

    def test_task_sent_as_a_run_ends_continues_the_task_loaded_before_it(self) -> None:
        one, _ = _seed_chat("chatE", ["one", "two"])

        tab = "tab-tree-late"

        def send_after_the_last_drain() -> None:
            # The run's loop is past its last look at the queue: the
            # task waits as a message of the tab for the daemon to run
            # it next, and is still a task of its own.
            with agent_state.STATE_LOCK:
                state = agent_state.find_by_tab(tab)
                assert state is not None
                state.followup_queue_closed = True
            self.server._handle_command({
                "type": "appendUserMessage", "tabId": tab,
                "prompt": "<task>four task</task>", "continueTaskId": one,
            })

        self.capture.during_first_run = send_after_the_last_drain
        self._run(tab, "three task", chatId="chatE")
        self._wait_for(lambda: len(self.capture.prompts) == 2)

        _, four = self.capture.prompts
        assert _context_tasks(four) == ["one task"], four
        assert _chat_of("four task") == "chatE"

    def test_follow_up_of_a_run_that_never_ran_keeps_the_task_loaded_before_it(self) -> None:
        one, _ = _seed_chat("chatV", ["one", "two"])

        tab = "tab-tree-unrun"
        # A run that ends before its agent starts (a ``/xxx help``
        # answer, a refusal) leaves the tasks queued on it to the
        # daemon, which runs them next: each still continues the task
        # loaded before it was sent.
        with agent_state.STATE_LOCK:
            state = agent_state.AgentState(f"unrun-{tab}", tab_id=tab, server_owned=True)
            state.chat_id = "chatV"
            agent_state.register(state)
            routed = self.server._route_prompt_to_owner(
                state, "<task>four task</task>", tab, None, continued_task_id=one,
            )
            assert routed is True
            task_runner._end_run_before_loop(state)
            reruns, redispatch, _ = self.server._release_run_state(
                state, {"tabId": tab}, tab, "", "claude-fable-5",
            )
        assert redispatch
        assert [str(prompt) for prompt, _ in reruns] == ["<task>four task</task>"]
        self.server._redispatch_leftover_prompts(
            {"type": "run", "model": "claude-fable-5", "workDir": self.tmpdir,
             "tabId": tab, "chatId": "chatV"},
            reruns,
        )
        self._wait_for(lambda: len(self.capture.prompts) == 1)

        (four,) = self.capture.prompts
        assert _context_tasks(four) == ["one task"], four
        assert _chat_of("four task") == "chatV"

    def test_help_exchange_continues_the_loaded_task(self) -> None:
        one, _ = _seed_chat("chatH", ["one", "two"])

        tab = "tab-tree-help"
        _open_by_task_id(self.server, tab, "chatH", one)
        # ``/xxx help`` is answered without an agent, by a row of the chat.
        self.server._run_task({
            "type": "run", "prompt": "/sync help",
            "model": "claude-fable-5", "workDir": self.tmpdir,
            "tabId": tab, "chatId": "chatH",
        })
        assert not self.capture.prompts, "a help exchange runs no agent"
        assert _lineage_of("/sync help") == ["one task", "/sync help"]
        assert _chat_of("/sync help") == "chatH"

        _run_and_wait(self.server, tab, "three task", self.tmpdir)
        prompt = self.capture.prompts[0]
        assert _context_tasks(prompt) == ["one task", "/sync help"], prompt


class TestChainContextCleared(_ChainContextBase):
    """The loaded task is discarded on re-open / newChat / close."""

    def test_resume_without_task_id_clears_seed(self) -> None:
        one, _ = _seed_chat("chatA", ["one", "two"])

        tab = "tab-clear-resume"
        _open_by_task_id(self.server, tab, "chatA", one)
        self.server._handle_command({
            "type": "resumeSession", "chatId": "chatA", "tabId": tab,
        })
        _run_and_wait(self.server, tab, "post-clear check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert _context_tasks(prompt) == ["one task", "two task"], prompt

    def test_unknown_task_id_falls_back_to_chat_context(self) -> None:
        _seed_chat("chatE", ["koala"])

        tab = "tab-fallback"
        _open_by_task_id(self.server, tab, "chatE", "missing-task-id")
        _run_and_wait(self.server, tab, "fallback check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert _context_tasks(prompt) == ["koala task"], prompt
        assert _lineage_of("fallback check") == ["koala task", "fallback check"]

    def test_new_chat_clears_seed(self) -> None:
        (one,) = _seed_chat("chatA", ["one"])

        tab = "tab-clear-newchat"
        _open_by_task_id(self.server, tab, "chatA", one)
        self.server._handle_command({"type": "newChat", "tabId": tab})
        _run_and_wait(self.server, tab, "fresh chat check", self.tmpdir)

        prompt = self.capture.prompts[0]
        assert "Previous tasks and results" not in prompt, prompt
        assert prompt.startswith("# Task\n"), prompt
        # The first task of a chat is the root of its tree.
        assert _lineage_of("fresh chat check") == ["fresh chat check"]

    def test_close_tab_clears_seed(self) -> None:
        (one,) = _seed_chat("chatA", ["one"])

        tab = "tab-clear-close"
        _open_by_task_id(self.server, tab, "chatA", one)
        assert self.server._tab_opened_task_ids.get(tab) == one
        self.server._handle_command({"type": "closeTab", "tabId": tab})
        assert tab not in self.server._tab_opened_task_ids


class TestChainContextPrimitives(_ChainContextBase):
    """Direct behavior of the persistence helper and agent seed API."""

    def test_lineage_of_no_or_unknown_task_is_empty(self) -> None:
        assert _load_task_lineage("") == []
        assert _load_task_lineage("nope") == []

    def test_lineage_runs_from_the_root_to_the_task(self) -> None:
        rows = _seed_dispatch_chain([
            ("root", "root-res", "chatZ"),
            ("mid", "mid-res", "chatZ"),
            ("leaf", "leaf-res", "chatZ"),
        ])
        ctx = _load_task_lineage(rows[-1][0])
        assert [e["task"] for e in ctx] == ["root", "mid", "leaf"]
        assert [e["result"] for e in ctx] == [
            "root-res", "mid-res", "leaf-res",
        ]

    def test_resume_from_task_id_empty_is_noop(self) -> None:
        agent = ChatSorcarAgent("chain-noop")
        agent.resume_from_task_id("")
        assert agent.build_chat_prompt("go") == "# Task\ngo"

    def test_resume_from_task_id_seeds_one_prompt_only(self) -> None:
        rows = _seed_dispatch_chain([
            ("root", "root-res", "chatZ"),
            ("leaf", "leaf-res", "chatZ"),
        ])
        agent = ChatSorcarAgent("chain-seed")
        agent.resume_from_task_id(rows[-1][0])
        first = agent.build_chat_prompt("go")
        assert _context_tasks(first) == ["root", "leaf"], first
        second = agent.build_chat_prompt("go again")
        assert second == "# Task\ngo again"


if __name__ == "__main__":
    unittest.main()
