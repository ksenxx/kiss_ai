# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: concurrent ``_cmd_run`` start race on one tab (BUG-C).

The winning submit installs ``state.task_thread`` under
``_state_lock`` and calls ``thread.start()`` only AFTER releasing the
lock and publishing the tab in the registry (a file write and a
broadcast — a wide window).  A second concurrent submit for the same
tab arriving in that window must NOT start a second task: it must
observe the installed ``task_thread`` and queue its prompt as steering
instead.  A replay of another chat into the tab arriving in that
window owns the tab or loses it by the order of the two registry
publications, and the run's startup frames follow the owner.

The tests make the races deterministic by holding the first
``_cmd_run`` before or right after its registry publication until the
other command has executed on another thread.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import CodeType
from typing import Any, cast

from kiss.agents.sorcar import persistence as th
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.server import agent_state
from kiss.server.server import AgentServer
from kiss.tests.server.pause import Pause, Pauses


class TestRunStartRace(unittest.TestCase):
    """Another connection's command lands in the start window of a run."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt3-race-")
        self.server = AgentServer()
        self.events: list[dict[str, Any]] = []
        self.run_inputs: list[str] = []
        self._events_lock = threading.Lock()
        self.server.printer.broadcast = self._record  # type: ignore[method-assign]

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        run_inputs = self.run_inputs

        def stub_run(self_agent: object, **kwargs: object) -> str:
            run_inputs.append(str(kwargs))
            return "success: true\nsummary: ok\n"

        self._parent_class.run = stub_run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run
        agent_state.agent_states.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _record(self, event: dict[str, Any]) -> None:
        with self._events_lock:
            self.events.append(event)

    def _frames(self, kind: str) -> list[dict[str, Any]]:
        with self._events_lock:
            return [e for e in self.events if e.get("type") == kind]

    def _run_cmd(self, tab_id: str) -> dict[str, Any]:
        work_dir = Path(self.tmpdir) / "plain"
        work_dir.mkdir(exist_ok=True)
        return {
            "type": "run",
            "prompt": "bughunt3 race task",
            "tabId": tab_id,
            "workDir": str(work_dir),
            "useWorktree": False,
            "autoCommit": False,
            "model": "",
        }

    def test_concurrent_second_submit_is_dropped(self) -> None:
        tab_id = "race-tab"
        cmd = self._run_cmd(tab_id)
        t1 = threading.Thread(
            target=self.server._cmd_run, args=(dict(cmd),), daemon=True,
        )
        with Pause(AgentServer._publish_tabs_state.__code__, bool) as gate:
            t1.start()
            assert gate.paused.wait(timeout=30), "first _cmd_run never published its tab"
            with self.server._state_lock:
                first_state = agent_state.find_by_tab(tab_id)
                assert first_state is not None
                first_thread = first_state.task_thread
                first_stop = first_state.stop_event
            self.server._cmd_run(dict(cmd))
            with self.server._state_lock:
                racing_state = agent_state.find_by_tab(tab_id)
                assert racing_state is not None
                queued = list(racing_state.pending_user_messages)
            assert not self._frames("clear"), (
                "BUG: a second concurrent submit passed the busy guard "
                "during the start window and started a task of its own"
            )
            assert racing_state is first_state
            assert racing_state.task_thread is first_thread
            assert racing_state.stop_event is first_stop
            assert queued == ["bughunt3 race task"], (
                "the racing submit must be queued as steering, not dropped"
            )
        t1.join(timeout=30)
        self._wait_until_idle(tab_id)

    def _replay_into_start_window(self, hold: CodeType) -> tuple[str, str, str]:
        """Replay another chat's task into a tab whose run is held at *hold*.

        Returns:
            The tab, the run's chat and the replayed chat.
        """
        other_task, other_chat = th._add_task("an older conversation")
        tab_id = "rebound-tab"
        starter = threading.Thread(
            target=self.server._cmd_run, args=(self._run_cmd(tab_id),), daemon=True,
        )
        with Pause(hold, bool) as gate:
            starter.start()
            assert gate.paused.wait(timeout=30)
            run_state = agent_state.find_by_tab(tab_id)
            assert run_state is not None
            self.server._replay_session(other_chat, tab_id, task_id=str(other_task))
        starter.join(timeout=30)
        self._wait_until_idle(tab_id)
        return tab_id, run_state.chat_id, other_chat

    def test_replay_published_after_the_run_keeps_the_tab(self) -> None:
        """The run's startup frames do not wipe the newer replay's transcript."""
        tab_id, _run_chat, other_chat = self._replay_into_start_window(
            AgentServer._publish_tabs_state.__code__,
        )
        assert not self._frames("clear")
        assert len(self._frames("setTaskText")) == 1, "only the submit's own echo"
        assert self.server._tab_chat_views[tab_id] == other_chat
        assert (tab_id, other_chat) in [
            (tab, chat) for tab, chat, _task in self.server.tab_registry.bound_tabs()
        ]

    def test_run_published_after_a_replay_takes_the_tab_back(self) -> None:
        """The registry's last word is the run's: so are the view, the frames and the context."""
        tab_id, run_chat, _other_chat = self._replay_into_start_window(
            AgentServer._registry_update_tab.__code__,
        )
        assert [e.get("chat_id") for e in self._frames("clear")] == [run_chat]
        assert self.server._tab_chat_views[tab_id] == run_chat
        assert self.run_inputs
        assert not [text for text in self.run_inputs if "an older conversation" in text], (
            "the run was seeded with the task the superseded replay pinned"
        )

    def test_run_published_after_a_replay_not_committed_yet_keeps_its_context(self) -> None:
        """A replay that pinned its task and published, its view still uncommitted, loses too."""
        other_task, other_chat = th._add_task("an older conversation")
        tab_id = "staged-tab"
        starter = threading.Thread(
            target=self.server._cmd_run, args=(self._run_cmd(tab_id),), daemon=True,
        )
        replayer = threading.Thread(
            target=self.server._replay_session,
            args=(other_chat, tab_id),
            kwargs={"task_id": str(other_task)},
            daemon=True,
        )
        run_gate = Pause(AgentServer._registry_update_tab.__code__, bool)
        commit_gate = Pause(AgentServer._commit_replay_publication.__code__, bool)
        with Pauses(run_gate, commit_gate):
            starter.start()
            assert run_gate.paused.wait(timeout=30)
            run_state = agent_state.find_by_tab(tab_id)
            assert run_state is not None
            replayer.start()
            assert commit_gate.paused.wait(timeout=30), "the replay never published"
            run_gate.release()
            starter.join(timeout=30)
            self._wait_until_idle(tab_id)
        replayer.join(timeout=30)
        assert not replayer.is_alive()
        assert self.server._tab_chat_views[tab_id] == run_state.chat_id
        assert self.run_inputs
        assert not [text for text in self.run_inputs if "an older conversation" in text], (
            "the run was seeded with the task the superseded replay had staged"
        )

    def _wait_until_idle(self, tab_id: str) -> None:
        """Wait for the tab's run, and the re-run of a prompt it left queued, to end."""
        # The stubbed first run never drains the queued prompt, so the
        # runner re-submits it as the tab's next run once the first
        # worker has torn down; wait for that sequential second run.
        deadline = time.time() + 30
        while time.time() < deadline:
            with self.server._state_lock:
                st = agent_state.find_by_tab(tab_id)
                thread = st.task_thread if st is not None else None
            if thread is not None:
                thread.join(timeout=max(0.0, deadline - time.time()))
                continue
            time.sleep(0.2)
            with self.server._state_lock:
                st = agent_state.find_by_tab(tab_id)
                if st is None or st.task_thread is None:
                    break


if __name__ == "__main__":
    unittest.main()
