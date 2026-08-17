# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The model picker shows the user's pick, except while an agent overrides it.

The picker is per-tab: it shows the model that tab's user last picked,
and that is the model the tab's next task runs with.  A running agent
may call ``set_model`` on itself, and then the tabs watching that task
show what it is actually running — but only until the task ends, and
without ever overwriting the user's own choice underneath.

One transient event carries this, ``modelPick``, whose ``source`` says
what the model means: ``"agent"`` while the agent has the picker, and
``"restore"`` when it gives it back.

These tests drive the real daemon: a real :class:`VSCodeServer`, the
real task pipeline with a real agent, and the real ``JsonPrinter``
broadcast routing (via :class:`MemoryPrinter`, which mirrors it).
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vscode_config
from kiss.agents.sorcar.persistence import _load_last_model
from kiss.agents.sorcar.sorcar_agent import _broadcast_subagent_done
from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.core.models.model_info import get_available_models
from kiss.core.models.model_info import model as model_factory
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.tests.server._memory_printer import MemoryPrinter


def _model_picks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the ``modelPick`` events, in emission order."""
    return [e for e in events if e.get("type") == "modelPick"]


def _init_git_repo(tmpdir: str) -> None:
    """Create a git repo with one commit at *tmpdir*."""
    subprocess.run(["git", "init", tmpdir], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=tmpdir, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True,
    )
    Path(tmpdir, ".gitkeep").touch()
    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True,
    )


class _SwitchingAgent(WorktreeSorcarAgent):
    """Real agent whose task body is one real ``set_model`` call.

    Only ``perform_task`` — the agentic loop — is replaced, so
    ``run`` still does everything the daemon relies on: it allocates
    the ``task_history`` row, binds the printer, and sets the thread-
    local task id.  The task then ends by raising ``KeyboardInterrupt``,
    driving the pipeline's normal stop/cleanup path, which is where the
    picker has to be handed back to the user.
    """

    def __init__(self, target_model: str) -> None:
        super().__init__("model picker override agent")
        self._target_model = target_model
        self.started = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        self.switched = threading.Event()
        self.switch_result = ""

    def perform_task(
        self, tools: list, attachments: list | None = None,
    ) -> str:
        """Switch this agent's model, then end the task."""
        self.started.set()
        self.proceed.wait(timeout=60)
        self._use_web_tools = False
        # The real loop builds the live model here; do the same so
        # `set_model` runs its actual model-swap path rather than the
        # "no live model yet" shortcut.
        self.model = model_factory(self.model_name)
        for tool in self._get_tools():
            if getattr(tool, "__name__", "") == "set_model":
                self.switch_result = tool(self._target_model)
                break
        self.switched.set()
        raise KeyboardInterrupt("stopped by test")


class _QuietAgent(WorktreeSorcarAgent):
    """Real agent that runs a task without ever touching the picker."""

    def __init__(self) -> None:
        super().__init__("quiet agent")

    def perform_task(
        self, tools: list, attachments: list | None = None,
    ) -> str:
        """End the task without switching models."""
        raise KeyboardInterrupt("stopped by test")


class TestAgentModelPickFanout(unittest.TestCase):
    """``broadcast_agent_model_pick`` reaches every tab watching the task."""

    def _emit(self, printer: MemoryPrinter, task_id: str, tab_id: str,
              model: str) -> None:
        """Broadcast an agent pick from a worker thread bound to *task_id*."""

        def runner() -> None:
            printer._thread_local.task_id = task_id
            printer.broadcast_agent_model_pick(model, tab_id)

        worker = threading.Thread(target=runner)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "broadcast worker hung")

    def test_launcher_and_viewers_each_get_one_copy(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "launcher")
        printer.subscribe_tab("TASK1", "viewer")

        self._emit(printer, "TASK1", "launcher", "gpt-5.6-sol")

        picks = _model_picks(printer.emitted)
        self.assertEqual(
            sorted(p["tabId"] for p in picks),
            ["launcher", "viewer"],
            "every tab watching the task must show the agent's model, "
            f"exactly once each; got {picks}",
        )
        for pick in picks:
            self.assertEqual(pick["model"], "gpt-5.6-sol")
            self.assertEqual(pick["source"], "agent")

    def test_launcher_gets_it_without_any_subscriber(self) -> None:
        """A task nobody else is watching still updates its own tab."""
        printer = MemoryPrinter()

        self._emit(printer, "TASK1", "launcher", "gpt-5.6-sol")

        self.assertEqual(
            [p["tabId"] for p in _model_picks(printer.emitted)], ["launcher"],
        )

    def test_a_tab_joining_late_is_told_the_agent_model(self) -> None:
        """A window that opens the running task after the switch would
        otherwise sit on the wrong label until the task ended."""
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "launcher")
        self._emit(printer, "TASK1", "launcher", "gpt-5.6-sol")
        printer.emitted.clear()

        printer.subscribe_tab("TASK1", "latecomer")

        picks = _model_picks(printer.emitted)
        self.assertEqual(
            [(p["tabId"], p["source"], p["model"]) for p in picks],
            [("latecomer", "agent", "gpt-5.6-sol")],
            f"the joining tab was not caught up; got {picks}",
        )

    def test_a_tab_joining_a_quiet_task_is_told_nothing(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "launcher")

        printer.subscribe_tab("TASK1", "latecomer")

        self.assertEqual(_model_picks(printer.emitted), [])

    def test_a_caught_up_tab_gets_its_picker_back(self) -> None:
        printer = MemoryPrinter()
        self._emit(printer, "TASK1", "launcher", "gpt-5.6-sol")
        printer.subscribe_tab("TASK1", "latecomer")
        printer.emitted.clear()

        printer.restore_model_pick("claude-opus-5", "latecomer")

        picks = _model_picks(printer.emitted)
        self.assertEqual([p["source"] for p in picks], ["restore"])

    def test_a_finished_task_stops_catching_tabs_up(self) -> None:
        """Once the task is gone its override must not be handed to a
        tab that opens the conversation from history."""
        printer = MemoryPrinter()
        self._emit(printer, "TASK1", "launcher", "gpt-5.6-sol")
        printer.cleanup_task("TASK1", subscriber_linger_seconds=0)
        printer.emitted.clear()

        printer.subscribe_tab("TASK1", "latecomer")

        self.assertEqual(_model_picks(printer.emitted), [])

    def test_a_closed_tab_stops_gating_the_restore(self) -> None:
        printer = MemoryPrinter()
        self._emit(printer, "TASK1", "launcher", "gpt-5.6-sol")
        printer.cleanup_tab("launcher")
        printer.emitted.clear()

        printer.restore_model_pick("claude-opus-5", "launcher")

        self.assertEqual(_model_picks(printer.emitted), [])

    def test_override_is_never_recorded_into_the_task(self) -> None:
        """The override is a live view, not history.

        Recording it would make a replayed conversation re-apply a
        picker label months after the agent that chose it exited.
        """
        printer = MemoryPrinter()
        printer._recordings["TASK1"] = []
        printer.subscribe_tab("TASK1", "launcher")

        self._emit(printer, "TASK1", "launcher", "gpt-5.6-sol")

        self.assertEqual(
            printer._recordings["TASK1"],
            [],
            "modelPick must stay out of the task's event log",
        )

    def test_a_daemon_with_no_model_at_all_says_nothing(self) -> None:
        """Nothing sensible to show beats showing an empty picker."""
        printer = MemoryPrinter()
        self._emit(printer, "TASK1", "launcher", "gpt-5.6-sol")
        printer.emitted.clear()

        printer.restore_model_pick("", "launcher")

        self.assertEqual(_model_picks(printer.emitted), [])

    def test_empty_model_emits_nothing(self) -> None:
        printer = MemoryPrinter()
        printer.subscribe_tab("TASK1", "launcher")

        self._emit(printer, "TASK1", "launcher", "")

        self.assertEqual(_model_picks(printer.emitted), [])


class TestSubagentPickerHandback(unittest.TestCase):
    """A finished sub-agent hands its tab's picker back too."""

    def test_switching_subagent_tab_is_restored(self) -> None:
        printer = MemoryPrinter()

        def switch() -> None:
            printer._thread_local.task_id = "SUBTASK"
            printer.broadcast_agent_model_pick("gpt-5.6-sol", "tab__sub_0")

        worker = threading.Thread(target=switch)
        worker.start()
        worker.join(timeout=5)
        printer.emitted.clear()

        _broadcast_subagent_done(printer, ["tab__sub_0"], "claude-opus-5")

        picks = _model_picks(printer.emitted)
        self.assertEqual(
            [(p["tabId"], p["source"], p["model"]) for p in picks],
            [("tab__sub_0", "restore", "claude-opus-5")],
            f"the sub-agent tab was left on the agent's model; got {picks}",
        )
        self.assertEqual(
            printer._model_override_tabs,
            set(),
            "the finished sub-agent tab must not stay in the override set",
        )

    def test_quiet_subagent_tab_costs_nothing(self) -> None:
        printer = MemoryPrinter()

        _broadcast_subagent_done(printer, ["tab__sub_0"], "claude-opus-5")

        self.assertEqual(_model_picks(printer.emitted), [])


class _ServerTestCase(unittest.TestCase):
    """Base: a real ``VSCodeServer`` on an isolated DB / config / repo."""

    def setUp(self) -> None:
        models = get_available_models()
        if not models:
            self.skipTest("no model API key configured")
        self.model = models[0]
        # The agent switches to a genuinely different, genuinely
        # available model, so `set_model` runs its real swap path.
        self.agent_target = models[1] if len(models) > 1 else ""
        agent_state.agent_states.clear()
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-modelpick-")
        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved_db = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR = kiss_dir
        th._DB_PATH = kiss_dir / "sorcar.db"
        th._db_conn = None
        self._saved_config = (
            vars(vscode_config).get("CONFIG_DIR"),
            vars(vscode_config).get("CONFIG_PATH"),
        )
        vscode_config.CONFIG_DIR = kiss_dir
        vscode_config.CONFIG_PATH = kiss_dir / "config.json"
        _init_git_repo(self.tmpdir)
        self.server = VSCodeServer()
        self.events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()
        real_broadcast = self.server.printer.broadcast

        def capture(event: dict[str, Any]) -> None:
            with self._events_lock:
                self.events.append(dict(event))
            real_broadcast(event)

        self.server.printer.broadcast = capture  # type: ignore[assignment]

    def tearDown(self) -> None:
        agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
        (th._DB_PATH, th._db_conn, th._KISS_DIR) = self._saved_db
        saved_dir, saved_path = self._saved_config
        # The module resolves these lazily when unset, so an absent
        # original must be deleted rather than pinned to a stale value.
        if saved_dir is None:
            del vscode_config.CONFIG_DIR
        else:
            vscode_config.CONFIG_DIR = saved_dir
        if saved_path is None:
            del vscode_config.CONFIG_PATH
        else:
            vscode_config.CONFIG_PATH = saved_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def picks(self) -> list[dict[str, Any]]:
        """Snapshot the captured ``modelPick`` events."""
        with self._events_lock:
            return _model_picks(self.events)


class TestAgentOverrideAndRestore(_ServerTestCase):
    """An agent's switch shows up live, and is undone when it finishes."""

    def _user_picks(self, tab_id: str, model: str) -> None:
        """Pick *model* in *tab_id* exactly as clicking the picker does."""
        self.server._cmd_select_model(
            {"type": "selectModel", "model": model, "tabId": tab_id},
        )

    def _start_task(
        self, tab_id: str, agent: WorktreeSorcarAgent,
    ) -> threading.Thread:
        """Start a real task for *agent* in *tab_id* and return its worker."""
        state = agent_state.AgentState(
            f"pretask-{tab_id}",
            agent=agent,
            tab_id=tab_id,
            server_owned=True,
            stop_event=threading.Event(),
        )
        state.user_answer_queue = queue.Queue(maxsize=1)
        agent_state.register(state)
        worker = threading.Thread(
            target=self.server._run_task,
            args=({
                "type": "run",
                "prompt": "switch your model",
                "model": self.server._tab_model(tab_id),
                "workDir": self.tmpdir,
                "tabId": tab_id,
            },),
            daemon=True,
        )
        state.task_thread = worker
        worker.start()
        return worker

    def _run_switching_task(self, tab_id: str, target: str) -> None:
        """Run a real task whose agent switches to *target*, then stops."""
        agent = _SwitchingAgent(target)
        worker = self._start_task(tab_id, agent)
        worker.join(timeout=120)
        self.assertFalse(worker.is_alive(), "task worker never finished")
        self.assertTrue(agent.switched.is_set(), "agent never switched model")
        self.assertIn(
            "Model changed from",
            agent.switch_result,
            f"set_model did not perform a live swap: {agent.switch_result!r}",
        )

    def test_override_then_restore_in_the_running_tab(self) -> None:
        if not self.agent_target:
            self.skipTest("needs two available models to switch between")
        self._user_picks("tab-run", self.model)

        self._run_switching_task("tab-run", self.agent_target)

        picks = [p for p in self.picks() if p["tabId"] == "tab-run"]
        self.assertEqual(
            [p["source"] for p in picks],
            ["agent", "restore"],
            "the running tab must show the agent's model while the task "
            f"runs and the user's pick after it ends; got {picks}",
        )
        self.assertEqual(picks[0]["model"], self.agent_target)
        self.assertEqual(
            picks[1]["model"],
            self.model,
            "the restored model must be the one picked in THIS tab",
        )

    def test_restore_reaches_a_viewer_watching_the_same_task(self) -> None:
        """A second window watching the task must not be left behind on
        the agent's model after the task ends."""
        if not self.agent_target:
            self.skipTest("needs two available models to switch between")
        self._user_picks("tab-run", self.model)
        self._user_picks("tab-viewer", self.model)
        agent = _SwitchingAgent(self.agent_target)
        agent.proceed.clear()
        worker = self._start_task("tab-run", agent)
        self.assertTrue(agent.started.wait(timeout=60), "task never started")

        # A second window joins the live task, exactly as a
        # history-resume click does.
        task_id = getattr(agent, "_last_task_id", None)
        self.assertTrue(task_id, "task id was never allocated")
        self.server.printer.subscribe_tab(task_id, "tab-viewer")
        agent.proceed.set()

        worker.join(timeout=120)
        self.assertFalse(worker.is_alive(), "task worker never finished")

        viewer_picks = [p for p in self.picks() if p["tabId"] == "tab-viewer"]
        self.assertEqual(
            [p["source"] for p in viewer_picks],
            ["agent", "restore"],
            "the viewer must see the agent's model while the task runs "
            f"and its own pick after it ends; got {viewer_picks}",
        )
        self.assertEqual(viewer_picks[0]["model"], self.agent_target)
        self.assertEqual(viewer_picks[-1]["model"], self.model)

    def test_agent_switch_does_not_touch_the_user_preference(self) -> None:
        """The whole point: the agent borrows the picker, it does not
        take it."""
        if not self.agent_target:
            self.skipTest("needs two available models to switch between")
        self._user_picks("tab-run", self.model)

        self._run_switching_task("tab-run", self.agent_target)

        self.assertEqual(_load_last_model(), self.model)
        self.assertEqual(
            self.server._tab_model("tab-run"),
            self.model,
            "the tab must still be set to run the user's model next time",
        )

    def test_an_agent_that_never_switches_costs_nothing(self) -> None:
        """A task that leaves the picker alone must not put a single
        picker event on the wire."""
        self._user_picks("tab-run", self.model)

        worker = self._start_task("tab-run", _QuietAgent())
        worker.join(timeout=120)
        self.assertFalse(worker.is_alive(), "task worker never finished")

        self.assertEqual(self.picks(), [])

    def test_a_tab_the_daemon_no_longer_knows_still_gets_its_picker_back(
        self,
    ) -> None:
        """A viewer whose tab state is already gone must still be handed
        back a model rather than left on the agent's."""
        self.server.printer._model_override_tabs.add("tab-gone")

        self.server._restore_user_model_pick("tab-gone")

        picks = self.picks()
        self.assertEqual([p["tabId"] for p in picks], ["tab-gone"])
        self.assertEqual(picks[0]["source"], "restore")
        self.assertTrue(picks[0]["model"])

    def test_another_tab_picker_is_left_alone(self) -> None:
        """The picker is per-tab: a task in one tab must not repaint
        another tab's picker."""
        if not self.agent_target:
            self.skipTest("needs two available models to switch between")
        self._user_picks("tab-run", self.model)
        self._user_picks("tab-other", self.model)

        self._run_switching_task("tab-run", self.agent_target)

        self.assertEqual(
            [p for p in self.picks() if p["tabId"] == "tab-other"], [],
        )


if __name__ == "__main__":
    unittest.main()
