# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Live end-to-end test: mid-task ``set_model`` must not leak globally.

Reproduces the chat-webview scenario with REAL LLM calls (no mocks):

1. The user picks model A in the model picker (``_cmd_select_model`` —
   this is the only action allowed to persist the global ``last_model``
   preference in ``~/.kiss/config.json``).
2. Task 1 runs on the daemon's real task-runner path
   (``_TaskRunnerMixin._run_task``) with model A; the prompt instructs
   the agent to call the ``set_model`` tool to switch itself to model B
   mid-task, then finish.
3. After task 1 the global model preference — the persisted
   ``last_model``, the daemon-wide ``_default_model`` (what the picker
   shows), and the tab's ``selected_model`` — must all still be model A.
4. Task 2 is submitted the way the chat webview does it (the frontend
   sends the picker's current model, which mirrors the daemon's
   ``_default_model``); the agent that executes task 2 must run on
   model A, not on the model B that task 1's agent switched itself to.

The agent under test is the production ``WorktreeSorcarAgent`` that
``_get_tab`` allocates; both tasks perform real LLM calls against real
provider endpoints (Anthropic for model A, Gemini for model B).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.vscode.vscode_config as vscode_config
from kiss.agents.sorcar import persistence as sorcar_persistence
from kiss.agents.sorcar.persistence import _load_last_model
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.vscode.server import VSCodeServer
from kiss.core.models.model_info import get_available_models
from kiss.tests.conftest import (
    requires_anthropic_api_key,
    requires_gemini_api_key,
)

MODEL_A = "claude-haiku-4-5"
MODEL_B = "gemini-2.5-flash"

TAB_ID = "tab-set-model-live"

TASK_1_PROMPT = (
    "Call the set_model tool exactly once with "
    f"model_name='{MODEL_B}'. After set_model returns, immediately call "
    "finish with success='true' and summary='model switched'. Do not "
    "use any other tool and do not do anything else."
)

TASK_2_PROMPT = (
    "Immediately call finish with success='true' and summary='ok'. "
    "Do not use any other tool and do not do anything else."
)


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path) -> Any:
    """Redirect VS Code config writes to a per-test temp directory."""
    orig_dir = vscode_config.CONFIG_DIR
    orig_path = vscode_config.CONFIG_PATH
    test_dir = tmp_path / ".kiss"
    test_dir.mkdir()
    vscode_config.CONFIG_DIR = test_dir
    vscode_config.CONFIG_PATH = test_dir / "config.json"
    yield
    vscode_config.CONFIG_DIR = orig_dir
    vscode_config.CONFIG_PATH = orig_path


@pytest.fixture(autouse=True)
def _isolate_sorcar_db(tmp_path: Path) -> Any:
    """Redirect the Sorcar SQLite DB to a per-test temp file."""
    orig_kiss_dir = sorcar_persistence._KISS_DIR
    orig_db_path = sorcar_persistence._DB_PATH
    test_dir = tmp_path / "sorcar_db_dir"
    test_dir.mkdir()
    sorcar_persistence._KISS_DIR = test_dir
    sorcar_persistence._DB_PATH = test_dir / "sorcar.db"
    sorcar_persistence._close_db()
    yield
    sorcar_persistence._close_db()
    sorcar_persistence._KISS_DIR = orig_kiss_dir
    sorcar_persistence._DB_PATH = orig_db_path


def _make_server() -> tuple[VSCodeServer, list[dict[str, Any]]]:
    """Spin up a real :class:`VSCodeServer` whose broadcasts land in a list."""
    server = VSCodeServer()
    events: list[dict[str, Any]] = []
    lock = threading.Lock()
    original_broadcast = server.printer.broadcast

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(dict(event))
        original_broadcast(event)

    server.printer.broadcast = capture  # type: ignore[assignment]
    return server, events


class _ModelSampler:
    """Polls the live agent's ``model_name`` while a task runs.

    ``_run_task`` disposes ``tab.agent`` when the task ends, so the
    only way to observe which model the task actually ran on is to
    sample the live agent (and its inner per-session executor) while
    the run is in flight.  Real LLM round-trips take seconds, so a
    2 ms poll cannot miss the model that served them.
    """

    def __init__(self, tab: _RunningAgentState) -> None:
        """Start sampling *tab*'s live agent on a daemon thread.

        Args:
            tab: The per-tab state whose ``agent`` should be sampled.
        """
        self._tab = tab
        self.samples: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        """Record every distinct model name the live agent exposes."""
        while not self._stop.is_set():
            agent = self._tab.agent
            if agent is not None:
                name = getattr(agent, "model_name", "")
                if name:
                    self.samples.add(name)
                executor = getattr(agent, "_current_executor", None)
                if executor is not None:
                    model = getattr(executor, "model", None)
                    if model is not None:
                        self.samples.add(model.model_name)
            time.sleep(0.002)

    def stop(self) -> set[str]:
        """Stop sampling and return the collected model names."""
        self._stop.set()
        self._thread.join(timeout=5)
        return self.samples


def _persisted_task_models() -> list[str]:
    """Return the ``model`` column of every task_history row, oldest first."""
    # Use the persistence module's connection helper so a fresh
    # per-test database is migrated before the first read.
    conn = sorcar_persistence._get_db()
    rows = conn.execute(
        "SELECT model FROM task_history ORDER BY timestamp ASC, rowid ASC"
    ).fetchall()
    return [str(model or "") for (model,) in rows]


@requires_anthropic_api_key
@requires_gemini_api_key
class TestSetModelDoesNotLeakIntoNextWebviewTask:
    """Mid-task ``set_model`` must never change the picker/global model."""

    def setup_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def teardown_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_next_task_runs_on_picker_model(self, tmp_path: Any) -> None:
        """Task 2 in the same chat webview runs on the picker's model A."""
        available = get_available_models()
        assert MODEL_A in available, f"{MODEL_A} not available"
        assert MODEL_B in available, f"{MODEL_B} not available"

        server, events = _make_server()
        baseline = _persisted_task_models()

        # 1. The user picks model A in the model picker.  This is the
        #    ONLY action that persists the global last_model.
        server._cmd_select_model({"tabId": TAB_ID, "model": MODEL_A})
        assert _load_last_model() == MODEL_A
        tab = _RunningAgentState.running_agent_states[TAB_ID]
        assert tab.selected_model == MODEL_A

        # 2. Task 1: the webview submits with the picker's model (A);
        #    the agent switches itself to model B mid-task via the
        #    real set_model tool, then finishes.
        sampler1 = _ModelSampler(tab)
        server._run_task({
            "tabId": TAB_ID,
            "prompt": TASK_1_PROMPT,
            "model": MODEL_A,
            "workDir": str(tmp_path),
        })
        samples1 = sampler1.stop()

        # The switch must actually have happened (otherwise this test
        # proves nothing): the live agent ran on B at some point.
        assert MODEL_B in samples1, (
            f"set_model to {MODEL_B} never took effect in task 1; "
            f"observed models: {sorted(samples1)}; events tail: "
            f"{[e.get('type') for e in events][-20:]}"
        )

        # 3. The global model preference must be untouched by set_model.
        assert _load_last_model() == MODEL_A, (
            "set_model leaked into the persisted last_model "
            f"(global config): {_load_last_model()!r}"
        )
        assert server._default_model == MODEL_A, (
            "set_model leaked into the daemon-wide default model "
            f"(the model picker's selection): {server._default_model!r}"
        )
        assert tab.selected_model == MODEL_A, (
            "set_model leaked into the tab's selected model: "
            f"{tab.selected_model!r}"
        )

        # 4. Task 2, submitted the way the chat webview does it: the
        #    frontend's picker mirrors the daemon's default model.
        picker_model = server._default_model
        sampler2 = _ModelSampler(tab)
        server._run_task({
            "tabId": TAB_ID,
            "prompt": TASK_2_PROMPT,
            "model": picker_model,
            "workDir": str(tmp_path),
        })
        samples2 = sampler2.stop()

        assert MODEL_A in samples2, (
            f"task 2 never ran on the picker model {MODEL_A}; "
            f"observed models: {sorted(samples2)}"
        )
        assert MODEL_B not in samples2, (
            f"task 2 ran on {MODEL_B}, which task 1's set_model "
            "switched to — the mid-task model change leaked into the "
            "next task in the same chat webview"
        )

        # 5. The persisted per-task metadata must record the LAUNCH
        #    model of each task (A for both), not the mid-task switch.
        persisted = _persisted_task_models()[len(baseline):]
        assert persisted == [MODEL_A, MODEL_A], (
            f"persisted task models {persisted} != [{MODEL_A!r}, {MODEL_A!r}]"
        )


@requires_anthropic_api_key
@requires_gemini_api_key
class TestSetModelDoesNotLeakIntoPersistedTaskModel:
    """Direct ``ChatSorcarAgent.run`` (CLI / remote / sub-agent path).

    Unlike webview tasks (which pass ``_skip_persistence=True`` and let
    the daemon's task runner persist the launch-time model),
    ``ChatSorcarAgent.run`` persists the task's model itself in its
    ``finally`` block.  A mid-task ``set_model`` mutates
    ``self.model_name``, and that mutated name must NOT be recorded as
    the task's model — the recorded model is what the chat webview's
    History sidebar shows and what any global consumer reads.
    """

    def setup_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def teardown_method(self) -> None:
        _RunningAgentState.running_agent_states.clear()

    def test_persisted_model_is_launch_model(self, tmp_path: Any) -> None:
        """A run that switches to model B must still be recorded as A."""
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
        from kiss.agents.sorcar.persistence import _save_last_model

        available = get_available_models()
        assert MODEL_A in available, f"{MODEL_A} not available"
        assert MODEL_B in available, f"{MODEL_B} not available"

        # The user's global picker preference.
        _save_last_model(MODEL_A)
        baseline = _persisted_task_models()

        agent = ChatSorcarAgent("set-model-persist-live")
        result = agent.run(
            prompt_template=TASK_1_PROMPT,
            model_name=MODEL_A,
            work_dir=str(tmp_path),
            web_tools=False,
        )
        assert "success" in result

        # The switch must actually have happened for the test to prove
        # anything: the agent's live model name is B after the run.
        assert agent.model_name == MODEL_B, (
            f"set_model to {MODEL_B} never took effect; agent still on "
            f"{agent.model_name!r}"
        )

        # Global picker preference untouched.
        assert _load_last_model() == MODEL_A, (
            "set_model leaked into the persisted last_model "
            f"(global config): {_load_last_model()!r}"
        )

        # The task row must record the model the task was LAUNCHED
        # with (A), not the model the agent switched itself to (B).
        persisted = _persisted_task_models()[len(baseline):]
        assert persisted == [MODEL_A], (
            f"persisted task model {persisted} != [{MODEL_A!r}] — the "
            "mid-task set_model leaked into the task's recorded model"
        )
