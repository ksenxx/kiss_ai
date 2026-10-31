# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``ChatSorcarAgent.run``'s setup window and printer.

Two defects of ``ChatSorcarAgent.run``:

* A run stopped (or failing) after its row was allocated but before
  ``super().run`` was entered never banked the pre-run classifier's
  spend: only ``SorcarAgent.run``'s cleanup folded it, so the row and
  the agent's totals missed a model call that was paid for, and the
  next run's classification reset dropped the record for good.
* With ``printer=None`` the run's setup and cleanup hooks used the
  agent's PREVIOUS printer (``self.printer`` is whatever the last run
  bound, while ``super().run`` binds the printer from the argument), so
  a second ``verbose=True`` run on one agent skipped persisting its
  ``task_settings`` event, and a run after one with a server printer
  broadcast its events through that old printer.

A last test checks that the local and UTC halves of the "Starting time"
task setting describe the same instant.

The runs are real: every model call (classifier and agent) goes to a
local OpenAI-compatible stand-in server, and the stop is the server's
own ``KeyboardInterrupt`` injection landing in the setup window through
the ``_on_task_id_allocated`` hook the task runner uses.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import pytest

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _flush_chat_events,
    _get_db,
    _load_chat_events_by_task_id,
)
from kiss.agents.sorcar.task_classifier import clear_classification_cache
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    CapturePrinter,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
)

_CLASSIFIER_SWITCH = "KISS_DISABLE_TASK_CLASSIFIER"


@pytest.fixture
def home() -> Iterator[IsolatedKissHome]:
    """An isolated KISS home (database, config) with a scratch repo."""
    isolated = IsolatedKissHome("kiss-chat-setup-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


@pytest.fixture
def classifier_on(home: IsolatedKissHome) -> Iterator[None]:
    """Lift the suite-wide classifier kill switch; use the LLM route."""
    saved = os.environ.get(_CLASSIFIER_SWITCH)
    os.environ[_CLASSIFIER_SWITCH] = "0"
    home.write_config(classify_with_decisions=False)
    clear_classification_cache()
    try:
        yield
    finally:
        clear_classification_cache()
        if saved is None:
            os.environ.pop(_CLASSIFIER_SWITCH, None)
        else:
            os.environ[_CLASSIFIER_SWITCH] = saved


def _verdict(requests: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
    """Record *request* and answer it with a 52-token "not simple" verdict."""
    requests.append(request)
    return {
        "id": "chatcmpl-sweep1030",
        "object": "chat.completion",
        "created": 0,
        "model": STANDIN_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": json.dumps({"is_simple": False})},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
    }


def _finish(request: dict[str, Any]) -> dict[str, Any]:
    """Answer every agent model call with an immediate ``finish``."""
    return finish_response("done")


def _stop_this_thread(task_id: str, chat_id: str) -> None:
    """Inject the server's Stop into the calling (task) thread."""
    inject_keyboard_interrupt(threading.get_ident())


def _run_until_stopped(
    agent: ChatSorcarAgent,
    work_dir: str,
    model_config: dict[str, Any],
    raised: list[BaseException],
) -> None:
    """Run *agent* with a Stop injected at row allocation; keep what it raised."""
    try:
        agent.run(
            prompt_template="refactor the parser module and update its tests",
            model_name=STANDIN_MODEL,
            model_config=model_config,
            work_dir=work_dir,
            web_tools=False,
            use_memory=False,
            max_budget=1.0,
            _on_task_id_allocated=_stop_this_thread,
        )
    except BaseException as e:  # noqa: BLE001 – the outcome is under test
        raised.append(e)


def _event_types(task_id: str) -> list[str]:
    """The persisted event types of task *task_id*, in order."""
    _flush_chat_events()
    session = _load_chat_events_by_task_id(task_id)
    assert session is not None, task_id
    events = session.get("events")
    assert isinstance(events, list), session
    return [str(e.get("type")) for e in events]


def _run_headless(agent: ChatSorcarAgent, prompt: str, home: IsolatedKissHome,
                  server: StandInModelServer, **kwargs: Any) -> str:
    """Run *agent* on *prompt* against *server*; return the run's task id."""
    agent.run(
        prompt_template=prompt,
        model_name=STANDIN_MODEL,
        model_config=server.model_config,
        work_dir=str(home.repo),
        web_tools=False,
        use_memory=False,
        max_budget=1.0,
        **kwargs,
    )
    return agent.last_task_id


@pytest.mark.usefixtures("classifier_on")
def test_stop_during_setup_still_banks_classifier_spend(home: IsolatedKissHome) -> None:
    """The classifier's paid call lands on the row although ``super().run`` never ran."""
    requests: list[dict[str, Any]] = []
    server = StandInModelServer(functools.partial(_verdict, requests))
    agent = ChatSorcarAgent("setup-stop")
    raised: list[BaseException] = []
    # A task thread, as in the daemon: the injected Stop never reaches pytest.
    worker = threading.Thread(
        target=_run_until_stopped,
        args=(agent, str(home.repo), server.model_config, raised),
    )
    try:
        worker.start()
        worker.join(timeout=120)
    finally:
        server.stop()
    assert not worker.is_alive()
    assert len(raised) == 1 and isinstance(raised[0], KeyboardInterrupt), raised
    # Only the classifier reached the model: the Stop landed in setup.
    assert len(requests) == 1
    budget, tokens, steps = agent.usage_snapshot()
    assert (tokens, steps) == (52, 1)
    assert budget > 0
    assert agent._classifier_spend is None
    row = _get_db().execute(
        "SELECT cost, tokens, steps, result FROM task_history WHERE id = ?",
        (agent.last_task_id,),
    ).fetchone()
    assert tuple(row) == (pytest.approx(budget), 52, 1, "Task interrupted")


def test_second_verbose_run_persists_its_task_settings(home: IsolatedKissHome) -> None:
    """``verbose=True`` leaves a console printer on the agent; it is not the next run's."""
    server = StandInModelServer(_finish)
    agent = ChatSorcarAgent("verbose-rerun")
    try:
        task_ids = [
            _run_headless(agent, prompt, home, server, verbose=True)
            for prompt in ("first console task", "second console task")
        ]
    finally:
        server.stop()
    for task_id in task_ids:
        assert _event_types(task_id).count("task_settings") == 1, task_id


def test_run_without_printer_leaves_the_previous_printer_alone(
    home: IsolatedKissHome,
) -> None:
    """A run after one with a server printer emits nothing through that printer."""
    server = StandInModelServer(_finish)
    agent = ChatSorcarAgent("printer-then-none")
    previous = CapturePrinter()
    try:
        _run_headless(agent, "task with a printer", home, server, printer=previous)
        seen = len(previous.captured)
        second = _run_headless(agent, "task without a printer", home, server)
    finally:
        server.stop()
    assert len(previous.captured) == seen, previous.captured[seen:]
    assert _event_types(second).count("task_settings") == 1


def test_starting_time_halves_name_the_same_instant() -> None:
    """The local and UTC halves of "Starting time" come from one clock read.

    The child runs in the POSIX fixed-offset zone ``IST-5:30`` (UTC+05:30),
    which glibc, macOS and the Windows CRT read without a zoneinfo
    database, and reports the offset its clock applies.
    """
    code = (
        "from datetime import datetime\n"
        "from kiss.agents.sorcar.relentless_agent import RelentlessAgent\n"
        "agent = RelentlessAgent('tz')\n"
        "agent.max_budget = 1.0\n"
        "print(agent._system_prompt_task_settings()['Starting time'])\n"
        "print(datetime.now().astimezone().utcoffset())\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TZ": "IST-5:30"},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    setting, offset = done.stdout.strip().splitlines()[-2:]
    assert offset == "5:30:00", f"the child's clock ignored TZ=IST-5:30: {offset}"
    local, utc = setting.removesuffix(" UTC)").split(" (")
    assert local.endswith(" IST"), local
    fmt = "%Y-%m-%d %H:%M:%S"
    local_time = datetime.strptime(local.removesuffix(" IST"), fmt)
    assert local_time - datetime.strptime(utc, fmt) == timedelta(hours=5, minutes=30)
