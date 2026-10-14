# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: a ``/ask`` whose worker thread cannot start still answers.

``_dispatch_ask_side_channel`` runs the answering session on a daemon
thread.  ``Thread.start()`` raises ``RuntimeError: can't start new
thread`` under thread exhaustion; before the fix the exception escaped
``_cmd_append_user_message`` AFTER the ``/ask`` line had been echoed
into the transcript, so the tab showed the question and never an
answer.  Now the failure is reported as a failed ``ask_answer`` panel
that travels the same path as a real answer: it is sent live to the
viewer, recorded into the running task's in-memory recording (so a
``resumeSession`` replay shows it) and persisted into the task's
``events`` rows.

The whole scenario runs in a child Python process, from ``_child_main``
below, against the real delivery path: a real :class:`RemoteAccessServer`
on a loopback endpoint, a real webview-like local client, a real
``run`` command that starts a real ``ChatSorcarAgent`` task, and the
``/ask`` line typed through the real ``appendUserMessage`` command.
The only replaced piece is the model boundary (``SorcarAgent``'s base
class ``run``, the LLM loop), which blocks until the test releases it
so the task stays live while ``/ask`` is typed — the same stub the
other daemon-level ``/ask`` suites use (``test_ask_subagent_tab_closes``);
it is not part of the code under test.

Thread exhaustion is reproduced for real: right before the ``/ask``
the child lowers its own soft ``RLIMIT_NPROC`` to 1 (Linux counts
threads against that limit, so every later ``clone`` fails with
``EAGAIN``).  Threads that already exist keep running, so the daemon's
loop thread, the task thread, the persistence writer and the
executor threads warmed up beforehand all stay available; the only
thread the ``/ask`` needs and cannot get is its own worker.  The limit
is per-process and never touches the test runner, and it is restored
once the failed answer has arrived.  Root bypasses ``RLIMIT_NPROC``,
so the test is skipped there.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import vscode_config
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import is_root, posix_only
from kiss.tests.local_ws import make_test_tls, open_local_connection
from kiss.tests.server.conftest import init_repo

# ``resource`` is imported inside the child-process body: a module-level
# import fails collection on Windows before the skip mark applies.  Only
# Linux counts threads against RLIMIT_NPROC: on macOS the worker thread
# starts regardless and the restore of the limit fails with "not allowed
# to raise maximum limit", so the scenario cannot be reproduced there.
pytestmark = [
    posix_only("RLIMIT_NPROC"),
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="only Linux counts threads against RLIMIT_NPROC",
    ),
]

TAB_ID = "webtab-audit1013-asker"
PARENT_MARKER = "long running parent task audit1013"
QUESTION = "what is the task doing audit1013?"
_RESULT_PREFIX = "RESULT "
_CHILD_ENTRY = (
    "from kiss.tests.server.test_audit1013_ask_thread_start_failure "
    "import _child_main; _child_main()"
)

# Released by the child once the failed answer has been delivered.
_PARENT_RELEASE = threading.Event()


def _blocking_model_boundary(self_agent: Any, **kwargs: Any) -> str:
    """Stand in for the LLM loop: block until released, then report success."""
    _PARENT_RELEASE.wait(timeout=60)
    self_agent.total_tokens_used = 5
    self_agent.budget_used = 0.001
    self_agent.total_steps = 1
    raw = "success: true\nis_continue: false\nsummary: done\n"
    printer = kwargs.get("printer") or getattr(self_agent, "printer", None)
    if printer is not None:
        printer.print(
            raw, type="result", step_count=1, total_tokens=5, cost="$0.0010",
        )
    return raw


async def _drain(reader: Any, received: list[dict[str, Any]]) -> None:
    """Append every JSON line the viewer receives to *received*."""
    while True:
        line = await reader.readline()
        if not line:
            return
        received.append(json.loads(line))


async def _send(writer: Any, cmd: dict[str, Any]) -> None:
    writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
    await writer.drain()


def _run_on(loop: asyncio.AbstractEventLoop, coro: Any) -> Any:
    """Run *coro* on the daemon's loop thread and return its result."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)


async def _warm_executor(loop: asyncio.AbstractEventLoop) -> None:
    """Create a few default-executor threads so later commands reuse them.

    ``RemoteAccessServer._run_cmd`` runs each client command in the
    loop's default executor, which creates its threads lazily; after
    the limit drops that creation would fail before the ``/ask``
    command ever reached the handler.
    """
    await asyncio.gather(
        *[loop.run_in_executor(None, time.sleep, 0.2) for _ in range(4)],
    )


def _wait_for(predicate: Any, what: str, timeout: float = 15.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = predicate()
        if found:
            return found
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _find(received: list[dict[str, Any]], etype: str, start: int = 0) -> Any:
    """First event of *etype* addressed to the viewer's tab, from index *start*."""
    return next(
        (
            e for e in list(received)[start:]
            if e.get("type") == etype and e.get("tabId") == TAB_ID
        ),
        None,
    )


def _child_main() -> None:
    """Run the scenario and print one ``RESULT <json>`` line for the parent."""
    tmpdir = tempfile.mkdtemp(prefix="audit1013_ask_")
    repo = str(Path(tmpdir) / "repo")
    init_repo(repo)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir()
    _persistence._KISS_DIR = kiss_dir
    _persistence._DB_PATH = kiss_dir / "history.db"
    _persistence._db_conn = None
    vscode_config.CONFIG_DIR = kiss_dir
    vscode_config.CONFIG_PATH = kiss_dir / "config.json"
    cast(Any, SorcarAgent.__mro__[1]).run = _blocking_model_boundary

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    certfile, keyfile, _ca = make_test_tls(Path(tmpdir))
    server = RemoteAccessServer(
        local_endpoint_file=Path(tmpdir) / "sorcar-local.json",
        certfile=str(certfile), keyfile=str(keyfile), work_dir=repo,
    )

    _run_on(loop, server.start_private_async())
    reader, writer = _run_on(loop, open_local_connection(server, limit=16 * 1024 * 1024))
    received: list[dict[str, Any]] = []
    asyncio.run_coroutine_threadsafe(_drain(reader, received), loop)
    _wait_for(
        lambda: server._printer._local_clients, "the viewer's registration",
    )

    _run_on(loop, _send(writer, {
        "type": "run", "tabId": TAB_ID, "prompt": PARENT_MARKER,
        "workDir": repo, "useWorktree": False, "autoCommit": False,
    }))
    settings = _wait_for(
        lambda: next(
            (
                e for e in list(received)
                if e.get("type") == "task_settings" and e.get("tabId") == TAB_ID
                and (e.get("settings") or {}).get("task_id")
            ),
            None,
        ),
        "the parent's persisted task id",
    )
    task_id = str(settings["settings"]["task_id"])
    _run_on(loop, _warm_executor(loop))

    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    before_ask = len(received)
    resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
    try:
        _run_on(loop, _send(writer, {
            "type": "appendUserMessage", "tabId": TAB_ID,
            "prompt": f"/ask {QUESTION}",
        }))
        live = _wait_for(
            lambda: _find(received, "ask_answer", before_ask),
            "the live ask_answer",
        )
    finally:
        resource.setrlimit(resource.RLIMIT_NPROC, (soft, hard))
    live_index = received.index(live)
    echo = _find(received, "prompt", before_ask)

    # Replay of the still-running task: what a webview reload does.
    before_replay = len(received)
    _run_on(loop, _send(writer, {
        "type": "resumeSession", "taskId": task_id, "tabId": TAB_ID,
    }))
    replay = _wait_for(
        lambda: _find(received, "task_events", before_replay),
        "the task_events replay",
    )
    _persistence._flush_chat_events(task_id)
    stored = _persistence._load_chat_events_by_task_id(task_id) or {}

    _PARENT_RELEASE.set()
    _wait_for(lambda: _find(received, "result"), "the parent's result")
    _run_on(loop, server.stop_async())
    _persistence._close_db()
    print(_RESULT_PREFIX + json.dumps({
        "task_id": task_id,
        "echo_index": received.index(echo) if echo else -1,
        "live_index": live_index,
        "live": live,
        "replayed": replay.get("events"),
        "stored": stored.get("events"),
    }, default=str))


def _check_answer(answer: dict[str, Any], task_id: str, live: bool) -> None:
    """Assert *answer* is the failed panel; only the *live* copy is tab-stamped.

    The recorded and persisted copies drop ``tabId`` so a replay can
    address them to whichever tab is viewing the task.
    """
    assert answer["success"] is False
    assert answer["question"] == QUESTION
    assert "could not be started" in answer["text"]
    assert "can't start new thread" in answer["text"]
    assert answer.get("tabId") == (TAB_ID if live else None)
    assert str(answer["taskId"]) == task_id


@pytest.mark.skipif(is_root(), reason="root is exempt from RLIMIT_NPROC")
def test_ask_reports_failed_answer_when_worker_thread_cannot_start() -> None:
    """Thread exhaustion yields a failed ``ask_answer`` live, on replay and on disk."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_ENTRY], capture_output=True, text=True,
        timeout=120, check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_RESULT_PREFIX)]
    assert lines, f"child produced no result\nstdout={proc.stdout}\nstderr={proc.stderr}"
    result = json.loads(lines[-1][len(_RESULT_PREFIX):])
    task_id = result["task_id"]

    # Live: the echoed ``/ask`` line precedes exactly one failed answer.
    _check_answer(result["live"], task_id, live=True)
    assert 0 <= result["echo_index"] < result["live_index"]

    # Replay of the running task carries the echo and the answer, in order.
    replayed = [e for e in result["replayed"] if e.get("type") in ("prompt", "ask_answer")]
    assert [e["type"] for e in replayed][-2:] == ["prompt", "ask_answer"], replayed
    _check_answer(replayed[-1], task_id, live=False)

    # Persisted under the task's events rows.
    stored = [e for e in result["stored"] if e.get("type") == "ask_answer"]
    assert len(stored) == 1, result["stored"]
    _check_answer(stored[0], task_id, live=False)
