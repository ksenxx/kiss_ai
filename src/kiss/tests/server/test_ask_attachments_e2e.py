# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""E2E: a file attached to a ``/ask`` message reaches the answering session.

A running task's tab sends ``appendUserMessage`` with ``/ask <question>``
and an attachment.  The side channel dispatches the ``ask_sea`` run
through the daemon with the message's files, so the answering agent's
first model request carries the image as a content part.

A real local daemon serves the parent task and the answering session;
only the model endpoint is a scripted local stand-in server.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.sorcar_agent import _persisted_task_id
from kiss.server import agent_state, sorcar
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
    tool_call_response,
)

PNG = bytes([137, 80, 78, 71, 13, 10, 26, 10]) + b"\x00" * 8
PNG_B64 = base64.b64encode(PNG).decode("ascii")


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    home = IsolatedKissHome(prefix="kiss-ask-attachments-")
    home.write_config(classify_tasks=False)
    try:
        yield home
    finally:
        home.cleanup()


@pytest.fixture
def daemon(
    env: IsolatedKissHome, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[RemoteAccessServer]:
    """A real local daemon; the side channel finds it through ``KISS_SORCAR_LOCAL``."""
    endpoint_file = str(env.tmpdir / "sorcar-local.json")
    monkeypatch.setenv("KISS_SORCAR_LOCAL", endpoint_file)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server = RemoteAccessServer(local_endpoint_file=endpoint_file, work_dir=str(env.repo))
    asyncio.run_coroutine_threadsafe(server.start_private_async(), loop).result(timeout=30)
    try:
        yield server
    finally:
        asyncio.run_coroutine_threadsafe(server.stop_async(), loop).result(timeout=15)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def _running_owner(timeout: float = 30.0) -> agent_state.AgentState:
    """The parent's state once its task row exists (the side channel needs the id)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for st in agent_state.snapshot():
            if st.running() and _persisted_task_id(st.agent):
                return st
        time.sleep(0.1)
    raise AssertionError("the parent task never allocated its row")


def test_ask_message_files_reach_the_answering_session(
    env: IsolatedKissHome, daemon: RemoteAccessServer,
) -> None:
    child_seen = threading.Event()
    child_requests: list[dict[str, Any]] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        text = request_text(request)
        if "ASK-QUESTION" in text:
            child_requests.append(request)
            child_seen.set()
            return finish_response("child-done")
        # The parent: one cheap step, then wait for the answer before
        # finishing so its tab is still running while the file travels.
        messages = request.get("messages", [])
        if not any(m.get("role") == "tool" for m in messages):
            return tool_call_response("Bash", {"command": "echo parent-step"})
        child_seen.wait(timeout=90)
        return finish_response("parent-done")

    model = StandInModelServer(responder)
    # The answering session is dispatched without a model: it runs the
    # home's default model on the home's custom endpoint (the stand-in).
    env.write_config(
        last_model=STANDIN_MODEL, custom_endpoint=model.url, custom_api_key="kiss-test-key",
    )
    outcome: dict[str, Any] = {}

    def run_parent() -> None:
        outcome["result"] = sorcar.run(
            "PARENT-TASK wait for the question",
            work_dir=str(env.repo),
            model=STANDIN_MODEL,
            model_config=model.model_config,
            is_parallel=False,
            endpoint_file=str(env.tmpdir / "sorcar-local.json"),
            timeout=300,
        )

    parent = threading.Thread(target=run_parent, daemon=True)
    parent.start()
    try:
        owner = _running_owner()
        daemon._agent_server._cmd_append_user_message({
            "tabId": owner.tab_id,
            "prompt": "/ask ASK-QUESTION what does the picture show?",
            "attachments": [{"name": "shot.png", "mimeType": "image/png", "data": PNG_B64}],
        })
        assert child_seen.wait(timeout=90), "the answering session never ran"
        parent.join(timeout=120)
        assert not parent.is_alive(), "the parent never finished"
    finally:
        model.stop()

    assert outcome["result"].success is True, outcome
    assert "parent-done" in outcome["result"].text
    # The answering session's first request carries the picture as an
    # image content part (a data URL of the PNG bytes).
    wire = json.dumps(child_requests[0])
    assert "image_url" in wire, wire[:2000]
    assert PNG_B64 in wire
    # The parent's own tab never queued the message as a steering text.
    assert owner.pending_user_messages == []
    assert owner.pending_user_attachments == []
