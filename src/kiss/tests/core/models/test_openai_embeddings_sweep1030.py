# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end: embeddings requests of the OpenAI-compatible transport.

* **Wire name.**  Chat and Responses requests send the provider's id
  (``_provider_model_name``: the ``openrouter/`` routing prefix and a
  thinking alias stripped), but embeddings sent the catalog name, so an
  ``openrouter/<vendor>/<id>`` embedding model asked OpenRouter for an id
  it does not serve.
* **Stop during the connect.**  The embeddings call runs inline on the
  task thread (the memory tools), where a Stop arrives only as the
  runner's forced ``KeyboardInterrupt``, and an interrupt cannot land
  while the thread sits in the C ``connect()``.  The request's plain
  60 s timeout covered the connect phase too, so against an endpoint
  that drops SYNs a Stop waited up to a minute; the connect phase is now
  capped at ``stream_abort.CONNECT_TIMEOUT``, as on the streamed path.
  The endpoint is a real listening socket whose accept queue is full, so
  the kernel drops further SYNs; the interrupt is the runner's own
  mechanism (``tool_interrupt.inject_async_exception``).
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from typing import Any

import httpx
import pytest

from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.stream_abort import CONNECT_TIMEOUT
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.core.models.openai_sse_harness import Reply, Request, ScriptedOpenAIServer

# Margin above CONNECT_TIMEOUT for the interrupt to unwind; far below the
# 60 s request timeout that used to cover the connect phase.
_UNWIND_MARGIN = 15.0


def _embeddings_reply(request: Request) -> Reply:
    """Answer an embeddings request with one 2-d vector per input, ``[0.5, index]``."""
    inputs = request.body["input"]
    count = 1 if isinstance(inputs, str) else len(inputs)
    return Reply(
        json_body={
            "object": "list",
            "model": request.body["model"],
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.5, float(i)]}
                for i in range(count)
            ],
            "usage": {"prompt_tokens": count, "total_tokens": count},
        }
    )


def test_openrouter_embeddings_send_the_providers_own_id() -> None:
    with ScriptedOpenAIServer(_embeddings_reply) as server:
        model = OpenAICompatibleModel(
            "openrouter/qwen/qwen3-embedding-8b", base_url=server.base_url, api_key="sk-test"
        )
        assert model.get_embedding("x") == [0.5, 0.0]
        assert model.get_embeddings(["a", "b"]) == [[0.5, 0.0], [0.5, 1.0]]
        model.get_embedding("y", embedding_model="openrouter/openai/text-embedding-3-small")
        model.get_embedding("z", embedding_model="text-embedding-3-small")
    assert [request.body["model"] for request in server.requests] == [
        "qwen/qwen3-embedding-8b",
        "qwen/qwen3-embedding-8b",
        "openai/text-embedding-3-small",
        "text-embedding-3-small",
    ]


def test_a_configured_timeout_object_is_sent_as_is() -> None:
    """Only a plain number gets the connect cap; an ``httpx.Timeout`` is the caller's own."""
    with ScriptedOpenAIServer(_embeddings_reply) as server:
        model = OpenAICompatibleModel(
            "text-embedding-3-small",
            base_url=server.base_url,
            api_key="sk-test",
            model_config={"timeout": httpx.Timeout(5.0, connect=2.0)},
        )
        assert model.get_embedding("x") == [0.5, 0.0]


def _syn_dropping_listener() -> tuple[socket.socket, list[socket.socket]]:
    """Return a listener whose accept queue is full, and the connections filling it.

    Further connects then hang in the kernel (their SYNs are dropped)
    until their own timeout, like an endpoint behind a black hole.

    The backlog is 1, not 0: macOS reads a backlog of 0 as the default
    (``kern.ipc.somaxconn``, 128 connections), which the fillers below
    never reach.  A queue of 1 is full after one connection on macOS and
    two on Linux (which admits ``backlog + 1``).  Windows answers a full
    queue with a reset instead of silence, so the test skips there.

    Returns:
        ``(listener, fillers)``; the caller closes all of them.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    fillers: list[socket.socket] = []
    for _ in range(16):
        probe = socket.socket()
        probe.settimeout(0.5)
        try:
            probe.connect(listener.getsockname())
        except TimeoutError:
            probe.close()
            return listener, fillers
        except ConnectionRefusedError:
            probe.close()
            for sock in [*fillers, listener]:
                sock.close()
            pytest.skip(
                "this kernel resets connections to a full accept queue instead of dropping them"
            )
        fillers.append(probe)
    raise AssertionError("the listener's accept queue never filled up")


def _embed_into(model: OpenAICompatibleModel, outcome: dict[str, Any]) -> None:
    """Thread body: embed one text and store the vector or the error in *outcome*."""
    try:
        outcome["vector"] = model.get_embedding("x")
    except BaseException as error:  # noqa: BLE001 - the error is the result
        outcome["error"] = error


def _wait_until_connecting(thread: threading.Thread) -> None:
    """Wait (at most 10 s) until *thread* sits in ``socket.create_connection``."""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(thread.ident or -1)
        if frame is not None and frame.f_code.co_name == "create_connection":
            return
        time.sleep(0.01)
    raise AssertionError("the embeddings request never started connecting")


def test_a_stop_during_the_embeddings_connect_lands_within_the_connect_timeout() -> None:
    listener, fillers = _syn_dropping_listener()
    host, port = listener.getsockname()
    try:
        model = OpenAICompatibleModel(
            "text-embedding-3-small", base_url=f"http://{host}:{port}/v1", api_key="sk-test"
        )
        outcome: dict[str, Any] = {}
        thread = threading.Thread(target=_embed_into, args=(model, outcome), daemon=True)
        thread.start()
        _wait_until_connecting(thread)
        stopped_at = time.monotonic()
        assert inject_async_exception(thread.ident or -1, KeyboardInterrupt) == 1
        thread.join(CONNECT_TIMEOUT + _UNWIND_MARGIN)
        elapsed = time.monotonic() - stopped_at
        assert not thread.is_alive(), f"the Stop still waits on the connect after {elapsed:.1f}s"
        assert isinstance(outcome.get("error"), KeyboardInterrupt), outcome
    finally:
        for sock in [*fillers, listener]:
            sock.close()
