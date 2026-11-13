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
* **Stop during the connect.**  The embeddings call ran inline on the
  task thread (the memory tools), where a Stop arrives only as the
  runner's forced ``KeyboardInterrupt``, and an interrupt cannot land
  while the thread sits in the C ``connect()``.  The request's plain
  60 s timeout covered the connect phase too, so against an endpoint
  that drops SYNs a Stop waited up to a minute; the connect phase is
  capped at ``stream_abort.CONNECT_TIMEOUT``, as on the streamed path,
  and the request now runs on a worker thread, so the interrupt lands
  in the caller's poll.  The endpoint is a real listening socket whose
  accept queue is full, so the kernel drops further SYNs; the interrupt
  is the runner's own mechanism (``tool_interrupt.inject_async_exception``).
* **The SDK's sleep before its retry.**  A 429's ``Retry-After`` (up to
  a minute) is slept out inside the SDK, where no socket is pending for
  an abort to cut: the Stop and the timeout now end the caller at once
  and leave the sleeping request on its worker thread.
* **One request's abort is its own.**  The abort watchdog shuts down
  every socket of the pool it watches, and that pool
  (``AbortableNetworkBackend``) then shuts down every later connection
  the moment it is up.  With one client shared by every embeddings call
  of a model instance, a timed-out or stopped request cut the
  concurrent requests of the same instance and left the retained
  instance (the memory tools keep theirs) unable to embed anything ever
  again.  Each call now sends through a client of its own.
* **``timeout: None``.**  A valid SDK override (no SDK timeout; the
  watchdog still bounds the request) that used to be dereferenced for
  its ``read`` attribute before the request was sent.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from kiss.core import stop_signal
from kiss.core.kiss_error import KISSError
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.stream_abort import CONNECT_TIMEOUT
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    assert_left_during_the_retry_sleep,
    retry_later,
)

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


def _embeddings_with_indices(indices: list[int]) -> Callable[[Request], Reply]:
    """A responder whose rows carry *indices*, each row's vector ``[row position]``."""

    def respond(request: Request) -> Reply:
        return Reply(
            json_body={
                "object": "list",
                "model": request.body["model"],
                "data": [
                    {"object": "embedding", "index": index, "embedding": [float(row)]}
                    for row, index in enumerate(indices)
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        )

    return respond


@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [1, 1]])
def test_embeddings_whose_indices_are_not_the_inputs_are_rejected(indices: list[int]) -> None:
    """Rows with a duplicate, missing or out-of-range index are a malformed
    reply, not vectors to hand out positionally for the wrong inputs."""
    with ScriptedOpenAIServer(_embeddings_with_indices(indices)) as server:
        model = OpenAICompatibleModel(
            "text-embedding-3-small", base_url=server.base_url, api_key="sk-test"
        )
        with pytest.raises(KISSError, match="Embedding generation failed.*indices"):
            model.get_embeddings(["a", "b"])
    with ScriptedOpenAIServer(_embeddings_with_indices([1, 0])) as server:
        model = OpenAICompatibleModel(
            "text-embedding-3-small", base_url=server.base_url, api_key="sk-test"
        )
        assert model.get_embeddings(["a", "b"]) == [[1.0], [0.0]], "rows reordered by index"


def test_a_trickling_embeddings_body_is_cut_at_the_timeout() -> None:
    """The timeout bounds the whole request, not only the silence between two reads.

    A peer that keeps sending a byte before each read timeout would
    otherwise hold the thread (a memory search worker, joined at process
    exit) for as long as it trickles.  The cut request is the only
    casualty: the next request of the same model instance reaches the
    endpoint and succeeds (the memory tools keep their instance for the
    whole daemon).
    """
    hold = threading.Event()

    def respond(request: Request) -> Reply:
        if request.body["input"] == "x":
            return Reply(sse_chunks=[b"{"], hold=hold)
        return _embeddings_reply(request)

    with ScriptedOpenAIServer(respond) as server:
        model = OpenAICompatibleModel(
            "text-embedding-3-small",
            base_url=server.base_url,
            api_key="sk-test",
            model_config={"timeout": 0.5},
        )
        outcome: dict[str, Any] = {}
        thread = threading.Thread(target=_embed_into, args=(model, outcome), daemon=True)
        started = time.monotonic()
        thread.start()
        thread.join(5.0)
        try:
            assert not thread.is_alive(), "the trickling reply held the request past its timeout"
            assert time.monotonic() - started >= 0.5
            assert isinstance(outcome.get("error"), KISSError), outcome
        finally:
            hold.set()
        assert model.get_embedding("after the timeout") == [0.5, 0.0]
        assert [request.body["input"] for request in server.requests] == ["x", "after the timeout"]


def test_a_stopped_request_leaves_its_concurrent_peer_and_the_next_request_alone() -> None:
    """A Stop cuts the stopped request only.

    Memory search embeds the query while the index sync embeds the
    changed pages, through the same model instance: A (stopped, its
    reply trickling) and B (unstopped, its reply pending) are in flight
    together, A's Stop lands, B still completes, and so does a later
    request of the same instance.
    """
    release_a, release_b = threading.Event(), threading.Event()
    arrived_a, arrived_b = threading.Event(), threading.Event()

    def respond(request: Request) -> Reply:
        if request.body["input"] == "A":
            arrived_a.set()
            return Reply(json_body=_embeddings_reply(request).json_body, trickle=release_a)
        if request.body["input"] == "B":
            arrived_b.set()
            return Reply(json_body=_embeddings_reply(request).json_body, headers_gate=release_b)
        return _embeddings_reply(request)

    with ScriptedOpenAIServer(respond) as server:
        model = OpenAICompatibleModel(
            "text-embedding-3-small",
            base_url=server.base_url,
            api_key="sk-test",
            model_config={"timeout": 20.0},
        )
        stop_a, stop_b = threading.Event(), threading.Event()
        outcome_a: dict[str, Any] = {}
        outcome_b: dict[str, Any] = {}
        thread_a = threading.Thread(
            target=_embed_into, args=(model, outcome_a, "A", stop_a), daemon=True
        )
        thread_b = threading.Thread(
            target=_embed_into, args=(model, outcome_b, "B", stop_b), daemon=True
        )
        try:
            thread_a.start()
            assert arrived_a.wait(5.0), "A never reached the endpoint"
            thread_b.start()
            assert arrived_b.wait(5.0), "B never reached the endpoint"
            stop_a.set()
            thread_a.join(5.0)
            assert not thread_a.is_alive(), "the Stop did not cut A"
            assert isinstance(outcome_a.get("error"), KeyboardInterrupt), outcome_a
            release_b.set()
            thread_b.join(5.0)
            assert not thread_b.is_alive(), "A's Stop held B"
            assert not stop_b.is_set()
            assert outcome_b == {"vector": [0.5, 0.0]}, "A's Stop cut B's request"
        finally:
            release_a.set()
            release_b.set()
        assert model.get_embedding("C") == [0.5, 0.0]
        assert sorted(request.body["input"] for request in server.requests) == ["A", "B", "C"]


@pytest.mark.parametrize("stop", [False, True], ids=["timeout", "stop"])
def test_the_sdks_retry_sleep_does_not_hold_an_embeddings_request(stop: bool) -> None:
    """The Stop and the timeout end a request whose SDK sleeps out a ``Retry-After``.

    The caller (a memory search worker, joined at process exit) leaves
    while the SDK still sleeps, and the request given up is not sent again.
    """
    with ScriptedOpenAIServer(retry_later) as server:
        model = OpenAICompatibleModel(
            "text-embedding-3-small",
            base_url=server.base_url,
            api_key="sk-test",
            model_config={"timeout": 1.0},
        )
        stop_event = threading.Event()
        outcome: dict[str, Any] = {}
        caller = threading.Thread(
            target=_embed_into, args=(model, outcome, "x", stop_event), daemon=True
        )
        assert_left_during_the_retry_sleep(caller, stop_event if stop else None)
        assert isinstance(outcome.get("error"), KeyboardInterrupt if stop else KISSError), outcome
        assert len(server.requests) == 1, "the request given up was sent again"


@pytest.mark.parametrize("timeout", [httpx.Timeout(5.0, connect=2.0), None])
def test_a_configured_timeout_object_is_sent_as_is(timeout: httpx.Timeout | None) -> None:
    """Only a plain number gets the connect cap; an ``httpx.Timeout`` or ``None`` goes as is."""
    with ScriptedOpenAIServer(_embeddings_reply) as server:
        model = OpenAICompatibleModel(
            "text-embedding-3-small",
            base_url=server.base_url,
            api_key="sk-test",
            model_config={"timeout": timeout},
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


def _embed_into(
    model: OpenAICompatibleModel,
    outcome: dict[str, Any],
    text: str = "x",
    stop: threading.Event | None = None,
) -> None:
    """Thread body: embed *text* and store the vector or the error in *outcome*.

    Args:
        model: The model under test.
        outcome: Receives ``"vector"`` or ``"error"``.
        text: The text to embed.
        stop: When given, bound as this thread's stop event, as the
            runner binds a task thread's.
    """
    stop_signal.set_thread_stop_event(stop)
    try:
        outcome["vector"] = model.get_embedding(text)
    except BaseException as error:  # noqa: BLE001 - the error is the result
        outcome["error"] = error
    finally:
        stop_signal.set_thread_stop_event(None)


def _wait_until_connecting() -> None:
    """Wait (at most 10 s) until a thread (the request's) sits in ``socket.create_connection``."""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        frames = sys._current_frames().values()
        if any(frame.f_code.co_name == "create_connection" for frame in frames):
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
        _wait_until_connecting()
        stopped_at = time.monotonic()
        assert inject_async_exception(thread.ident or -1, KeyboardInterrupt) == 1
        thread.join(CONNECT_TIMEOUT + _UNWIND_MARGIN)
        elapsed = time.monotonic() - stopped_at
        assert not thread.is_alive(), f"the Stop still waits on the connect after {elapsed:.1f}s"
        assert isinstance(outcome.get("error"), KeyboardInterrupt), outcome
    finally:
        for sock in [*fillers, listener]:
            sock.close()
