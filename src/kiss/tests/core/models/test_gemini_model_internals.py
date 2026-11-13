# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for GeminiModel internal conversions.

These tests avoid mocks while exercising internal transformations and
conversation handling without making external API calls.
"""

import functools
import gc
import json
import os
import socket
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from kiss.core import stop_signal
from kiss.core.kiss_error import KISSError
from kiss.core.models.gemini_model import (
    GeminiModel,
    _EmbeddingClient,
    _ResponseTrackingHttpxClient,
)
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.core.models.gemini_sse_harness import chunk, serve, text_part
from kiss.tests.server.pause import Pause


class TestGeminiModelConversationConversion:
    """Tests for GeminiModel conversation conversion and helpers."""

    def _model(self) -> GeminiModel:
        model = GeminiModel("gemini-3-flash-preview", api_key="test")
        model.model_config = {}
        return model

    def test_convert_conversation_with_tools_and_signatures(self):
        model = self._model()
        model._thought_signatures = {"call_1": b"sig-1"}
        model.conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Answer"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "tool_a", "arguments": {"x": 1}}},
                    {"id": "call_2", "function": {"name": "tool_b", "arguments": {"y": 2}}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"result": "ok"}'},
            {"role": "tool", "tool_call_id": "call_2", "content": {"result": "ok2"}},
            {"role": "tool", "tool_call_id": "call_missing", "content": "not json"},
            {"role": "system", "content": "ignored"},
        ]

        contents = model._convert_conversation_to_gemini_contents(model._chat_messages())
        assert len(contents) >= 4
        assert any(content.role == "model" for content in contents)
        assert any(content.role == "user" for content in contents)

    def test_a_tool_result_nested_too_deep_is_sent_as_its_text(self, monkeypatch):
        """A result too deep to decode, or to send parsed, goes out as text (audit A/C4-F7).

        ``json.loads`` raises ``RecursionError`` near 10,000 open brackets, and
        the SDK builds a request with one Python call per nesting level: either
        result stayed in the conversation and failed every later request.
        """
        shallow = [1, {"still": ["parsed"]}]
        results = ["[" * 100_000, "[" * 2_000 + "]" * 2_000, json.dumps(shallow)]
        endpoint = serve()
        base_url, script = next(endpoint)
        try:
            monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
            model = GeminiModel("gemini-3-flash-preview", api_key="test")
            model.initialize("Read the files.")
            calls = [{"id": name, "function": {"name": "Read", "arguments": {}}} for name in "abc"]
            model.conversation.append({"role": "assistant", "content": "", "tool_calls": calls})
            for name, result in zip("abc", results):
                model.conversation.append({"role": "tool", "tool_call_id": name, "content": result})
            script.play([chunk([text_part("All read.")], finish_reason="STOP")])
            assert model.generate()[0] == "All read."
        finally:
            endpoint.close()
        turns = script.requests[-1]["contents"][-3:]
        sent = [turn["parts"][0]["functionResponse"]["response"] for turn in turns]
        assert sent == [{"result": results[0]}, {"result": results[1]}, {"result": shallow}]

    def test_a_non_finite_number_is_sent_as_its_name(self, monkeypatch):
        """A NaN or infinity in a tool result or a replayed call's arguments goes out as a string.

        Python's decoder accepts ``NaN``, ``Infinity`` and a number that
        overflows to one, and the SDK writes them back as bare tokens: not
        JSON, so Gemini answered 400 ``Invalid JSON payload`` to that request
        and to every later one of the conversation.
        """
        endpoint = serve()
        base_url, script = next(endpoint)
        try:
            monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
            model = GeminiModel("gemini-3-flash-preview", api_key="test")
            model.initialize("Train, then report.")
            call = {"id": "a", "function": {"name": "train", "arguments": '{"lr": NaN}'}}
            model.conversation.append({"role": "assistant", "content": "", "tool_calls": [call]})
            result = '{"loss": NaN, "best": [1e999, -Infinity], "epochs": 3}'
            model.conversation.append({"role": "tool", "tool_call_id": "a", "content": result})
            script.play([chunk([text_part("Diverged.")], finish_reason="STOP")])
            assert model.generate()[0] == "Diverged."
        finally:
            endpoint.close()
        request = script.requests[-1]
        json.dumps(request, allow_nan=False)  # raises for a bare NaN or Infinity on the wire
        call_turn, result_turn = request["contents"][-2:]
        assert call_turn["parts"][0]["functionCall"]["args"] == {"lr": "NaN"}
        assert result_turn["parts"][0]["functionResponse"]["response"] == {
            "loss": "NaN",
            "best": ["Infinity", "-Infinity"],
            "epochs": 3,
        }

    def test_extract_token_counts_no_usage(self):
        model = self._model()

        class Dummy:
            usage_metadata = None

        assert model.extract_input_output_token_counts_from_response(Dummy()) == (0, 0, 0, 0)

    def test_extract_token_counts_clamps_cached_prompt_difference(self):
        model = self._model()
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
                thoughts_token_count=2,
                cached_content_token_count=20,
            )
        )
        assert model.extract_input_output_token_counts_from_response(response) == (0, 7, 20, 0)


class TestGeminiClientReuse:
    """``initialize()`` keeps one ``genai.Client`` across runs."""

    def test_client_is_built_once_per_api_key(self):
        model = GeminiModel("gemini-3-flash-preview", api_key="test")
        assert model.client is None
        model.initialize("first run")
        client = model.client
        assert client is not None
        model.initialize("second run")
        assert model.client is client
        assert model.conversation == [{"role": "user", "content": "second run"}]

        # A key overridden between construction and initialize() (the
        # Sorcar agent's same-provider switch) must reach the client.
        model.api_key = "other-key"
        model.initialize("third run")
        rebuilt = model.client
        assert rebuilt is not None and rebuilt is not client
        assert rebuilt._api_client.api_key == "other-key"


class _TricklingEmbeddings(BaseHTTPRequestHandler):
    """``batchEmbedContents`` trickling JSON whitespace until the text's event, then the vector."""

    arrived: dict[str, threading.Event]
    release: dict[str, threading.Event]

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Answer one embeddings request; the bytes reset every byte-level read timeout."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["requests"][0]["content"]["parts"][0]["text"]
        self.send_response_only(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.arrived[text].set()
        try:
            while not self.release[text].wait(0.02):
                self.wfile.write(b" ")
                self.wfile.flush()
            self.wfile.write(b'{"embeddings": [{"values": [0.1, 0.2]}]}')
        except OSError:
            pass  # the client hung up


def test_a_trickling_embedding_ends_at_its_own_stop_or_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Stop or the deadline cuts an embedding whose reply trickles, and no other (audit C4, F2).

    The call went out bare through the instance's shared pool, where the
    trickled bytes kept the read timeout from firing.  The memory tools
    embed from several threads through one instance, so A's Stop must
    leave B, in flight beside it, to complete.
    """
    arrived = {text: threading.Event() for text in "ABC"}
    release = {text: threading.Event() for text in "ABC"}
    handler = type("_Handler", (_TricklingEmbeddings,), {"arrived": arrived, "release": release})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    model = GeminiModel("gemini-embedding-001", api_key="test")
    impatient = GeminiModel("gemini-embedding-001", api_key="test", model_config={"timeout": 0.5})
    for embedder in (model, impatient):
        embedder.initialize("")  # as the memory tools' ModelEmbedder does
    stop = threading.Event()
    # A's thread has the stop event bound the way the runner binds a task thread's.
    stopped = ThreadPoolExecutor(1, initializer=stop_signal.set_thread_stop_event, initargs=(stop,))
    others = ThreadPoolExecutor(2)
    try:
        a = stopped.submit(model.get_embedding, "A")
        b = others.submit(model.get_embedding, "B")
        c = others.submit(impatient.get_embedding, "C")
        assert arrived["A"].wait(10) and arrived["B"].wait(10), "a request never arrived"
        stop.set()
        assert isinstance(a.exception(10), KeyboardInterrupt), "the Stop did not cut A"
        assert isinstance(c.exception(10), KISSError), "the deadline did not cut C"
        release["B"].set()
        assert b.result(10) == [0.1, 0.2], "A's Stop or C's deadline cut B"
    finally:
        for event in release.values():
            event.set()
        stopped.shutdown()
        others.shutdown()
        server.shutdown()
        server.server_close()


class _WholeEmbeddings(BaseHTTPRequestHandler):
    """``batchEmbedContents`` answering in one write, on a connection it keeps open."""

    protocol_version = "HTTP/1.1"
    accepted: list[object]
    closed: threading.Semaphore
    seen: list[tuple[str, str]]
    pair: threading.Barrier
    stop: threading.Event

    def setup(self) -> None:
        """Count the accepted connection: one handler serves every request on it."""
        super().setup()
        self.accepted.append(self.client_address)

    def finish(self) -> None:
        """Count the connection as closed: its handler is done once the client hung up."""
        super().finish()
        self.closed.release()

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Record the text and its key; ``pair`` waits for its twin, ``stop`` sets the Stop."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["requests"][0]["content"]["parts"][0]["text"]
        self.seen.append((text, self.headers["x-goog-api-key"]))
        if text == "pair":
            self.pair.wait(10)
        if text == "stop":
            self.stop.set()
        reply = b'{"embeddings": [{"values": [0.1, 0.2]}]}'
        head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
        self.wfile.write(head % len(reply) + reply)


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_WholeEmbeddings]]:
    """Serve a :class:`_WholeEmbeddings` with a state of its own as the Gemini endpoint."""

    class Handler(_WholeEmbeddings):
        accepted, seen = [], []
        pair, stop = threading.Barrier(2), threading.Event()
        closed = threading.Semaphore(0)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
        yield Handler
        server.shutdown()


def test_sequential_embeddings_share_one_connection(endpoint: type[_WholeEmbeddings]) -> None:
    """A batch embeds over one connection, as before the per-call client (review R2 of C4, F2).

    Two embeddings in flight together (neither is answered before both
    arrived) have a client each, one of them kept; a replaced key, a new one.
    """
    model = GeminiModel("gemini-embedding-001", api_key="test")
    model.initialize("")  # as the memory tools' ModelEmbedder does
    assert model.get_embeddings(list("abcde")) == [[0.1, 0.2]] * 5
    assert len(endpoint.accepted) == 1, "a connection per embedding"
    with ThreadPoolExecutor(2) as together:
        assert list(together.map(model.get_embedding, ["pair", "pair"])) == [[0.1, 0.2]] * 2
    assert model.get_embedding("f") == [0.1, 0.2]
    assert len(endpoint.accepted) == 2, "the pair's kept client was not reused"
    model.api_key = "other"
    model.initialize("")
    assert model.get_embedding("g") == [0.1, 0.2]
    assert endpoint.seen[-1] == ("g", "other"), "sent with the replaced key"


@pytest.mark.parametrize("already_set", [True, False])
def test_an_embedding_ends_at_a_stop_set_before_it_or_just_before_its_reply(
    endpoint: type[_WholeEmbeddings], already_set: bool
) -> None:
    """A Stop set before the call, or by the endpoint before its whole reply, is raised.

    The watchdog first looks 0.1 s into a request: the embedding came
    back either way, and the Stop set beforehand did not keep the request
    from being sent (review R3 of C4, F2).
    """
    if already_set:
        endpoint.stop.set()
    model = GeminiModel("gemini-embedding-001", api_key="test")
    model.initialize("")
    with ThreadPoolExecutor(
        1, initializer=stop_signal.set_thread_stop_event, initargs=(endpoint.stop,)
    ) as task:  # bound to the Stop the way the runner binds a task thread
        error = task.submit(model.get_embedding, "stop").exception(10)
    assert isinstance(error, KeyboardInterrupt), "the Stop was ignored"
    assert len(endpoint.seen) == (0 if already_set else 1), "sent after the Stop"


def _always(_frame_locals: dict[str, object]) -> bool:
    """A :class:`Pause` matcher: hold the first thread that enters the hooked function."""
    return True


def test_a_stop_between_taking_the_idle_client_and_the_request_leaves_it_closed(
    endpoint: type[_WholeEmbeddings],
) -> None:
    """A Stop once the idle client is out of the slot closes that client (review r2 of C4, F2).

    It was taken before the ``try`` that cleans up, so the Stop left it
    neither in the slot nor closed: while the failed call's traceback was
    kept, its connection stayed open beside the next call's.  The call is
    held in the one function it enters with the taken client and no
    request yet: ``close()``, for a client whose key was replaced.
    """
    model = GeminiModel("gemini-embedding-001", api_key="test")
    model.initialize("")
    assert model.get_embedding("warm") == [0.1, 0.2]
    model.api_key = "other"
    stop = threading.Event()
    with (
        ThreadPoolExecutor(
            1, initializer=stop_signal.set_thread_stop_event, initargs=(stop,)
        ) as task,  # bound to the Stop the way the runner binds a task thread
        Pause(_EmbeddingClient.close.__code__, _always) as closing,
    ):
        thread_id = task.submit(threading.get_ident).result(10)
        failed = task.submit(model.get_embedding, "stopped")
        assert closing.paused.wait(10), "the idle client was not being closed"
        stop.set()
        assert inject_async_exception(thread_id, KeyboardInterrupt) == 1  # the runner's Stop
    assert isinstance(failed.exception(10), KeyboardInterrupt), "the Stop was ignored"
    # ``failed`` still holds the traceback, and through it the frames that held the client.
    assert model.get_embedding("next") == [0.1, 0.2]
    assert endpoint.closed.acquire(timeout=10), "the client taken at the Stop kept its connection"
    assert model.get_embedding("again") == [0.1, 0.2]
    assert len(endpoint.accepted) == 2, "the slot held no usable client"


def _looking_up(port: int | None, frame_locals: dict[str, object]) -> bool:
    """A :class:`Pause` matcher for ``socket.getaddrinfo``: the lookup of a connect to *port*."""
    return frame_locals.get("port") == port


@pytest.mark.parametrize("stopped", [False, True], ids=["deadline", "stop"])
def test_an_embedding_ends_while_its_endpoint_is_still_being_looked_up(
    endpoint: type[_WholeEmbeddings], stopped: bool
) -> None:
    """The deadline and a Stop end an embedding whose endpoint's address is not known yet.

    Until the lookup returns the request has no socket for the watchdog
    to shut down: the call waited for as long as the resolver did.  The
    lookup is held at its entry, as a resolver that does not answer holds it.
    """
    config = {} if stopped else {"timeout": 0.3}
    model = GeminiModel("gemini-embedding-001", api_key="test", model_config=config)
    model.initialize("")
    looking_up = functools.partial(_looking_up, urlsplit(os.environ["GOOGLE_GEMINI_BASE_URL"]).port)
    with (
        ThreadPoolExecutor(
            1, initializer=stop_signal.set_thread_stop_event, initargs=(endpoint.stop,)
        ) as task,  # bound to the Stop the way the runner binds a task thread
        Pause(socket.getaddrinfo.__code__, looking_up) as held,
    ):
        call = task.submit(model.get_embedding, "unsent")
        assert held.paused.wait(10), "the request never reached its address lookup"
        if stopped:
            endpoint.stop.set()
        error = call.exception(5)
    assert isinstance(error, KeyboardInterrupt if stopped else KISSError), error
    assert endpoint.seen == [], "the request was sent after it was cut"


class _AfterReply:
    """A :class:`Pause` matcher: hold the *step*-th ``stop_requested`` once *replying* is set."""

    def __init__(self, replying: threading.Event, step: int) -> None:
        self._replying, self._step = replying, step

    def __call__(self, _frame_locals: dict[str, object]) -> bool:
        if self._replying.is_set():
            self._step -= 1
        return self._replying.is_set() and self._step == 0


@pytest.mark.parametrize("held", [1, 2, "close"])
def test_a_stop_injected_once_the_reply_is_back_leaves_the_client_idle_or_closed(
    endpoint: type[_WholeEmbeddings], held: int | str
) -> None:
    """A Stop injected while the embedding's client is put away loses no client (audit L11-F1).

    The calling thread put it away itself, in a ``finally`` the Stop could
    land in: the client was then neither idle nor closed, and its connection
    stayed open for as long as the failed call's traceback was kept.  The
    Stop is injected while a thread is held at each Python call made once
    the reply is on its way: the two Stop checks (the client goes back to
    the slot), and ``close()`` when the endpoint itself sets the Stop.
    """
    model = GeminiModel("gemini-embedding-001", api_key="test")
    closing = held == "close"
    hook = _EmbeddingClient.close if closing else stop_signal.stop_requested
    with (
        ThreadPoolExecutor(
            1,
            initializer=stop_signal.set_thread_stop_event,
            initargs=(endpoint.stop if closing else None,),
        ) as task,
        Pause(hook.__code__, _always if closing else _AfterReply(endpoint.stop, int(held))) as at,
    ):
        thread_id = task.submit(threading.get_ident).result(10)
        failed = task.submit(model.get_embedding, "stop")
        assert at.paused.wait(10), "no thread reached the step to hold"
        assert inject_async_exception(thread_id, KeyboardInterrupt) == 1  # the runner's Stop
    assert isinstance(failed.exception(10), KeyboardInterrupt), "the Stop was ignored"
    # ``failed`` still holds the traceback, and through it the frames that held the client.
    if closing:
        assert endpoint.closed.acquire(timeout=10), "the client's connection was left open"
    with ThreadPoolExecutor(2) as together:  # one of the pair through the idle client, if any
        assert list(together.map(model.get_embedding, ["pair", "pair"])) == [[0.1, 0.2]] * 2
    assert len(endpoint.accepted) == (3 if closing else 2), "the client was not idle"


def test_an_embedding_without_a_key_fails_as_any_other(
    endpoint: type[_WholeEmbeddings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client that cannot be built is a ``KISSError`` too, and its pool closed (audit A/C4-F6)."""
    names = "GOOGLE_API_KEY GEMINI_API_KEY GOOGLE_GENAI_USE_VERTEXAI GOOGLE_GENAI_USE_ENTERPRISE"
    for name in names.split():  # where else the SDK finds a key, or Vertex credentials
        monkeypatch.delenv(name, raising=False)
    model = GeminiModel("gemini-embedding-001", api_key="")
    before = [pool for pool in gc.get_objects() if isinstance(pool, _ResponseTrackingHttpxClient)]
    with pytest.raises(KISSError, match="No API key") as failure:
        model.get_embedding("unsent")
    assert type(failure.value.__cause__) is ValueError, "the SDK's error is not the cause"
    assert endpoint.seen == [], "sent without a key"
    # The failure's traceback keeps the frame that built the pool, and so the pool.
    pools = [pool for pool in gc.get_objects() if isinstance(pool, _ResponseTrackingHttpxClient)]
    built = [pool for pool in pools if all(pool is not other for other in before)]
    assert [pool.is_closed for pool in built] == [True], "the pool built for the call stayed open"
