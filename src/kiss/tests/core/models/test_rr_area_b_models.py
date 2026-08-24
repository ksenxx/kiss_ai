# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the area-B redundancy/race fixes.

Covers, with no mocks/patches/fakes (real SDKs against real local HTTP
servers, real threads, real sockets):

* B-R1 — ``get_embedding`` lives once on ``OpenAICompatibleBase`` and
  works identically through both transports.
* B-R2 — the shared ``__init__`` (thinking-level defaulting,
  ``base_url``/``api_key``/``_api_model_name``, ``__str__``) behaves
  identically for v1 and v2.
* B-R4 — ``_emit_as_thinking`` lives once on ``CLITextModel`` and
  brackets the token with a thinking start/end pair for both CLI
  adapters.
* B-R5 — ``OpenAICompatibleModel.initialize()`` reuses the SDK client
  (and its connection pool) across calls, rebuilding only when the
  client's constructor inputs change.
* B-R6 — the id-only unanswered-function-call scan is a thin view over
  the names variant and reports the same outstanding calls.
* B-R7 — the audio format/MIME/extension tables are derived from the
  one canonical table in ``model.py`` and carry the exact values the
  three hand-written tables used to.
* B-R9 — ``_stream_stall_timeout`` is read once in ``Model.__init__``
  for every adapter.
* B-RC1 — ``_AbortableStream`` snapshots its own httpx response, so a
  late watchdog abort can never observe (and shut down) a LIVE retry's
  socket through the shared tracking client.
* B-RC2 — the v1 Chat Completions streaming loop closes its
  ``stop_aware_events`` generator deterministically, so a loop body
  that raises (a token callback propagating an error) does not leave a
  daemon watchdog thread armed over a stranded connection.
* B-RC3 — ``AnthropicModel._create_message`` disarms the watchdog
  before reading its flags / collecting the final message.  The exact
  race window (an abort claimed between the last stream event and the
  old flag check) cannot be reproduced deterministically without
  injecting a pause into ``_create_message`` between the loop exit and
  the flag read — i.e. without instrumenting the code under test — so
  the tests here verify the surrounding behaviour (stall detection
  still raises the retryable ``TimeoutError``, the success path still
  returns, and no watchdog thread survives either path).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.gemini_model import (
    GeminiModel,
    _AbortableStream,
    _ResponseTrackingHttpxClient,
)
from kiss.core.models.model import (
    _AUDIO_FORMAT_TO_MIME,
    _AUDIO_MIME_TO_EXT,
    _AUDIO_MIME_TO_FORMAT,
    Attachment,
    _audio_mime_to_format,
)
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2
from kiss.core.models.stream_abort import DEFAULT_STREAM_STALL_TIMEOUT
from kiss.tests.core.models.openai_sse_harness import (
    Reply,
    Request,
    ScriptedOpenAIServer,
    chat_chunk,
)

_MODEL = "rr-area-b-model-under-test"


def _echo(text: str) -> str:
    """Echo *text* back.

    Args:
        text: The text to echo.

    Returns:
        The same text.
    """
    return text


def _live_threads(*names: str) -> list[str]:
    """Return the live threads whose name is one of *names*."""
    return [t.name for t in threading.enumerate() if t.name in names]


# ---------------------------------------------------------------------------
# B-R1: shared get_embedding
# ---------------------------------------------------------------------------


_EMBEDDING_BODY = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 1.0]}],
    "model": "text-embedding-x",
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}


def _embedding_responder(request: Request) -> Reply:
    """Answer /embeddings with a fixed vector, anything else with a 404."""
    if request.path.endswith("/embeddings"):
        return Reply(json_body=_EMBEDDING_BODY)
    return Reply(status=404, json_body={"error": {"message": "wrong path"}})


class TestSharedGetEmbedding:
    """B-R1: both transports embed through the one base implementation."""

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_embedding_round_trip(self, cls: type) -> None:
        """Both transports return the server's vector."""
        with ScriptedOpenAIServer(_embedding_responder) as server:
            model = cls(_MODEL, base_url=server.base_url, api_key="k")
            model.initialize("seed")
            vector = model.get_embedding("hello", embedding_model="text-embedding-x")
            assert vector == [0.25, -0.5, 1.0]
            assert server.requests[-1].body["model"] == "text-embedding-x"
            assert server.requests[-1].body["input"] == "hello"

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_embedding_failure_wraps_kisserror(self, cls: type) -> None:
        """A provider rejection surfaces as KISSError naming the model."""

        def reject(request: Request) -> Reply:
            return Reply(
                status=400, json_body={"error": {"message": "no such model"}}
            )

        with ScriptedOpenAIServer(reject) as server:
            model = cls(_MODEL, base_url=server.base_url, api_key="k")
            model.initialize("seed")
            with pytest.raises(KISSError, match="Embedding generation failed"):
                model.get_embedding("hello")


# ---------------------------------------------------------------------------
# B-R2: shared __init__ behaviour
# ---------------------------------------------------------------------------


def _catalog_model_with_thinking() -> str:
    """Return a catalog model name that declares a default thinking level."""
    from kiss.core.models.model_info import MODEL_INFO

    for name, info in MODEL_INFO.items():
        if info.thinking and not name.startswith(("cc/", "codex/")):
            return name
    pytest.skip("no catalog model declares a thinking level")


class TestSharedInit:
    """B-R2: v1 and v2 share one thinking-defaulting / naming __init__."""

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_thinking_level_defaulted_from_catalog(self, cls: type) -> None:
        """A catalog thinking level becomes the default reasoning_effort."""
        from kiss.core.models.model_info import MODEL_INFO

        name = _catalog_model_with_thinking()
        model = cls(name, base_url="http://127.0.0.1:1/v1", api_key="k")
        assert model.model_config["reasoning_effort"] == MODEL_INFO[name].thinking

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_explicit_effort_wins_over_catalog_default(self, cls: type) -> None:
        """A caller-set reasoning_effort is never overwritten."""
        name = _catalog_model_with_thinking()
        model = cls(
            name,
            base_url="http://127.0.0.1:1/v1",
            api_key="k",
            model_config={"reasoning_effort": "low"},
        )
        assert model.model_config["reasoning_effort"] == "low"

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_native_reasoning_effort_suppresses_default(self, cls: type) -> None:
        """A native reasoning.effort dict suppresses the alias default."""
        name = _catalog_model_with_thinking()
        model = cls(
            name,
            base_url="http://127.0.0.1:1/v1",
            api_key="k",
            model_config={"reasoning": {"effort": "high"}},
        )
        assert "reasoning_effort" not in model.model_config

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_unknown_model_gets_no_default(self, cls: type) -> None:
        """Custom endpoint models outside the catalog get no effort key."""
        model = cls(_MODEL, base_url="http://127.0.0.1:1/v1", api_key="k")
        assert "reasoning_effort" not in model.model_config

    @pytest.mark.parametrize("cls", [OpenAICompatibleModel, OpenAICompatibleModel2])
    def test_str_and_repr_show_endpoint(self, cls: type) -> None:
        """Both transports render class, model name and base_url."""
        model = cls(_MODEL, base_url="http://127.0.0.1:9/v1", api_key="k")
        rendered = str(model)
        assert cls.__name__ in rendered
        assert _MODEL in rendered
        assert "http://127.0.0.1:9/v1" in rendered
        assert repr(model) == rendered

    def test_openrouter_prefix_stripped_for_api_name(self) -> None:
        """The openrouter/ routing prefix never goes over the wire."""
        model = OpenAICompatibleModel(
            "openrouter/some-vendor/some-model",
            base_url="http://127.0.0.1:1/v1",
            api_key="k",
        )
        assert model._api_model_name == "some-vendor/some-model"


# ---------------------------------------------------------------------------
# B-R5: client reuse in v1 initialize()
# ---------------------------------------------------------------------------


_CHAT_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def _chat_responder(request: Request) -> Reply:
    """Answer every chat completion with a fixed non-streamed body."""
    return Reply(json_body=_CHAT_BODY)


class TestClientReuse:
    """B-R5: v1 keeps one SDK client (and pool) across initialize() calls."""

    def test_initialize_reuses_client_and_connection(self) -> None:
        """Two runs share one client object and one TCP connection."""
        with ScriptedOpenAIServer(_chat_responder) as server:
            model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
            model.initialize("first run")
            first_client = model.client
            assert model.generate()[0] == "ok"
            model.initialize("second run")
            assert model.client is first_client
            assert model.generate()[0] == "ok"
            assert len(server.connection_keys) == 1

    def test_header_change_rebuilds_client(self) -> None:
        """Changing extra_headers between runs rebuilds the client."""
        with ScriptedOpenAIServer(_chat_responder) as server:
            model = OpenAICompatibleModel(_MODEL, base_url=server.base_url, api_key="k")
            model.initialize("first run")
            first_client = model.client
            model.model_config = dict(model.model_config)
            model.model_config["extra_headers"] = {"X-Area-B": "yes"}
            model.initialize("second run")
            assert model.client is not first_client
            assert model.generate()[0] == "ok"

    def test_base_url_change_rebuilds_client(self) -> None:
        """Repointing base_url between runs rebuilds the client."""
        with (
            ScriptedOpenAIServer(_chat_responder) as first,
            ScriptedOpenAIServer(_chat_responder) as second,
        ):
            model = OpenAICompatibleModel(_MODEL, base_url=first.base_url, api_key="k")
            model.initialize("first run")
            first_client = model.client
            assert model.generate()[0] == "ok"
            model.base_url = second.base_url
            model.initialize("second run")
            assert model.client is not first_client
            assert model.generate()[0] == "ok"
            assert first.requests and second.requests


# ---------------------------------------------------------------------------
# B-R6: unanswered function-call scan
# ---------------------------------------------------------------------------


class TestUnansweredFunctionCallScan:
    """B-R6: the id-only scan is a view over the single names-scan loop."""

    def _model(self) -> OpenAICompatibleModel2:
        model = OpenAICompatibleModel2(
            _MODEL, base_url="http://127.0.0.1:1/v1", api_key="k"
        )
        model.conversation = [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "call_a", "name": "f", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_b", "name": "g", "arguments": "{}"},
            {"type": "function_call", "call_id": "", "name": "no_id", "arguments": "{}"},
            "not-a-dict",
            {"type": "function_call_output", "call_id": "call_a", "output": "done"},
        ]
        return model

    def test_id_scan_matches_names_scan(self) -> None:
        """Both scans report the same outstanding calls, in order."""
        model = self._model()
        with_names = model._unanswered_function_calls_from_conversation_with_names()
        assert with_names == [{"name": "g", "call_id": "call_b"}]
        assert model._unanswered_function_calls_from_conversation() == ["call_b"]

    def test_all_answered_is_empty(self) -> None:
        """Answering the last call empties both scans."""
        model = self._model()
        model.conversation.append(
            {"type": "function_call_output", "call_id": "call_b", "output": "done"}
        )
        assert model._unanswered_function_calls_from_conversation() == []
        assert model._unanswered_function_calls_from_conversation_with_names() == []


# ---------------------------------------------------------------------------
# B-R7: one canonical audio table
# ---------------------------------------------------------------------------


class TestCanonicalAudioTables:
    """B-R7: derived tables carry exactly the old hand-written values."""

    def test_mime_to_format_values(self) -> None:
        """The canonical table matches the old v1 table."""
        assert _AUDIO_MIME_TO_FORMAT == {
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/ogg": "ogg",
            "audio/webm": "webm",
            "audio/flac": "flac",
            "audio/aac": "aac",
            "audio/mp4": "mp4",
        }

    def test_derived_format_to_mime_matches_old_anthropic_table(self) -> None:
        """First-MIME-wins inversion reproduces the old Anthropic table."""
        assert _AUDIO_FORMAT_TO_MIME == {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
            "flac": "audio/flac",
            "aac": "audio/aac",
            "mp4": "audio/mp4",
        }

    def test_derived_mime_to_ext_matches_old_model_table(self) -> None:
        """Extension derivation reproduces the old model.py table."""
        assert _AUDIO_MIME_TO_EXT == {
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/ogg": ".ogg",
            "audio/webm": ".webm",
            "audio/flac": ".flac",
            "audio/aac": ".aac",
            "audio/mp4": ".m4a",
        }

    def test_format_helper_mapped_and_fallback(self) -> None:
        """The helper maps known MIME types and falls back to the subtype."""
        assert _audio_mime_to_format("audio/mpeg") == "mp3"
        assert _audio_mime_to_format("audio/x-wav") == "wav"
        assert _audio_mime_to_format("audio/amr") == "amr"
        assert _audio_mime_to_format("noslash") == "noslash"

    def test_v1_attachment_conversion_still_uses_table(self) -> None:
        """An mp3 attachment becomes an input_audio part with format mp3."""
        part = OpenAICompatibleModel._attachment_to_content_part(
            Attachment(data=b"\xff\xfbaudio", mime_type="audio/mpeg")
        )
        assert part is not None
        assert part["type"] == "input_audio"
        assert part["input_audio"]["format"] == "mp3"


# ---------------------------------------------------------------------------
# B-R4: shared _emit_as_thinking
# ---------------------------------------------------------------------------


class TestSharedEmitAsThinking:
    """B-R4: both CLI adapters bracket the text in one thinking pair."""

    @pytest.mark.parametrize(
        ("cls", "name"),
        [(ClaudeCodeModel, "cc/opus"), (CodexModel, "codex/default")],
    )
    def test_emit_brackets_token(self, cls: type, name: str) -> None:
        """The token streams between thinking(True) and thinking(False)."""
        events: list[Any] = []
        model = cls(
            name,
            token_callback=lambda tok: events.append(("token", tok)),
            thinking_callback=lambda flag: events.append(("thinking", flag)),
        )
        model._emit_as_thinking("$ ls\n")
        assert events == [
            ("thinking", True),
            ("token", "$ ls\n"),
            ("thinking", False),
        ]
        assert model._thinking_open is False

    def test_emit_without_callbacks_is_a_noop(self) -> None:
        """No callbacks bound: emitting must not raise."""
        model = ClaudeCodeModel("cc/opus")
        model._emit_as_thinking("quiet")
        assert model._thinking_open is False


# ---------------------------------------------------------------------------
# B-R9: stall timeout read once in Model.__init__
# ---------------------------------------------------------------------------


class TestStallTimeoutHoist:
    """B-R9: every adapter reads stream_stall_timeout from Model.__init__."""

    def test_configured_value_reaches_every_adapter(self) -> None:
        """A configured timeout lands on all five adapter families."""
        config = {"stream_stall_timeout": 7}
        models: list[Any] = [
            OpenAICompatibleModel(
                _MODEL, base_url="http://127.0.0.1:1/v1", api_key="k",
                model_config=dict(config),
            ),
            OpenAICompatibleModel2(
                _MODEL, base_url="http://127.0.0.1:1/v1", api_key="k",
                model_config=dict(config),
            ),
            AnthropicModel(_MODEL, api_key="k", model_config=dict(config)),
            GeminiModel(_MODEL, api_key="k", model_config=dict(config)),
            ClaudeCodeModel("cc/opus", model_config=dict(config)),
        ]
        for model in models:
            assert model._stream_stall_timeout == 7.0, type(model).__name__

    def test_default_applies_when_unset(self) -> None:
        """Without the key, the shared default applies."""
        model = OpenAICompatibleModel(
            _MODEL, base_url="http://127.0.0.1:1/v1", api_key="k"
        )
        assert model._stream_stall_timeout == DEFAULT_STREAM_STALL_TIMEOUT


# ---------------------------------------------------------------------------
# B-RC2: v1 streaming loop closes its generator deterministically
# ---------------------------------------------------------------------------


_STREAM_CHUNK = chat_chunk(
    {
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": _MODEL,
        "choices": [
            {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
        ],
    }
)

_V1_WATCHDOG_NAMES = (
    "openai-stream-abort-watchdog",
    "openai-tools-stream-abort-watchdog",
)


class _RaisingCallback:
    """A token callback that raises on its first invocation."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, token: str) -> None:
        """Record the call and raise.

        Args:
            token: The streamed token (ignored).
        """
        self.calls += 1
        raise RuntimeError("printer exploded")


class TestV1StreamGeneratorClosed:
    """B-RC2: a raising loop body must not strand watchdog or connection."""

    def _run(self, invoke: Callable[[OpenAICompatibleModel], Any]) -> None:
        release = threading.Event()

        def responder(request: Request) -> Reply:
            return Reply(sse_chunks=[_STREAM_CHUNK], hold=release)

        try:
            with ScriptedOpenAIServer(responder) as server:
                callback = _RaisingCallback()
                model = OpenAICompatibleModel(
                    _MODEL,
                    base_url=server.base_url,
                    api_key="k",
                    model_config={"stream_stall_timeout": 60},
                    token_callback=callback,
                )
                model.initialize("stream then explode")
                with pytest.raises(RuntimeError, match="printer exploded") as excinfo:
                    invoke(model)
                # The traceback is alive in `excinfo`, so an abandoned
                # (non-closed) generator could NOT have been collected
                # yet: a surviving watchdog thread here is the leak.
                assert excinfo.value is not None
                assert _live_threads(*_V1_WATCHDOG_NAMES) == []
                assert server.client_disconnected.wait(timeout=10.0), (
                    "the streamed connection was never released"
                )
                assert callback.calls == 1
        finally:
            release.set()

    def test_plain_generate_path(self) -> None:
        """The tool-less streaming loop cleans up when the body raises."""
        self._run(lambda model: model.generate())

    def test_tools_path(self) -> None:
        """The adaptive (tools) streaming loop cleans up the same way."""
        self._run(
            lambda model: model.generate_and_process_with_tools({"_echo": _echo})
        )


class TestGeminiStreamGeneratorClosed:
    """B-RC2 (Gemini): the mirrored events.close() fix in _stream_turn."""

    def test_raising_token_callback_leaves_no_watchdog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising loop body must not strand the gemini watchdog."""
        from kiss.tests.core.models.gemini_sse_harness import chunk as gemini_chunk
        from kiss.tests.core.models.gemini_sse_harness import serve, text_part

        endpoint = serve()
        base_url, script = next(endpoint)
        try:
            monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
            script.play(
                [gemini_chunk([text_part("hello")])], after="keepalive"
            )
            callback = _RaisingCallback()
            model = GeminiModel(
                "gemini-rr-area-b",
                api_key="test-key",
                model_config={"stream_stall_timeout": 60},
                token_callback=callback,
            )
            model.initialize("stream then explode")
            with pytest.raises(RuntimeError, match="printer exploded") as excinfo:
                model.generate()
            # The traceback is alive in `excinfo`, so an abandoned
            # (non-closed) generator could NOT have been collected yet.
            assert _live_threads("gemini-stream-abort-watchdog") == []
            assert callback.calls == 1
            assert excinfo.value is not None
        finally:
            script.release.set()
            for _rest in endpoint:  # run the fixture generator's teardown
                pass  # pragma: no cover — serve() yields exactly once


# ---------------------------------------------------------------------------
# B-RC1: _AbortableStream snapshots its own response
# ---------------------------------------------------------------------------


def _null_responder(request: Request) -> Reply:
    """Answer every request with an empty JSON object."""
    return Reply(json_body={})


def _issue(client: httpx.Client, url: str) -> httpx.Response:
    """Send one real POST through *client*, returning the response."""
    request = client.build_request("POST", url, json={})
    response = client.send(request)
    response.read()
    return response


class TestAbortableStreamSnapshot:
    """B-RC1: an abort can only ever reach THIS stream's response."""

    def test_construction_forgets_previous_request(self) -> None:
        """A dead prior response is never exposed as this stream's."""
        with ScriptedOpenAIServer(_null_responder) as server:
            client = _ResponseTrackingHttpxClient(stall_timeout=30.0)
            url = server.base_url + "/anything"
            previous = _issue(client, url)
            assert client.last_response is previous
            stream = _AbortableStream(iter([]), client)
            assert client.last_response is None
            assert stream.response is None
            client.close()

    def test_first_chunk_snapshots_and_retry_cannot_replace(self) -> None:
        """After the first chunk, a retry's response is never observed."""
        with ScriptedOpenAIServer(_null_responder) as server:
            client = _ResponseTrackingHttpxClient(stall_timeout=30.0)
            url = server.base_url + "/anything"

            def lazy_request_chunks() -> Generator[str]:
                # Mirrors the SDK generator: the request is only issued
                # once the first chunk is pulled.
                _issue(client, url)
                yield "chunk-1"
                yield "chunk-2"

            stream = _AbortableStream(lazy_request_chunks(), client)
            iterator = iter(stream)
            assert next(iterator) == "chunk-1"
            own_response = stream.response
            assert own_response is not None
            retry = _issue(client, url)  # a later request on the SAME client
            assert client.last_response is retry
            assert stream.response is own_response
            stream.close()
            assert stream.response is own_response
            client.close()

    def test_property_read_caches_before_close(self) -> None:
        """A pre-close property read freezes the current response."""
        with ScriptedOpenAIServer(_null_responder) as server:
            client = _ResponseTrackingHttpxClient(stall_timeout=30.0)
            url = server.base_url + "/anything"
            stream = _AbortableStream(iter(["only"]), client)
            own_response = _issue(client, url)  # "this stream's" request
            assert stream.response is own_response
            later = _issue(client, url)
            assert client.last_response is later
            assert stream.response is own_response
            client.close()

    def test_close_before_request_freezes_none(self) -> None:
        """A stream closed before its request never adopts a later one."""
        with ScriptedOpenAIServer(_null_responder) as server:
            client = _ResponseTrackingHttpxClient(stall_timeout=30.0)
            url = server.base_url + "/anything"
            def no_chunks() -> Generator[str]:
                return
                yield  # pragma: no cover — makes this a generator

            stream = _AbortableStream(no_chunks(), client)
            stream.close()
            _issue(client, url)  # a retry after the dead stream was closed
            assert stream.response is None
            client.close()


# ---------------------------------------------------------------------------
# B-RC3: Anthropic watchdog disarmed before flags / final message
# ---------------------------------------------------------------------------


def _anthropic_sse(event_type: str, payload: dict[str, Any]) -> bytes:
    """Render one Anthropic Messages SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _anthropic_success_stream(model_name: str) -> list[bytes]:
    """A complete one-text-block Anthropic turn."""
    return [
        _anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_rr_b",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
        ),
        _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "done"},
            },
        ),
        _anthropic_sse(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ),
        _anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        _anthropic_sse("message_stop", {"type": "message_stop"}),
    ]


class _AnthropicScript:
    """Per-test policy for the local Anthropic-shaped server."""

    def __init__(self) -> None:
        self.stall = False
        self.release = threading.Event()


_ANTHROPIC_SCRIPT = _AnthropicScript()


class _AnthropicHandler(BaseHTTPRequestHandler):
    """Serves /v1/messages: a full turn, or message_start then silence."""

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Stream the scripted reply."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        model_name = body.get("model", _MODEL)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = _anthropic_success_stream(model_name)
        if _ANTHROPIC_SCRIPT.stall:
            chunks = chunks[:1]
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()
        if _ANTHROPIC_SCRIPT.stall:
            _ANTHROPIC_SCRIPT.release.wait(timeout=60.0)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default stderr access log."""


class _DaemonServer(ThreadingHTTPServer):
    """Threading server whose request threads never block interpreter exit."""

    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture
def anthropic_server() -> Generator[str]:
    """Run the local Anthropic-shaped server for one test."""
    _ANTHROPIC_SCRIPT.stall = False
    _ANTHROPIC_SCRIPT.release = threading.Event()
    server = _DaemonServer(("127.0.0.1", 0), _AnthropicHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    _ANTHROPIC_SCRIPT.release.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5.0)


class TestAnthropicWatchdogOrder:
    """B-RC3: stop-before-flags keeps both outcomes correct and leak-free.

    The narrow race itself — an abort CLAIMED between the last stream
    event and the old pre-stop flag check, poisoning a pooled socket —
    needs a pause injected exactly there inside ``_create_message``,
    which is impossible without instrumenting the code under test (a
    test double).  These tests pin the observable contract around the
    reordering instead: the success path returns with the watchdog
    already disarmed, and the stall path still raises the retryable
    ``TimeoutError``.
    """

    def test_success_path_returns_with_watchdog_disarmed(
        self, monkeypatch: pytest.MonkeyPatch, anthropic_server: str
    ) -> None:
        """A complete turn returns its text and leaves no watchdog."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", anthropic_server)
        tokens: list[str] = []
        model = AnthropicModel(
            _MODEL,
            api_key="k",
            model_config={"stream_stall_timeout": 30},
            token_callback=tokens.append,
        )
        model.initialize("say done")
        content, _response = model.generate()
        assert content == "done"
        assert tokens == ["done"]
        assert _live_threads("anthropic-stream-abort-watchdog") == []

    def test_stall_after_message_start_raises_retryable_timeout(
        self, monkeypatch: pytest.MonkeyPatch, anthropic_server: str
    ) -> None:
        """Silence after message_start still raises the stall TimeoutError."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", anthropic_server)
        _ANTHROPIC_SCRIPT.stall = True
        model = AnthropicModel(
            _MODEL,
            api_key="k",
            model_config={"stream_stall_timeout": 1.0},
            token_callback=lambda tok: None,
        )
        model.initialize("stall out")
        with pytest.raises(TimeoutError, match="stalled"):
            model.generate()
        assert _live_threads("anthropic-stream-abort-watchdog") == []


# ---------------------------------------------------------------------------
# Review finding 4a: the adaptive tools+effort verdict key follows base_url
# ---------------------------------------------------------------------------


_RESPONSES_TEXT_BODY = {
    "id": "resp-1",
    "object": "response",
    "created_at": 0,
    "model": _MODEL,
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "output": [
        {
            "type": "message",
            "id": "msg-1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "resp-ok", "annotations": []}],
        }
    ],
    "usage": {
        "input_tokens": 5,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 7,
    },
}


def _effort_rejecting_responder(request: Request) -> Reply:
    """Reject tools + reasoning_effort with the 400 real endpoints send."""
    if "reasoning_effort" in request.body:
        return Reply(
            status=400,
            json_body={
                "error": {
                    "message": "reasoning_effort is not supported with tools",
                    "type": "invalid_request_error",
                }
            },
        )
    return Reply(json_body=_CHAT_BODY)


def _accepting_responder(request: Request) -> Reply:
    """Accept every chat completion, reasoning_effort included."""
    return Reply(json_body=_CHAT_BODY)


class TestEffortVerdictKeyFollowsEndpoint:
    """Reconfiguring base_url must not leak the old endpoint's verdict.

    The adaptive tools+``reasoning_effort`` verdict is cached per
    ``(base_url, api_model_name)``.  Before the fix the key was computed
    once in ``__init__``: after repointing the model at endpoint B (the
    same reconfiguration that rebuilds the SDK client), B inherited A's
    cached rejection — its requests silently lost ``reasoning_effort``
    without a single probe — and any verdict B earned was recorded under
    A's key.
    """

    def test_new_endpoint_is_probed_and_old_verdict_survives(self) -> None:
        """B gets its own optimistic probe; A's rejection stays cached."""
        with (
            ScriptedOpenAIServer(_effort_rejecting_responder) as server_a,
            ScriptedOpenAIServer(_accepting_responder) as server_b,
        ):
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server_a.base_url,
                api_key="k",
                model_config={"reasoning_effort": "high"},
            )
            model.initialize("use the tool")
            calls, content, _ = model.generate_and_process_with_tools({"echo": _echo})
            assert (calls, content) == ([], "ok")
            # The probe carried the effort, the retry dropped it.
            assert [("reasoning_effort" in r.body) for r in server_a.requests] == [
                True,
                False,
            ]

            # Repoint the SAME model instance at endpoint B.
            model.base_url = server_b.base_url
            model.initialize("use the tool again")
            calls, content, _ = model.generate_and_process_with_tools({"echo": _echo})
            assert (calls, content) == ([], "ok")
            # B was probed optimistically: its very first request kept
            # the effort instead of inheriting A's cached rejection.
            assert len(server_b.requests) == 1
            assert server_b.requests[0].body["reasoning_effort"] == "high"

            # Back on A: the rejection learned for A is still in force,
            # so the effort is dropped up front -- no second 400 probe.
            model.base_url = server_a.base_url
            model.initialize("and once more")
            requests_before = len(server_a.requests)
            calls, content, _ = model.generate_and_process_with_tools({"echo": _echo})
            assert (calls, content) == ([], "ok")
            new_requests = server_a.requests[requests_before:]
            assert len(new_requests) == 1
            assert "reasoning_effort" not in new_requests[0].body


# ---------------------------------------------------------------------------
# Review finding 4b: the Responses delegate follows endpoint reconfiguration
# ---------------------------------------------------------------------------


def _responses_responder(request: Request) -> Reply:
    """Answer /responses with a fixed text turn, anything else with a 404."""
    if request.path.endswith("/responses"):
        return Reply(json_body=_RESPONSES_TEXT_BODY)
    return Reply(status=404, json_body={"error": {"message": "wrong path"}})


class TestDelegateFollowsEndpointReconfiguration:
    """A delegated tool turn must hit the CURRENT endpoint and credentials.

    The v1 model caches its Responses delegate for the connection pool.
    Before the fix the delegate froze the ``base_url`` / ``api_key`` /
    ``extra_headers`` it was constructed with: after the parent was
    repointed at endpoint B with fresh credentials, every delegated tool
    turn kept POSTing to endpoint A with A's key.
    """

    def test_delegated_turn_moves_with_the_parent(self) -> None:
        """After reconfiguration the next delegated request reaches B."""
        with (
            ScriptedOpenAIServer(_responses_responder) as server_a,
            ScriptedOpenAIServer(_responses_responder) as server_b,
        ):
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server_a.base_url,
                api_key="key-a",
                model_config={"use_responses_api": True, "reasoning_effort": "high"},
            )
            model.initialize("first tool turn")
            calls, content, _ = model.generate_and_process_with_tools({"echo": _echo})
            assert (calls, content) == ([], "resp-ok")
            assert server_a.requests[-1].path.endswith("/responses")
            assert server_a.requests[-1].headers["authorization"] == "Bearer key-a"
            delegate = model._responses_delegate
            assert delegate is not None

            # Reconfigure endpoint, credentials and extra headers -- the
            # same inputs whose change rebuilds the parent's own client.
            model.base_url = server_b.base_url
            model.api_key = "key-b"
            model.model_config = dict(model.model_config)
            model.model_config["extra_headers"] = {"X-Kiss-Rr": "b"}
            model.initialize("second tool turn")
            calls, content, _ = model.generate_and_process_with_tools({"echo": _echo})
            assert (calls, content) == ([], "resp-ok")

            # The delegated request reached B with B's credentials and
            # headers; A saw nothing after the reconfiguration.
            assert len(server_b.requests) == 1
            request_b = server_b.requests[0]
            assert request_b.path.endswith("/responses")
            assert request_b.headers["authorization"] == "Bearer key-b"
            assert request_b.headers["x-kiss-rr"] == "b"
            assert len(server_a.requests) == 1
            # The pool was kept: the same delegate served both turns.
            assert model._responses_delegate is delegate

    def test_unchanged_endpoint_keeps_the_delegate_client(self) -> None:
        """Without reconfiguration the delegate's client is not rebuilt."""
        with ScriptedOpenAIServer(_responses_responder) as server:
            model = OpenAICompatibleModel(
                _MODEL,
                base_url=server.base_url,
                api_key="key-a",
                model_config={"use_responses_api": True, "reasoning_effort": "high"},
            )
            model.initialize("turn one")
            model.generate_and_process_with_tools({"echo": _echo})
            assert model._responses_delegate is not None
            first_client = model._responses_delegate.client
            model.initialize("turn two")
            model.generate_and_process_with_tools({"echo": _echo})
            assert model._responses_delegate.client is first_client
            assert len(server.connection_keys) == 1


# ---------------------------------------------------------------------------
# Review finding 5: gemini converts the conversation once per request
# ---------------------------------------------------------------------------


class _ConversionCountingGemini(GeminiModel):
    """GeminiModel that counts runs of its one conversion hook.

    ``_chat_messages`` already exists as the documented single place
    :func:`responses_items_to_chat_messages` runs for a request; this
    subclass only counts the calls and delegates, so the code under test
    is exercised unchanged (no mocks, no behaviour doubles).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.conversions = 0

    def _chat_messages(self) -> list[dict[str, Any]]:
        """Count and delegate to the real conversion."""
        self.conversions += 1
        return super()._chat_messages()


class TestGeminiConvertOncePerRequest:
    """The conversation is converted exactly once per generate call.

    ``generate`` / ``generate_and_process_with_tools`` convert the
    conversation once and hand the result to all three consumers
    (`_tool_call_id_to_name_map`, the contents builder and the system
    instruction resolver); this pins that count so a regression back to
    per-consumer conversion is caught, and proves the single conversion
    still produces the full request payload for a long mixed
    conversation.
    """

    _CONVERSATION: list[dict[str, Any]] = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "echo hi please"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "hi"},
        {"role": "user", "content": "now answer in one word"},
    ]

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, with_tools: bool
    ) -> tuple[int, dict[str, Any]]:
        """Run one generation over the long conversation; return count+request."""
        from kiss.tests.core.models.gemini_sse_harness import (
            chunk as gemini_chunk,
        )
        from kiss.tests.core.models.gemini_sse_harness import serve, text_part

        endpoint = serve()
        base_url, script = next(endpoint)
        try:
            monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
            script.play([gemini_chunk([text_part("done")])])
            model = _ConversionCountingGemini("gemini-rr-area-b", api_key="k")
            model.initialize("placeholder")
            model.conversation = [dict(m) for m in self._CONVERSATION]
            if with_tools:
                calls, content, _ = model.generate_and_process_with_tools(
                    {"echo": _echo}
                )
                assert (calls, content) == ([], "done")
            else:
                content, _ = model.generate()
                assert content == "done"
            return model.conversions, script.requests[-1]
        finally:
            script.release.set()
            for _rest in endpoint:
                pass  # pragma: no cover -- serve() yields exactly once

    @pytest.mark.parametrize("with_tools", [False, True])
    def test_one_conversion_per_request(
        self, monkeypatch: pytest.MonkeyPatch, with_tools: bool
    ) -> None:
        """Exactly one conversion, and the full conversation still arrives."""
        conversions, request = self._run(monkeypatch, with_tools)
        assert conversions == 1

        # The single conversion produced the complete request: system
        # instruction, both user turns, the tool call and its reply.
        system = request["systemInstruction"]["parts"][0]["text"]
        assert system == "be terse"
        contents = request["contents"]
        roles = [entry["role"] for entry in contents]
        assert roles == ["user", "model", "user", "user"]
        assert contents[0]["parts"][0]["text"] == "echo hi please"
        assert contents[1]["parts"][0]["functionCall"]["name"] == "echo"
        assert contents[2]["parts"][0]["functionResponse"]["name"] == "echo"
        assert contents[3]["parts"][0]["text"] == "now answer in one word"
