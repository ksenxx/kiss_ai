# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Eyes for text-only models: a vision model describes what a tool returned.

The model under test is ``deepseek-ai/DeepSeek-V4-Pro`` (a text-only name)
played by the scripted local chat-completions server; the description is
a real call to the first configured model of ``VISION_MODELS`` and is
skipped when none is configured.  The other tests need no network.
"""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image, ImageDraw, ImageFont

from kiss.agents.sorcar.useful_tools import UsefulTools
from kiss.core import config as config_module
from kiss.core import stop_signal
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import BudgetExceededError, KISSError
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.core.models.model import Attachment, hand_off_conversation
from kiss.core.models.model_families import model_supports_images
from kiss.core.models.model_info import (
    _NATIVE_PROVIDERS,
    OPENAI_COMPATIBLE_PROVIDERS,
    calculate_cost,
)
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.vision_bridge import describe_attachments, vision_model
from kiss.tests.agents.sorcar.local_model_server import serve, tool_call_body
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.anthropic_sse_harness import ScriptedAnthropicServer, sse
from kiss.tests.core.test_audit1018_codex_bridge_stop import _call, _reply

TEXT_ONLY_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
"""A catalog name of a model that cannot see; the local server answers for it."""

LOCAL_VISION_MODEL = "claude-sonnet-5-5"
"""The first of ``VISION_MODELS``: the one the bridge picks when only the Anthropic key is set."""

VISION_INPUT_TOKENS = 1_200
VISION_OUTPUT_TOKENS = 80
"""Usage the scripted vision endpoint reports for every description."""

VISION_COST = calculate_cost(LOCAL_VISION_MODEL, VISION_INPUT_TOKENS, VISION_OUTPUT_TOKENS)
"""What one scripted description costs at the catalog's rates for the vision model."""


def _vision_turn(text: str) -> list[bytes]:
    """SSE chunks of one streamed Anthropic turn answering *text* with the scripted usage."""
    message: dict[str, Any] = {
        "id": "msg_vision",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": LOCAL_VISION_MODEL,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": VISION_INPUT_TOKENS, "output_tokens": 1},
    }
    block_start = {"type": "content_block_start", "index": 0}
    delta = {"type": "content_block_delta", "index": 0}
    return [
        sse("message_start", {"type": "message_start", "message": message}),
        sse("content_block_start", {**block_start, "content_block": {"type": "text", "text": ""}}),
        sse("content_block_delta", {**delta, "delta": {"type": "text_delta", "text": text}}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": VISION_OUTPUT_TOKENS},
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    ]


def _route_the_bridge_to(monkeypatch: pytest.MonkeyPatch, server: ScriptedAnthropicServer) -> None:
    """Make *server* the vision endpoint: only the Anthropic key is set, and the SDK's URL is it.

    No code is patched: the bridge picks ``claude-sonnet-5-5`` because it
    is the first configured vision model, and the real Anthropic adapter
    builds its client without a ``base_url``, which the SDK then reads
    from ``ANTHROPIC_BASE_URL``.
    """
    _blank_every_provider_key(monkeypatch)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "sk-ant-local-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
    assert vision_model() == LOCAL_VISION_MODEL


def _write_png(path: Path) -> bytes:
    """Write a 480x240 white PNG with the black text ``KISS 42`` and a red box."""
    image = Image.new("RGB", (480, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 60), "KISS 42", fill="black", font=ImageFont.load_default(size=72))
    draw.rectangle((340, 150, 440, 220), fill="red")
    image.save(path, format="PNG")
    return path.read_bytes()


def _write_pdf(path: Path) -> bytes:
    """Write a one-page PDF (a rendered 600x300 page) with the text ``INVOICE 7781``."""
    page = Image.new("RGB", (600, 300), "white")
    draw = ImageDraw.Draw(page)
    draw.text((40, 100), "INVOICE 7781", fill="black", font=ImageFont.load_default(size=64))
    page.save(path, format="PDF")
    return path.read_bytes()


def _tool_message_texts(request: dict[str, Any]) -> list[str]:
    """Return the text of every ``tool`` message of a chat-completions request."""
    texts = []
    for message in request["messages"]:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        texts.append(content if isinstance(content, str) else json.dumps(content))
    return texts


def _blank_every_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every HTTP provider credential, so no vision model is available."""
    for provider in OPENAI_COMPATIBLE_PROVIDERS:
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, provider.api_key_name, "")
    for _prefix, _label, api_key_name in _NATIVE_PROVIDERS:
        if api_key_name is not None:
            monkeypatch.setattr(config_module.DEFAULT_CONFIG, api_key_name, "")


def test_text_only_model_gets_a_description_instead_of_base64(tmp_path: Path) -> None:
    """A Read of a PNG by a text-only model reaches it as words, not bytes.

    The scripted server plays the text-only model: it calls ``Read`` on the
    PNG, then ``finish``.  The second request's tool message must carry the
    vision model's description (the text in the picture, the header that
    says who described it and why), and neither the sentinel block nor the
    base64 payload.
    """
    if vision_model() is None:
        pytest.skip("no vision model is configured")
    png_path = tmp_path / "chart.png"
    png_bytes = _write_png(png_path)
    script = [
        tool_call_body("Read", {"file_path": str(png_path)}, prompt_tokens=100),
        tool_call_body("finish", {"result": "done"}, prompt_tokens=100),
    ]
    with serve(script) as (url, requests):
        KISSAgent("vision-bridge-test").run(
            model_name=TEXT_ONLY_MODEL,
            prompt_template="Read {path} and report what it shows.",
            arguments={"path": str(png_path)},
            tools=[UsefulTools().Read],
            model_config={"base_url": url, "api_key": "local"},
            max_budget=1.0,
            verbose=False,
            print_prompts=False,
        )
    assert len(requests) >= 2
    tool_texts = _tool_message_texts(requests[1])
    assert len(tool_texts) == 1, requests[1]["messages"]
    text = tool_texts[0]
    assert f"[Description of image/png attachment 1 by {vision_model()}; " in text
    assert f"the model you run on ({TEXT_ONLY_MODEL}) cannot see images]" in text
    assert "KISS" in text.upper()
    assert "42" in text
    assert "<<KISS_BINARY_ATTACHMENT" not in text
    assert base64.b64encode(png_bytes).decode("ascii")[:64] not in text
    assert "could not be described" not in text


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("claude-sonnet-5-5", True),
        ("gpt-6-luna", True),
        ("gemini-3.5-flash-lite", True),
        ("x-ai/grok-5", True),
        ("openrouter/anthropic/claude-sonnet-5.5", True),
        ("bedrock/anthropic.claude-sonnet-5-5-v1:0", True),
        ("Qwen/Qwen3-VL-8B-Instruct", True),
        ("Qwen/Qwen2.5-VL-72B-Instruct", True),
        ("openrouter/cloudflare/clef-omni", True),
        ("openrouter/deepseek/deepseek-v4-flash-vision-exp", True),
        ("nvidia/NVIDIA-Nemotron-3.5-Super-VL", True),
        ("deepseek-ai/DeepSeek-V4-Pro", False),
        ("moonshotai/Kimi-K3", False),
        ("Qwen/Qwen3-235B-A22B-Instruct", False),
        ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", False),
        ("mistralai/Mistral-Small-24B-Instruct", False),
        ("local/anything", False),
        ("devlin", False),
    ],
)
def test_model_supports_images_by_name(model_name: str, expected: bool) -> None:
    """The four vision families and the vision name segments see; the rest do not."""
    assert model_supports_images(model_name) is expected


def test_a_vision_model_keeps_its_attachments(tmp_path: Path) -> None:
    """An adapter of a model that sees returns the attachments untouched.

    Built with a dummy key and never sent a request: the split happens
    before any network call.
    """
    png_path = tmp_path / "shot.png"
    png_bytes = _write_png(png_path)
    adapter = AnthropicModel(model_name="claude-sonnet-5-5", api_key="dummy")
    assert adapter.supports_images()
    text, attachments = adapter.tool_result_text_and_attachments(
        {"result": UsefulTools().Read(str(png_path))}
    )
    assert [(a.mime_type, a.data) for a in attachments] == [("image/png", png_bytes)]
    assert "cannot see images" not in text
    assert "shot.png" in text


def test_a_cli_transport_gets_the_description(tmp_path: Path) -> None:
    """Claude Code's model sees, but its one-text-prompt transport cannot carry bytes.

    The CLI adapter therefore goes through the bridge like a text-only model:
    the tool text carries the description and no attachment is left over.
    """
    if vision_model() is None:
        pytest.skip("no vision model is configured")
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    adapter = ClaudeCodeModel("cc/opus")
    assert not adapter.supports_images()
    text, attachments = adapter.tool_result_text_and_attachments(
        {"result": UsefulTools().Read(str(png_path))}
    )
    assert attachments == []
    assert f"[Description of image/png attachment 1 by {vision_model()}; " in text
    assert "the model you run on (cc/opus) cannot see images]" in text
    assert "42" in text


def test_a_codex_native_tool_result_gets_the_description(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tool Codex calls natively over the MCP bridge goes through the bridge too.

    The native executor stripped the attachments itself and recorded the
    stripped text, so neither Codex (in the MCP reply) nor the next turn
    or a model switch (in the replayed conversation) ever got the vision
    description: a screenshot was ``[1 attachment(s) ... could not be
    shown]`` for good.  Now the reply carries the description and the
    usage line like any tool result, and the replay records that text
    as it was served (the usage line once).
    """
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    usage_line = "Steps: 3/10 (the footer the framework shows the model)"

    def read_the_png(call: dict[str, Any]) -> tuple[str, str]:
        return call["name"], UsefulTools().Read(str(png_path))

    schema = [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]
    with ScriptedAnthropicServer(_vision_turn("A white 480x240 PNG reading KISS 42.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        adapter = CodexModel("codex/default")
        adapter.usage_info_for_messages = usage_line
        with adapter.native_tools_exposed(schema, read_the_png):
            bridge = adapter._tool_bridge
            assert bridge is not None
            reply = _reply(_call(bridge, "Read"))
        vision_requests = vision.requests
    assert len(vision_requests) == 1
    served = reply["text"]
    assert "could not be shown" not in served
    assert f"[Description of image/png attachment 1 by {LOCAL_VISION_MODEL}; " in served
    assert "the model you run on (codex/default) cannot see images]" in served
    assert "KISS 42" in served
    assert served.count(usage_line) == 1
    tool_message = adapter.conversation[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == served
    assert tool_message["tool_call_id"] == adapter.conversation[-2]["tool_calls"][0]["id"]


def test_a_stop_during_the_description_of_a_codex_native_result_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Stop landing on the bridge's thread while the picture is described is a reply.

    The agent's executor binds the task's stop event to the bridge's
    thread and turns the ``KeyboardInterrupt`` a Stop raises into an
    error Codex is told about; the description runs after the executor
    returned, on the same thread, under the same binding.  Left to
    escape, the interrupt killed the handler thread: Codex got no reply
    and the call was never recorded.
    """
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    stop = threading.Event()
    stop.set()

    def read_the_png_under_a_stop(call: dict[str, Any]) -> tuple[str, str]:
        stop_signal.set_thread_stop_event(stop)
        return call["name"], UsefulTools().Read(str(png_path))

    schema = [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]
    with ScriptedAnthropicServer(_vision_turn("never read")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        adapter = CodexModel("codex/default")
        with adapter.native_tools_exposed(schema, read_the_png_under_a_stop):
            bridge = adapter._tool_bridge
            assert bridge is not None
            reply = _reply(_call(bridge, "Read"))
    stopped = "KISSError: KISS Error: Tool Read stopped: Agent stop requested"
    assert reply == {"error": stopped}
    assert adapter.conversation[-1] == {
        "role": "tool",
        "tool_call_id": adapter.conversation[-2]["tool_calls"][0]["id"],
        "content": stopped,
    }


class _Credited(dict[str, tuple[float, int]]):
    """The (cost, tokens) the bridge credited, by call key: a retried credit counts once."""

    def credit(self, cost: float, tokens: int, key: str, model: str) -> None:
        self.setdefault(key, (cost, tokens))


def test_no_vision_model_configured_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With every provider key blank the text says the picture could not be described."""
    _blank_every_provider_key(monkeypatch)
    assert vision_model() is None
    attachments = [
        Attachment(_write_png(tmp_path / "shot.png"), "image/png"),
        Attachment(_write_pdf(tmp_path / "scan.pdf"), "application/pdf"),
    ]
    credited = _Credited()
    described = describe_attachments(
        attachments, TEXT_ONLY_MODEL, "Read shot.png", credit=credited.credit
    )
    assert credited == {}  # no call was made, so none is billed
    assert described.split("\n\n") == [
        f"[{mime} attachment {index} could not be described: no vision model is "
        f"configured; the model you run on ({TEXT_ONLY_MODEL}) cannot see images, "
        "use a vision model or an OCR tool]"
        for index, mime in ((1, "image/png"), (2, "application/pdf"))
    ]


def test_a_pdf_is_described_too(tmp_path: Path) -> None:
    """A PDF goes to the vision model as a document and comes back as words."""
    if vision_model() is None:
        pytest.skip("no vision model is configured")
    pdf_bytes = _write_pdf(tmp_path / "scan.pdf")
    credited = _Credited()
    text = describe_attachments(
        [Attachment(pdf_bytes, "application/pdf")], TEXT_ONLY_MODEL, credit=credited.credit
    )
    assert text.startswith(f"[Description of application/pdf attachment 1 by {vision_model()}; ")
    assert "INVOICE" in text.upper()
    assert "7781" in text
    assert "could not be described" not in text
    ((cost, tokens),) = credited.values()  # one call
    assert cost > 0
    assert tokens > 0


def test_a_failed_vision_call_becomes_a_failure_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A vision model that answers with an error leaves a line, not an exception.

    Only the Anthropic key is set, to a value the API rejects (a 401 in one
    round trip, no retry); offline the connection error takes the same path.
    """
    _blank_every_provider_key(monkeypatch)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "sk-ant-not-a-key")
    assert vision_model() == "claude-sonnet-5-5"
    png_path = tmp_path / "shot.png"
    credited = _Credited()
    text = describe_attachments(
        [Attachment(_write_png(png_path), "image/png")], TEXT_ONLY_MODEL, credit=credited.credit
    )
    assert text.startswith(
        "[image/png attachment 1 could not be described: claude-sonnet-5-5 failed: "
    )
    assert text.endswith(
        f"; the model you run on ({TEXT_ONLY_MODEL}) cannot see images, "
        "use a vision model or an OCR tool]"
    )
    # The call was made (and counted); a rejected request is billed nothing.
    assert list(credited.values()) == [(0.0, 0)]


def test_a_text_only_adapter_routes_attachments_through_the_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The split for a text-only model empties the attachments and keeps the usage line last.

    Built without a vision model so the bridge's failure line stands in
    for the description; the usage line the framework appends must still
    close the text.
    """
    _blank_every_provider_key(monkeypatch)
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    adapter = OpenAICompatibleModel(
        model_name=TEXT_ONLY_MODEL, api_key="dummy", base_url="http://127.0.0.1:9/v1"
    )
    adapter.usage_info_for_messages = "Steps: 1/10"
    assert not adapter.supports_images()
    text, attachments = adapter.tool_result_text_and_attachments(
        {"result": UsefulTools().Read(str(png_path))}
    )
    assert attachments == []
    assert "<<KISS_BINARY_ATTACHMENT" not in text
    assert "[image/png attachment 1 could not be described: no vision model is configured" in text
    assert text.endswith("\n\nSteps: 1/10")
    assert text.index("shot.png") < text.index("could not be described")
    assert adapter.helper_usage()[1:] == (0.0, 0)


def _run_text_only_agent(
    png_path: Path, script: list[bytes], max_budget: float
) -> tuple[KISSAgent, list[dict[str, Any]], Exception | None]:
    """Run a text-only agent that Reads *png_path* against the scripted chat server.

    Returns:
        The agent, the chat requests the server received, and the error
        the run raised (``None`` when it finished).
    """
    agent = KISSAgent("vision-bridge-billing")
    error: Exception | None = None
    with serve(script) as (url, requests):
        try:
            agent.run(
                model_name=TEXT_ONLY_MODEL,
                prompt_template="Read {path} and report what it shows.",
                arguments={"path": str(png_path)},
                tools=[UsefulTools().Read],
                model_config={"base_url": url, "api_key": "local"},
                max_budget=max_budget,
                max_steps=5,
                verbose=False,
                print_prompts=False,
            )
        except Exception as raised:
            error = raised
    return agent, requests, error


def test_the_bridge_call_is_billed_to_the_calling_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """What the vision model charged joins the calling agent's totals and its next footer.

    The text-only model (the scripted chat server) Reads the PNG, then a
    text file, then finishes; the scripted vision endpoint describes the
    PNG for ``VISION_INPUT_TOKENS`` / ``VISION_OUTPUT_TOKENS``.  Before
    the fix the agent's ``budget_used`` held the three chat calls only
    (review of 2026-10-31).
    """
    png_path = tmp_path / "shot.png"
    png_bytes = _write_png(png_path)
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("plain notes\n")
    script = [
        tool_call_body("Read", {"file_path": str(png_path)}, prompt_tokens=100),
        tool_call_body("Read", {"file_path": str(notes_path)}, prompt_tokens=100),
        tool_call_body("finish", {"result": "done"}, prompt_tokens=100),
    ]
    chat_call_cost = calculate_cost(TEXT_ONLY_MODEL, 100, 100)
    with ScriptedAnthropicServer(_vision_turn("A white 480x240 PNG reading KISS 42.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        agent, requests, error = _run_text_only_agent(png_path, script, max_budget=1.0)
        vision_requests = vision.requests
    assert error is None
    assert len(requests) == 3
    # The bridge made one call and sent the picture itself.
    assert len(vision_requests) == 1
    sent = json.dumps(vision_requests[0])
    assert base64.b64encode(png_bytes).decode("ascii") in sent
    assert '"media_type": "image/png"' in sent
    # Totals: three chat calls plus the description, at each model's own rate.
    assert VISION_COST > 0
    assert agent.budget_used == pytest.approx(3 * chat_call_cost + VISION_COST)
    assert agent.total_tokens_used == 3 * 200 + VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS
    assert agent.step_count == 3  # the description is no step of the agent
    # The footer under the described result was written before the
    # description was made; the next one (under the notes, read by the
    # third call) carries the description's cost and tokens.
    described_result = _tool_message_texts(requests[1])[0]
    assert "KISS 42" in described_result
    assert f"Budget: ${chat_call_cost:.4f}/$1.00" in described_result
    notes_result = _tool_message_texts(requests[2])[-1]
    assert "plain notes" in notes_result
    assert f"Budget: ${2 * chat_call_cost + VISION_COST:.4f}/$1.00" in notes_result
    assert f"Total tokens: {400 + VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS:,}" in notes_result


def test_the_bridge_spend_counts_against_the_budget_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cap the description pushes the agent over stops it before its next model call.

    The cap sits above the first chat call and below that call plus the
    description: the reviewer's reproduction (cap 0.01, bridge spend
    unbilled) ran on to its second call.
    """
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    script = [
        tool_call_body("Read", {"file_path": str(png_path)}, prompt_tokens=100),
        tool_call_body("finish", {"result": "done"}, prompt_tokens=100),
    ]
    chat_call_cost = calculate_cost(TEXT_ONLY_MODEL, 100, 100)
    cap = chat_call_cost + VISION_COST / 2
    with ScriptedAnthropicServer(_vision_turn("A white PNG reading KISS 42.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        agent, requests, error = _run_text_only_agent(png_path, script, max_budget=cap)
        assert len(vision.requests) == 1
    assert isinstance(error, BudgetExceededError), error
    assert len(requests) == 1, "the agent made another model call past its cap"
    assert agent.budget_used == pytest.approx(chat_call_cost + VISION_COST)
    assert agent.budget_used > cap


def test_pending_bridge_spend_moves_with_the_conversation_on_fail_over(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The helper spend record follows a model hand-off; a reset starts a new epoch.

    The agent folds the spend before its next limit check; a fail-over
    in between hands the conversation (and the spend record, epoch
    included) to the new model, so nothing is lost; a new conversation
    starts clean.
    """
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    old = OpenAICompatibleModel(
        model_name=TEXT_ONLY_MODEL, api_key="dummy", base_url="http://127.0.0.1:9/v1"
    )
    with ScriptedAnthropicServer(_vision_turn("A white PNG reading KISS 42.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        text, attachments = old.tool_result_text_and_attachments(
            {"result": UsefulTools().Read(str(png_path))}
        )
    assert attachments == []
    assert "KISS 42" in text
    new = OpenAICompatibleModel(
        model_name=TEXT_ONLY_MODEL, api_key="dummy", base_url="http://127.0.0.1:9/v2"
    )
    hand_off_conversation(old, new)
    assert new.helper_usage() == old.helper_usage()  # the same record, epoch included
    epoch, cost, tokens = new.helper_usage()
    assert cost == pytest.approx(VISION_COST)
    assert tokens == VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS
    assert new.helper_usage() == (epoch, cost, tokens)  # reading never consumes
    with ScriptedAnthropicServer(_vision_turn("A white PNG reading KISS 42.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        new.tool_result_text_and_attachments({"result": UsefulTools().Read(str(png_path))})
    assert new.helper_usage()[1] == pytest.approx(2 * VISION_COST)
    new.reset_conversation()
    assert new.helper_usage()[0] is not epoch
    assert new.helper_usage()[1:] == (0.0, 0)


def test_a_description_made_in_the_last_step_is_billed_before_the_run_ends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The spend of a description made in the last allowed step reaches the saved totals.

    The fold ran before each limit check only, so a description made
    in the step the step limit ends (no next check) was never billed:
    ``budget_used`` and the trajectory held the chat calls only, under
    a cap the description had crossed (review of 2026-10-31, finding
    4).  The run now folds the pending spend on every exit.
    """
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    script = [tool_call_body("Read", {"file_path": str(png_path)}, prompt_tokens=100)]
    chat_call_cost = calculate_cost(TEXT_ONLY_MODEL, 100, 100)
    cap = chat_call_cost + VISION_COST / 2
    agent = KISSAgent("vision-bridge-last-step")
    with ScriptedAnthropicServer(_vision_turn("A white PNG reading KISS 42.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        with serve(script) as (url, requests):
            with pytest.raises(KISSError, match="exceeded 1 steps"):
                agent.run(
                    model_name=TEXT_ONLY_MODEL,
                    prompt_template="Read {path} and report what it shows.",
                    arguments={"path": str(png_path)},
                    tools=[UsefulTools().Read],
                    model_config={"base_url": url, "api_key": "local"},
                    max_budget=cap,
                    max_steps=1,
                    verbose=False,
                    print_prompts=False,
                )
        assert len(vision.requests) == 1 and len(requests) == 1
    totals = agent._usage_totals
    assert totals.helper_mark == agent.model.helper_usage(), "nothing is left unbilled"
    assert agent.budget_used == pytest.approx(chat_call_cost + VISION_COST)
    assert agent.total_tokens_used == 200 + VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS
    assert agent.budget_used > cap
    saved = yaml.safe_load(agent.get_trajectory_path().read_text(encoding="utf-8"))
    assert saved["budget_used"] == pytest.approx(agent.budget_used)
    assert saved["tokens_used"] == agent.total_tokens_used


def _stop_after_billing(monkeypatch: pytest.MonkeyPatch, name: str, stop_on: int) -> None:
    """Make the *stop_on*-th (1-based) run of a KISSAgent called *name* end in a stop.

    The real call runs and is billed, then the stop lands before ``run``
    returns: the window a user's Stop (an injected ``KeyboardInterrupt``)
    hits while the helper is answering.
    """
    real_run = KISSAgent.run
    calls = {"n": 0}

    def run(self: KISSAgent, *args: Any, **kwargs: Any) -> str:
        result: str = real_run(self, *args, **kwargs)
        if self.name == name:
            calls["n"] += 1
            if calls["n"] == stop_on:
                raise KeyboardInterrupt("stop")
        return result

    monkeypatch.setattr(KISSAgent, "run", run)


def test_a_stop_during_a_description_keeps_every_billed_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stop that ends the second description loses neither call's spend.

    ``_describe_one`` caught ``Exception`` only and the spend travelled
    in the returned ``Described``, so a stop dropped the interrupted
    call's spend and the earlier completed one's (review of 2026-11-01,
    finding 1).  Each call is now credited as it ends, and the stop
    still propagates.
    """
    credited: dict[str, tuple[float, int]] = {}  # by key: a retried credit counts once

    def credit(cost: float, tokens: int, key: str, model: str) -> None:
        credited.setdefault(key, (cost, tokens))

    png = Attachment(_write_png(tmp_path / "shot.png"), "image/png")
    with ScriptedAnthropicServer(_vision_turn("A white PNG.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        _stop_after_billing(monkeypatch, "Vision bridge", stop_on=2)
        with pytest.raises(KeyboardInterrupt):
            describe_attachments([png, png], TEXT_ONLY_MODEL, credit=credit)
        assert len(vision.requests) == 2
    assert len(credited) == 2
    assert sum(c for c, _ in credited.values()) == pytest.approx(2 * VISION_COST)
    assert sum(t for _, t in credited.values()) == 2 * (VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS)


def test_a_stop_during_a_description_still_bills_the_calling_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stopped description's spend reaches the agent's (and the trajectory's) totals."""
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    script = [tool_call_body("Read", {"file_path": str(png_path)}, prompt_tokens=100)]
    chat_call_cost = calculate_cost(TEXT_ONLY_MODEL, 100, 100)
    agent = KISSAgent("vision-bridge-stop")
    with ScriptedAnthropicServer(_vision_turn("A white PNG reading KISS 42.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        _stop_after_billing(monkeypatch, "Vision bridge", stop_on=1)
        with serve(script) as (url, _requests):
            with pytest.raises(KeyboardInterrupt):
                agent.run(
                    model_name=TEXT_ONLY_MODEL,
                    prompt_template="Read {path} and report what it shows.",
                    arguments={"path": str(png_path)},
                    tools=[UsefulTools().Read],
                    model_config={"base_url": url, "api_key": "local"},
                    max_budget=1.0,
                    max_steps=5,
                    verbose=False,
                    print_prompts=False,
                )
    assert agent.budget_used == pytest.approx(chat_call_cost + VISION_COST)
    assert agent.total_tokens_used == 200 + VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS


def test_the_helper_fold_bills_each_dollar_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Folding is idempotent, bills only new spend, and never re-bills an older run's.

    ``take_helper_usage`` cleared the model's figures before the fold
    stored the totals, so a stop between the two stores lost the spend
    (review of 2026-11-01, finding 3).  The model's record is now
    cumulative and never consumed; the totals carry the mark of what
    they include, advanced in the same single store.
    """
    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    agent = KISSAgent("helper-fold")
    agent.model = OpenAICompatibleModel(
        model_name=TEXT_ONLY_MODEL, api_key="dummy", base_url="http://127.0.0.1:9/v1"
    )
    agent.reset_usage()
    with ScriptedAnthropicServer(_vision_turn("A PNG.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        read = {"result": UsefulTools().Read(str(png_path))}
        agent.model.tool_result_text_and_attachments(read)
        agent._bill_helper_usage()
        agent._bill_helper_usage()  # a retried fold adds nothing
        assert agent.budget_used == pytest.approx(VISION_COST)
        agent.model.tool_result_text_and_attachments(read)
        agent._bill_helper_usage()
    assert agent.budget_used == pytest.approx(2 * VISION_COST)
    assert agent.total_tokens_used == 2 * (VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS)
    # A new run: the spend the model still holds is the old run's.
    agent.reset_usage()
    agent._bill_helper_usage()
    assert agent.budget_used == 0.0
    agent.model.reset_conversation()
    agent._bill_helper_usage()
    assert (agent.budget_used, agent.total_tokens_used) == (0.0, 0)


@pytest.mark.parametrize("stop_after_store", [False, True])
def test_a_stop_inside_the_credit_is_retried_and_counted_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stop_after_store: bool
) -> None:
    """A stop landing inside the call's own credit still credits it exactly once.

    The credit runs in a ``finally``; a stop there (before or after its
    store) used to lose the completed call's spend (review of
    2026-11-01, round 2, finding 1).  The bridge credits again with the
    same key before the stop propagates, and the model adds a key once.
    """
    png = Attachment(_write_png(tmp_path / "shot.png"), "image/png")
    model = OpenAICompatibleModel(
        model_name=TEXT_ONLY_MODEL, api_key="dummy", base_url="http://127.0.0.1:9/v1"
    )
    calls: list[str] = []

    def credit(cost: float, tokens: int, key: str, helper: str) -> None:
        calls.append(key)
        if len(calls) == 1 and not stop_after_store:
            raise KeyboardInterrupt("stop before the store")
        model._credit_helper_usage(cost, tokens, key, helper)
        if len(calls) == 1:
            raise KeyboardInterrupt("stop after the store")

    with ScriptedAnthropicServer(_vision_turn("A white PNG.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        with pytest.raises(KeyboardInterrupt):
            describe_attachments([png], TEXT_ONLY_MODEL, credit=credit)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert model.helper_usage()[1:] == (
        pytest.approx(VISION_COST),
        VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS,
    )
    # Listed once under the vision model, for the task's split by model.
    assert [(key, helper) for key, helper, _, _ in model.helper_calls()] == [
        (calls[0], vision_model())
    ]


def test_the_session_bank_folds_helper_spend_a_stop_left_unfolded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Helper spend whose terminal fold a stop interrupted still reaches the task totals.

    A stop inside ``KISSAgent.run``'s final fold left the description's
    spend on the model only, and the Relentless session bank banked the
    executor without it (review of 2026-11-01, round 2, finding 3).
    The bank folds first (``_fold_helper_spend``, idempotent).
    """
    from kiss.agents.sorcar import relentless_agent

    png_path = tmp_path / "shot.png"
    _write_png(png_path)
    agent = KISSAgent("helper-bank")
    agent.model = OpenAICompatibleModel(
        model_name=TEXT_ONLY_MODEL, api_key="dummy", base_url="http://127.0.0.1:9/v1"
    )
    agent.reset_usage()
    with ScriptedAnthropicServer(_vision_turn("A PNG.")) as vision:
        _route_the_bridge_to(monkeypatch, vision)
        agent.model.tool_result_text_and_attachments({"result": UsefulTools().Read(str(png_path))})
    assert agent.budget_used == 0.0  # the run's own fold was interrupted
    relentless_agent._fold_helper_spend(agent)
    relentless_agent._fold_helper_spend(agent)  # the stop retry adds nothing
    assert relentless_agent._session_usage(agent)[:2] == (
        pytest.approx(VISION_COST),
        VISION_INPUT_TOKENS + VISION_OUTPUT_TOKENS,
    )
