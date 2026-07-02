# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end offline regression tests for the core models simplification.

Pins the observable behavior of ``kiss.core.models.model``,
``kiss.core.models.model_info`` and ``kiss.core.models.__init__`` before and
after simplification.  No mocks, no network: every test calls the real code
with real values.
"""

import json
from typing import Any

import pytest

import kiss.core.models as models_pkg
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    Attachment,
    Model,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
    _strip_text_based_tool_calls,
    encode_binary_attachment,
    flatten_content_to_text,
    parse_binary_attachments,
    responses_items_to_chat_messages,
)
from kiss.core.models.model_info import (
    MODEL_INFO,
    _match_openai_compatible_provider,
    _strip_provider_prefix,
    calculate_cost,
    get_default_model,
    get_fast_model,
    get_flaky_reason,
    get_max_context_length,
    get_model_provider,
    get_most_expensive_model,
    is_model_flaky,
    model,
    openai_compatible_provider_for_base_url,
    rank_model_suggestions,
)


class _ConcreteModel(Model):
    """Minimal concrete Model used to exercise base-class helpers."""

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Store the prompt as the first user message."""
        self.conversation = [{"role": "user", "content": prompt}]

    def generate(self) -> tuple[str, Any]:
        """Not used in these offline tests."""
        raise NotImplementedError

    def generate_and_process_with_tools(self, function_map, tools_schema=None):
        """Not used in these offline tests."""
        raise NotImplementedError

    def extract_input_output_token_counts_from_response(self, response):
        """Not used in these offline tests."""
        raise NotImplementedError

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Not used in these offline tests."""
        raise NotImplementedError


def _sample_tool(path: str, count: int = 3, tags: list[str] | None = None) -> str:
    """Read a file.

    Args:
        path: The file path to read.
        count (int): How many lines.
        tags: Optional tag list.
    """
    return path


# ---------------------------------------------------------------------------
# model_info: routing
# ---------------------------------------------------------------------------


def test_model_routing_by_prefix() -> None:
    """model() routes each name prefix to the documented provider class."""
    assert type(model("claude-test-model")).__name__ == "AnthropicModel"
    assert type(model("gemini-test-model")).__name__ == "GeminiModel"
    assert type(model("text-embedding-004")).__name__ == "GeminiModel"
    for name in ("gpt-4o", "openrouter/foo/bar", "glm-4", "kimi-k2", "moonshot-v1",
                 "meta-llama/Llama-3", "Qwen/qwen-x", "o3-mini"):
        assert type(model(name)).__name__ == "OpenAICompatibleModel", name


def test_model_base_url_override_bypasses_routing() -> None:
    """A base_url in model_config forces an OpenAICompatibleModel."""
    m = model("claude-test-model", model_config={"base_url": "http://localhost:1/v1",
                                                 "api_key": "k", "extra": 1})
    from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

    assert isinstance(m, OpenAICompatibleModel)
    assert m.base_url == "http://localhost:1/v1"
    assert m.api_key == "k"
    assert m.model_config == {"extra": 1}


def test_model_unknown_name_raises() -> None:
    """Unrecognized model names raise KISSError."""
    with pytest.raises(KISSError, match="Unknown model name"):
        model("totally-unknown-model-xyz")


def test_strip_provider_prefix() -> None:
    """Harbor-style provider prefixes are stripped only for redundant routes."""
    assert _strip_provider_prefix("openai/gpt-5.4") == "gpt-5.4"
    assert _strip_provider_prefix("anthropic/claude-opus-4-6") == "claude-opus-4-6"
    assert _strip_provider_prefix("google/gemini-2.5-pro") == "gemini-2.5-pro"
    # KISS-owned prefixes are NOT stripped.
    assert _strip_provider_prefix("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert _strip_provider_prefix("openrouter/openai/gpt-4o") == "openrouter/openai/gpt-4o"
    assert _strip_provider_prefix("gpt-4o") == "gpt-4o"
    assert _strip_provider_prefix("anthropic/other") == "anthropic/other"


def _provider_name(provider: Any) -> str:
    """Return the provider's name, failing the test when provider is None."""
    assert provider is not None
    return str(provider.name)


def test_provider_registry_lookup() -> None:
    """Provider registry routes by prefix and by base_url host substring."""
    assert _provider_name(_match_openai_compatible_provider("openrouter/x")) == "openrouter"
    assert _provider_name(_match_openai_compatible_provider("gpt-4o")) == "openai"
    assert _provider_name(_match_openai_compatible_provider("openai/gpt-oss-120b")) == "together"
    assert _match_openai_compatible_provider("text-embedding-004") is None
    assert _match_openai_compatible_provider("codex/default") is None
    assert _match_openai_compatible_provider("claude-x") is None
    assert _provider_name(
        openai_compatible_provider_for_base_url("https://openrouter.ai/api/v1")
    ) == "openrouter"
    assert openai_compatible_provider_for_base_url("http://my-gateway.local/v1") is None


def test_get_model_provider_labels() -> None:
    """get_model_provider mirrors the model() dispatch order."""
    assert get_model_provider("cc/opus") == "Claude Code CLI"
    assert get_model_provider("codex/default") == "Codex CLI"
    assert get_model_provider("openrouter/x/y") == "OpenRouter"
    assert get_model_provider("claude-x") == "Anthropic"
    assert get_model_provider("gemini-x") == "Gemini"
    assert get_model_provider("text-embedding-004") == "Gemini"
    assert get_model_provider("glm-5") == "Z.AI"
    assert get_model_provider("kimi-k2") == "Moonshot"
    assert get_model_provider("gpt-4o") == "OpenAI"
    assert get_model_provider("openai/gpt-oss-120b") == "Together"
    assert get_model_provider("meta-llama/Llama-3") == "Together"
    assert get_model_provider("something-else") == "Unknown"


# ---------------------------------------------------------------------------
# model_info: pricing / context lengths
# ---------------------------------------------------------------------------


def test_calculate_cost_basic_and_unknown() -> None:
    """Cost math is (tokens * price) / 1M; unknown models allow only zero usage."""
    name = next(
        n for n, i in MODEL_INFO.items()
        if n.startswith("claude-") and i.is_generation_supported
    )
    info = MODEL_INFO[name]
    expected = (1000 * info.input_price_per_1M + 500 * info.output_price_per_1M) / 1e6
    assert calculate_cost(name, 1000, 500) == pytest.approx(expected)
    # Provider-prefixed lookup falls back to the stripped name.
    assert calculate_cost(f"anthropic/{name}", 1000, 500) == pytest.approx(expected)
    assert calculate_cost("no-such-model", 0, 0) == 0.0
    with pytest.raises(KISSError, match="unknown model"):
        calculate_cost("no-such-model", 1, 0)


def test_calculate_cost_cache_pricing() -> None:
    """Cache read/write tokens are billed at the ModelInfo cache prices."""
    name = next(
        n for n, i in MODEL_INFO.items()
        if n.startswith("claude-") and i.is_generation_supported
        and i.cache_read_price_per_1M is not None
    )
    info = MODEL_INFO[name]
    cr = info.cache_read_price_per_1M
    cw = info.cache_write_price_per_1M
    cw1h = info.cache_write_1h_price_per_1M
    assert cr is not None and cw is not None and cw1h is not None
    expected = (
        100 * info.input_price_per_1M
        + 10 * info.output_price_per_1M
        + 1000 * cr
        + 200 * cw
        + 50 * cw1h
    ) / 1e6
    assert calculate_cost(name, 100, 10, 1000, 200, 50) == pytest.approx(expected)


def test_anthropic_and_openai_cache_multipliers() -> None:
    """Bundled entries get the documented provider cache-pricing multipliers."""
    from kiss.core.models.model_info import PACKAGE_MODEL_INFO_PATH

    raw = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())
    claude = next(
        n for n, e in raw.items()
        if n.startswith("claude-") and e.get("gen", True)
        and "cache_read_price_per_1M" not in e
    )
    info = MODEL_INFO[claude]
    inp = info.input_price_per_1M
    assert info.cache_read_price_per_1M == pytest.approx(inp * 0.1)
    assert info.cache_write_price_per_1M == pytest.approx(inp * 1.25)
    assert info.cache_write_1h_price_per_1M == pytest.approx(inp * 2.0)
    gpt = next(
        n for n, e in raw.items()
        if n.startswith("gpt-4o") and e.get("gen", True)
        and "cache_read_price_per_1M" not in e
    )
    ginfo = MODEL_INFO[gpt]
    assert ginfo.cache_read_price_per_1M == pytest.approx(ginfo.input_price_per_1M * 0.5)
    assert ginfo.cache_write_price_per_1M == 0.0


def test_long_context_pricing_gpt55() -> None:
    """gpt-5.5 (non-pro) switches to long-context prices above 200k tokens."""
    name = next(
        (n for n in MODEL_INFO
         if n.startswith("gpt-5.5") and "-pro" not in n
         and MODEL_INFO[n].is_generation_supported),
        None,
    )
    if name is None:
        pytest.skip("no gpt-5.5 entry in MODEL_INFO")
    cost = calculate_cost(name, 300_000, 1000)
    assert cost == pytest.approx((300_000 * 10.00 + 1000 * 45.00) / 1e6)


def test_get_max_context_length() -> None:
    """Context lengths resolve directly and via provider-prefix stripping."""
    name = next(n for n in MODEL_INFO if n.startswith("gpt-"))
    assert get_max_context_length(name) == MODEL_INFO[name].context_length
    assert get_max_context_length(f"openai/{name}") == MODEL_INFO[name].context_length
    with pytest.raises(KeyError, match="not found in MODEL_INFO"):
        get_max_context_length("no-such-model")


def test_flaky_model_helpers() -> None:
    """Flaky-model markers round-trip through the helper functions."""
    assert is_model_flaky("openrouter/baidu/ernie-4.5-21b-a3b")
    assert get_flaky_reason("openrouter/baidu/ernie-4.5-21b-a3b") != ""
    assert not is_model_flaky("claude-x")
    assert get_flaky_reason("claude-x") == ""


def test_default_fast_and_expensive_model_pickers() -> None:
    """Model pickers return a string (a model name or 'No model')."""
    for picker in (get_fast_model, get_default_model):
        picked = picker()
        assert isinstance(picked, str) and picked
        if picked != "No model":
            assert picked in MODEL_INFO or picked.startswith(("cc/", "codex/"))
    expensive = get_most_expensive_model()
    assert isinstance(expensive, str)
    if expensive:
        assert MODEL_INFO[expensive].is_function_calling_supported


def test_rank_model_suggestions() -> None:
    """Prefix matches come before substring matches, each group sorted."""
    names = ["gpt-4o", "claude-opus", "openrouter/gpt-x", "claude-haiku"]
    assert rank_model_suggestions("", names) == names
    assert rank_model_suggestions("claude", names) == ["claude-haiku", "claude-opus"]
    assert rank_model_suggestions("gpt", names) == ["gpt-4o", "openrouter/gpt-x"]
    assert rank_model_suggestions("zzz", names) == []


# ---------------------------------------------------------------------------
# package __init__: lazy imports
# ---------------------------------------------------------------------------


def test_lazy_class_imports() -> None:
    """Every advertised model class is importable lazily; bad names raise."""
    for cls_name in ("AnthropicModel", "ClaudeCodeModel", "CodexModel",
                     "OpenAICompatibleModel", "OpenAICompatibleModel2", "GeminiModel"):
        cls = getattr(models_pkg, cls_name)
        assert cls is not None and cls.__name__ == cls_name
    with pytest.raises(AttributeError):
        models_pkg.NoSuchModelClass  # noqa: B018
    assert issubclass(models_pkg.AnthropicModel, models_pkg.Model)


# ---------------------------------------------------------------------------
# model.py: text-based tool calling helpers
# ---------------------------------------------------------------------------


def test_parse_text_based_tool_calls() -> None:
    """Bare, fenced, nested and duplicated tool_calls JSON all parse correctly."""
    content = (
        'thinking...\n{"tool_calls": [{"name": "a", "arguments": {"x": [1, {"y": "}"}]}}]}\n'
        '```json\n{"tool_calls": [{"name": "a", "arguments": {"x": [1, {"y": "}"}]}}, '
        '{"name": "b", "arguments": {}}]}\n```'
    )
    calls = _parse_text_based_tool_calls(content)
    assert [(c["name"], c["arguments"]) for c in calls] == [
        ("a", {"x": [1, {"y": "}"}]}), ("b", {}),
    ]
    assert all(c["id"].startswith("call_") for c in calls)
    assert _parse_text_based_tool_calls("no json here {broken") == []


def test_strip_text_based_tool_calls() -> None:
    """Tool-call JSON and leftover empty fences are removed; prose is kept."""
    content = 'Before.\n```json\n{"tool_calls": [{"name": "a", "arguments": {}}]}\n```\nAfter.'
    assert _strip_text_based_tool_calls(content) == "Before.\n\nAfter."
    assert _strip_text_based_tool_calls('{"tool_calls": [{"name": "a", "arguments": {}}]}') == ""
    assert _strip_text_based_tool_calls("just prose {not json") == "just prose {not json"


def test_build_text_based_tools_prompt() -> None:
    """The tools prompt lists each tool with its params and first doc line."""
    assert _build_text_based_tools_prompt({}) == ""
    prompt = _build_text_based_tools_prompt({"_sample_tool": _sample_tool})
    assert "**_sample_tool**: Read a file." in prompt
    assert "- path (str)" in prompt
    assert "- count (int)" in prompt
    assert '"tool_calls"' in prompt


# ---------------------------------------------------------------------------
# model.py: schema building helpers
# ---------------------------------------------------------------------------


def test_function_to_openai_tool_schema() -> None:
    """Signatures + docstrings convert to a full OpenAI tool schema."""
    m = _ConcreteModel("m")
    schema = m._build_openai_tools_schema({"_sample_tool": _sample_tool})
    assert len(schema) == 1
    fn = schema[0]["function"]
    assert schema[0]["type"] == "function"
    assert fn["name"] == "_sample_tool"
    assert fn["description"] == "Read a file."
    props = fn["parameters"]["properties"]
    assert props["path"] == {"type": "string", "description": "The file path to read."}
    assert props["count"] == {"type": "integer", "description": "How many lines."}
    assert props["tags"] == {
        "type": "array", "items": {"type": "string"}, "description": "Optional tag list.",
    }
    assert fn["parameters"]["required"] == ["path"]
    # Pre-built schemas pass through unchanged.
    assert m._resolve_openai_tools_schema({}, schema) is schema
    assert m._resolve_openai_tools_schema({"_sample_tool": _sample_tool}, None) == schema


def test_python_type_to_json_schema_variants() -> None:
    """Type-annotation conversion covers unions, containers and fallbacks."""
    m = _ConcreteModel("m")
    conv = m._python_type_to_json_schema
    assert conv(str) == {"type": "string"}
    assert conv(bool) == {"type": "boolean"}
    assert conv(float) == {"type": "number"}
    assert conv(int | None) == {"type": "integer"}
    assert conv(int | str) == {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert conv(list) == {"type": "string"}  # bare list has no origin -> fallback
    assert conv(list[int]) == {"type": "array", "items": {"type": "integer"}}
    assert conv(dict[str, int]) == {"type": "object"}
    assert conv(object) == {"type": "string"}


# ---------------------------------------------------------------------------
# model.py: conversation helpers
# ---------------------------------------------------------------------------


def test_add_message_and_usage_info() -> None:
    """Usage info is appended to user messages and tool results only."""
    m = _ConcreteModel("m")
    m.add_message_to_conversation("assistant", "hi")
    assert m.conversation[-1] == {"role": "assistant", "content": "hi"}
    m.set_usage_info_for_messages("[usage: 42]")
    m.add_message_to_conversation("user", "hello")
    assert m.conversation[-1]["content"] == "hello\n\n[usage: 42]"
    m.reset_conversation()
    assert m.conversation == [] and m.usage_info_for_messages == ""


def test_tool_call_id_matching_and_results() -> None:
    """Tool results attach to the last assistant message's tool_call ids."""
    m = _ConcreteModel("m")
    m.initialize("go")
    m.add_message_to_conversation("assistant", "calling")
    m._replace_last_assistant_with_tool_calls(
        "calling",
        [{"id": "id1", "name": "f", "arguments": {"a": 1}},
         {"id": "id2", "name": "g", "arguments": {}}],
    )
    last = m.conversation[-1]
    assert last["role"] == "assistant" and last["content"] == "calling"
    assert [tc["id"] for tc in last["tool_calls"]] == ["id1", "id2"]
    assert json.loads(last["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
    assert m._find_tool_call_ids_from_last_assistant() == [("f", "id1"), ("g", "id2")]
    m.add_function_results_to_conversation_and_return(
        [("f", {"result": "r1"}), ("g", {"result": "r2"}), ("h", {"result": "r3"})]
    )
    tools = m.conversation[-3:]
    assert [(t["role"], t["tool_call_id"], t["content"]) for t in tools] == [
        ("tool", "id1", "r1"), ("tool", "id2", "r2"), ("tool", "call_h_2", "r3"),
    ]


def test_find_tool_calls_from_responses_items() -> None:
    """Trailing Responses-API function_call items are recognized, answered ones skipped."""
    m = _ConcreteModel("m")
    m.conversation = [
        {"role": "user", "content": "go"},
        {"type": "reasoning", "summary": []},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
    ]
    assert m._find_tool_call_ids_from_last_assistant() == [("g", "c2")]
    # Anthropic-style content-block tool_use ids are also found.
    m.conversation = [
        {"role": "assistant", "content": [
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
        ]},
    ]
    assert m._find_tool_call_ids_from_last_assistant() == [("f", "t1")]
    m.conversation = [{"role": "assistant", "content": "plain"}]
    assert m._find_tool_call_ids_from_last_assistant() == []


def test_responses_items_to_chat_messages() -> None:
    """Responses-API items convert to Chat-Completions messages losslessly."""
    conv = [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "hello"}]},
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "calling"}]},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a": 1}'},
        {"type": "function_call_output", "call_id": "c1", "output": "res"},
        {"role": "assistant", "content": "already chat format"},
    ]
    out = responses_items_to_chat_messages(conv)
    assert out[0] == {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    assert out[1]["role"] == "assistant"
    assert out[1]["tool_calls"][0] == {
        "id": "c1", "type": "function",
        "function": {"name": "f", "arguments": '{"a": 1}'},
    }
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "res"}
    assert out[3] == {"role": "assistant", "content": "already chat format"}


# ---------------------------------------------------------------------------
# model.py: attachments and content flattening
# ---------------------------------------------------------------------------


def test_binary_attachment_roundtrip() -> None:
    """encode_binary_attachment payloads decode back to the original bytes."""
    payload = b"\x89PNG fake bytes"
    text = "before " + encode_binary_attachment("image/png", payload) + " after"
    plain, attachments = parse_binary_attachments(text)
    assert plain == f"before [attached image/png, {len(payload)} bytes] after"
    assert len(attachments) == 1
    assert attachments[0].mime_type == "image/png" and attachments[0].data == payload
    assert parse_binary_attachments("no sentinels") == ("no sentinels", [])


def test_attachment_from_file(tmp_path) -> None:
    """Attachment.from_file detects MIME type and builds data URLs."""
    png = tmp_path / "pic.png"
    png.write_bytes(b"pngdata")
    att = Attachment.from_file(str(png))
    assert att.mime_type == "image/png" and att.data == b"pngdata"
    assert att.to_data_url() == "data:image/png;base64," + att.to_base64()
    bad = tmp_path / "notes.xyz123"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported MIME type"):
        Attachment.from_file(str(bad))


def test_flatten_content_to_text() -> None:
    """Content blocks flatten to readable text; hidden blocks are dropped."""
    assert flatten_content_to_text("plain") == "plain"
    assert flatten_content_to_text(None) == ""
    assert flatten_content_to_text(42) == "42"
    blocks = [
        {"type": "thinking", "thinking": "secret"},
        {"type": "text", "text": "visible"},
        {"type": "tool_use", "name": "f", "input": {"a": 1}},
        {"type": "tool_result", "content": [{"type": "text", "text": "res"}]},
        {"type": "image", "source": {}},
        "raw",
    ]
    assert flatten_content_to_text(blocks) == (
        'visible\n[Tool Call] f({"a": 1})\nres\n[image attachment omitted]\nraw'
    )
