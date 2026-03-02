"""Tests for Model base class concrete methods (add_message_to_conversation,
add_function_results_to_conversation_and_return, compact_conversation) moved from subclasses."""

from kiss.core.models.anthropic_model import AnthropicModel
from kiss.core.models.gemini_model import GeminiModel
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel


def _make_anthropic() -> AnthropicModel:
    m = AnthropicModel("claude-haiku-4-5", api_key="test")
    m.conversation = []
    return m


def _make_gemini() -> GeminiModel:
    m = GeminiModel("gemini-3-flash-preview", api_key="test")
    m.conversation = []
    return m


def _make_openai() -> OpenAICompatibleModel:
    m = OpenAICompatibleModel("gpt-4.1-mini", base_url="http://localhost", api_key="test")
    m.conversation = []
    return m


class TestAddMessageToConversation:
    """Test add_message_to_conversation inherited from Model base class."""

    def test_user_message_no_usage_info(self):
        for m in [_make_anthropic(), _make_gemini(), _make_openai()]:
            m.add_message_to_conversation("user", "hello")
            assert m.conversation[-1] == {"role": "user", "content": "hello"}

    def test_user_message_with_usage_info(self):
        for m in [_make_anthropic(), _make_gemini(), _make_openai()]:
            m.set_usage_info_for_messages("Steps: 1/10")
            m.add_message_to_conversation("user", "hello")
            assert m.conversation[-1]["content"] == "hello\n\nSteps: 1/10"

    def test_assistant_message_no_usage_info_appended(self):
        for m in [_make_anthropic(), _make_gemini(), _make_openai()]:
            m.set_usage_info_for_messages("Steps: 1/10")
            m.add_message_to_conversation("assistant", "response")
            assert m.conversation[-1]["content"] == "response"

    def test_empty_usage_info_not_appended(self):
        for m in [_make_anthropic(), _make_gemini(), _make_openai()]:
            m.set_usage_info_for_messages("")
            m.add_message_to_conversation("user", "hello")
            assert m.conversation[-1]["content"] == "hello"


class TestAddFunctionResultsBaseClass:
    """Test add_function_results_to_conversation_and_return from Model base class
    (used by OpenAI and Gemini models)."""

    def test_matches_tool_calls_by_index(self):
        for m in [_make_gemini(), _make_openai()]:
            m.conversation = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc_1", "function": {"name": "tool_a", "arguments": "{}"}},
                        {"id": "tc_2", "function": {"name": "tool_b", "arguments": "{}"}},
                    ],
                }
            ]
            m.add_function_results_to_conversation_and_return([
                ("tool_a", {"result": "res_a"}),
                ("tool_b", {"result": "res_b"}),
            ])
            assert m.conversation[1]["tool_call_id"] == "tc_1"
            assert m.conversation[1]["content"] == "res_a"
            assert m.conversation[2]["tool_call_id"] == "tc_2"
            assert m.conversation[2]["content"] == "res_b"

    def test_fallback_ids_when_no_tool_calls(self):
        for m in [_make_gemini(), _make_openai()]:
            m.conversation = [{"role": "assistant", "content": "no tools"}]
            m.add_function_results_to_conversation_and_return([
                ("tool_x", {"result": "ok"}),
            ])
            assert m.conversation[-1]["tool_call_id"] == "call_tool_x_0"

    def test_usage_info_appended_to_results(self):
        for m in [_make_gemini(), _make_openai()]:
            m.set_usage_info_for_messages("Tokens: 100")
            m.conversation = [{"role": "assistant", "content": ""}]
            m.add_function_results_to_conversation_and_return([
                ("fn", {"result": "done"}),
            ])
            assert "Tokens: 100" in m.conversation[-1]["content"]

    def test_result_dict_without_result_key(self):
        for m in [_make_gemini(), _make_openai()]:
            m.conversation = [{"role": "assistant", "content": ""}]
            m.add_function_results_to_conversation_and_return([
                ("fn", {"other": "data"}),
            ])
            assert "other" in m.conversation[-1]["content"]

    def test_more_results_than_tool_calls(self):
        for m in [_make_gemini(), _make_openai()]:
            m.conversation = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tc_1", "function": {"name": "fn", "arguments": "{}"}},
                    ],
                }
            ]
            m.add_function_results_to_conversation_and_return([
                ("fn", {"result": "r1"}),
                ("fn2", {"result": "r2"}),
            ])
            assert m.conversation[1]["tool_call_id"] == "tc_1"
            assert m.conversation[2]["tool_call_id"] == "call_fn2_1"


class TestAnthropicOverride:
    """Anthropic has its own add_function_results_to_conversation_and_return override."""

    def test_anthropic_uses_tool_result_blocks(self):
        m = _make_anthropic()
        m.conversation = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "fn_a", "input": {}},
                ],
            }
        ]
        m.add_function_results_to_conversation_and_return([
            ("fn_a", {"result": "ok"}),
        ])
        last = m.conversation[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        assert last["content"][0]["type"] == "tool_result"
        assert last["content"][0]["tool_use_id"] == "toolu_1"

    def test_anthropic_fallback_tool_use_id(self):
        m = _make_anthropic()
        m.conversation = [{"role": "assistant", "content": "plain text"}]
        m.add_function_results_to_conversation_and_return([
            ("fn_a", {"result": "ok"}),
        ])
        last = m.conversation[-1]
        assert last["content"][0]["tool_use_id"] == "toolu_fn_a_0"


class TestEstimateConversationTokens:

    def test_empty_conversation(self):
        for m in [_make_anthropic(), _make_gemini(), _make_openai()]:
            m.conversation = []
            assert m._estimate_conversation_tokens() == 0

    def test_string_content(self):
        m = _make_openai()
        m.conversation = [{"role": "user", "content": "x" * 400}]
        assert m._estimate_conversation_tokens() == 100

    def test_list_content_blocks(self):
        m = _make_anthropic()
        m.conversation = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "a" * 800},
            ]}
        ]
        tokens = m._estimate_conversation_tokens()
        assert tokens >= 200


class TestCompactConversation:

    def _big_conversation_openai(self):
        m = _make_openai()
        big = "x" * 4000
        m.conversation = [
            {"role": "user", "content": "initial prompt"},
            {"role": "assistant", "content": "thinking..."},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "more thinking..."},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "even more..."},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "latest response"},
            {"role": "tool", "content": "latest result"},
        ]
        return m

    def test_no_compaction_when_under_threshold(self):
        m = self._big_conversation_openai()
        original = [msg.get("content") for msg in m.conversation]
        m.compact_conversation(1_000_000)
        after = [msg.get("content") for msg in m.conversation]
        assert original == after

    def test_compaction_truncates_old_tool_results(self):
        m = self._big_conversation_openai()
        m.compact_conversation(100)
        assert len(m.conversation[2]["content"]) < 4000
        assert "truncated" in m.conversation[2]["content"]

    def test_compaction_preserves_first_message(self):
        m = self._big_conversation_openai()
        m.compact_conversation(100)
        assert m.conversation[0]["content"] == "initial prompt"

    def test_compaction_preserves_recent_messages(self):
        m = self._big_conversation_openai()
        m.compact_conversation(100)
        assert m.conversation[-1]["content"] == "latest result"
        assert m.conversation[-2]["content"] == "latest response"

    def test_compaction_anthropic_tool_results(self):
        m = _make_anthropic()
        big = "y" * 4000
        m.conversation = [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "z" * 5000, "signature": "sig123"},
                {"type": "text", "text": "response"},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": big},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "step2"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": big},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "step3"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t3", "content": big},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "latest"}]},
            {"role": "user", "content": "final"},
        ]
        m.compact_conversation(100)
        tr = m.conversation[2]["content"][0]
        assert len(tr["content"]) < 4000
        assert "truncated" in tr["content"]

    def test_compaction_never_modifies_thinking_blocks(self):
        m = _make_anthropic()
        big_thinking = "z" * 5000
        big = "x" * 4000
        m.conversation = [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": big_thinking, "signature": "sig"},
                {"type": "text", "text": "hi"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": big},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "step2"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": big},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "step3"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t3", "content": big},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "latest"},
        ]
        m.compact_conversation(100)
        thinking = m.conversation[1]["content"][0]
        assert thinking["thinking"] == big_thinking
        assert thinking["signature"] == "sig"

    def test_compaction_short_conversation_unchanged(self):
        m = _make_openai()
        m.conversation = [
            {"role": "user", "content": "x" * 4000},
            {"role": "assistant", "content": "y" * 4000},
        ]
        orig_lens = [len(msg["content"]) for msg in m.conversation]
        m.compact_conversation(100)
        after_lens = [len(msg["content"]) for msg in m.conversation]
        assert orig_lens == after_lens

    def test_compaction_preserves_system_and_user_message_openai(self):
        m = _make_openai()
        big = "x" * 4000
        m.conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Do the task: " + "a" * 2000},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "next"},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "more"},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "latest"},
            {"role": "tool", "content": "small result"},
        ]
        m.compact_conversation(100)
        assert m.conversation[0]["content"] == "You are a helpful assistant."
        assert m.conversation[1]["content"].startswith("Do the task: ")
        assert len(m.conversation[1]["content"]) > 300
        assert len(m.conversation[3]["content"]) < 4000
        assert "truncated" in m.conversation[3]["content"]

    def test_compaction_preserves_user_message_gemini(self):
        m = _make_gemini()
        big = "x" * 4000
        m.conversation = [
            {"role": "user", "content": "Do the task: " + "a" * 2000},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "next"},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "more"},
            {"role": "tool", "content": big},
            {"role": "assistant", "content": "latest"},
            {"role": "tool", "content": "small result"},
        ]
        m.compact_conversation(100)
        assert m.conversation[0]["content"].startswith("Do the task: ")
        assert len(m.conversation[0]["content"]) > 300
        assert len(m.conversation[2]["content"]) < 4000
        assert "truncated" in m.conversation[2]["content"]

    def test_compaction_truncates_long_text_blocks(self):
        m = _make_anthropic()
        big_text = "w" * 4000
        m.conversation = [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": [
                {"type": "text", "text": big_text},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 4000},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": big_text},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "x" * 4000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "step3"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t3", "content": "x" * 4000},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "latest"}]},
            {"role": "user", "content": "final"},
        ]
        m.compact_conversation(100)
        text_block = m.conversation[1]["content"][0]
        assert len(text_block["text"]) < 4000
        assert "truncated" in text_block["text"]
