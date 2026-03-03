"""Tests for Model base class concrete methods (add_message_to_conversation,
add_function_results_to_conversation_and_return) moved from subclasses."""

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
            m.add_function_results_to_conversation_and_return(
                [
                    ("tool_a", {"result": "res_a"}),
                    ("tool_b", {"result": "res_b"}),
                ]
            )
            assert m.conversation[1]["tool_call_id"] == "tc_1"
            assert m.conversation[1]["content"] == "res_a"
            assert m.conversation[2]["tool_call_id"] == "tc_2"
            assert m.conversation[2]["content"] == "res_b"

    def test_fallback_ids_when_no_tool_calls(self):
        for m in [_make_gemini(), _make_openai()]:
            m.conversation = [{"role": "assistant", "content": "no tools"}]
            m.add_function_results_to_conversation_and_return(
                [
                    ("tool_x", {"result": "ok"}),
                ]
            )
            assert m.conversation[-1]["tool_call_id"] == "call_tool_x_0"

    def test_usage_info_appended_to_results(self):
        for m in [_make_gemini(), _make_openai()]:
            m.set_usage_info_for_messages("Tokens: 100")
            m.conversation = [{"role": "assistant", "content": ""}]
            m.add_function_results_to_conversation_and_return(
                [
                    ("fn", {"result": "done"}),
                ]
            )
            assert "Tokens: 100" in m.conversation[-1]["content"]

    def test_result_dict_without_result_key(self):
        for m in [_make_gemini(), _make_openai()]:
            m.conversation = [{"role": "assistant", "content": ""}]
            m.add_function_results_to_conversation_and_return(
                [
                    ("fn", {"other": "data"}),
                ]
            )
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
            m.add_function_results_to_conversation_and_return(
                [
                    ("fn", {"result": "r1"}),
                    ("fn2", {"result": "r2"}),
                ]
            )
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
        m.add_function_results_to_conversation_and_return(
            [
                ("fn_a", {"result": "ok"}),
            ]
        )
        last = m.conversation[-1]
        assert last["role"] == "user"
        assert isinstance(last["content"], list)
        assert last["content"][0]["type"] == "tool_result"
        assert last["content"][0]["tool_use_id"] == "toolu_1"

    def test_anthropic_fallback_tool_use_id(self):
        m = _make_anthropic()
        m.conversation = [{"role": "assistant", "content": "plain text"}]
        m.add_function_results_to_conversation_and_return(
            [
                ("fn_a", {"result": "ok"}),
            ]
        )
        last = m.conversation[-1]
        assert last["content"][0]["tool_use_id"] == "toolu_fn_a_0"
