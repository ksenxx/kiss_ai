"""Integration tests for GeminiModel internal conversions.

These tests avoid mocks while exercising internal transformations and
conversation handling without making external API calls.
"""
from kiss.core.models.gemini_model import GeminiModel


class TestGeminiModelConversationConversion:
    """Tests for GeminiModel conversation conversion and helpers."""

    def _model(self) -> GeminiModel:
        model = GeminiModel("gemini-3-flash-preview", api_key="test")
        model.model_config = {}
        return model

    def test_convert_conversation_with_tools_and_signatures(self):
        """Test conversion covers tool calls, signatures, and unknown roles."""
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

        contents = model._convert_conversation_to_gemini_contents()
        assert len(contents) >= 4
        assert any(content.role == "model" for content in contents)
        assert any(content.role == "user" for content in contents)

    def test_add_function_results_fallback_ids_and_usage_info(self):
        """Test tool results use fallback IDs and usage info is appended."""
        model = self._model()
        model.set_usage_info_for_messages("Usage: 10 tokens")
        model.conversation = []

        model.add_function_results_to_conversation_and_return(
            [("tool_a", {"result": "ok"})]
        )
        assert model.conversation[-1]["tool_call_id"].startswith("call_tool_a_")
        assert "Usage: 10 tokens" in model.conversation[-1]["content"]

    def test_add_message_appends_usage_info(self):
        """Test adding user messages appends usage info."""
        model = self._model()
        model.set_usage_info_for_messages("Usage: 5 tokens")
        model.conversation = []

        model.add_message_to_conversation("user", "Hi")
        assert "Usage: 5 tokens" in model.conversation[-1]["content"]

    def test_extract_token_counts_no_usage(self):
        """Test token count extraction returns zeros without usage metadata."""
        model = self._model()

        class Dummy:
            usage_metadata = None

        assert model.extract_input_output_token_counts_from_response(Dummy()) == (0, 0)

    def test_extract_token_counts_with_usage(self):
        """Test token count extraction with usage metadata."""
        model = self._model()

        class Usage:
            prompt_token_count = 7
            candidates_token_count = 3

        class Dummy:
            usage_metadata = Usage()

        assert model.extract_input_output_token_counts_from_response(Dummy()) == (7, 3)


