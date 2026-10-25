# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for GeminiModel internal conversions.

These tests avoid mocks while exercising internal transformations and
conversation handling without making external API calls.
"""

from types import SimpleNamespace

from kiss.core.models.gemini_model import GeminiModel


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
