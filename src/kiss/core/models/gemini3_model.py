# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Gemini 3 model implementation for KISS agent."""

from collections.abc import Callable
from typing import Any, cast

from google import genai
from google.genai import types

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model


class Gemini3Model(Model):
    """Google Gemini model implementation."""

    client: genai.Client | None

    def _get_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> list[types.FunctionDeclaration]:
        return [
            types.FunctionDeclaration(
                name=s["name"], description=s["description"], parameters=cast(Any, s["schema"])
            )
            for s in self._build_function_schema(function_map)
        ]

    def _extract_function_calls_from_response(self, response: Any) -> list[dict[str, Any]]:
        if not response.candidates:
            return []
        calls = []
        for c in response.candidates:
            if not c.content or not c.content.parts:
                continue
            for p in c.content.parts:
                if p.function_call:
                    calls.append(
                        {
                            "name": p.function_call.name,
                            "arguments": self._parse_function_args(p.function_call.args),
                        }
                    )
        return calls

    def _extract_text_from_response(self, response: Any) -> str:
        if not response.candidates:
            return ""
        return "".join(
            p.text
            for c in response.candidates
            if c.content and c.content.parts
            for p in c.content.parts
            if p.text
        )

    def _get_initial_prompt_text(self) -> str:
        if not self.conversation or not self.conversation[0].parts:
            return ""
        return getattr(self.conversation[0].parts[0], "text", "") or ""

    def initialize(self, prompt: str) -> None:
        self.client = genai.Client(api_key=DEFAULT_CONFIG.agent.api_keys.GEMINI_API_KEY)
        self.conversation = [
            types.Content(
                role="user", parts=[types.Part.from_text(text=self._append_usage_info(prompt))]
            )
        ]

    def generate_content_with_tools(self, function_map: dict[str, Callable[..., Any]]) -> Any:
        assert self.client
        decls = self._get_tools(function_map)
        return self.client.models.generate_content(
            model=self.model_name,
            contents=self.conversation,
            config=types.GenerateContentConfig(
                tools=cast(Any, [types.Tool(function_declarations=decls)] if decls else None),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.AUTO
                    )
                ),
                temperature=1.0,
            ),
        )

    def add_model_response_to_conversation(self, response: Any) -> None:
        parts = [
            p
            for c in (response.candidates or [])
            if c.content and c.content.parts
            for p in c.content.parts
        ]
        if parts:
            self.conversation.append(types.Content(role="model", parts=parts))

    def add_message_to_conversation(self, role: str, content: str) -> None:
        self.conversation.append(
            types.Content(role=role, parts=[types.Part.from_text(text=content)])
        )

    def generate(self) -> tuple[str, Any]:
        assert self.client
        prompt = self._get_initial_prompt_text()
        if not prompt:
            raise KISSError("No prompt provided.")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        )
        c = response.candidates[0] if response.candidates else None
        if not c or not c.content or not c.content.parts:
            raise KISSError("No response from Gemini model.")
        return "".join(str(p.text) for p in c.content.parts if p.text), response

    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        meta = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        return (
            (
                int(getattr(meta, "prompt_token_count", 0) or 0),
                int(getattr(meta, "candidates_token_count", 0) or 0),
            )
            if meta
            else (0, 0)
        )

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        if not function_results:
            return
        parts = [types.Part.from_function_response(name=n, response=r) for n, r in function_results]
        self.conversation.append(types.Content(role="user", parts=parts))

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generate embeddings using Gemini's embedding API.

        Args:
            text: The text to embed.
            embedding_model: The embedding model to use (default: text-embedding-004).

        Returns:
            A list of floats representing the embedding vector.
        """
        assert self.client
        model = embedding_model or "text-embedding-004"
        result = self.client.models.embed_content(model=model, contents=text)
        if result.embeddings is None or len(result.embeddings) == 0:
            raise KISSError("No embeddings returned from Gemini.")
        embedding_values = result.embeddings[0].values
        if embedding_values is None:
            raise KISSError("Embedding values are None.")
        return list(embedding_values)
