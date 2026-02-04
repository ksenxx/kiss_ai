# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Gemini model implementation for Google's GenAI models."""

from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Model


class GeminiModel(Model):
    """A model that uses Google's GenAI API (Gemini)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_config: dict[str, Any] | None = None,
    ):
        """Initialize a GeminiModel instance.

        Args:
            model_name: The name of the Gemini model to use.
            api_key: The Google API key for authentication.
            model_config: Optional dictionary of model configuration parameters.
        """
        super().__init__(model_name, model_config=model_config)
        self.api_key = api_key
        # Store thought signatures from function calls for use in responses
        self._thought_signatures: dict[str, bytes] = {}

    def __str__(self) -> str:
        """Return string representation of the model.

        Returns:
            str: A string describing the model class and name.
        """
        return f"{self.__class__.__name__}(name={self.model_name})"

    __repr__ = __str__

    def initialize(self, prompt: str) -> None:
        """Initializes the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt to start the conversation.
        """
        self.client = genai.Client(api_key=self.api_key)
        self.conversation = [{"role": "user", "content": prompt}]
        self._thought_signatures = {}  # Reset thought signatures for new conversation

    def _convert_conversation_to_gemini_contents(self) -> list[types.Content]:
        """Converts the internal conversation format to Gemini contents.

        Returns:
            list[types.Content]: The conversation in Gemini API format.
        """
        contents = []
        for msg in self.conversation:
            role = msg["role"]
            content = msg.get("content", "")

            parts = []

            if role == "user":
                gemini_role = "user"
                if isinstance(content, str):
                    parts.append(types.Part.from_text(text=content))
                elif isinstance(content, list):
                    # Handle list of blocks (e.g. tool results)
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            pass

            elif role == "assistant":
                gemini_role = "model"
                if isinstance(content, str) and content:
                    parts.append(types.Part.from_text(text=content))

                # Handle tool calls
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        args = fn.get("arguments")
                        args = args if isinstance(args, dict) else {}
                        call_id = tc.get("id")
                        # Include thought_signature if we have one for this tool call
                        thought_sig = self._thought_signatures.get(call_id) if call_id else None
                        if thought_sig:
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=fn.get("name"), args=args
                                    ),
                                    thought_signature=thought_sig,
                                )
                            )
                        else:
                            parts.append(
                                types.Part.from_function_call(name=fn.get("name"), args=args)
                            )

            elif role == "tool":
                gemini_role = "user"  # Function responses come from the 'user' side in Gemini chat

                tool_call_id = msg.get("tool_call_id")
                func_name = "unknown"
                for prev_msg in reversed(self.conversation):
                    if prev_msg == msg:
                        continue  # Skip current
                    if prev_msg["role"] == "assistant" and prev_msg.get("tool_calls"):
                        for tc in prev_msg["tool_calls"]:
                            if tc["id"] == tool_call_id:
                                func_name = tc["function"]["name"]
                                break
                    if func_name != "unknown":
                        break

                import json

                try:
                    if isinstance(content, str):
                        response_dict = json.loads(content)
                    else:
                        response_dict = {"result": content}
                except json.JSONDecodeError:
                    response_dict = {"result": content}

                # Include thought_signature if we have one for this tool call
                # (required for Gemini 3.x models with thinking enabled)
                thought_sig = self._thought_signatures.get(tool_call_id)
                if thought_sig:
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=func_name, response=response_dict
                            ),
                            thought_signature=thought_sig,
                        )
                    )
                else:
                    parts.append(
                        types.Part.from_function_response(name=func_name, response=response_dict)
                    )

            else:
                continue

            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        return contents

    def generate(self) -> tuple[str, Any]:
        """Generates content from prompt without tools.

        Returns:
            tuple[str, Any]: A tuple of (generated_text, raw_response).
        """
        contents = self._convert_conversation_to_gemini_contents()

        config = types.GenerateContentConfig(
            max_output_tokens=self.model_config.get("max_tokens"),
            temperature=self.model_config.get("temperature"),
            top_p=self.model_config.get("top_p"),
            stop_sequences=self.model_config.get("stop"),
        )

        response = self.client.models.generate_content(
            model=self.model_name, contents=contents, config=config
        )

        content = response.text or ""
        self.conversation.append({"role": "assistant", "content": content})
        return content, response

    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generates content with tools, processes the response, and adds it to conversation.

        Args:
            function_map: Dictionary mapping function names to callable functions.

        Returns:
            tuple[list[dict[str, Any]], str, Any]: A tuple of
                (function_calls, response_text, raw_response).
        """

        # Convert tools to Gemini format
        # Gemini expects a list of Tool objects
        gemini_tools = []
        declarations = []

        # We can reuse _build_openai_tools_schema but need to adapt it
        openai_tools = self._build_openai_tools_schema(function_map)

        for tool in openai_tools:
            fn = tool["function"]
            declarations.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description"),
                    parameters=fn.get("parameters"),
                )
            )

        if declarations:
            gemini_tools = [types.Tool(function_declarations=declarations)]

        contents = self._convert_conversation_to_gemini_contents()

        config = types.GenerateContentConfig(
            max_output_tokens=self.model_config.get("max_tokens"),
            temperature=self.model_config.get("temperature"),
            top_p=self.model_config.get("top_p"),
            stop_sequences=self.model_config.get("stop"),
            tools=gemini_tools if gemini_tools else None,  # type: ignore[arg-type]
        )

        response = self.client.models.generate_content(
            model=self.model_name, contents=contents, config=config
        )

        function_calls: list[dict[str, Any]] = []
        content = ""

        # Extract content and function calls from response
        # response.candidates[0].content.parts
        has_parts = (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        )
        if has_parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content += part.text
                if part.function_call:
                    # Create a unique ID for the tool call
                    # (Gemini doesn't provide one natively per call usually)
                    # OpenAI uses IDs to map responses. We need to generate one.
                    import uuid

                    call_id = f"call_{uuid.uuid4().hex[:8]}"

                    # Store thought_signature if present (required for Gemini 3.x models)
                    if part.thought_signature:
                        self._thought_signatures[call_id] = part.thought_signature

                    function_calls.append(
                        {
                            "id": call_id,
                            "name": part.function_call.name,
                            "arguments": part.function_call.args,
                        }
                    )

        self.conversation.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": fc["id"],
                        "type": "function",
                        "function": {"name": fc["name"], "arguments": fc["arguments"]},
                    }
                    for fc in function_calls
                ]
                if function_calls
                else None,
            }
        )

        return function_calls, content, response

    def add_function_results_to_conversation_and_return(
        self, function_results: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """Adds function results to the conversation state.

        Args:
            function_results: List of tuples containing (function_name, result_dict).
        """
        # Find tool call IDs from the last assistant message
        tool_call_map: dict[str, str] = {}
        for msg in reversed(self.conversation):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_call_map = {tc["function"]["name"]: tc["id"] for tc in msg["tool_calls"]}
                break

        for func_name, result_dict in function_results:
            result_content = result_dict.get("result", str(result_dict))
            if self.usage_info_for_messages:
                result_content = f"{result_content}\n\n{self.usage_info_for_messages}"

            # Add as a tool message (OpenAI style) which we convert later
            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_map.get(func_name, f"call_{func_name}"),
                    "content": result_content,
                }
            )

    def add_message_to_conversation(self, role: str, content: str) -> None:
        """Adds a message to the conversation state.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant').
            content: The message content.
        """
        if role == "user" and self.usage_info_for_messages:
            content = f"{content}\n\n{self.usage_info_for_messages}"
        self.conversation.append({"role": role, "content": content})

    def extract_input_output_token_counts_from_response(self, response: Any) -> tuple[int, int]:
        """Extracts input and output token counts from an API response.

        Args:
            response: The raw Gemini API response object.

        Returns:
            tuple[int, int]: A tuple of (input_tokens, output_tokens).
        """
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            return prompt_tokens, output_tokens
        return 0, 0

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Generates an embedding vector for the given text.

        Args:
            text: The text to generate an embedding for.
            embedding_model: Optional model name. Defaults to "text-embedding-004".

        Returns:
            list[float]: The embedding vector as a list of floats.

        Raises:
            KISSError: If embedding generation fails.
        """
        model_to_use = embedding_model or "text-embedding-004"
        try:
            response = self.client.models.embed_content(model=model_to_use, contents=text)
            # Response has embeddings (list), we take the first one
            return list(response.embeddings[0].values)
        except Exception as e:
            raise KISSError(f"Embedding generation failed for model {model_to_use}: {e}") from e
