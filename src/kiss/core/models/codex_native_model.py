# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Native OpenAI Codex backend transport via /backend-api/codex/responses."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from kiss._version import __version__
from kiss.core.kiss_error import KISSError
from kiss.core.models.codex_cli_model import (
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
)
from kiss.core.models.codex_oauth import OpenAICodexOAuthManager
from kiss.core.models.model import Attachment, Model, TokenCallback


class CodexNativeModel(Model):
    """Model adapter that calls the Codex backend directly over HTTPS."""

    CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        oauth_manager: OpenAICodexOAuthManager | None = None,
    ) -> None:
        super().__init__(model_name, model_config=model_config, token_callback=token_callback)
        self.oauth_manager = oauth_manager or OpenAICodexOAuthManager()
        self.session_id = uuid.uuid4().hex

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name})"

    __repr__ = __str__

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        self.conversation = []
        system_instruction = self.model_config.get("system_instruction")
        if system_instruction:
            self.conversation.append({"role": "system", "content": system_instruction})
        self.conversation.append(
            {
                "role": "user",
                "content": prompt,
                "attachments": list(attachments or []),
            }
        )

    def _latest_image_attachments(self) -> list[Attachment]:
        for msg in reversed(self.conversation):
            if msg.get("role") != "user":
                continue
            attachments = msg.get("attachments")
            if not isinstance(attachments, list) or not attachments:
                continue
            images = [
                a
                for a in attachments
                if isinstance(a, Attachment) and a.mime_type.startswith("image/")
            ]
            non_images = [
                a
                for a in attachments
                if isinstance(a, Attachment) and not a.mime_type.startswith("image/")
            ]
            if non_images:
                raise KISSError("Codex native backend currently supports image attachments only.")
            return images
        return []

    def _conversation_for_prompt(self) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for msg in self.conversation:
            role = str(msg.get("role", "user"))
            entry: dict[str, Any] = {"role": role, "content": str(msg.get("content", ""))}
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                entry["tool_calls"] = tool_calls
            tool_call_id = msg.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                entry["tool_call_id"] = tool_call_id
            attachments = msg.get("attachments")
            if isinstance(attachments, list) and attachments:
                entry["attachments"] = [
                    a.mime_type for a in attachments if isinstance(a, Attachment)
                ]
            rendered.append(entry)
        return rendered

    def _build_generation_prompt(self, tools_prompt: str | None = None) -> str:
        sections = [
            "You are acting as a backend chat-completion model for an external orchestrator.",
            "Return only the next assistant response as plain text.",
            "Do not wrap the response in markdown fences.",
        ]
        if tools_prompt:
            sections.append("When a tool call is needed, follow the exact JSON contract below.")
            sections.append(tools_prompt)
        sections.append("Conversation JSON:")
        sections.append(json.dumps(self._conversation_for_prompt(), ensure_ascii=False, indent=2))
        return "\n\n".join(sections)

    @staticmethod
    def _extract_text_from_response_body(body: dict[str, Any]) -> str:
        output_text = body.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        output = body.get("output")
        if not isinstance(output, list):
            return ""
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") in {"output_text", "text"}:
                            text = part.get("text")
                            if isinstance(text, str) and text:
                                chunks.append(text)
            elif item_type == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
        return "".join(chunks)

    @staticmethod
    def _normalize_usage(body: dict[str, Any]) -> dict[str, int]:
        usage_raw = body.get("usage")
        if not isinstance(usage_raw, dict):
            return {}

        input_details = usage_raw.get("input_tokens_details")
        if not isinstance(input_details, dict):
            input_details = usage_raw.get("prompt_tokens_details")
        if not isinstance(input_details, dict):
            input_details = {}

        input_tokens = usage_raw.get("input_tokens", usage_raw.get("prompt_tokens", 0))
        if not isinstance(input_tokens, (int, float)):
            input_tokens = 0

        cached = usage_raw.get("cache_read_input_tokens")
        if not isinstance(cached, (int, float)):
            cached = usage_raw.get("cache_read_tokens")
        if not isinstance(cached, (int, float)):
            cached = usage_raw.get("cached_tokens")
        if not isinstance(cached, (int, float)):
            cached = input_details.get("cached_tokens", 0)
        if not isinstance(cached, (int, float)):
            cached = 0

        output_tokens = usage_raw.get("output_tokens", usage_raw.get("completion_tokens", 0))
        if not isinstance(output_tokens, (int, float)):
            output_tokens = 0

        return {
            "input_tokens": int(input_tokens),
            "cached_input_tokens": int(cached),
            "output_tokens": int(output_tokens),
        }

    def _build_request_body(self, prompt: str) -> dict[str, Any]:
        content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
        for att in self._latest_image_attachments():
            content.append({"type": "input_image", "image_url": att.to_data_url()})

        instructions = self.model_config.get("codex_instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            instructions = self.model_config.get("system_instruction")
        if not isinstance(instructions, str) or not instructions.strip():
            instructions = "You are a backend assistant for an external orchestrator."

        body: dict[str, Any] = {
            "model": self.model_name,
            "input": [{"role": "user", "content": content}],
            "instructions": instructions,
            "stream": True,
            "store": False,
        }
        reasoning_effort = self.model_config.get("reasoning_effort")
        if isinstance(reasoning_effort, str) and reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
            body["include"] = ["reasoning.encrypted_content"]
        return body

    def _request_headers(self, access_token: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "originator": "kiss-ai",
            "session_id": self.session_id,
            "User-Agent": f"kiss-ai/{__version__}",
        }
        account_id = self.oauth_manager.get_account_id()
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id
        return headers

    def _default_timeout_seconds(self) -> float:
        """Return transport timeout tuned by Codex model tier."""
        return 20.0 if "spark" in self.model_name else 90.0

    def _post_responses_request(self, body: dict[str, Any]) -> dict[str, Any]:
        access_token = self.oauth_manager.get_access_token()
        if not access_token:
            raise KISSError(
                "Codex OAuth credentials are unavailable. Run `codex login` or set "
                "KISS_CODEX_AUTH_FILE to a valid auth.json."
            )

        timeout_seconds = float(
            self.model_config.get("timeout_seconds", self._default_timeout_seconds())
        )
        body_bytes = json.dumps(body).encode("utf-8")

        for attempt in range(2):
            request = urllib.request.Request(
                self.CODEX_RESPONSES_URL,
                data=body_bytes,
                headers=self._request_headers(access_token),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    return self._parse_sse_response(response)
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace").strip()
                auth_error = exc.code in {401, 403}
                if attempt == 0 and auth_error:
                    refreshed = self.oauth_manager.force_refresh_access_token()
                    if refreshed:
                        access_token = refreshed
                        continue
                raise KISSError(
                    f"Codex native API failed with status {exc.code}: {err_body or exc.reason}"
                ) from exc
            except urllib.error.URLError as exc:
                raise KISSError(f"Codex native API request failed: {exc.reason}") from exc

        raise KISSError("Codex native API request failed after token refresh retry.")

    def _parse_sse_response(self, response: Any) -> dict[str, Any]:
        content_parts: list[str] = []
        usage: dict[str, int] = {}
        completed_response: dict[str, Any] = {}
        early_tool_exit = bool(self.model_config.get("early_tool_exit", True))

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            event_type = event.get("type")
            if event_type in {"response.text.delta", "response.output_text.delta"}:
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    content_parts.append(delta)
                    if early_tool_exit:
                        candidate = "".join(content_parts)
                        if _parse_text_based_tool_calls(candidate):
                            return {
                                "output_text": candidate,
                                "usage": usage,
                                "response": completed_response,
                            }
                continue

            if event_type in {"response.text.done", "response.output_text.done"}:
                text_done = event.get("text")
                if isinstance(text_done, str) and text_done:
                    content_parts.append(text_done)
                    if early_tool_exit and _parse_text_based_tool_calls(text_done):
                        return {
                            "output_text": text_done,
                            "usage": usage,
                            "response": completed_response,
                        }
                continue

            if event_type in {"response.error", "response.failed", "error"}:
                message = ""
                err_obj = event.get("error")
                if isinstance(err_obj, dict):
                    err_msg = err_obj.get("message")
                    if isinstance(err_msg, str):
                        message = err_msg
                if not message:
                    raw_msg = event.get("message")
                    if isinstance(raw_msg, str):
                        message = raw_msg
                raise KISSError(f"Codex native API stream error: {message or 'unknown error'}")

            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage = self._normalize_usage({"usage": event_usage})

            if event_type in {"response.done", "response.completed"}:
                maybe_response = event.get("response")
                if isinstance(maybe_response, dict):
                    completed_response = maybe_response
                    maybe_usage = maybe_response.get("usage")
                    if isinstance(maybe_usage, dict):
                        usage = self._normalize_usage({"usage": maybe_usage})

        content = "".join(content_parts)
        if not content and completed_response:
            content = self._extract_text_from_response_body(
                {
                    "output_text": completed_response.get("output_text"),
                    "output": completed_response.get("output"),
                }
            )

        return {
            "output_text": content,
            "usage": usage,
            "response": completed_response,
        }

    def _run_codex_request(self, prompt: str) -> tuple[str, dict[str, Any]]:
        body = self._build_request_body(prompt)
        response_body = self._post_responses_request(body)
        content = self._extract_text_from_response_body(response_body)
        usage = self._normalize_usage(response_body)
        metadata = {"usage": usage, "response": response_body}
        return content, metadata

    def generate(self) -> tuple[str, Any]:
        prompt = self._build_generation_prompt()
        content, response = self._run_codex_request(prompt)
        self.conversation.append({"role": "assistant", "content": content})
        if content and self.token_callback is not None:
            self._invoke_token_callback(content)
        return content, response

    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        tools_prompt = _build_text_based_tools_prompt(function_map)
        prompt = self._build_generation_prompt(tools_prompt)
        content, response = self._run_codex_request(prompt)
        if content and self.token_callback is not None:
            self._invoke_token_callback(content)

        function_calls = _parse_text_based_tool_calls(content)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if function_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": fc["id"],
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": fc.get("arguments", {}),
                    },
                }
                for fc in function_calls
            ]
        self.conversation.append(assistant_msg)
        return function_calls, content, response

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        usage: dict[str, int] = {}
        if isinstance(response, dict):
            maybe_usage = response.get("usage")
            if isinstance(maybe_usage, dict):
                usage = maybe_usage
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cache_read = int(usage.get("cached_input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        non_cached_input = max(0, input_tokens - cache_read)
        return non_cached_input, output_tokens, cache_read, 0

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        model_to_use = embedding_model or self.model_name
        raise KISSError(
            f"Embedding generation is not supported by Codex native backend ({model_to_use})."
        )
