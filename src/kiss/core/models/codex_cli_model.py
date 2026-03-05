# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Codex CLI-backed model implementation.

This adapter enables running OpenAI-family models through the local `codex`
CLI authentication flow (for example "Logged in using ChatGPT"), without
requiring OPENAI_API_KEY.
"""

from __future__ import annotations

import inspect
import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Attachment, Model, TokenCallback


def _build_text_based_tools_prompt(function_map: dict[str, Callable[..., Any]]) -> str:
    """Build a text-based tools description for Codex CLI prompting."""
    if not function_map:
        return ""

    tools_desc: list[str] = []
    for func_name, func in function_map.items():
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Function {func_name}"

        params: list[str] = []
        for param_name, param in sig.parameters.items():
            param_type = param.annotation
            type_name = getattr(param_type, "__name__", str(param_type))
            if type_name == "_empty":
                type_name = "any"
            params.append(f"    - {param_name} ({type_name})")

        params_str = "\n".join(params) if params else "    (no parameters)"
        first_line = doc.split(chr(10))[0]
        tools_desc.append(f"- **{func_name}**: {first_line}\n  Parameters:\n{params_str}")

    return f"""
## Available Tools

To call a tool, output a JSON object in the following format:

```json
{{"tool_calls": [{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}]}}
```

You can call multiple tools at once by including multiple objects in the tool_calls array.

### Tools:
{chr(10).join(tools_desc)}

IMPORTANT: When you want to call a tool, output ONLY the JSON object with tool_calls.
Do not include any other text before or after the JSON.
When you have the final answer, call the `finish` tool with your result.
"""


def _parse_text_based_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse tool calls from text output."""
    function_calls: list[dict[str, Any]] = []

    json_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{[^{}]*\"tool_calls\"[^{}]*\[[^\]]*\][^{}]*\})",
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                    for tc in data["tool_calls"]:
                        if "name" in tc:
                            function_calls.append(
                                {
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "name": tc["name"],
                                    "arguments": tc.get("arguments", {}),
                                }
                            )
                    if function_calls:
                        return function_calls
            except json.JSONDecodeError:
                continue

    try:
        data = json.loads(content.strip())
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            for tc in data["tool_calls"]:
                if "name" in tc:
                    function_calls.append(
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": tc["name"],
                            "arguments": tc.get("arguments", {}),
                        }
                    )
    except json.JSONDecodeError:
        pass

    return function_calls


class CodexCliModel(Model):
    """Model adapter that delegates generation to `codex exec --json`."""

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
    ) -> None:
        super().__init__(model_name, model_config=model_config, token_callback=token_callback)
        self.codex_path = shutil.which("codex")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name})"

    __repr__ = __str__

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        if not self.codex_path:
            raise KISSError(
                "Codex CLI not found in PATH. Install Codex CLI and run `codex login` first."
            )
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

    @staticmethod
    def _collect_usage(stdout: str) -> dict[str, int]:
        usage: dict[str, int] = {}
        for line in stdout.splitlines():
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                raw_usage = event["usage"]
                usage = {
                    "input_tokens": int(raw_usage.get("input_tokens", 0) or 0),
                    "cached_input_tokens": int(raw_usage.get("cached_input_tokens", 0) or 0),
                    "output_tokens": int(raw_usage.get("output_tokens", 0) or 0),
                }
        return usage

    @staticmethod
    def _collect_last_agent_message(stdout: str) -> str:
        last_text = ""
        for line in stdout.splitlines():
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            if item.get("type") == "agent_message":
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    last_text = maybe_text
        return last_text

    @staticmethod
    def _attachment_suffix(mime_type: str) -> str:
        ext = mimetypes.guess_extension(mime_type) or ""
        return ext if ext else ".bin"

    def _latest_image_attachments(self) -> list[Attachment]:
        for msg in reversed(self.conversation):
            if msg.get("role") != "user":
                continue
            attachments = msg.get("attachments")
            if isinstance(attachments, list) and attachments:
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
                    raise KISSError("Codex CLI model currently supports image attachments only.")
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
            sections.append(
                "When a tool call is needed, follow the exact JSON contract below."
            )
            sections.append(tools_prompt)
        sections.append("Conversation JSON:")
        sections.append(json.dumps(self._conversation_for_prompt(), ensure_ascii=False, indent=2))
        return "\n\n".join(sections)

    def _run_codex_exec(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if not self.codex_path:
            raise KISSError(
                "Codex CLI not found in PATH. Install Codex CLI and run `codex login` first."
            )

        timeout_seconds = float(self.model_config.get("timeout_seconds", 300))
        work_dir = str(self.model_config.get("codex_work_dir", Path.cwd()))

        image_paths: list[str] = []
        temp_files: list[str] = []
        try:
            for att in self._latest_image_attachments():
                suffix = self._attachment_suffix(att.mime_type)
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(att.data)
                    temp_files.append(f.name)
                    image_paths.append(f.name)

            cmd: list[str] = [
                self.codex_path,
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--json",
                "--model",
                self.model_name,
                "-C",
                work_dir,
            ]
            for image_path in image_paths:
                cmd.extend(["--image", image_path])
            cmd.append("-")

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise KISSError(f"Codex CLI timed out after {timeout_seconds:.1f}s") from e
        except OSError as e:
            raise KISSError(f"Failed to execute Codex CLI: {e}") from e
        finally:
            for path in temp_files:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

        if result.returncode != 0:
            error_text = (result.stderr or result.stdout or "").strip()
            fallback = error_text or "unknown error"
            raise KISSError(
                f"Codex CLI failed with exit code {result.returncode}: {fallback}"
            )

        response_text = self._collect_last_agent_message(result.stdout)
        usage = self._collect_usage(result.stdout)
        metadata = {"usage": usage, "stdout": result.stdout, "stderr": result.stderr}
        return response_text, metadata

    def generate(self) -> tuple[str, Any]:
        prompt = self._build_generation_prompt()
        content, response = self._run_codex_exec(prompt)
        self.conversation.append({"role": "assistant", "content": content})
        if content and self.token_callback is not None:
            self._invoke_token_callback(content)
        return content, response

    def generate_and_process_with_tools(
        self, function_map: dict[str, Callable[..., Any]]
    ) -> tuple[list[dict[str, Any]], str, Any]:
        tools_prompt = _build_text_based_tools_prompt(function_map)
        prompt = self._build_generation_prompt(tools_prompt)
        content, response = self._run_codex_exec(prompt)
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
            f"Embedding generation is not supported by Codex CLI model backend ({model_to_use})."
        )
