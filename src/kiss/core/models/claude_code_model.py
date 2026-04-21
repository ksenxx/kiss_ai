# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Claude Code model implementation — uses the ``claude`` CLI as an LLM backend.

This lets you use Claude models through a Claude Code subscription at
subsidized per-token pricing. The model invokes ``claude --print --tools ""``
in single-shot mode, so **no agentic tool use** is involved.

For agentic use, tool descriptions are injected into the prompt and the
model's text output is parsed for tool-call JSON — the same approach used
for DeepSeek R1 in :mod:`kiss.core.models.openai_compatible_model`.
"""

import json
import logging
import shutil
import subprocess
from collections.abc import Callable, Iterable
from typing import Any

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    _build_text_based_tools_prompt,
    _parse_text_based_tool_calls,
    _strip_text_based_tool_calls,
)

logger = logging.getLogger(__name__)


def _find_claude_cli() -> str:
    """Locate the ``claude`` executable on PATH.

    Returns:
        Absolute path to the ``claude`` binary.

    Raises:
        KISSError: If the ``claude`` CLI is not installed.
    """
    path = shutil.which("claude")
    if path is None:
        raise KISSError(
            "Claude Code CLI ('claude') not found on PATH. "
            "Install it from https://docs.anthropic.com/en/docs/claude-code"
        )
    return path


class ClaudeCodeModel(Model):
    """A model that delegates to the Claude Code CLI for LLM completions.

    Model names use the ``cc/`` prefix.  The part after the prefix is passed
    as the ``--model`` flag to the ``claude`` CLI (e.g. ``cc/opus`` →
    ``--model opus``).

    Tool calling is supported via text-based prompting: tool descriptions
    are injected into the system prompt and the model's text output is
    parsed for JSON ``tool_calls`` blocks.  Embeddings are not available.
    """

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize a ClaudeCodeModel instance.

        Args:
            model_name: Full model name including ``cc/`` prefix (e.g. ``cc/opus``).
            model_config: Optional configuration. Recognised keys:
                - ``system_instruction`` (str): System prompt for the session.
                - ``timeout`` (int): Subprocess timeout in seconds (default 300).
            token_callback: Optional callback invoked with each streamed text token.
            thinking_callback: Optional callback invoked with ``True`` when a
                thinking block starts and ``False`` when it ends.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        # Strip the "cc/" prefix for the --model flag sent to claude CLI
        self._cli_model = model_name[3:] if model_name.startswith("cc/") else model_name

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initialize the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt.
            attachments: Not supported — ignored with a warning if provided.
        """
        if attachments:  # pragma: no cover – attachments not used in practice
            logger.warning("ClaudeCodeModel does not support attachments; they will be ignored.")
        self.conversation = [{"role": "user", "content": prompt}]

    def _build_prompt(self) -> str:
        """Build a single prompt string from the conversation history.

        For multi-turn conversations, formats all messages into a single
        text block since the Claude CLI is stateless.  Tool-result messages
        (``role == "tool"``) are rendered as ``[Tool Result]: …``.

        Returns:
            The assembled prompt string.
        """
        if len(self.conversation) == 1:
            return str(self.conversation[0]["content"])
        parts: list[str] = []
        for msg in self.conversation:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"[User]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
            elif role == "tool":
                parts.append(f"[Tool Result]: {content}")
        return "\n\n".join(parts)

    def _build_cli_args(self) -> list[str]:
        """Build the ``claude`` CLI argument list.

        Always uses ``stream-json`` output format so that tokens can be
        streamed incrementally and the process can be terminated early
        (e.g. when a second assistant message is detected).

        Returns:
            List of CLI arguments.
        """
        cli = _find_claude_cli()
        args = [
            cli,
            "--print",
            "--tools", "",
            "--bare",
            "--no-session-persistence",
            "--model", self._cli_model,
        ]
        system_instruction = self.model_config.get("system_instruction")
        if system_instruction:
            args.extend(["--system-prompt", system_instruction])
        args.extend([
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ])
        return args

    def generate(self) -> tuple[str, Any]:
        """Generate a response using the Claude Code CLI.

        Always uses streaming so tokens are delivered incrementally and the
        process is terminated before a second assistant message is produced.

        Returns:
            tuple[str, Any]: (generated_text, parsed_json_response).

        Raises:
            KISSError: If the CLI invocation fails.
        """
        prompt = self._build_prompt()
        timeout = self.model_config.get("timeout", 300)
        args = self._build_cli_args()

        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as e:  # pragma: no cover – requires broken PATH
            raise KISSError(f"Failed to start Claude Code CLI: {e}") from e

        assert proc.stdin is not None
        proc.stdin.write(prompt)
        proc.stdin.close()

        assert proc.stdout is not None
        content, result_json = self._parse_stream_events(proc.stdout)

        proc.wait(timeout=timeout)
        if proc.returncode != 0:  # pragma: no cover – requires CLI failure
            stderr = proc.stderr.read() if proc.stderr else ""
            raise KISSError(
                f"Claude Code CLI failed (exit {proc.returncode}): {stderr.strip()}"
            )

        self.conversation.append({"role": "assistant", "content": content})
        return content, result_json

    def _parse_stream_events(self, lines: Iterable[str]) -> tuple[str, dict[str, Any]]:
        """Parse stream-json events, stopping before a second assistant message.

        Iterates over newline-delimited JSON events from the Claude CLI.
        Thinking blocks are streamed via the thinking callback, and text
        blocks via the token callback.  If a second ``assistant`` event is
        encountered, parsing stops immediately — the content from the
        second (and subsequent) messages is discarded.

        Also handles ``content_block_start`` / ``content_block_delta`` /
        ``content_block_stop`` events emitted by the CLI with
        ``--include-partial-messages``.

        A ``result`` event (if received before a second assistant) is used
        as the authoritative final content.

        Args:
            lines: An iterable of JSON strings (one event per line).

        Returns:
            Tuple of ``(content, result_json)`` where *content* is the text
            from the first assistant message and *result_json* is the parsed
            ``result`` event dict (or ``{}`` if none was received).
        """
        content = ""
        result_json: dict[str, Any] = {}
        assistant_count = 0
        current_block_type = ""
        seen_assistant_id: str | None = None
        # When --include-partial-messages is set, the CLI streams tokens via
        # content_block_* events AND emits redundant ``assistant`` snapshots
        # with the same accumulated content.  Once a content_block_* event
        # has been observed, the content_block_* stream is the authoritative
        # source — re-processing content from ``assistant`` snapshots would
        # duplicate thinking_start/end boundaries, causing the UI to collapse
        # the thoughts panel into a bare "Thinking (click to expand)" bar.
        saw_content_block = False
        # Defer thinking_start until actual thinking content (thinking_delta)
        # arrives.  Claude opus sends thinking blocks with only
        # signature_delta events and no readable content — emitting
        # thinking_start/end for those would show an empty "Thinking" bar.
        thinking_started = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            # Unwrap stream_event containers (CLI wraps content_block_*
            # and message_* events inside {"type":"stream_event","event":{...}})
            if event_type == "stream_event":
                event = event.get("event", {})
                event_type = event.get("type")

            if event_type == "assistant":
                msg = event.get("message", {})
                msg_id = msg.get("id")
                # --include-partial-messages sends multiple assistant events
                # for the same message (same id); only count genuinely new
                # messages.  Events without an id are always treated as new.
                if msg_id is None or msg_id != seen_assistant_id:
                    assistant_count += 1
                    if assistant_count > 1:
                        break
                    seen_assistant_id = msg_id
                # Skip content processing when the content_block_* stream
                # has already (or will) carry the same tokens — avoids
                # duplicate callbacks and premature thinking_end emission.
                if saw_content_block:
                    continue
                for block in msg.get("content", []):
                    block_type = block.get("type")
                    if block_type == "thinking":
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            self._invoke_thinking_callback(True)
                            self._invoke_token_callback(thinking_text)
                            self._invoke_thinking_callback(False)
                    elif block_type == "text":
                        text = block.get("text", "")
                        if text:
                            content += text
                            self._invoke_token_callback(text)
            elif event_type == "content_block_start":
                saw_content_block = True
                block = event.get("content_block", {})
                current_block_type = block.get("type", "")
                # Defer thinking_start — only emit when actual thinking
                # content arrives (see thinking_delta handling below).
                thinking_started = False
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")
                if delta_type == "thinking_delta":
                    thinking_text = delta.get("thinking", "")
                    if thinking_text:
                        if not thinking_started:
                            self._invoke_thinking_callback(True)
                            thinking_started = True
                        self._invoke_token_callback(thinking_text)
                elif delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        content += text
                        self._invoke_token_callback(text)
            elif event_type == "content_block_stop":
                if current_block_type == "thinking" and thinking_started:
                    self._invoke_thinking_callback(False)
                    thinking_started = False
                current_block_type = ""
            elif event_type == "result":
                result_json = event
                content = event.get("result", content)

        return content, result_json

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate with text-based tool calling via the Claude Code CLI.

        Tool descriptions are injected into the system prompt.  The model's
        text output is parsed for JSON ``tool_calls`` blocks, which are
        returned to the framework for execution — the CLI itself runs in
        pure LLM mode (``--tools ""``), **not** as an agent.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Ignored (text-based tool calling builds its own prompt).

        Returns:
            Tuple of ``(function_calls, content, response)``.
        """
        tools_prompt = _build_text_based_tools_prompt(function_map)

        # Use a local copy to avoid mutating shared model_config
        original_config = self.model_config
        config = dict(original_config)
        original_system = config.get("system_instruction", "")
        config["system_instruction"] = (
            (original_system + "\n\n" + tools_prompt).strip()
        )
        self.model_config = config

        # Buffer streamed *text* tokens so tool-call JSON can be stripped
        # before it reaches the UI.  Thinking tokens are passed through to
        # the original callback immediately (tool-call JSON never appears
        # inside thinking blocks), so the thoughts panel receives them in
        # real time.
        original_token_cb = self.token_callback
        original_thinking_cb = self.thinking_callback
        buffer: list[str] = []
        in_thinking = False

        if original_token_cb is not None:
            def _thinking_wrapper(is_start: bool) -> None:
                nonlocal in_thinking
                in_thinking = is_start
                if original_thinking_cb is not None:
                    original_thinking_cb(is_start)

            def _token_wrapper(token: str) -> None:
                if in_thinking:
                    original_token_cb(token)
                else:
                    buffer.append(token)

            self.token_callback = _token_wrapper
            self.thinking_callback = _thinking_wrapper

        try:
            content, response = self.generate()
        finally:
            self.model_config = original_config
            self.token_callback = original_token_cb
            self.thinking_callback = original_thinking_cb

        # generate() appended a plain assistant message — replace it with
        # one that includes tool_calls if any were found in the text.
        function_calls = _parse_text_based_tool_calls(content)

        # Emit cleaned text (without tool-call JSON) to the UI
        if original_token_cb is not None:
            cleaned = _strip_text_based_tool_calls(content) if function_calls else content
            if cleaned:
                original_token_cb(cleaned)

        if function_calls:
            self._replace_last_assistant_with_tool_calls(content, function_calls)

        return function_calls, content, response

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        """Extract token counts from the Claude Code CLI JSON response.

        Args:
            response: The parsed JSON response from the CLI.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).
        """
        if not isinstance(response, dict):
            return 0, 0, 0, 0
        usage = response.get("usage", {})
        return (
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
        )

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Not supported — Claude Code CLI does not provide embeddings.

        Raises:
            KISSError: Always.
        """
        raise KISSError("ClaudeCodeModel does not support embeddings.")
