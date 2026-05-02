# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Codex model implementation — uses the ``codex`` CLI as an LLM backend.

This lets you use OpenAI Codex models through a ChatGPT subscription at
subsidized per-token pricing.  The model invokes
``codex exec --json --skip-git-repo-check --sandbox read-only`` in
single-shot mode and consumes the JSONL event stream emitted on stdout.

For agentic use, tool descriptions are injected into the prompt and the
model's text output is parsed for tool-call JSON — the same approach used
by :class:`kiss.core.models.claude_code_model.ClaudeCodeModel` and by
DeepSeek R1 in :mod:`kiss.core.models.openai_compatible_model`.
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


def _find_codex_cli() -> str:
    """Locate the ``codex`` executable on PATH.

    Returns:
        Absolute path to the ``codex`` binary.

    Raises:
        KISSError: If the ``codex`` CLI is not installed.
    """
    path = shutil.which("codex")
    if path is None:
        raise KISSError(
            "Codex CLI ('codex') not found on PATH. "
            "Install it from https://github.com/openai/codex"
        )
    return path


class CodexModel(Model):
    """A model that delegates to the OpenAI Codex CLI for LLM completions.

    Model names use the ``codex/`` prefix.  The part after the prefix is
    passed as the ``-m`` flag to the ``codex`` CLI (e.g.
    ``codex/gpt-5-codex`` → ``-m gpt-5-codex``).  The special name
    ``codex/default`` invokes the CLI without ``-m``, letting Codex use
    the model configured for the active ChatGPT account.

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
        """Initialize a CodexModel instance.

        Args:
            model_name: Full model name including ``codex/`` prefix
                (e.g. ``codex/gpt-5-codex``, ``codex/default``).
            model_config: Optional configuration. Recognised keys:
                - ``system_instruction`` (str): Prepended to the user prompt
                  (the Codex CLI has no native system-prompt flag).
                - ``timeout`` (int): Subprocess timeout in seconds (default 300).
            token_callback: Optional callback invoked with each streamed
                text token.
            thinking_callback: Optional callback invoked with ``True`` when
                a thinking block starts and ``False`` when it ends.
        """
        super().__init__(
            model_name,
            model_config=model_config,
            token_callback=token_callback,
            thinking_callback=thinking_callback,
        )
        self._cli_model = (
            model_name[len("codex/"):] if model_name.startswith("codex/") else model_name
        )

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Initialize the conversation with an initial user prompt.

        Args:
            prompt: The initial user prompt.
            attachments: Not supported — ignored with a warning if provided.
        """
        if attachments:  # pragma: no cover – attachments not used in practice
            logger.warning("CodexModel does not support attachments; they will be ignored.")
        self.conversation = [{"role": "user", "content": prompt}]

    def _build_prompt(self) -> str:
        """Build a single prompt string from the conversation history.

        The Codex CLI is stateless across invocations, so multi-turn
        conversations are flattened into a single text block.
        Tool-result messages (``role == "tool"``) are rendered as
        ``[Tool Result]: …``.  When ``system_instruction`` is set in
        ``model_config`` it is prepended as ``[System]: …``.

        Returns:
            The assembled prompt string.
        """
        parts: list[str] = []
        system_instruction = self.model_config.get("system_instruction")
        if system_instruction:
            parts.append(f"[System]: {system_instruction}")
        if len(self.conversation) == 1 and not system_instruction:
            return str(self.conversation[0]["content"])
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
        """Build the ``codex exec`` CLI argument list.

        Returns:
            List of CLI arguments.
        """
        cli = _find_codex_cli()
        args = [
            cli,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
        ]
        if self._cli_model and self._cli_model != "default":
            args.extend(["-m", self._cli_model])
        return args

    def generate(self) -> tuple[str, Any]:
        """Generate a response using the Codex CLI.

        The Codex CLI emits newline-delimited JSON events to stdout.  Each
        ``item.completed`` event of type ``agent_message`` carries the full
        assistant text for that item; the token callback is invoked once
        per such item.  The terminal ``turn.completed`` event provides the
        usage information.

        Returns:
            tuple[str, Any]: (generated_text, parsed_response).  The
            response is a dict ``{"usage": {...}, "thread_id": "..."}``.

        Raises:
            KISSError: If the CLI invocation fails or the model emits a
                ``turn.failed`` / ``error`` event.
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
            raise KISSError(f"Failed to start Codex CLI: {e}") from e

        assert proc.stdin is not None
        proc.stdin.write(prompt)
        proc.stdin.close()

        assert proc.stdout is not None
        content, result_json, error_message = self._parse_stream_events(proc.stdout)

        proc.wait(timeout=timeout)
        if error_message is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise KISSError(
                f"Codex CLI failed: {error_message}"
                + (f"\nstderr: {stderr.strip()}" if stderr.strip() else "")
            )
        if proc.returncode != 0:  # pragma: no cover – non-error non-zero exit
            stderr = proc.stderr.read() if proc.stderr else ""
            raise KISSError(
                f"Codex CLI failed (exit {proc.returncode}): {stderr.strip()}"
            )

        self.conversation.append({"role": "assistant", "content": content})
        return content, result_json

    def _parse_stream_events(
        self, lines: Iterable[str]
    ) -> tuple[str, dict[str, Any], str | None]:
        """Parse the JSONL event stream from ``codex exec --json``.

        Recognised events:
            - ``thread.started``: records ``thread_id`` in the result.
            - ``turn.started``: ignored.
            - ``item.completed`` with ``item.type == "agent_message"``:
              text is appended to *content* and forwarded via the token
              callback.
            - ``item.completed`` with ``item.type == "agent_reasoning"``:
              text is forwarded via the token callback wrapped in a
              thinking start/end pair.
            - ``turn.completed``: ``usage`` is recorded in the result.
            - ``error`` / ``turn.failed``: collected into *error_message*.

        Args:
            lines: An iterable of JSON strings (one event per line).

        Returns:
            Tuple of ``(content, result_json, error_message)``.  The
            *error_message* is ``None`` on success.
        """
        content = ""
        result_json: dict[str, Any] = {}
        error_message: str | None = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            if event_type == "thread.started":
                result_json["thread_id"] = event.get("thread_id", "")
            elif event_type == "item.completed":
                item = event.get("item", {})
                item_type = item.get("type")
                text = item.get("text", "")
                if item_type == "agent_message" and text:
                    content += text
                    self._invoke_token_callback(text)
                elif item_type == "agent_reasoning" and text:
                    self._invoke_thinking_callback(True)
                    self._invoke_token_callback(text)
                    self._invoke_thinking_callback(False)
            elif event_type == "turn.completed":
                result_json["usage"] = event.get("usage", {})
            elif event_type in ("error", "turn.failed"):
                err = event.get("error") or {}
                error_message = (
                    event.get("message")
                    or (err.get("message") if isinstance(err, dict) else None)
                    or str(event)
                )

        return content, result_json, error_message

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Generate with text-based tool calling via the Codex CLI.

        Tool descriptions are injected into ``system_instruction`` (which
        the model prepends to the user prompt).  The model's text output
        is parsed for JSON ``tool_calls`` blocks, which are returned to
        the framework for execution.  The CLI is run as a stateless LLM,
        not as an agent.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Ignored — text-based tool calling builds its own prompt.

        Returns:
            Tuple of ``(function_calls, content, response)``.
        """
        tools_prompt = _build_text_based_tools_prompt(function_map)

        original_config = self.model_config
        config = dict(original_config)
        original_system = config.get("system_instruction", "")
        config["system_instruction"] = (
            (original_system + "\n\n" + tools_prompt).strip()
        )
        self.model_config = config

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

        function_calls = _parse_text_based_tool_calls(content)

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
        """Extract token counts from the Codex CLI response dict.

        Codex reports ``input_tokens`` (total prompt tokens, including
        cached), ``cached_input_tokens`` (subset that was cache-served),
        and ``output_tokens`` (which already includes
        ``reasoning_output_tokens``).  KISS expects ``input_tokens`` to
        exclude cache-read tokens, so we subtract.  Codex provides no
        cache-write count.

        Args:
            response: The dict returned by :meth:`generate`.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens).
        """
        if not isinstance(response, dict):
            return 0, 0, 0, 0
        usage = response.get("usage", {})
        total_input = usage.get("input_tokens", 0)
        cache_read = usage.get("cached_input_tokens", 0)
        non_cached_input = max(total_input - cache_read, 0)
        return (
            non_cached_input,
            usage.get("output_tokens", 0),
            cache_read,
            0,
        )

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Not supported — Codex CLI does not provide embeddings.

        Raises:
            KISSError: Always.
        """
        raise KISSError("CodexModel does not support embeddings.")
