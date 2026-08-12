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

The subprocess supervision both CLI-backed adapters share
(:class:`~kiss.core.models.model._CLIProcess`,
:class:`~kiss.core.models.model._ToolCallFilteredStream`) lives beside
:class:`~kiss.core.models.model.CLITextModel` in
:mod:`kiss.core.models.model`, which this module and
:mod:`kiss.core.models.codex_model` both import it from.
"""

import contextlib
import json
import logging
import shutil
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from kiss.core.kiss_error import KISSError
from kiss.core.models.anthropic_model import cache_creation_tokens
from kiss.core.models.model import (
    CLITextModel,
    ThinkingCallback,
    TokenCallback,
    _cli_stall_error,
    _CLIProcess,
    _iter_balanced_json_objects,
    _iter_tool_calls_lists,
    _parse_text_based_tool_calls,
    _StreamReadTimeoutError,
    _ToolCallFilteredStream,
    flatten_content_to_text,
)

logger = logging.getLogger(__name__)


def _dict_field(record: Any, name: str) -> Any:
    """Return key *name* of the dict *record*, or ``None`` when absent."""
    return record.get(name) if isinstance(record, dict) else None



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


def _iter_stream_json_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield parsed stream-json events, unwrapping ``stream_event`` wrappers.

    Blank lines and lines that are not valid JSON are skipped.

    Args:
        lines: An iterable of JSON strings (one event per line).

    Yields:
        The parsed event dicts, with ``stream_event`` wrappers replaced by
        their inner event.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "stream_event":
            event = event.get("event", {})
        yield event


def _find_consecutive_tool_calls_end(content: str) -> int:
    """Return the end position of the last consecutive ``tool_calls`` JSON block.

    Scans *content* for balanced JSON objects containing ``tool_calls``
    lists.  Consecutive blocks (separated only by whitespace) are all
    included.  The sequence stops when non-whitespace text appears between
    blocks or a non-tool-calls JSON object is encountered after the first
    tool_calls block.

    This lets the parser collect **all** tool calls the model outputs
    back-to-back, rather than only the first block.  Blocks separated by
    hallucinated text (e.g. ``(no output)``) are *not* consecutive, so
    only the first group is captured.

    Args:
        content: The accumulated text to scan.

    Returns:
        End position (exclusive) of the last consecutive tool_calls JSON
        object, or ``-1`` if none found.
    """
    last_tc_end = -1
    for start, end, parsed in _iter_balanced_json_objects(content):
        if _iter_tool_calls_lists(parsed):
            if last_tc_end == -1:
                last_tc_end = end
            else:
                between = content[last_tc_end:start]
                if between.strip():
                    break
                last_tc_end = end
        elif last_tc_end != -1:
            break
    return last_tc_end


class ClaudeCodeModel(CLITextModel):
    """A model that delegates to the Claude Code CLI for LLM completions.

    Model names use the ``cc/`` prefix.  The part after the prefix is passed
    as the ``--model`` flag to the ``claude`` CLI (e.g. ``cc/opus`` →
    ``--model opus``).

    Tool calling is supported via text-based prompting: tool descriptions
    are injected into the system prompt and the model's text output is
    parsed for JSON ``tool_calls`` blocks.  Embeddings are not available.
    """

    _cli_model_name = "ClaudeCodeModel"
    _cli_logger = logger

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
        self._cli_model = model_name[3:] if model_name.startswith("cc/") else model_name
        self._last_thinking_content: str = ""
        self._pre_result_content: str = ""
        self._stopped_for_tool_calls: bool = False

    def _build_prompt(self) -> str:
        """Build a single prompt string from the conversation history.

        For multi-turn conversations, formats all messages into a single
        text block since the Claude CLI is stateless.  Tool-result messages
        (``role == "tool"``) are rendered as ``[Tool Result]: …``.

        Returns:
            The assembled prompt string.
        """
        if len(self.conversation) == 1:
            return flatten_content_to_text(self.conversation[0]["content"])
        return self._conversation_as_dialogue()

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
            "--disable-slash-commands",
            "--tools", "",
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

        On a tool-bearing turn — one wrapped in
        :class:`~kiss.core.models.model._ToolCallFilteredStream`, which both
        CLI adapters install — the response is truncated at the end of the
        first run of complete ``tool_calls`` JSON blocks and parsing stops,
        so a reasoning model that keeps going cannot hallucinate its own
        tool results into the content.  The CLI is **not** killed at that
        instant: the stream is first drained to its terminal ``result``
        event, the only carrier of usage and cost (issue #34).  That drain
        gives up at *timeout* rather than failing a step whose tool call is
        already parsed, and the process is terminated as soon as it finishes
        either way.

        Returns:
            tuple[str, Any]: (generated_text, parsed_json_response).

        Raises:
            KISSError: If the CLI could not be started or exited with a
                failure status.
            TimeoutError: If the CLI produced no complete turn before the
                deadline.  Retryable, unlike ``KISSError``.
            KeyboardInterrupt: If the user stopped the task mid-turn.
        """
        prompt = self._build_prompt()
        timeout = self.model_config.get("timeout", 300)
        args = self._build_cli_args()
        self._stopped_for_tool_calls = False
        stop_on_tool_calls = self._tool_bearing_turn

        with _CLIProcess(args, "Claude Code CLI", timeout) as proc:
            try:
                # Inside the handlers: sending the prompt is bounded by
                # the same deadline and Stop signal as reading the reply.
                proc.send_prompt(prompt)
                content, result_json = self._parse_stream_events(
                    proc.lines(), stop_on_tool_calls=stop_on_tool_calls
                )
            except _StreamReadTimeoutError:
                self._close_thinking_if_open()
                raise _cli_stall_error("Claude Code CLI", timeout) from None
            except KeyboardInterrupt:
                self._close_thinking_if_open()
                raise
            if not self._stopped_for_tool_calls:
                status = proc.wait_for_exit()
                if status not in (0, -15, None):
                    raise KISSError(
                        f"Claude Code CLI failed (exit {status}): "
                        f"{proc.stderr_text().strip()}"
                    )

        self.conversation.append({"role": "assistant", "content": content})
        return content, result_json

    def _parse_stream_events(
        self,
        lines: Iterable[str],
        stop_on_tool_calls: bool = False,
    ) -> tuple[str, dict[str, Any]]:
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

        When *stop_on_tool_calls* is ``True``, the parser watches for
        complete ``{"tool_calls": [...]}`` JSON blocks in the accumulated
        text content.  As soon as one is found the content is truncated to
        the end of that first block and parsing stops, preventing reasoning
        models from hallucinating tool results in an unbounded stream.

        Args:
            lines: An iterable of JSON strings (one event per line).
            stop_on_tool_calls: When ``True``, stop as soon as a complete
                ``tool_calls`` JSON block is detected in the text content.

        Returns:
            Tuple of ``(content, result_json)`` where *content* is the text
            from the first assistant message and *result_json* is the parsed
            ``result`` event dict (or ``{}`` if none was received).
        """
        content = ""
        thinking_content = ""
        pre_result_content = ""
        result_json: dict[str, Any] = {}
        assistant_count = 0
        current_block_type = ""
        seen_assistant_id: str | None = None
        saw_content_block = False
        thinking_started = False
        seen_tool_calls_hint = False
        found_tool_calls = False
        last_tc_end = -1

        events = _iter_stream_json_events(lines)
        for event in events:
            event_type = event.get("type")

            if event_type == "assistant":
                msg = event.get("message", {})
                msg_id = msg.get("id")
                if msg_id is None or msg_id != seen_assistant_id:
                    assistant_count += 1
                    if assistant_count > 1:
                        break
                    seen_assistant_id = msg_id
                if saw_content_block:
                    continue
                for block in msg.get("content", []):
                    block_type = block.get("type")
                    if block_type == "thinking":
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            thinking_content += thinking_text
                            self._invoke_thinking_callback(True)
                            self._invoke_token_callback(thinking_text)
                            self._invoke_thinking_callback(False)
                    elif block_type == "text":
                        text = block.get("text", "")
                        if text:
                            content += text
                            self._invoke_token_callback(text)
                if stop_on_tool_calls and content:
                    tc_end = _find_consecutive_tool_calls_end(content)
                    if tc_end > 0:
                        content = content[:tc_end]
                        self._stopped_for_tool_calls = True
                        break
            elif event_type == "content_block_start":
                saw_content_block = True
                block = event.get("content_block", {})
                current_block_type = block.get("type", "")
                thinking_started = False
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")
                if delta_type == "thinking_delta":
                    thinking_text = delta.get("thinking", "")
                    if thinking_text:
                        thinking_content += thinking_text
                        if not thinking_started:
                            self._invoke_thinking_callback(True)
                            thinking_started = True
                        self._invoke_token_callback(thinking_text)
                elif delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        content += text
                        self._invoke_token_callback(text)
                        if stop_on_tool_calls:
                            if not seen_tool_calls_hint and "tool_calls" in content:
                                seen_tool_calls_hint = True
                            if seen_tool_calls_hint and "}" in text:
                                tc_end = _find_consecutive_tool_calls_end(content)
                                if tc_end > 0:
                                    last_tc_end = tc_end
                                    found_tool_calls = True
                            if found_tool_calls:
                                trailing = content[last_tc_end:].strip()
                                if trailing and not trailing.startswith("{"):
                                    content = content[:last_tc_end]
                                    self._stopped_for_tool_calls = True
                                    break
            elif event_type == "content_block_stop":
                if current_block_type == "thinking" and thinking_started:
                    self._invoke_thinking_callback(False)
                    thinking_started = False
                current_block_type = ""
            elif event_type == "result":
                result_json = event
                pre_result_content = content
                if not found_tool_calls:
                    content = event.get("result", content)

        if found_tool_calls and not self._stopped_for_tool_calls:
            content = content[:last_tc_end]
            self._stopped_for_tool_calls = True

        if self._stopped_for_tool_calls and not result_json:
            # The tool call is already parsed; this drain only rescues the
            # usage counts the terminal event carries (issue #34).  A CLI
            # that overruns the deadline here costs cost-accuracy for one
            # step, never the step itself.
            with contextlib.suppress(_StreamReadTimeoutError):
                for event in events:
                    if event.get("type") == "result":
                        result_json = event
                        break

        self._last_thinking_content = thinking_content
        self._pre_result_content = pre_result_content
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

        Thinking tokens stream to the callbacks as they arrive, but the
        assistant text is held back and re-emitted once the turn ends,
        stripped of the ``tool_calls`` JSON the framework parses out of
        it — otherwise the raw block would be rendered in the chat panel
        before every tool card.  :class:`CodexModel` filters its stream
        the same way.

        Args:
            function_map: Dictionary mapping function names to callable functions.
            tools_schema: Ignored (text-based tool calling builds its own prompt).

        Returns:
            Tuple of ``(function_calls, content, response)``.
        """
        original_config = self._install_tools_prompt_in_system_instruction(function_map)
        try:
            with _ToolCallFilteredStream(self):
                content, response = self.generate()
        finally:
            self.model_config = original_config

        all_sources = [content]
        if self._pre_result_content:
            all_sources.append(self._pre_result_content)
        if self._last_thinking_content:
            all_sources.append(self._last_thinking_content)
        combined = "\n".join(all_sources)
        function_calls = _parse_text_based_tool_calls(combined)

        if function_calls:
            self._replace_last_assistant_with_tool_calls(content, function_calls)

        return function_calls, content, response

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int, int]:
        """Extract token counts from the Claude Code CLI JSON response.

        Args:
            response: The parsed JSON response from the CLI.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens,
            cache_write_5m_tokens, cache_write_1h_tokens).  The last element
            is the Anthropic one-hour cache-write token count.
        """
        if not isinstance(response, dict):
            return 0, 0, 0, 0, 0
        usage = response.get("usage") or {}
        cache_write_5m, cache_write_1h = cache_creation_tokens(usage, _dict_field)
        return (
            usage.get("input_tokens") or 0,
            usage.get("output_tokens") or 0,
            usage.get("cache_read_input_tokens") or 0,
            cache_write_5m,
            cache_write_1h,
        )
