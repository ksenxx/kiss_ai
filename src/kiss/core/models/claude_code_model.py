# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Claude Code model implementation — uses the ``claude`` CLI as an LLM backend.

This lets you use Claude models through a Claude Code subscription at
subsidized per-token pricing.  The model invokes
``claude --print --dangerously-skip-permissions`` in single-shot mode and
consumes the stream-json event stream emitted on stdout.  Like
:class:`kiss.core.models.codex_model.CodexModel`, the CLI runs **agentically**:
its native tools (Bash, Edit, …) stay enabled and permission prompts are
bypassed because KISS is the outer agent and the user has already authorized
KISS to act on their behalf; running claude with tools disabled would prevent
any file modifications the user requested.  Tool activity (tool invocations
and their results) is streamed to the UI as thinking blocks, the same way
CodexModel surfaces ``command_execution`` events.

For KISS-level tool calling, tool descriptions are injected into the prompt
and the model's text output is parsed for tool-call JSON — the same approach
used by :class:`kiss.core.models.codex_model.CodexModel` and for DeepSeek R1
in :mod:`kiss.core.models.openai_compatible_model`.

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


# Appended to the text-based tools prompt so the agentic CLI never tries to
# invoke a KISS framework tool through its native tool-use mechanism (native
# invocation of an unregistered tool fails, and weaker models then apologize
# instead of falling back to the plain-text JSON protocol).
_KISS_TOOLS_ARE_NOT_NATIVE_NOTE = """

IMPORTANT: The tools listed above are provided by an outer framework and are
NOT part of your native tool set. NEVER attempt to invoke them through your
native tool-use mechanism — doing so fails with "no such tool". To call one,
print the tool_calls JSON object above as plain text in your reply and stop.
Your own native tools (Bash, Read, Edit, ...) remain available as usual.
"""


def _accumulate_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    """Add one message's usage counters into *total* in place.

    Numeric fields are summed; the nested ``cache_creation`` dict is
    merged field-by-field.  Non-numeric fields are ignored.

    Args:
        total: The running aggregate, updated in place.
        usage: One ``message_delta`` event's ``usage`` dict.
    """
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif key == "cache_creation" and isinstance(value, dict):
            sub = total.setdefault("cache_creation", {})
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    sub[sub_key] = sub.get(sub_key, 0) + sub_value


def _tool_result_text(content: Any) -> str:
    """Flatten a ``tool_result`` content payload to plain text.

    The Claude CLI reports tool results either as a plain string or as a
    list of content blocks (``{"type": "text", "text": …}``).

    Args:
        content: The ``content`` field of a ``tool_result`` block.

    Returns:
        The flattened text, or an empty string when nothing is textual.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""


class ClaudeCodeModel(CLITextModel):
    """A model that delegates to the Claude Code CLI for LLM completions.

    Model names use the ``cc/`` prefix.  The part after the prefix is passed
    as the ``--model`` flag to the ``claude`` CLI (e.g. ``cc/opus`` →
    ``--model opus``).

    The CLI runs agentically (native tools enabled, permission checks
    bypassed), mirroring :class:`~kiss.core.models.codex_model.CodexModel`.
    Native tool invocations and their results are surfaced as thinking
    blocks so the user sees real-time progress.

    KISS-level tool calling is supported via text-based prompting: tool
    descriptions are injected into the system prompt and the model's text
    output is parsed for JSON ``tool_calls`` blocks.  Embeddings are not
    available.
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
        (e.g. when a KISS ``tool_calls`` block is detected).

        The CLI runs agentically: its native tools stay enabled and
        ``--dangerously-skip-permissions`` bypasses approval prompts,
        matching the sandbox bypass CodexModel passes to ``codex exec``.

        Returns:
            List of CLI arguments.
        """
        cli = _find_claude_cli()
        args = [
            cli,
            "--print",
            "--disable-slash-commands",
            "--dangerously-skip-permissions",
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

        Always uses streaming so tokens are delivered incrementally.  The
        CLI runs agentically, so a single call may span several assistant
        messages interleaved with native tool executions; text from every
        assistant message is accumulated and native tool activity streams
        as thinking blocks.

        On a tool-bearing turn — one wrapped in
        :class:`~kiss.core.models.model._ToolCallFilteredStream`, which both
        CLI adapters install — the response is truncated at the end of the
        first run of complete ``tool_calls`` JSON blocks and parsing stops,
        so a reasoning model that keeps going cannot hallucinate its own
        tool results into the content.  The CLI **is** killed at that
        instant: with native tools enabled, draining the child to its
        terminal ``result`` event (the old issue #34 usage rescue) would
        let it keep executing tools after the framework already ended the
        turn.  Usage for such a turn is instead aggregated from the
        per-message ``message_delta`` events seen before the stop.

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

    def _emit_as_thinking(self, text: str) -> None:
        """Forward *text* to the token callback wrapped in a thinking block.

        Args:
            text: The text to stream inside a thinking start/end pair.
        """
        self._invoke_thinking_callback(True)
        self._invoke_token_callback(text)
        self._invoke_thinking_callback(False)

    def _emit_tool_use_as_thinking(
        self, block: dict[str, Any], seen_tool_use_ids: set[str]
    ) -> None:
        """Surface a native ``tool_use`` block as a thinking-stream line.

        Bash invocations render as ``$ command`` (matching CodexModel's
        ``command_execution`` rendering); other tools render as
        ``ToolName({...input...})``.  Repeated snapshots of the same block
        (the CLI re-sends assistant messages as they grow) are deduplicated
        by tool-use id.

        Args:
            block: The ``tool_use`` content block from an assistant event.
            seen_tool_use_ids: Ids already emitted; updated in place.
        """
        tool_id = block.get("id")
        if tool_id:
            if tool_id in seen_tool_use_ids:
                return
            seen_tool_use_ids.add(tool_id)
        name = block.get("name", "")
        tool_input = block.get("input") or {}
        if name == "Bash" and tool_input.get("command"):
            self._emit_as_thinking(f"$ {tool_input['command']}\n")
        elif name:
            self._emit_as_thinking(f"{name}({json.dumps(tool_input)})\n")

    def _parse_stream_events(
        self,
        lines: Iterable[str],
        stop_on_tool_calls: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Parse stream-json events from an agentic Claude Code run.

        Iterates over newline-delimited JSON events from the Claude CLI.
        Thinking blocks are streamed via the thinking callback, and text
        blocks via the token callback.  An agentic run emits multiple
        assistant messages interleaved with native tool activity; text from
        every assistant message is accumulated.  Native ``tool_use`` blocks
        and the ``tool_result`` payloads carried by ``user`` events are
        forwarded via the token callback wrapped in thinking start/end
        pairs — the same treatment CodexModel gives ``command_execution``
        events — so the user sees real-time progress.

        Also handles ``content_block_start`` / ``content_block_delta`` /
        ``content_block_stop`` events emitted by the CLI with
        ``--include-partial-messages``.

        A ``result`` event is used as the authoritative final content.

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
        current_block_type = ""
        saw_content_block = False
        thinking_started = False
        seen_tool_calls_hint = False
        found_tool_calls = False
        last_tc_end = -1
        seen_tool_use_ids: set[str] = set()
        aggregated_usage: dict[str, Any] = {}
        current_tool_block: dict[str, Any] | None = None
        current_tool_json = ""

        events = _iter_stream_json_events(lines)
        for event in events:
            event_type = event.get("type")

            if event_type == "assistant":
                msg = event.get("message", {})
                for block in msg.get("content", []):
                    block_type = block.get("type")
                    if block_type == "tool_use":
                        self._emit_tool_use_as_thinking(block, seen_tool_use_ids)
                    elif saw_content_block:
                        continue
                    elif block_type == "thinking":
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
                current_tool_json = ""
                current_tool_block = (
                    block if current_block_type == "tool_use" else None
                )
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
                elif delta_type == "input_json_delta":
                    current_tool_json += delta.get("partial_json", "")
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
                elif current_block_type == "tool_use" and current_tool_block:
                    # Streams that never send the complete assistant
                    # snapshot still surface the tool call: reassemble the
                    # input from the input_json_delta chunks.  The shared
                    # id set deduplicates against snapshot-borne emissions
                    # in either arrival order.
                    if current_tool_block.get("id"):
                        block = dict(current_tool_block)
                        if current_tool_json:
                            with contextlib.suppress(json.JSONDecodeError):
                                block["input"] = json.loads(current_tool_json)
                        self._emit_tool_use_as_thinking(block, seen_tool_use_ids)
                    current_tool_block = None
                    current_tool_json = ""
                current_block_type = ""
            elif event_type == "user":
                msg = event.get("message", {})
                blocks = msg.get("content")
                if isinstance(blocks, list):
                    for block in blocks:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                        ):
                            output = _tool_result_text(block.get("content"))
                            if output:
                                self._emit_as_thinking(output)
            elif event_type == "message_delta":
                usage = event.get("usage")
                if isinstance(usage, dict):
                    _accumulate_usage(aggregated_usage, usage)
            elif event_type == "result":
                result_json = event
                pre_result_content = content
                if not found_tool_calls and not content.strip():
                    # Fallback only: an agentic run's terminal ``result``
                    # carries just the LAST assistant message, so replacing
                    # accumulated multi-message text with it would silently
                    # drop everything said before the final native tool use.
                    content = event.get("result", content)

        if found_tool_calls and not self._stopped_for_tool_calls:
            content = content[:last_tc_end]
            self._stopped_for_tool_calls = True

        if self._stopped_for_tool_calls and not result_json and aggregated_usage:
            # The turn ended at the KISS tool_calls block, before the
            # terminal ``result`` event.  The CLI child is agentic now, so
            # it must be killed immediately — draining it to the terminal
            # event (the old issue #34 rescue) would let it keep executing
            # native tools after the framework already ended the turn.
            # The per-message ``message_delta`` usage aggregated above
            # keeps the token accounting instead.
            result_json = {"usage": aggregated_usage}

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
        returned to the framework for execution.  The CLI additionally runs
        as an agent with its native tools enabled, exactly like
        :class:`CodexModel`; KISS-level tools and native CLI tools coexist.

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
        self.model_config["system_instruction"] += _KISS_TOOLS_ARE_NOT_NATIVE_NOTE
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
