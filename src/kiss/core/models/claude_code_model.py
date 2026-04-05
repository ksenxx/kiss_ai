# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Claude Code model implementation — uses the ``claude`` CLI as an LLM backend.

This lets you use Claude models through a Claude Code subscription at
subsidized per-token pricing. The model invokes ``claude --print --tools ""``
in single-shot mode, so **no agentic tool use** is involved.
"""

import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import Attachment, Model, TokenCallback

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

    Only text generation is supported — tool/function calling and embeddings
    are **not** available through this backend.
    """

    def __init__(
        self,
        model_name: str,
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
    ):
        """Initialize a ClaudeCodeModel instance.

        Args:
            model_name: Full model name including ``cc/`` prefix (e.g. ``cc/opus``).
            model_config: Optional configuration. Recognised keys:
                - ``system_instruction`` (str): System prompt for the session.
                - ``timeout`` (int): Subprocess timeout in seconds (default 300).
            token_callback: Optional callback invoked with each streamed text token.
        """
        super().__init__(model_name, model_config=model_config, token_callback=token_callback)
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
        text block since the Claude CLI is stateless.

        Returns:
            The assembled prompt string.
        """
        if len(self.conversation) == 1:
            return str(self.conversation[0]["content"])
        parts: list[str] = []
        for msg in self.conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"[User]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
        return "\n\n".join(parts)

    def _build_cli_args(self, use_streaming: bool = False) -> list[str]:
        """Build the ``claude`` CLI argument list.

        Args:
            use_streaming: If True, use stream-json output format.

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
        if use_streaming:
            args.extend([
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
            ])
        else:
            args.extend(["--output-format", "json"])
        return args

    def generate(self) -> tuple[str, Any]:
        """Generate a response using the Claude Code CLI.

        Returns:
            tuple[str, Any]: (generated_text, parsed_json_response).

        Raises:
            KISSError: If the CLI invocation fails.
        """
        prompt = self._build_prompt()
        timeout = self.model_config.get("timeout", 300)

        if self.token_callback is not None:
            return self._generate_streaming(prompt, timeout)
        return self._generate_blocking(prompt, timeout)

    def _generate_blocking(self, prompt: str, timeout: int) -> tuple[str, Any]:
        """Run claude CLI and get the full JSON response.

        Args:
            prompt: The prompt text to send.
            timeout: Subprocess timeout in seconds.

        Returns:
            tuple[str, Any]: (generated_text, parsed_json_response).
        """
        args = self._build_cli_args(use_streaming=False)
        try:
            result = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:  # pragma: no cover – requires real timeout
            raise KISSError(f"Claude Code CLI timed out after {timeout}s") from e

        if result.returncode != 0:  # pragma: no cover – requires CLI failure
            raise KISSError(
                f"Claude Code CLI failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as e:  # pragma: no cover – requires malformed output
            raise KISSError(f"Failed to parse Claude Code CLI output: {e}") from e

        content = response.get("result", "")
        self.conversation.append({"role": "assistant", "content": content})
        return content, response

    def _generate_streaming(self, prompt: str, timeout: int) -> tuple[str, Any]:
        """Run claude CLI with streaming output and invoke token callback.

        Args:
            prompt: The prompt text to send.
            timeout: Subprocess timeout in seconds.

        Returns:
            tuple[str, Any]: (generated_text, parsed_result_json).
        """
        args = self._build_cli_args(use_streaming=True)
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
        content = ""
        result_json: dict[str, Any] = {}
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "assistant":
                # Extract text from content blocks
                msg = event.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            content += text
                            self._invoke_token_callback(text)
            elif event_type == "result":
                result_json = event
                # The result contains the complete text; use it as authoritative
                content = event.get("result", content)

        proc.wait(timeout=timeout)
        if proc.returncode != 0:  # pragma: no cover – requires CLI failure
            stderr = proc.stderr.read() if proc.stderr else ""
            raise KISSError(
                f"Claude Code CLI failed (exit {proc.returncode}): {stderr.strip()}"
            )

        self.conversation.append({"role": "assistant", "content": content})
        return content, result_json

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Run Claude Code CLI as a full agent with its built-in tools.

        The CLI executes the entire agentic loop internally (Bash, Edit,
        Read, Write) and returns the final result.  A synthetic ``finish``
        tool call is returned so the framework's loop terminates cleanly.

        Args:
            function_map: Ignored — the CLI uses its own built-in tools.
            tools_schema: Ignored — the CLI uses its own tool definitions.

        Returns:
            Tuple of ``([finish_call], "", response_json)``.
        """
        prompt = self._build_prompt()
        timeout = self.model_config.get("timeout", 3600)

        cli = _find_claude_cli()
        args = [
            cli,
            "--print",
            "--dangerously-skip-permissions",
            "--bare",
            "--no-session-persistence",
            "--model", self._cli_model,
            "--output-format", "json",
        ]

        system_instruction = self.model_config.get("system_instruction")
        if system_instruction:
            args.extend(["--system-prompt", system_instruction])

        try:
            result = subprocess.run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:  # pragma: no cover
            raise KISSError(f"Claude Code CLI timed out after {timeout}s") from e

        if result.returncode != 0:  # pragma: no cover
            raise KISSError(
                f"Claude Code CLI failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as e:  # pragma: no cover
            raise KISSError(f"Failed to parse Claude Code CLI output: {e}") from e

        content = response.get("result", "")
        self.conversation.append({"role": "assistant", "content": content})

        finish_call: dict[str, Any] = {
            "name": "finish",
            "arguments": {
                "success": "true",
                "is_continue": "false",
                "summary": content,
            },
        }
        return [finish_call], "", response

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
