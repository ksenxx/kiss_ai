# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Compact formatter implementation using terminal output."""

import ast
import re
import sys
from typing import Any

import markdown_it
from rich.console import Console
from rich.markdown import Markdown

from kiss.core import config as config_module
from kiss.core.formatter import Formatter

LINE_LENGTH = 160

_TOOL_CALL_RE = re.compile(r"```python\s*\n(.+?)\n```", re.DOTALL)
_USAGE_RE = re.compile(r"(####?\s*Usage Information.*)", re.DOTALL)
_MD_PARSER = markdown_it.MarkdownIt().enable("strikethrough")
_MULTI_SPACE = re.compile(r" {2,}")


def _strip_markdown(text: str) -> str:
    parts: list[str] = []
    for token in _MD_PARSER.parse(text):
        for child in token.children or []:
            if child.content:
                parts.append(child.content)
        if not token.children and token.content:
            parts.append(token.content)
    return _MULTI_SPACE.sub(" ", " ".join(parts)).strip()


def _parse_tool_desc(call_body: str) -> str:
    try:
        tree = ast.parse(call_body, mode="eval")
        if not isinstance(tree.body, ast.Call):
            return call_body
    except SyntaxError:
        return call_body

    call = tree.body
    func_name = (
        call.func.id  # type: ignore[union-attr]
        if isinstance(call.func, ast.Name)
        else ast.dump(call.func)
    )

    parts = []
    for kw in call.keywords:
        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str):
                return kw.value.value
            continue
        val = (
            str(kw.value.value)
            if isinstance(kw.value, ast.Constant)
            else ast.unparse(kw.value)
        )
        if len(val) > LINE_LENGTH:
            val = val[:LINE_LENGTH] + "..."
        parts.append(f"{kw.arg}={val}")
    return f"{func_name}({', '.join(parts)})" if parts else func_name


def _extract_parts(content: str) -> tuple[str, str, str]:
    tool_match = _TOOL_CALL_RE.search(content)
    usage_match = _USAGE_RE.search(content)

    boundary = (
        tool_match.start()
        if tool_match
        else usage_match.start()
        if usage_match
        else len(content)
    )
    thought = _strip_markdown(content[:boundary])

    tool_desc = (
        _parse_tool_desc(tool_match.group(1).strip())
        if tool_match
        else ""
    )
    usage = usage_match.group(1).strip() if usage_match else ""
    usage = usage.replace("#### Usage Information", "[usage]:").strip()
    return thought, tool_desc, usage


class CompactFormatter(Formatter):
    """Compact formatter that displays truncated single-line messages."""

    def __init__(self) -> None:
        self.color = sys.stdout.isatty()
        self._console = Console() if self.color else None
        self._stderr_console = Console(stderr=True) if self.color else None

    def format_message(self, message: dict[str, Any]) -> str:
        if not config_module.DEFAULT_CONFIG.agent.verbose:
            return ""
        role = message.get("role", "unknown")
        content = message.get("content", "")

        if role == "user":
            flat = " ".join(content.split()) if content else ""
            prefix = f"[{role}]: {flat}" if flat else f"[{role}]: (empty)"
            return prefix[:LINE_LENGTH] + "..." if len(prefix) > LINE_LENGTH else prefix

        thought, tool_desc, usage = _extract_parts(content)
        lines: list[str] = []
        if thought:
            prefix = f"[{role}]: {thought}"
            if len(prefix) > LINE_LENGTH:
                lines.append(prefix[:LINE_LENGTH] + "...")
            else:
                lines.append(prefix)
        if tool_desc:
            lines.append(f"[action]: {tool_desc}")
        if usage:
            lines.append(usage)
        return "\n".join(lines) if lines else f"[{role}]: (empty)"

    def format_messages(self, messages: list[dict[str, Any]]) -> str:
        if config_module.DEFAULT_CONFIG.agent.verbose:
            return "\n".join(self.format_message(m) for m in messages)
        return ""

    def print_message(self, message: dict[str, Any]) -> None:
        if config_module.DEFAULT_CONFIG.agent.verbose:
            print(self.format_message(message))

    def print_messages(self, messages: list[dict[str, Any]]) -> None:
        if config_module.DEFAULT_CONFIG.agent.verbose:
            print(self.format_messages(messages))

    def _print(self, message: str, style: str, stderr: bool = False) -> None:
        if not config_module.DEFAULT_CONFIG.agent.verbose:
            return
        console = self._stderr_console if stderr else self._console
        if console:
            console.print(message, style=style)
        else:
            print(message, file=sys.stderr if stderr else sys.stdout)

    def print_status(self, message: str) -> None:
        self._print(message, "green")

    def print_error(self, message: str) -> None:
        self._print(message, "red", stderr=True)

    def print_warning(self, message: str) -> None:
        self._print(message, "yellow")

    def print_label_and_value(self, label: str, value: str) -> None:
        if config_module.DEFAULT_CONFIG.agent.verbose:
            md = Markdown(f"__**{label}**__: {value}")
            if self._console:
                self._console.print(md)
            else:
                print(md)
