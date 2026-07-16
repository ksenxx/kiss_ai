# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests covering tool-call argument rendering in the sorcar CLI.

When the sorcar CLI runs in interactive mode it connects to the local
``sorcar web`` daemon over a Unix socket and receives newline-delimited
JSON events.  Tool-call events are emitted by
:class:`kiss.server.json_printer.JsonPrinter._format_tool_call`
as a *flat* dict whose payload keys (``path``, ``command``,
``description``, ``content``, ``old_string``, ``new_string``,
``extras``) sit at the top level — there is no ``input`` key.

The CLI side
(:class:`kiss.agents.sorcar.cli_client._EventDispatcher._render`)
must turn that flat payload back into a ``tool_input`` dict so the
shared :class:`kiss.core.print_to_console.ConsolePrinter` formatter can
render the argument panel.  A regression where ``_render`` looked up
the (never-emitted) ``input`` key caused every tool-call panel in the
terminal to show only the tool name and ``(no arguments)``.

These tests drive a real ``_EventDispatcher`` wired to a real
``ConsolePrinter`` writing to an in-memory buffer and assert that the
expected argument strings appear in the rendered output.  No mocks,
patches, or test doubles are used — the daemon event shape, the
dispatcher, and the formatter are all real production code.
"""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console

from kiss.agents.sorcar.cli_client import _EventDispatcher
from kiss.core.print_to_console import ConsolePrinter


def _make_dispatcher() -> tuple[_EventDispatcher, io.StringIO]:
    """Wire a real ConsolePrinter to a wide non-terminal StringIO buffer."""
    buf = io.StringIO()
    printer = ConsolePrinter(file=buf)
    # Force a wide non-terminal Console so Rich does not wrap long
    # command / path strings and the assertions below stay simple.
    printer._console = Console(  # noqa: SLF001 — test wiring
        highlight=False, file=buf, width=200, force_terminal=False, color_system=None,
    )
    return _EventDispatcher(printer), buf


def _dispatch(event: dict[str, Any]) -> str:
    dispatcher, buf = _make_dispatcher()
    dispatcher.dispatch(event)
    return buf.getvalue()


def test_bash_tool_call_renders_command_and_description() -> None:
    """A daemon Bash tool_call event must surface command + description."""
    out = _dispatch(
        {
            "type": "tool_call",
            "name": "Bash",
            "command": "ls -la /tmp/sorcar-demo",
            "description": "List demo directory",
        },
    )
    assert "Bash" in out
    assert "ls -la /tmp/sorcar-demo" in out, (
        f"Bash command argument missing from CLI render:\n{out}"
    )
    assert "List demo directory" in out, (
        f"Bash description argument missing from CLI render:\n{out}"
    )
    assert "(no arguments)" not in out, (
        f"Renderer fell back to '(no arguments)' even though the daemon "
        f"sent a command + description:\n{out}"
    )


def test_edit_tool_call_renders_path_old_and_new_strings() -> None:
    """A daemon Edit tool_call event must surface path / old / new strings."""
    out = _dispatch(
        {
            "type": "tool_call",
            "name": "Edit",
            "path": "/repo/src/foo.py",
            "lang": "python",
            "old_string": "return 1",
            "new_string": "return 42",
        },
    )
    assert "Edit" in out
    assert "/repo/src/foo.py" in out, f"Edit path missing:\n{out}"
    assert "return 1" in out, f"Edit old_string missing:\n{out}"
    assert "return 42" in out, f"Edit new_string missing:\n{out}"
    assert "(no arguments)" not in out


def test_write_tool_call_renders_path_and_content() -> None:
    """A daemon Write tool_call event must surface path + content."""
    out = _dispatch(
        {
            "type": "tool_call",
            "name": "Write",
            "path": "/repo/hello.txt",
            "lang": "text",
            "content": "hello world from sorcar",
        },
    )
    assert "Write" in out
    assert "/repo/hello.txt" in out
    assert "hello world from sorcar" in out, f"Write content missing:\n{out}"
    assert "(no arguments)" not in out


def test_tool_call_renders_extras() -> None:
    """Extras (non-standard tool args) must appear as dim ``key: value`` lines."""
    out = _dispatch(
        {
            "type": "tool_call",
            "name": "go_to_url",
            "extras": {"url": "https://example.com/page"},
        },
    )
    assert "go_to_url" in out
    assert "url" in out
    assert "https://example.com/page" in out, (
        f"Extras (url) missing from CLI render:\n{out}"
    )
    assert "(no arguments)" not in out


def test_tool_call_with_no_args_still_renders_no_arguments_placeholder() -> None:
    """A bare tool_call event (just name) keeps the ``(no arguments)`` body."""
    out = _dispatch({"type": "tool_call", "name": "SomeTool"})
    assert "SomeTool" in out
    assert "(no arguments)" in out


def test_tool_call_with_non_dict_extras_does_not_crash() -> None:
    """A malformed ``extras`` value (non-dict) must be ignored without crashing.

    Production daemons always emit ``extras`` as a ``dict[str, str]``,
    but ``_render`` defensively guards against version-drift / hostile
    payloads.  This locks the defensive ``isinstance(extras, dict)``
    branch in place.
    """
    out = _dispatch(
        {
            "type": "tool_call",
            "name": "BadExtrasTool",
            "extras": "not-a-dict",
        },
    )
    assert "BadExtrasTool" in out
    # Malformed extras dropped → no arguments survive → placeholder body.
    assert "(no arguments)" in out
    assert "not-a-dict" not in out
