# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The ``add_to_system_prompt()`` agent-script getter adds to ``appendToSystemPrompt``.

``add_to_system_prompt()`` appends its text after whatever the field
already carries — the caller's text, then the text of the deprecated
``append_to_system_prompt()`` getter (ADDITIVE too, under the new SEA
contract: it no longer replaces the caller's text) — each part
separated by a blank line.  Exercised directly through
:func:`apply_agent_overrides`, the daemon-side loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiss.server.agent_file import AgentFileError, apply_agent_overrides


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "adder_sea.py"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_addition_alone_becomes_the_suffix(tmp_path: Path) -> None:
    """With no caller text the addition is the whole ``appendToSystemPrompt``."""
    script = _script(tmp_path, "def add_to_system_prompt():\n    return 'PROTOCOL'\n")
    cmd: dict[str, Any] = {"agentPath": script}
    assert apply_agent_overrides(cmd) == {"appendToSystemPrompt"}
    assert cmd["appendToSystemPrompt"] == "PROTOCOL"


def test_addition_follows_the_callers_text(tmp_path: Path) -> None:
    """The caller's ``appendToSystemPrompt`` survives, the addition comes after it."""
    script = _script(tmp_path, "def add_to_system_prompt():\n    return 'PROTOCOL'\n")
    cmd: dict[str, Any] = {"agentPath": script, "appendToSystemPrompt": "CALLER"}
    apply_agent_overrides(cmd)
    assert cmd["appendToSystemPrompt"] == "CALLER\n\nPROTOCOL"


def test_legacy_append_getter_is_additive_and_precedes_the_addition(tmp_path: Path) -> None:
    """``append_to_system_prompt()`` follows the caller's text; ``add_to_system_prompt()`` is last.

    The deprecated getter used to REPLACE the caller's text; it is now
    additive like ``add_to_system_prompt()``, so all three parts survive
    in caller -> legacy -> addition order.
    """
    script = _script(
        tmp_path,
        "def append_to_system_prompt():\n    return 'REPLACED'\n"
        "def add_to_system_prompt():\n    return 'PROTOCOL'\n",
    )
    cmd: dict[str, Any] = {"agentPath": script, "appendToSystemPrompt": "CALLER"}
    apply_agent_overrides(cmd)
    assert cmd["appendToSystemPrompt"] == "CALLER\n\nREPLACED\n\nPROTOCOL"


def test_empty_addition_and_non_string_caller_value(tmp_path: Path) -> None:
    """An empty addition changes nothing; a non-string wire value counts as empty."""
    script = _script(tmp_path, "def add_to_system_prompt():\n    return ''\n")
    cmd: dict[str, Any] = {"agentPath": script, "appendToSystemPrompt": "CALLER"}
    apply_agent_overrides(cmd)
    assert cmd["appendToSystemPrompt"] == "CALLER"
    script = _script(tmp_path, "def add_to_system_prompt():\n    return 'PROTOCOL'\n")
    cmd = {"agentPath": script, "appendToSystemPrompt": 5}
    apply_agent_overrides(cmd)
    assert cmd["appendToSystemPrompt"] == "PROTOCOL"


def test_wrong_type_is_rejected_and_leaves_the_command_untouched(tmp_path: Path) -> None:
    """A non-string return is an ``AgentFileError`` naming the getter and the type."""
    script = _script(tmp_path, "def add_to_system_prompt():\n    return 42\n")
    cmd: dict[str, Any] = {"agentPath": script, "appendToSystemPrompt": "CALLER"}
    with pytest.raises(AgentFileError, match=r"add_to_system_prompt\(\) .* must return a string"):
        apply_agent_overrides(cmd)
    assert cmd["appendToSystemPrompt"] == "CALLER"
