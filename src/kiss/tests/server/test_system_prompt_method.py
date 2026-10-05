# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The SEA ``system_prompt(self, system_prompt)`` method is staged as ``systemPromptHook``.

The method receives the assembled system prompt and returns the one the
run uses; :func:`apply_agent_overrides`, the daemon-side loader, stages
it as the ``systemPromptHook`` callable the daemon applies once the
prompt is assembled.  The caller's ``appendToSystemPrompt`` and the
``channel`` kind's preamble stay on the ``appendToSystemPrompt`` field,
which the hook never touches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.agent_file import CHANNEL_PREAMBLE, apply_agent_overrides
from kiss.agents.sorcar.sea_commands import SeaScriptError

_PROTOCOL_SEA = """
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return system_prompt + "\\n\\nPROTOCOL"
"""


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "adder_sea.py"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_method_is_staged_as_the_hook_and_receives_the_assembled_prompt(
    tmp_path: Path,
) -> None:
    """``system_prompt()`` becomes ``systemPromptHook``; nothing is evaluated at staging."""
    script = _script(tmp_path, _PROTOCOL_SEA)
    cmd: dict[str, Any] = {"agentPath": script}
    assert apply_agent_overrides(cmd) == {"systemPromptHook"}
    assert "appendToSystemPrompt" not in cmd
    assert "systemPrompt" not in cmd
    assert cmd["systemPromptHook"]("BASE") == "BASE\n\nPROTOCOL"


def test_callers_suffix_stays_on_its_field(tmp_path: Path) -> None:
    """The caller's ``appendToSystemPrompt`` survives untouched next to the hook."""
    script = _script(tmp_path, _PROTOCOL_SEA)
    cmd: dict[str, Any] = {"agentPath": script, "appendToSystemPrompt": "CALLER"}
    assert apply_agent_overrides(cmd) == {"systemPromptHook"}
    assert cmd["appendToSystemPrompt"] == "CALLER"
    assert cmd["systemPromptHook"]("BASE\n\nCALLER") == "BASE\n\nCALLER\n\nPROTOCOL"


def test_channel_preamble_goes_to_the_suffix_not_the_hook(tmp_path: Path) -> None:
    """A ``channel`` SEA's preamble follows the caller's text on ``appendToSystemPrompt``.

    The hook is independent of the suffix: it rewrites whatever prompt
    the daemon assembled (base plus suffix), so the preamble that keeps
    a channel SEA from recursing into ``run_agent`` is never displaced
    by the SEA's own protocol.
    """
    script = _script(
        tmp_path,
        """
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def settings(self, settings):
        return settings | {"kind": "channel"}

    def system_prompt(self, system_prompt):
        return system_prompt + "\\n\\nPROTOCOL"
""",
    )
    cmd: dict[str, Any] = {"agentPath": script, "appendToSystemPrompt": "CALLER"}
    overridden = apply_agent_overrides(cmd)
    assert {"systemPromptHook", "appendToSystemPrompt"} <= overridden
    assert cmd["appendToSystemPrompt"] == (
        "CALLER\n\n" + CHANNEL_PREAMBLE.format(name="adder")
    )
    # The daemon hands the hook the base prompt plus the staged suffix:
    # caller -> preamble -> protocol is the order the run ends up with.
    assembled = "BASE\n\n" + cmd["appendToSystemPrompt"]
    assert cmd["systemPromptHook"](assembled) == (
        "BASE\n\nCALLER\n\n" + CHANNEL_PREAMBLE.format(name="adder") + "\n\nPROTOCOL"
    )


def test_hook_may_replace_or_keep_the_prompt(tmp_path: Path) -> None:
    """The method owns the whole prompt: it may drop it or return it unchanged."""
    script = _script(
        tmp_path,
        """
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return "REPLACED" if "drop me" in system_prompt else system_prompt
""",
    )
    cmd: dict[str, Any] = {"agentPath": script, "appendToSystemPrompt": 5}
    apply_agent_overrides(cmd)
    assert cmd["appendToSystemPrompt"] == 5
    assert cmd["systemPromptHook"]("BASE, drop me") == "REPLACED"
    assert cmd["systemPromptHook"]("BASE") == "BASE"


def test_wrong_type_is_rejected_when_the_hook_runs(tmp_path: Path) -> None:
    """A non-string return is a ``SeaScriptError`` naming the method and the type.

    The method runs lazily, so staging succeeds and the command keeps
    the caller's suffix; the diagnostic comes from the hook.
    """
    script = _script(
        tmp_path,
        """
from kiss.agents.seas.base.base_sea import BaseSea


class Sea(BaseSea):
    def system_prompt(self, system_prompt):
        return 42
""",
    )
    cmd: dict[str, Any] = {"agentPath": script, "appendToSystemPrompt": "CALLER"}
    assert apply_agent_overrides(cmd) == {"systemPromptHook"}
    assert cmd["appendToSystemPrompt"] == "CALLER"
    with pytest.raises(
        SeaScriptError,
        match=r"system_prompt\(\) of agent script '.*adder_sea\.py' must return a string, got int",
    ):
        cmd["systemPromptHook"]("BASE")
