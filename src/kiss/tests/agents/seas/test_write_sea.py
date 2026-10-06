# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/write`` agent (:mod:`kiss.agents.seas.write.write_sea`).

The SEA class defines ``description()``, a ``system_prompt()`` that appends
its protocol and a ``settings()`` with ``timeout`` only, so the tests check
the things that matter: the slash command resolves to this file, a
``run_agent`` dispatch of it waits one hour by default (a shorter default
wait once stopped a README rewrite before it was returned), the daemon-side
loader stages its protocol as a system-prompt hook that appends, and a real
:class:`ChatSorcarAgent` run configured that way (against the scripted local
chat-completions server) sends the default system prompt with the protocol
added, never replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kiss.agents.seas.write import write_sea
from kiss.agents.seas.write.write_sea import WriteSea
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.agent_dispatch import resolve_timeout
from kiss.agents.sorcar.agent_file import apply_agent_overrides
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.agents.seas.sea_contract import assert_no_removed_getters
from kiss.tests.agents.sorcar.local_model_server import MODEL, finish_body, serve

_SEA_PATH = Path(write_sea.__file__).resolve()


def test_protocol_bans_the_tells_and_fixes_the_register() -> None:
    """The added protocol carries the user's constraints in words the model can act on."""
    assert WriteSea().system_prompt("BASE") == "BASE\n\n" + write_sea.SYSTEM_PROMPT
    protocol = write_sea.SYSTEM_PROMPT
    assert protocol.startswith("## Writing protocol (write)")
    for phrase in (
        "general reader",
        "American English",
        "Be concise",
        "read as if a careful person wrote it",
        "No em dashes",
        "No emoji",
        "delve, leverage",
        "moreover, furthermore",
        "Do not invent facts",
        "check every line against the lists above",
    ):
        assert phrase in protocol, phrase
    # The protocol practices what it preaches: the only "---" is the quoted token it bans.
    assert "\u2014" not in protocol
    assert protocol.count("---") == protocol.count('"---"') == 1


def test_description_is_one_sentence_naming_the_command() -> None:
    """``/write help`` returns the description without running the SEA."""
    text = WriteSea().description()
    assert text.count(". ") == 0 and text.endswith(".")
    assert "/write" in text
    assert sea_commands.help_text_if_command("/write help") == text.strip()


def test_slash_write_resolves_to_the_bundled_sea() -> None:
    """``/write <task>`` resolves to the task text and this file, whose ``timeout`` is an hour.

    The slash command runs the SEA directly (no ``run_agent`` relay), so
    the timeout matters on the ``run_agent`` path only: an empty
    ``timeout`` argument resolves to ``settings()["timeout"]`` and an
    explicit argument wins over it.
    """
    assert sea_commands.get_command("write") == _SEA_PATH
    hit = sea_commands.slash_command_task("/write a release note from CHANGELOG.md")
    assert hit is not None
    task_text, path = hit
    assert path == _SEA_PATH
    assert task_text == "a release note from CHANGELOG.md"
    assert WriteSea().settings({}) == {"timeout": write_sea.DISPATCH_TIMEOUT_SECONDS}
    assert write_sea.DISPATCH_TIMEOUT_SECONDS == 3600
    settings = sea_commands.sea_settings(path)
    assert settings == {"kind": "session", "timeout": 3600.0}
    assert resolve_timeout(None, settings) == 3600.0
    assert resolve_timeout(120.0, settings) == 120.0
    # The timeout lives in ``settings()`` alone: no removed getter
    # (``dispatch_timeout()`` included) is defined, as none would be read.
    assert_no_removed_getters(write_sea)


def test_loader_stages_the_protocol_as_a_system_prompt_hook() -> None:
    """The daemon-side loader stages a hook that appends the protocol, touching nothing else.

    The caller's ``appendToSystemPrompt`` is not the SEA's business: the
    hook receives the assembled prompt (default + suffix) and appends.
    """
    cmd: dict[str, Any] = {"agentPath": str(_SEA_PATH), "appendToSystemPrompt": "CALLER"}
    # The SEA pins no setting and has no ``prompt``: nothing is reported as
    # overridden (the four hooks are written on every run, not listed).
    assert apply_agent_overrides(cmd) == set()
    assert cmd["appendToSystemPrompt"] == "CALLER"
    hook = cmd["systemPromptHook"]
    assert hook("BASE\n\nCALLER") == "BASE\n\nCALLER\n\n" + write_sea.SYSTEM_PROMPT
    assert "prompt" not in cmd
    # ``tools`` is not overridden: the staged tools hook is the identity.
    assert cmd["toolsHook"]([print]) == [print]
    assert cmd["llmCallHook"]([{"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "x"}
    ]
    assert cmd["toolCallHook"]("Bash", {"command": "ls"}) == "OK"
    cmd = {"agentPath": str(_SEA_PATH)}
    apply_agent_overrides(cmd)
    assert cmd["systemPromptHook"]("BASE") == "BASE\n\n" + write_sea.SYSTEM_PROMPT


def test_agent_run_sends_the_default_prompt_with_the_protocol_added(tmp_path: Path) -> None:
    """A run configured as the daemon configures it keeps the default prompt and adds the protocol.

    The scripted model finishes at once with a paragraph; the test checks
    the system message of the real request: the default Sorcar prompt
    (``<identity>``) comes first, the writing protocol follows, and the
    prose reaches the summary untouched.
    """
    cmd: dict[str, Any] = {"agentPath": str(_SEA_PATH)}
    apply_agent_overrides(cmd)
    prose = "<p>The port was in use, so the test failed. Free it and rerun.</p>"
    with serve([finish_body(prose, prompt_tokens=500)]) as (url, requests):
        agent = ChatSorcarAgent("write-sea-test")
        result = agent.run(
            prompt_template="Explain in one paragraph why the test failed.",
            model_name=MODEL,
            work_dir=str(tmp_path),
            max_steps=3,
            max_budget=1.0,
            model_config={"base_url": url, "api_key": "local"},
            system_prompt_hook=cmd["systemPromptHook"],
            web_tools=False,
            use_memory=False,
            is_parallel=False,
            verbose=False,
        )
    parsed = yaml.safe_load(result)
    assert parsed["success"] is True
    assert parsed["summary"] == prose

    agentic = [r for r in requests if r.get("tools")]
    assert len(agentic) == 1, [list(r) for r in requests]
    system = str(next(m for m in agentic[0]["messages"] if m["role"] == "system")["content"])
    assert system.lstrip().startswith("<identity>"), system[:200]
    assert write_sea.SYSTEM_PROMPT in system
    assert system.index("<identity>") < system.index("## Writing protocol (write)")
