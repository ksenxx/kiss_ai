# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/machine`` agent
(:mod:`kiss.agents.seas.machine.machine_sea`).

The registry tools run against a real remote daemon in its own process
(:mod:`kiss.tests.remote_daemon`); the agent-level test drives a real
:class:`ChatSorcarAgent` ReAct loop against the scripted local
chat-completions server with the SEA's configuration, so the tool's
text really flows into the finish summary.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.seas.machine import machine_sea as sea
from kiss.agents.seas.machine.machine_sea import MachineSea
from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar import sea_commands
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body
from kiss.tests.remote_daemon import RemoteDaemon, start_remote_daemon
from kiss.tests.ssh_server import SSHD, SshServer, start_sshd

# Every test starts with no registered machine (conftest ``empty_machines``).
pytestmark = pytest.mark.usefixtures("empty_machines")


@pytest.fixture(scope="module")
def remote(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RemoteDaemon]:
    daemon = start_remote_daemon(
        tmp_path_factory.mktemp("sea-remote"), password="pw",
        seed=[{"chat_id": "chat-1", "task": "remote work"}],
    )
    yield daemon
    daemon.stop()


@pytest.fixture(scope="module")
def sshd(remote: RemoteDaemon, tmp_path_factory: pytest.TempPathFactory) -> Iterator[SshServer]:
    """An sshd whose logins see the remote daemon's home as ``KISS_HOME``."""
    if not SSHD:
        pytest.skip("no sshd on this host")
    server = start_sshd(tmp_path_factory.mktemp("sshd"), remote.home)
    env = pytest.MonkeyPatch()
    env.setenv(mach.SSH_CONFIG_ENV, str(server.config))
    yield server
    env.undo()
    server.stop()


def test_sea_methods_follow_the_contract() -> None:
    """A worker with no built-in tools, the three registry tools, a $1 budget."""
    agent = MachineSea()
    assert_worker_contract(
        sea, agent, {"tool_profile": "none", "locked": ["tool_profile"], "max_budget": 1.0}
    )
    assert agent.system_prompt("ASSEMBLED") == sea.SYSTEM_PROMPT
    assert agent.tools([]) == [sea.machine_add, sea.machine_remove, sea.machine_list]
    assert agent.tools([print]) == [print, sea.machine_add, sea.machine_remove, sea.machine_list]
    assert "/machine add user@host" in agent.description()
    assert "/machine add <name> <https://host:port> <password>" in agent.description()


def test_tools_add_list_and_remove_against_a_real_daemon(remote: RemoteDaemon) -> None:
    """The tools report what they did; a failed add stores nothing."""
    assert sea.machine_list() == "No remote machine is registered."
    assert sea.machine_add("box", remote.url, "nope") == (
        f"Not registered: box ({remote.url}) rejected the password"
    )
    assert mach.load_machines() == []
    assert sea.machine_add("box", f"127.0.0.1:{remote.port}", "pw") == (
        f"Registered machine 'box' at {remote.url}."
    )
    assert sea.machine_list() == f"box  {remote.url}  reachable"
    assert sea.machine_remove("other") == "No machine named 'other' is registered."
    assert sea.machine_remove("box") == "Removed machine 'box'."
    assert mach.load_machines() == []


def test_tools_add_an_ssh_machine_without_a_password(
    remote: RemoteDaemon, sshd: SshServer,
) -> None:
    """``add user@host`` enrolls this daemon over SSH; the host names the machine."""
    assert sea.machine_add("", sshd.login) == (
        f"Registered machine '127.0.0.1' at ssh {sshd.login} ({remote.url})."
    )
    assert mach.load_machines() == [mach.Machine("127.0.0.1", remote.url, ssh=sshd.login)]
    assert sea.machine_list() == f"127.0.0.1  ssh {sshd.login} ({remote.url})  reachable"
    rows = mach.fetch_history(mach.load_machines()[0], "", "", 10).rows
    assert [r["title"] for r in rows] == ["remote work"]
    mach.close_links()
    assert sea.machine_add("", f"nobody-x@127.0.0.1:{sshd.port}").startswith(
        f"Not registered: ssh nobody-x@127.0.0.1:{sshd.port} failed: "
    )
    assert sea.machine_remove("127.0.0.1") == "Removed machine '127.0.0.1'."
    assert mach.load_machines() == []


def test_agent_lists_the_machines_and_finishes_with_the_text(
    remote: RemoteDaemon, tmp_path: Path,
) -> None:
    """With the SEA's configuration the model gets only finish and the registry tools."""
    mach.add_machine("box", remote.url, "pw")
    expected = f"<p>box  {remote.url}  reachable</p>"
    script = [
        tool_call_body("machine_list", {}, prompt_tokens=300),
        finish_body(expected, prompt_tokens=400),
    ]
    run = sea_commands.evaluate_sea([MachineSea()], "list")
    assert run.tools_hook is not None and run.system_prompt_hook is not None
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)
    assert parsed["success"] is True
    assert parsed["summary"] == expected

    assert len(agentic) == 2
    for request in agentic:
        names = tool_names(request)
        assert names == {"finish", "machine_add", "machine_remove", "machine_list"}, names
        system = system_message(request)
        assert system.startswith(sea.SYSTEM_PROMPT)
    results = tool_results(agentic[1])
    assert len(results) == 1
    # The tool result is followed by the step/usage footer the agent appends.
    assert results[0].startswith(f"box  {remote.url}  reachable\n")


def test_agent_adds_an_ssh_machine_from_the_short_request(
    remote: RemoteDaemon, sshd: SshServer, tmp_path: Path,
) -> None:
    """``add user@host`` reaches ``machine_add`` with an empty name and no password."""
    expected = f"<p>Registered machine '127.0.0.1' at ssh {sshd.login} ({remote.url}).</p>"
    script = [
        tool_call_body(
            "machine_add", {"name": "", "address": sshd.login, "password": ""},
            prompt_tokens=300,
        ),
        finish_body(expected, prompt_tokens=400),
    ]
    run = sea_commands.evaluate_sea([MachineSea()], f"add {sshd.login}")
    assert run.tools_hook is not None and run.system_prompt_hook is not None
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)
    assert parsed["success"] is True and parsed["summary"] == expected
    assert mach.load_machines() == [mach.Machine("127.0.0.1", remote.url, ssh=sshd.login)]
    results = tool_results(agentic[1])
    assert results[0].startswith(
        f"Registered machine '127.0.0.1' at ssh {sshd.login} ({remote.url}).\n"
    )
