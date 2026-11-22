# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``/install`` SEA.

The machines are sandbox homes, each behind a private ``sshd`` of its
own (:mod:`kiss.tests.ssh_server`) whose logins have the home as
``HOME``: the SEA's own ``ssh`` logs in, and the checkout search and the
``install.sh`` run are the real remote commands; only the hosts are
local.  Each home's ``install.sh`` is a script that records how it was
started.
"""

from __future__ import annotations

import socket
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.seas.install import install_sea as sea_module
from kiss.agents.seas.install.install_sea import (
    InstallSea,
    find_checkout,
    install_machine,
    install_on_machines,
)
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.machines import SSH_CONFIG_ENV, Machine, save_machines
from kiss.tests.agents.seas.conftest import git, run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body
from kiss.tests.conftest import posix_only
from kiss.tests.ssh_server import SSHD, SshServer, start_sshd

pytestmark = posix_only("the machines are logins of a local sshd into a POSIX shell")

# Twenty numbered lines, a coloured one, the password box, then the exit
# code in the home's ``exit_code`` file (0 when absent).
INSTALL_SH = """#!/bin/bash
echo "$PWD $*" > "$HOME/ran"
for i in $(seq 1 20); do echo "line $i"; done
printf '\\033[1;32mdone\\033[0m\\n'
echo "*  Remote access password:  correct horse s3cret  *"
echo "warning on stderr" >&2
exit "$(cat "$HOME/exit_code" 2>/dev/null || echo 0)"
"""


def make_checkout(home: Path, name: str, origin: str) -> Path:
    """Create ``home/name``, a git checkout with *origin* and the stand-in ``install.sh``."""
    checkout = home / name
    checkout.mkdir(parents=True)
    git(checkout, "init", "-q")
    git(checkout, "remote", "add", "origin", origin)
    script = checkout / "install.sh"
    script.write_text(INSTALL_SH)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return checkout


class Remotes:
    """The machines of a test: their homes under :attr:`root`, an ``sshd`` in front of each."""

    def __init__(self, root: Path, env: pytest.MonkeyPatch) -> None:
        self.root = root
        self.env = env
        self.servers: list[SshServer] = []

    def machine(self, name: str) -> Machine:
        """Start the ``sshd`` whose logins have ``root/<name>`` as HOME; return the machine at it.

        The servers share their keys, so one client config (``$KISS_SSH_CONFIG``) logs in to all.
        """
        server = start_sshd(self.root.with_name("sshd"), None, home=self.root / name)
        self.servers.append(server)
        self.env.setenv(SSH_CONFIG_ENV, str(server.config))
        return Machine(name, ssh=server.login)


@pytest.fixture
def remotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Remotes]:
    """Isolate KISS_HOME and return the test's machines; their servers are stopped at the end.

    The homes and the servers' keys are under a directory named with a space, both quotes, a
    ``#`` and a ``%``: every path of a login has to survive them, in the servers' and the
    client's configuration as in the commands the SEA sends, whatever the runner's temp root is.
    """
    if not SSHD:
        pytest.skip("no sshd on this host")
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local" / ".kiss"))
    (tmp_path / "local" / ".kiss").mkdir(parents=True)
    machines = Remotes(tmp_path / "o'neil's \"lab\" #1 100%" / "remotes", monkeypatch)
    yield machines
    for server in machines.servers:
        server.stop()


def test_sea_methods_follow_the_user_contract() -> None:
    """The SEA pins the run: no built-in tools, its own prompt and system prompt, one tool."""
    sea = InstallSea()
    assert_worker_contract(
        sea_module,
        sea,
        {
            "tool_profile": "none",
            "max_budget": 1.0,
            "timeout": sea_module.INSTALL_TIMEOUT_SECONDS + 600,
            "locked": ["tool_profile"],
        },
    )
    assert "install.sh" in sea.description() and "/install" in sea.description()
    assert sea.prompt("") == sea_module.ADD_TO_PROMPT
    assert sea.prompt(" alpha ") == f"alpha\n\n{sea_module.ADD_TO_PROMPT}"
    assert sea.system_prompt("ignored") == sea_module.SYSTEM_PROMPT

    def other() -> None: ...

    assert sea.tools([other]) == [other, install_on_machines]
    assert sea_commands.bundled_commands()["install"] == Path(sea_module.__file__)


def test_find_checkout_matches_the_private_repo_in_any_url_form() -> None:
    """The first directory whose origin is the private repo wins; other repos are passed over."""
    listing = (
        "/h/s10s\thttps://github.com/SeamlessLabs/s10s.git\n"
        "/h/fork\thttps://github.com/someone/kiss.git\n"
        "/h/work\tgit@github.com:ksenxx/kiss\n"
        "/h/kiss\thttps://github.com/ksenxx/kiss.git\n"
        "garbage line\n"
    )
    assert find_checkout(listing) == "/h/work"
    assert find_checkout("/h/s10s\thttps://github.com/SeamlessLabs/s10s.git\n") == ""
    assert find_checkout("") == ""


def test_every_machine_is_installed_and_failures_are_reported(remotes: Remotes) -> None:
    """Each ssh machine runs install.sh in its private checkout; the rest say why they failed."""
    ok = make_checkout(remotes.root / "alpha", "kiss", "https://github.com/ksenxx/kiss.git")
    make_checkout(remotes.root / "alpha", "s10s", "https://github.com/SeamlessLabs/s10s.git")
    elsewhere = make_checkout(remotes.root / "beta", "code", "git@github.com:ksenxx/kiss.git")
    (remotes.root / "beta" / "exit_code").write_text("3")
    make_checkout(remotes.root / "gamma", "fork", "https://github.com/someone/kiss.git")
    alpha, beta, gamma = (remotes.machine(name) for name in ("alpha", "beta", "gamma"))
    # The local port of an open connection: taken, and with no listener, so every system
    # refuses a connection to it (macOS drops the ones to a socket that is only bound).
    with socket.create_server(("127.0.0.1", 0)) as anchor, socket.socket() as holder:
        holder.connect(anchor.getsockname())
        port = holder.getsockname()[1]
        dead = Machine("dead", ssh=f"me@127.0.0.1:{port}")
        web = Machine("web", url="https://web:8787", password="pw")
        save_machines([alpha, beta, gamma, dead, web])

        report = install_on_machines()

    assert report.startswith("Installed 1 of 5 machine(s).")
    assert (remotes.root / "alpha" / "ran").read_text() == f"{ok} --non-interactive\n"
    assert (remotes.root / "beta" / "ran").read_text() == f"{elsewhere} --non-interactive\n"
    assert not (remotes.root / "gamma" / "ran").exists()
    assert f"alpha ({alpha.ssh}): ok in {ok}\n" in report
    assert f"beta ({beta.ssh}): FAILED: install.sh exited 3 in {elsewhere}" in report
    assert f"gamma ({gamma.ssh}): FAILED: no clone of https://github.com/ksenxx/kiss.git" in report
    assert (
        f"dead ({dead.ssh}): FAILED: cannot connect: "
        f"ssh: connect to host 127.0.0.1 port {port}: Connection refused\n"
    ) in report
    assert "web (no ssh): FAILED: registered by URL" in report
    block = report.split("\n\n")[1].splitlines()  # alpha's
    assert len(block) == 1 + sea_module.TAIL_LINES
    assert block[-3:] == [
        "done",
        "*  Remote access password:  [redacted]",
        "warning on stderr",
    ]
    assert "horse" not in report and "s3cret" not in report
    assert "\x1b" not in report and "line 7\n" not in report


def test_named_machines_only_and_the_registry_checks(remotes: Remotes) -> None:
    """A subset installs only those machines; unknown names and an empty registry stop early."""
    assert install_on_machines() == (
        "No machine is registered (`/machine add user@host`); nothing to install."
    )
    make_checkout(remotes.root / "alpha", "kiss", "https://github.com/ksenxx/kiss.git")
    make_checkout(remotes.root / "beta", "kiss", "https://github.com/ksenxx/kiss.git")
    save_machines([remotes.machine("alpha"), remotes.machine("beta")])

    assert install_on_machines("alpha, nope,zed") == (
        "Not registered: nope, zed (see `/machine list`)."
    )
    assert install_on_machines(" beta ").startswith("Installed 1 of 1 machine(s).")
    assert (remotes.root / "beta" / "ran").exists()
    assert not (remotes.root / "alpha" / "ran").exists()


def test_timeouts_are_reported(remotes: Remotes) -> None:
    """A search or an install that outlives its limit is a failure naming the limit.

    The install prints for a minute, and ends once nobody reads it.  Its limit
    is 5 s because the limit bounds the search before it as well, and that is
    a whole ssh login.  The search with no answer is of a port that accepts
    the connection and never says a word.
    """
    checkout = make_checkout(remotes.root / "slow", "kiss", "https://github.com/ksenxx/kiss.git")
    (checkout / "install.sh").write_text(
        "#!/bin/bash\nfor i in $(seq 600); do echo installing || exit; sleep 0.1; done\n"
    )
    slow = remotes.machine("slow")
    assert install_machine(slow, timeout=5) == (
        f"slow ({slow.ssh}): FAILED: install.sh in {checkout} ran over 5 s."
    )
    with socket.create_server(("127.0.0.1", 0)) as silent:
        hang = Machine("hang", ssh=f"me@127.0.0.1:{silent.getsockname()[1]}")
        assert install_machine(hang, timeout=1) == (
            f"hang ({hang.ssh}): FAILED: no answer to the checkout search within 1 s."
        )


def test_agent_run_calls_the_tool_once_and_reports(tmp_path: Path, remotes: Remotes) -> None:
    """Under the SEA's configuration the model sees only the install tool and finish."""
    make_checkout(remotes.root / "alpha", "kiss", "https://github.com/ksenxx/kiss.git")
    alpha = remotes.machine("alpha")
    save_machines([alpha])
    script = [
        tool_call_body("install_on_machines", {"machines": ""}, prompt_tokens=500),
        finish_body("<p>alpha: ok.</p>", prompt_tokens=600),
    ]
    run = sea_commands.evaluate_sea([InstallSea()], "")
    assert run.prompt == sea_module.ADD_TO_PROMPT

    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)

    assert parsed["success"] is True and "alpha" in parsed["summary"]
    assert (remotes.root / "alpha" / "ran").exists()
    assert len(agentic) == 2, [list(r) for r in agentic]
    for request in agentic:
        assert tool_names(request) == {"install_on_machines", "finish"}, tool_names(request)
        assert system_message(request).startswith(sea_module.SYSTEM_PROMPT)
    [result] = tool_results(agentic[1])
    assert "Installed 1 of 1 machine(s)." in result and f"alpha ({alpha.ssh}): ok in" in result
