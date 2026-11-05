# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``/install`` SEA.

The machines are sandbox homes reached through a stand-in ``ssh`` on
PATH that runs the remote command in bash with ``HOME`` pointing at the
host's directory, so the checkout search and the ``install.sh`` run are
the real shell commands; only the network is missing.  Each home's
``install.sh`` is a script that records how it was started.
"""

from __future__ import annotations

import os
import stat
import subprocess
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
from kiss.agents.sorcar.machines import Machine, save_machines
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body
from kiss.tests.conftest import posix_only

pytestmark = posix_only("drives the SEA through a bash ssh stand-in")

# Options are dropped; the destination picks the host's HOME under
# $REMOTE_ROOT; ``dead`` refuses the connection like a real client (255)
# and ``hang`` never answers.
FAKE_SSH = """#!/bin/bash
while [ $# -gt 0 ]; do
    case "$1" in
        -o|-p|-F) shift 2 ;;
        -*) shift ;;
        *) break ;;
    esac
done
host="${1#*@}"
shift
[ "$host" = dead ] && { echo "ssh: connect to host dead: Connection refused" >&2; exit 255; }
[ "$host" = hang ] && exec sleep 5
export HOME="$REMOTE_ROOT/$host"
exec bash -c "$*"
"""

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
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", origin], check=True)
    script = checkout / "install.sh"
    script.write_text(INSTALL_SH)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return checkout


@pytest.fixture
def remotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put the stand-in ssh on PATH, isolate KISS_HOME and return the root of the host homes."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "ssh"
    fake.write_text(FAKE_SSH)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("REMOTE_ROOT", str(tmp_path / "remotes"))
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "local" / ".kiss"))
    (tmp_path / "local" / ".kiss").mkdir(parents=True)
    return tmp_path / "remotes"


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


def test_every_machine_is_installed_and_failures_are_reported(remotes: Path) -> None:
    """Each ssh machine runs install.sh in its private checkout; the rest say why they failed."""
    ok = make_checkout(remotes / "alpha", "kiss", "https://github.com/ksenxx/kiss.git")
    make_checkout(remotes / "alpha", "s10s", "https://github.com/SeamlessLabs/s10s.git")
    elsewhere = make_checkout(remotes / "beta", "code", "git@github.com:ksenxx/kiss.git")
    (remotes / "beta" / "exit_code").write_text("3")
    make_checkout(remotes / "gamma", "fork", "https://github.com/someone/kiss.git")
    save_machines(
        [
            Machine("alpha", ssh="me@alpha"),
            Machine("beta", ssh="me@beta"),
            Machine("gamma", ssh="me@gamma"),
            Machine("dead", ssh="me@dead"),
            Machine("web", url="https://web:8787", password="pw"),
        ]
    )

    report = install_on_machines()

    assert report.startswith("Installed 1 of 5 machine(s).")
    assert (remotes / "alpha" / "ran").read_text() == f"{ok} --non-interactive\n"
    assert (remotes / "beta" / "ran").read_text() == f"{elsewhere} --non-interactive\n"
    assert not (remotes / "gamma" / "ran").exists()
    assert f"alpha (me@alpha): ok in {ok}\n" in report
    assert f"beta (me@beta): FAILED: install.sh exited 3 in {elsewhere}" in report
    assert "gamma (me@gamma): FAILED: no clone of https://github.com/ksenxx/kiss.git" in report
    assert "dead (me@dead): FAILED: cannot connect: ssh: connect to host dead" in report
    assert "web (no ssh): FAILED: registered by URL" in report
    alpha = report.split("\n\n")[1].splitlines()
    assert len(alpha) == 1 + sea_module.TAIL_LINES
    assert alpha[-3:] == [
        "done",
        "*  Remote access password:  [redacted]",
        "warning on stderr",
    ]
    assert "horse" not in report and "s3cret" not in report
    assert "\x1b" not in report and "line 7\n" not in report


def test_named_machines_only_and_the_registry_checks(remotes: Path) -> None:
    """A subset installs only those machines; unknown names and an empty registry stop early."""
    assert install_on_machines() == (
        "No machine is registered (`/machine add user@host`); nothing to install."
    )
    make_checkout(remotes / "alpha", "kiss", "https://github.com/ksenxx/kiss.git")
    make_checkout(remotes / "beta", "kiss", "https://github.com/ksenxx/kiss.git")
    save_machines([Machine("alpha", ssh="me@alpha"), Machine("beta", ssh="me@beta")])

    assert install_on_machines("alpha, nope,zed") == (
        "Not registered: nope, zed (see `/machine list`)."
    )
    assert install_on_machines(" beta ").startswith("Installed 1 of 1 machine(s).")
    assert (remotes / "beta" / "ran").exists() and not (remotes / "alpha" / "ran").exists()


def test_timeouts_are_reported(remotes: Path) -> None:
    """A search or an install that outlives its limit is a failure naming the limit."""
    checkout = make_checkout(remotes / "slow", "kiss", "https://github.com/ksenxx/kiss.git")
    (checkout / "install.sh").write_text("#!/bin/bash\nsleep 5\n")
    assert install_machine(Machine("slow", ssh="me@slow"), timeout=1) == (
        f"slow (me@slow): FAILED: install.sh in {checkout} ran over 1 s."
    )
    assert install_machine(Machine("hang", ssh="me@hang"), timeout=1) == (
        "hang (me@hang): FAILED: no answer to the checkout search within 1 s."
    )


def test_agent_run_calls_the_tool_once_and_reports(tmp_path: Path, remotes: Path) -> None:
    """Under the SEA's configuration the model sees only the install tool and finish."""
    make_checkout(remotes / "alpha", "kiss", "https://github.com/ksenxx/kiss.git")
    save_machines([Machine("alpha", ssh="me@alpha")])
    script = [
        tool_call_body("install_on_machines", {"machines": ""}, prompt_tokens=500),
        finish_body("<p>alpha: ok.</p>", prompt_tokens=600),
    ]
    run = sea_commands.evaluate_sea([InstallSea()], "")
    assert run.prompt == sea_module.ADD_TO_PROMPT

    parsed, agentic = run_sea(run, script, tmp_path, max_steps=4)

    assert parsed["success"] is True and "alpha" in parsed["summary"]
    assert (remotes / "alpha" / "ran").exists()
    assert len(agentic) == 2, [list(r) for r in agentic]
    for request in agentic:
        assert tool_names(request) == {"install_on_machines", "finish"}, tool_names(request)
        assert system_message(request).startswith(sea_module.SYSTEM_PROMPT)
    [result] = tool_results(agentic[1])
    assert "Installed 1 of 1 machine(s)." in result and "alpha (me@alpha): ok in" in result
