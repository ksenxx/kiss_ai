# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Install agent — runs ``./install.sh`` in the private kiss checkout of every registered machine.

``/install`` reaches each machine registered with ``/machine add
user@host`` over ssh, finds the clone of :data:`PRIVATE_REPO_URL` in the
login's home (``~/kiss`` first, then any ``~/<dir>`` holding an
``install.sh``, matched by its ``origin`` URL), and runs
``./install.sh --non-interactive`` there.  Machines are installed
concurrently; ``/install <name>[,<name>...]`` installs only those.  A
machine registered by URL has no ssh login and is skipped.  The
remote-access password ``install.sh`` prints is redacted from the report.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.seas.sync.git_sync import repo_key
from kiss.agents.sorcar.machines import Machine, load_machines, ssh_command

PRIVATE_REPO_URL = "https://github.com/ksenxx/kiss.git"
"""The private kiss repository; a machine's checkout is the one whose ``origin`` is this."""

INSTALL_TIMEOUT_SECONDS = 3600
"""How long one machine's ``install.sh`` may run before it is reported as failed."""

SEARCH_TIMEOUT_SECONDS = 60
"""How long the search for a machine's checkout may take."""

TAIL_LINES = 15
"""Lines of each machine's install output kept in the report."""

FIND_CHECKOUTS = (
    'for d in "$HOME/kiss" "$HOME"/*/; do d="${d%/}"; [ -f "$d/install.sh" ] || continue; '
    'u="$(git -C "$d" remote get-url origin 2>/dev/null)" && printf "%s\\t%s\\n" "$d" "$u"; '
    "done; true"
)
"""Remote shell listing ``<dir>\\t<origin url>`` for each home directory holding an install.sh."""

ADD_TO_PROMPT = "Run ./install.sh in the private kiss checkout of every registered machine."
"""Appended to every task: a bare ``/install`` runs on this text alone."""

SYSTEM_PROMPT = """\
You reinstall KISS Sorcar on the user's registered remote machines.
1. Call `install_on_machines` once. Pass `machines` as the comma-separated machine names
   the message names, or "" for every registered machine.
2. Call `finish` with `success=true` when every machine reports ok, else `success=false`;
   `summary_in_html` lists each machine with its outcome and, for a failure, the reason and
   the output lines given. Do not retry and do not work around a failure.
"""
"""The whole base system prompt of the agent (replaces ``SYSTEM.md``)."""

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PASSWORD = re.compile(r"(?i)(password:\s*).*")
"""The rest of a line after ``password:``: a user-chosen password may contain spaces."""


def install_on_machines(machines: str = "") -> str:
    """Run ``./install.sh --non-interactive`` in the private kiss checkout of registered machines.

    Every machine is installed concurrently over its own ssh login.

    Args:
        machines: Comma-separated registered machine names; empty for
            every registered machine.

    Returns:
        One block per machine: ``ok`` or ``FAILED`` with the reason, the
        checkout used and the last lines of the install output.
    """
    registered = load_machines()
    if machines.strip():
        known = {machine.name: machine for machine in registered}
        wanted = [name.strip() for name in machines.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in known]
        if unknown:
            return "Not registered: " + ", ".join(unknown) + " (see `/machine list`)."
        registered = [known[name] for name in wanted]
    if not registered:
        return "No machine is registered (`/machine add user@host`); nothing to install."
    with ThreadPoolExecutor(max_workers=len(registered)) as pool:
        blocks = list(pool.map(install_machine, registered))
    failed = sum(1 for block in blocks if ": FAILED" in block.splitlines()[0])
    return f"Installed {len(blocks) - failed} of {len(blocks)} machine(s).\n\n" + "\n\n".join(
        blocks
    )


def install_machine(machine: Machine, timeout: int = INSTALL_TIMEOUT_SECONDS) -> str:
    """Find *machine*'s private kiss checkout, run its ``install.sh`` and return a report block.

    Args:
        machine: A registered machine.
        timeout: Seconds the install may run; the checkout search gets
            at most :data:`SEARCH_TIMEOUT_SECONDS` of them.
    """
    head = f"{machine.name} ({machine.ssh or 'no ssh'})"
    if not machine.ssh:
        return f"{head}: FAILED: registered by URL, not as an ssh login; skipped."
    search_timeout = min(SEARCH_TIMEOUT_SECONDS, timeout)
    try:
        found = _ssh(machine.ssh, FIND_CHECKOUTS, search_timeout)
    except subprocess.TimeoutExpired:
        return f"{head}: FAILED: no answer to the checkout search within {search_timeout} s."
    if found.returncode != 0:
        return f"{head}: FAILED: cannot connect: {_tail(found.stdout) or found.returncode}"
    checkout = find_checkout(found.stdout)
    if not checkout:
        return f"{head}: FAILED: no clone of {PRIVATE_REPO_URL} with an install.sh in the home."
    command = f"cd {shlex.quote(checkout)} && ./install.sh --non-interactive"
    try:
        done = _ssh(machine.ssh, command, timeout)
    except subprocess.TimeoutExpired:
        return f"{head}: FAILED: install.sh in {checkout} ran over {timeout} s."
    status = "ok" if done.returncode == 0 else f"FAILED: install.sh exited {done.returncode}"
    return f"{head}: {status} in {checkout}\n{_tail(done.stdout)}"


def find_checkout(listing: str) -> str:
    """Return the first directory of *listing* whose origin is :data:`PRIVATE_REPO_URL`, or "".

    Args:
        listing: ``<dir>\\t<origin url>`` lines as :data:`FIND_CHECKOUTS` prints them.
    """
    wanted = repo_key(PRIVATE_REPO_URL)
    for line in listing.splitlines():
        directory, _, url = line.partition("\t")
        if url and repo_key(url) == wanted:
            return directory
    return ""


def _ssh(login: str, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run *command* on *login* with stderr folded into stdout."""
    return subprocess.run(
        ssh_command(login, command=command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _tail(output: str) -> str:
    """The last :data:`TAIL_LINES` non-blank lines of *output*, uncoloured, password redacted."""
    lines = [line for line in _ANSI.sub("", output).splitlines() if line.strip()]
    return "\n".join(_PASSWORD.sub(r"\1[redacted]", line) for line in lines[-TAIL_LINES:])


class InstallSea(WorkerSea):
    """The ``/install`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/install help``."""
        return (
            "Runs `./install.sh --non-interactive` in the private kiss checkout (the clone of "
            f"{PRIVATE_REPO_URL}) of every machine registered over ssh (`/machine list`), "
            "concurrently, and reports each machine's outcome; use `/install` in the chat, "
            '`/install <machine>[,<machine>]` for some of them, or run_agent(agent="install", '
            'task="").'
        )

    def prompt(self, task: str) -> str:
        """Return the message followed by :data:`ADD_TO_PROMPT`."""
        return f"{task.strip()}\n\n{ADD_TO_PROMPT}".strip()

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A $1 worker whose only tool is :func:`install_on_machines`; ``tool_profile`` is locked.

        ``timeout`` leaves room for the slowest machine's install
        (:data:`INSTALL_TIMEOUT_SECONDS`) plus the two model calls.
        """
        return settings | {
            "tool_profile": "none",
            "max_budget": 1.0,
            "timeout": INSTALL_TIMEOUT_SECONDS + 600,
            "locked": ["tool_profile"],
        }

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Return *tools* plus :func:`install_on_machines`."""
        return [*tools, install_on_machines]
