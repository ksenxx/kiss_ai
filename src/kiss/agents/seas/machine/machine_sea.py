# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Machine agent — registers, forgets and lists remote Sorcar machines.

``/machine add [name] <address> [password]``, ``/machine remove <name>``
and ``/machine list`` edit the registry of
:mod:`kiss.agents.sorcar.machines` (``$KISS_HOME/machines.json``).  The
address is either ``user@host[:port]``, an SSH login (the daemon opens
an SSH tunnel to that machine's daemon port itself and logs in with
the machine's local token, so no password is needed: ``/machine add
ksen@10.0.0.5``), or ``https://host:port`` with the machine's remote
password.  The name defaults to the host.  Once a machine is
registered the CHATS panel lists its tasks next to the local ones
(de-duplicated by task id) and can be filtered by machine name.
"""

from __future__ import annotations

from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea
from kiss.agents.sorcar.machines import (
    MachineError,
    add_machine,
    load_machines,
    machine_status,
    remove_machine,
)

SYSTEM_PROMPT = """You maintain the registry of remote Sorcar machines.

The prompt is one request: `add [name] <address> [password]`, `remove <name>`
or `list` (also phrased in plain English). Do exactly that with one tool call:
- add: `machine_add(name, address, password)`. The address is `user@host[:port]`
  (an SSH login; no password) or `https://host:port` (with the machine's
  password). Pass the name as "" when only an address is given (`add ksen@10.0.0.5`),
  and the password as "" when omitted.
- remove: `machine_remove(name)`.
- list: `machine_list()`.
Then call `finish` with `success=true` and the tool's text as `summary_in_html`
(wrap it in `<p>`/`<ul>`; do not add anything to it). If the request is none of
these, or a required part is missing, finish with `success=false` and one
sentence saying what is expected. Never retry a failed tool call.
"""


class MachineSea(WorkerSea):
    """The ``/machine`` SEA."""

    def description(self) -> str:
        """Return the one-sentence help text shown by ``/machine help``."""
        return (
            "Registers, forgets or lists remote Sorcar machines whose tasks the CHATS panel "
            "shows next to the local ones: `/machine add user@host` (an SSH login; the daemon "
            "port is tunnelled over SSH, no password), `/machine add <name> <https://host:port> "
            "<password>`, `/machine remove <name>`, `/machine list`."
        )

    def system_prompt(self, system_prompt: str) -> str:
        """Return the agent's base system prompt (:data:`SYSTEM_PROMPT`)."""
        return SYSTEM_PROMPT

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A $1 worker with only the registry tools; ``tool_profile`` is locked."""
        return settings | {
            "tool_profile": "none",
            "locked": ["tool_profile"],
            "max_budget": 1.0,
        }

    def tools(self, tools: list[Any]) -> list[Any]:
        """Return *tools* plus the registry tools (add, remove, list)."""
        return tools + [machine_add, machine_remove, machine_list]


def machine_add(name: str, address: str, password: str = "") -> str:
    """Register a remote Sorcar machine (replacing one with the same name).

    Pins the machine's TLS certificate and logs in once before saving,
    so a wrong address or password is reported instead of stored.

    Args:
        name: Short label (letters, digits, ``.``, ``_``, ``-``); empty
            means the address's host.
        address: ``user@host[:port]``, an SSH login: the daemon tunnels
            the machine's daemon port over SSH (key-based login) and
            needs no password.  Or ``https://host:port`` of the remote
            ``kiss-web`` (port 8787 when omitted) with *password*.
        password: The machine's remote password for an ``https://``
            address; empty for an SSH login.

    Returns:
        A confirmation line, or the error line when the machine could not be
        registered.
    """
    try:
        machine = add_machine(name, address, password)
    except MachineError as exc:
        return f"Not registered: {exc}"
    return f"Registered machine '{machine.name}' at {machine.address()}."


def machine_remove(name: str) -> str:
    """Forget a registered machine.

    Args:
        name: The machine's name as shown by :func:`machine_list`.

    Returns:
        A confirmation line, or a line saying no such machine is registered.
    """
    if remove_machine(name):
        return f"Removed machine '{name}'."
    return f"No machine named '{name}' is registered."


def machine_list() -> str:
    """List the registered machines with their address and reachability.

    Returns:
        One line per machine (``name  address  reachable|<error>``, the
        address being ``ssh user@host`` or the URL), or a line saying
        none is registered.
    """
    machines = load_machines()
    if not machines:
        return "No remote machine is registered."
    return "\n".join(f"{m.name}  {m.address()}  {machine_status(m)}" for m in machines)
