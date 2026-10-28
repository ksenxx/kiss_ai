# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``plan_run_agent`` on a machine: a SEA reached by its path keeps its alias.

``dispatch_command`` used to walk the command registry twice on the
remote path (``remote_sea`` for ``sea_command`` and ``command_alias``
again for the ``ran:`` line's alias); it now reuses ``remote_sea``'s
name.  This pins that a call that names a registered SEA by its
script path still gets the alias hint, and that a call by name does
not (``alias == ""`` is covered in ``test_run_agent_machine``).
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar import machines as mach
from kiss.agents.sorcar.agent_dispatch import plan_run_agent
from kiss.agents.sorcar.sea_commands import get_command


def test_remote_sea_by_path_keeps_its_command_alias(tmp_path: Path) -> None:
    saved = mach.load_machines()
    mach.save_machines([mach.Machine(name="box", url="https://203.0.113.7:8787", password="x")])
    try:
        cron_path = get_command("cron")
        assert cron_path is not None
        plan = plan_run_agent(str(tmp_path), "list jobs", agent=str(cron_path), machine="box")
        assert isinstance(plan, dict)
        assert (plan["sea_path"], plan["sea_command"], plan["alias"]) == ("", "cron", "cron")
    finally:
        mach.save_machines(saved)
