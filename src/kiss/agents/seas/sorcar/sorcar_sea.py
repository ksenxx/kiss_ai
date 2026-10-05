# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The default Sorcar Extension Agent: a plain Sorcar session.

``run_agent`` runs this SEA when its ``agent`` argument is empty or
``"sorcar"``.  It pins no setting, so the sub-task is an ordinary
Sorcar session on the given task in the calling task's work directory
with the standard system prompt and tools.  It is ``hidden``: an empty
``agent`` is the way to ask for it, so it is no ``/sorcar`` command.
"""


from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea


class SorcarSea(BaseSea):
    """The plain Sorcar sub-agent."""

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Return the SEA's settings: hidden from the command list, nothing else."""
        return settings | {"hidden": True}

    def description(self) -> str:
        """Return the one-sentence help text of the plain sub-agent."""
        return (
            "Runs the task as a plain Sorcar sub-agent with the default system "
            "prompt, tools and work directory; ask for it with an empty `agent` "
            "argument (or `agent=\"sorcar\"`) of run_agent."
        )
