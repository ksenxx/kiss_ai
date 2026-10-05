# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The default Sorcar Extension Agent: a plain Sorcar session.

``run_agent`` runs this SEA when its ``agent`` argument is empty.  It
pins no setting, so the sub-task is an ordinary Sorcar session on the
given task in the calling task's work directory with the standard
system prompt and tools.  It is ``hidden``: an empty ``agent`` is the
way to ask for it, so it is no ``/dummy`` command and no agent name.
"""


def settings() -> dict:
    """Return the SEA's settings: hidden from the command list, nothing else."""
    return {"hidden": True}


def description() -> str:
    """Return the one-sentence help text shown by ``/dummy help``."""
    return (
        "Runs the task as a plain Sorcar sub-agent with the default system "
        "prompt, tools and work directory; use it as `/dummy <task>` or by "
        "calling run_agent with an empty `agent` argument."
    )
