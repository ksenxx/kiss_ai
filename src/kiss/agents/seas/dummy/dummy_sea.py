# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The default Sorcar Extension Agent: a plain Sorcar session.

``run_agent`` runs this SEA when its ``agent`` argument is empty.  It
defines no ``run`` parameter getters, so the sub-task is an ordinary
Sorcar session on the given task in the calling task's work directory
with the standard system prompt and tools.
"""


def description() -> str:
    """Return the one-sentence help text shown by ``/dummy help``."""
    return (
        "Runs the task as a plain Sorcar sub-agent with the default system "
        "prompt, tools and work directory; use it as `/dummy <task>` or by "
        "calling run_agent with an empty `agent` argument."
    )
