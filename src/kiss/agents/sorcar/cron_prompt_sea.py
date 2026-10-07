# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The SEA (SEA) of one scheduled run of a cron prompt job.

:func:`kiss.agents.sorcar.cron_agent._run_prompt_job` runs every LLM
cron job through ``run_agent(agent=<this file>, task=<preamble + job
prompt>, ...)``: the job's model, budget, work directory, worktree and
auto-commit choices travel as the tool's arguments and ``options``,
the task text is the prompt.  This file only pins what every
unattended run needs: no task classification (an unattended run must
not stall on it).  It is not registered as a slash command.
"""

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea


class CronPromptSea(BaseSea):
    """The ``/cron_prompt`` SEA."""

    def description(self) -> str:
        """Return the help text of this script."""
        return (
            "Runs one scheduled execution of a cron prompt job; launched by the "
            "cron agent, not meant to be invoked by hand."
        )

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Return the run settings of a scheduled prompt job."""
        return settings | {"auto_classify": False}


