# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The SEA of one scheduled run of a cron prompt job.

:func:`kiss.agents.sorcar.cron_agent._run_prompt_job` runs every LLM
cron job through ``run_agent(agent=<this file>, task=<preamble + job
prompt>, ...)``: the job's model, budget and timeout travel as the
tool's arguments, its work directory as ``options``, and the task
text is the prompt.  This file pins what every unattended run needs:
no task classification (an unattended run must not stall on it) and
the short system prompt of :data:`CRON_RUN_RULES` on top of the lite
base instead of the full ``SYSTEM.md`` — a scheduled run reads its
mailbox or runs a check, it does not need the 27 KB of development and
research rules on every turn.  It is not registered as a slash command.
"""

from typing import Any

from kiss.agents.seas.base.base_sea import BaseSea
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE

CRON_RUN_RULES = """

<scheduled_run>

## Scheduled (unattended) run

- Nobody is watching: never call ask_user_question or wait for approval; when an action is \
blocked, say so in the final summary and finish.
- The task message starts with a "# Run context" block: the current time, the schedule, \
the window start (the last successful run: cover everything since), the previous attempt \
and the tail of its result. Use it instead of running `date` or reading the job's log.
- Your final finish(summary_in_html=...) is delivered verbatim to the job's targets and \
appended to the job's log by the runner; reply with exactly [SILENT] when there is nothing \
worth reporting.
- Tools: Read a file before you Edit or Write it; keep scratch files under ./tmp/; batch \
independent shell commands into one Bash call; a command expected to run longer than a \
minute goes to Bash(background=true) + bash_job(wait).
- Do not create, modify or remove scheduled jobs.

</scheduled_run>
"""


def cron_run_system_prompt(system_prompt: str) -> str:
    """Return the run's system prompt: the lite base (in place of the full one) plus the cron rules.

    A suffix the daemon appended after the base (``system_prompt``
    setting, SEA layers) is kept.  A prompt that does not start with
    the full ``SYSTEM.md`` (a caller-supplied base) is returned with
    the cron rules appended.

    Args:
        system_prompt: The assembled system prompt of the run.
    """
    if system_prompt.startswith(SYSTEM_PROMPT):
        return SYSTEM_PROMPT_LITE + CRON_RUN_RULES + system_prompt[len(SYSTEM_PROMPT):]
    return system_prompt + CRON_RUN_RULES


class CronPromptSea(BaseSea):
    """The SEA a scheduled cron prompt job runs as (not a slash command)."""

    def description(self) -> str:
        """Return the help text of this script."""
        return (
            "Runs one scheduled execution of a cron prompt job; launched by the "
            "cron agent, not meant to be invoked by hand."
        )

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Return the run settings of a scheduled prompt job."""
        return settings | {"auto_classify": False}

    def system_prompt(self, system_prompt: str) -> str:
        """Return the short system prompt of a scheduled run (:func:`cron_run_system_prompt`)."""
        return cron_run_system_prompt(system_prompt)


