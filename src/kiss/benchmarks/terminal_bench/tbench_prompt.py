# Author: Koushik Sen (ksen@berkeley.edu)

"""Terminal-Bench system prompt for the sorcar agent."""

from pathlib import Path

_TBENCH_MD = (Path(__file__).parent / "tbench.md").read_text()

TBENCH_SYSTEM_PROMPT = """\
# FOCUS ON THE GIVEN TASK. ITS COMPLETION IS YOUR SOLE GOAL. BE RELENTLESS.

# Rules

- Write() for new files. Edit() for small changes.
- Run Bash commands synchronously using the `timeout_seconds` parameter.
  Use 300s (default) for builds/installations.
- Call finish(success=True, summary="detailed summary") immediately when task is complete.
- READ large files in chunks.

""" + _TBENCH_MD
