# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Terminal-Bench under the Harbor framework, with KISS Sorcar as the agent.

Modules:

- :mod:`.daemon`: a private KISS daemon for benchmark runs (no browser, no
  persistent memory, no task classification).
- :mod:`.harbor_agent`: the Harbor ``BaseAgent`` that drives the coding SEA
  on a trial container through that daemon.
- :mod:`.run`: the ``harbor run`` launcher with the Terminal-Bench settings.
- :mod:`.summarize`: per-task and overall scores of a finished Harbor job.
"""
