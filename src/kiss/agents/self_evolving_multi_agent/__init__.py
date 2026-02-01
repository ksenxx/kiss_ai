# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Self Evolving Multi Agent - A long-horizon coding agent with planning and error recovery."""

from kiss.agents.self_evolving_multi_agent.multi_agent import (
    SelfEvolvingMultiAgent,
    run_task,
)

__all__ = ["SelfEvolvingMultiAgent", "run_task"]
