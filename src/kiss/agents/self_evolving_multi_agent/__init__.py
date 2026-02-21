# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""
.. deprecated::
    This module is deprecated. Use ``kiss.agents.coding_agents.agent_optimizer`` instead.
"""

import warnings

warnings.warn(
    "kiss.agents.self_evolving_multi_agent is deprecated. "
    "Use kiss.agents.coding_agents.agent_optimizer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from kiss.agents.self_evolving_multi_agent.multi_agent import (
    SelfEvolvingMultiAgent,
    run_task,
)

__all__ = ["SelfEvolvingMultiAgent", "run_task"]
