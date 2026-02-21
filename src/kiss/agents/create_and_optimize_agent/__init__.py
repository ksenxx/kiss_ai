# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""DEPRECATED: Use kiss.agents.coding_agents.agent_optimizer instead.

This module is deprecated and will be removed in a future release.
Please migrate to kiss.agents.coding_agents.agent_optimizer.
"""

import warnings

from kiss.agents.create_and_optimize_agent.agent_evolver import (
    AgentEvolver,
    AgentVariant,
    EvolverPhase,
    EvolverProgress,
    create_progress_callback,
)
from kiss.agents.create_and_optimize_agent.config import (
    AgentCreatorConfig,
    EvolverConfig,
    ImproverConfig,
)
from kiss.agents.create_and_optimize_agent.improver_agent import ImprovementReport, ImproverAgent

warnings.warn(
    "kiss.agents.create_and_optimize_agent is deprecated. "
    "Use kiss.agents.coding_agents.agent_optimizer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AgentEvolver",
    "AgentVariant",
    "EvolverPhase",
    "EvolverProgress",
    "create_progress_callback",
    "ImproverAgent",
    "ImprovementReport",
    "AgentCreatorConfig",
    "ImproverConfig",
    "EvolverConfig",
]
