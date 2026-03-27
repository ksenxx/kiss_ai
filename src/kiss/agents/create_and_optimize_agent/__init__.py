# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""DEPRECATED: This module is deprecated and will be removed in a future release."""

import warnings

from kiss.agents.create_and_optimize_agent.agent_evolver import (
    AgentEvolver,
    AgentVariant,
    EvolverPhase,
    EvolverProgress,
    create_progress_callback,
)
from kiss.agents.create_and_optimize_agent.improver_agent import ImprovementReport, ImproverAgent

warnings.warn(
    "kiss.agents.create_and_optimize_agent is deprecated.",
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
]
