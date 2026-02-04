# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Agent Creator module for evolving and improving AI agents.

This module provides tools for:
1. ImproverAgent: Optimizes existing agent code for token efficiency and speed
2. AgentEvolver: Maintains a Pareto frontier of agent implementations

Example usage:

    from kiss.agents.create_and_optimize_agent import AgentEvolver, ImproverAgent

    # Create and evolve an agent for a task
    evolver = AgentEvolver(
        task_description="Build a code analysis assistant",
        max_generations=10,
    )
    best = evolver.evolve()

    # Or improve an existing agent
    improver = ImproverAgent()
    success, report = improver.improve(
        source_folder="/path/to/agent",
        work_dir="/path/to/improved",
        task_description="Build a code analysis assistant",
    )
"""

from kiss.agents.create_and_optimize_agent.agent_evolver import AgentEvolver, AgentVariant
from kiss.agents.create_and_optimize_agent.config import (
    AgentCreatorConfig,
    EvolverConfig,
    ImproverConfig,
)
from kiss.agents.create_and_optimize_agent.improver_agent import ImprovementReport, ImproverAgent

__all__ = [
    "AgentEvolver",
    "AgentVariant",
    "ImproverAgent",
    "ImprovementReport",
    "AgentCreatorConfig",
    "ImproverConfig",
    "EvolverConfig",
]
