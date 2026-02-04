# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration for Agent Creator - Improver and Evolver agents."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class ImproverConfig(BaseModel):
    """Configuration for the Improver agent that optimizes agent code."""

    model_name: str = Field(
        default="claude-sonnet-4-5",
        description="LLM model to use for the improver agent",
    )
    max_steps: int = Field(
        default=100,
        description="Maximum steps for the improver agent",
    )
    max_budget: float = Field(
        default=20.0,
        description="Maximum budget in USD for the improver agent",
    )


class EvolverConfig(BaseModel):
    """Configuration for the AgentEvolver that maintains the Pareto frontier."""

    model_name: str = Field(
        default="claude-sonnet-4-5",
        description="LLM model to use for agent creation and improvement",
    )
    max_generations: int = Field(
        default=10,
        description="Maximum number of improvement generations",
    )
    initial_frontier_size: int = Field(
        default=4,
        description="Initial size of the Pareto frontier",
    )
    max_frontier_size: int = Field(
        default=6,
        description="Maximum size of the Pareto frontier",
    )
    mutation_probability: float = Field(
        default=0.8,
        description="Probability of mutation vs crossover (1.0 = always mutate)",
    )
    initial_agent_max_steps: int = Field(
        default=50,
        description="Maximum steps for creating the initial agent",
    )
    initial_agent_max_budget: float = Field(
        default=50.0,
        description="Maximum budget in USD for creating the initial agent",
    )


class AgentCreatorConfig(BaseModel):
    """Combined configuration for agent creation and evolution."""

    improver: ImproverConfig = Field(
        default_factory=ImproverConfig,
        description="Configuration for the improver agent",
    )
    evolver: EvolverConfig = Field(
        default_factory=EvolverConfig,
        description="Configuration for the evolver agent",
    )


# Register config with the global DEFAULT_CONFIG
add_config("create_and_optimize_agent", AgentCreatorConfig)
