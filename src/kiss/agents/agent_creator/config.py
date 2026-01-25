# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration for Agent Creator - Improver and Evolver agents."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class ImproverConfig(BaseModel):
    """Configuration for the Improver agent that optimizes agent code."""

    # Model settings
    model: str = Field(
        default="claude-sonnet-4-5",
        description="LLM model to use for the improver agent",
    )

    # Agent execution limits
    max_steps: int = Field(
        default=150,
        description="Maximum steps for the improver agent per generation",
    )
    max_budget: float = Field(
        default=15.0,
        description="Maximum budget in USD for the improver agent per generation",
    )

    # Multi-generation improvement settings
    num_generations: int = Field(
        default=5,
        description="Number of improvement generations to run",
    )
    pareto_frontier_size: int = Field(
        default=4,
        description="Maximum size of the Pareto frontier to maintain",
    )
    mutation_probability: float = Field(
        default=0.5,
        description="Probability of sampling for mutation vs crossover",
    )

    # Pruning settings
    min_improvement_threshold: float = Field(
        default=0.01,
        description="Minimum improvement (1%) required to keep a variant",
    )
    prune_non_frontier: bool = Field(
        default=True,
        description="Whether to prune variants not in the Pareto frontier",
    )


class EvolverConfig(BaseModel):
    """Configuration for the AgentEvolver that maintains the Pareto frontier."""

    # Model settings
    model: str = Field(
        default="claude-sonnet-4-5",
        description="LLM model to use for the evolver orchestration",
    )

    # Evolution settings
    max_generations: int = Field(
        default=10,
        description="Maximum number of evolutionary generations",
    )
    population_size: int = Field(
        default=8,
        description="Maximum size of Pareto frontier",
    )
    pareto_size: int = Field(
        default=6,
        description="Maximum number of solutions to keep in the Pareto frontier",
    )

    # Sampling settings
    mutation_probability: float = Field(
        default=0.5,
        description="Probability of mutation vs crossover operation",
    )
    crossover_improvement_threshold: float = Field(
        default=0.95,
        description="Crossover must improve by this factor over both parents",
    )

    # Evaluation settings
    evaluation_model: str = Field(
        default="claude-sonnet-4-5",
        description="Model to use for evaluating agent performance",
    )
    evaluation_runs: int = Field(
        default=3,
        description="Number of evaluation runs to average for metrics",
    )
    evaluation_timeout: int = Field(
        default=600,
        description="Timeout in seconds for each evaluation run",
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
add_config("agent_creator", AgentCreatorConfig)
