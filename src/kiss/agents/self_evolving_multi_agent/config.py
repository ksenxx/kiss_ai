# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration for Self Evolving Multi Agent."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class SelfEvolvingMultiAgentConfig(BaseModel):
    """Configuration for the Self Evolving Multi Agent."""

    # Model settings
    model: str = Field(
        default="gemini-3-flash-preview",
        description="LLM model to use for the agent",
    )

    # Agent settings
    max_steps: int = Field(
        default=50,
        description="Maximum orchestrator steps",
    )
    max_budget: float = Field(
        default=2.0,
        description="Maximum budget in USD",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries on error",
    )

    # Sub-agent settings
    sub_agent_max_steps: int = Field(
        default=15,
        description="Maximum steps for sub-agents",
    )
    sub_agent_max_budget: float = Field(
        default=0.5,
        description="Maximum budget for sub-agents in USD",
    )

    # Docker settings
    docker_image: str = Field(
        default="python:3.12-slim",
        description="Docker image for execution",
    )
    workdir: str = Field(
        default="/workspace",
        description="Working directory in container",
    )

    # Dynamic tool settings
    enable_dynamic_tools: bool = Field(
        default=True,
        description="Enable dynamic tool creation",
    )
    max_dynamic_tools: int = Field(
        default=5,
        description="Maximum number of dynamic tools",
    )

    # Planning settings (for multi_agent.py)
    enable_planning: bool = Field(
        default=True,
        description="Enable planning capabilities",
    )
    max_plan_items: int = Field(
        default=10,
        description="Maximum items in a plan",
    )

    # Error recovery settings (for multi_agent.py)
    enable_error_recovery: bool = Field(
        default=True,
        description="Enable error recovery",
    )

    # Output settings
    verbose: bool = Field(
        default=True,
        description="Enable verbose output",
    )
    save_trajectories: bool = Field(
        default=True,
        description="Save agent trajectories",
    )

    # Evolver settings (for coding_agent_evolver.py)
    evolver_population_size: int = Field(
        default=4,
        description="Population size for evolution",
    )
    evolver_max_generations: int = Field(
        default=3,
        description="Maximum generations for evolution",
    )
    evolver_mutation_rate: float = Field(
        default=0.7,
        description="Mutation rate for evolution",
    )
    evolver_elite_size: int = Field(
        default=1,
        description="Elite size for evolution",
    )
    evolver_model: str = Field(
        default="gemini-3-flash-preview",
        description="Model for evolution",
    )

    # Test task settings
    test_task_timeout: int = Field(
        default=300,
        description="Timeout per task in seconds",
    )

    # Evolver output settings
    evolver_output: str = Field(
        default="evolved_agent.py",
        description="Output file for best evolved agent",
    )
    evolver_test_only: bool = Field(
        default=False,
        description="Only test the base agent without evolution",
    )


# Register config with the global DEFAULT_CONFIG
add_config("self_evolving_multi_agent", SelfEvolvingMultiAgentConfig)
