# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration for Self Evolving Multi Agent."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class SelfEvolvingMultiAgentConfig(BaseModel):
    """Configuration for the Self Evolving Multi Agent.

    This Pydantic model defines all configuration options for running
    the self-evolving multi-agent system, including model selection,
    agent limits, sub-agent settings, and Docker execution environment.

    Attributes:
        model: LLM model to use for the main agent.
        sub_agent_model: LLM model for sub-agents.
        evolver_model: LLM model for evolution/optimization.
        max_steps: Maximum orchestrator steps allowed.
        max_budget: Maximum budget in USD for the agent.
        max_retries: Maximum retries on error.
        sub_agent_max_steps: Maximum steps for sub-agents.
        sub_agent_max_budget: Maximum budget for sub-agents in USD.
        docker_image: Docker image for code execution.
        workdir: Working directory inside the Docker container.
    """

    # Model settings
    model: str = Field(
        default="gemini-3-flash-preview",
        description="LLM model to use for the agent",
    )

    sub_agent_model: str = Field(
        default="gemini-3-flash-preview",
        description="Model for sub-agents",
    )

    evolver_model: str = Field(
        default="gemini-3-flash-preview",
        description="Model for evolution",
    )

    # Agent settings
    max_steps: int = Field(
        default=100,
        description="Maximum orchestrator steps",
    )
    max_budget: float = Field(
        default=10.0,
        description="Maximum budget in USD",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries on error",
    )

    # Sub-agent settings
    sub_agent_max_steps: int = Field(
        default=50,
        description="Maximum steps for sub-agents",
    )
    sub_agent_max_budget: float = Field(
        default=2.0,
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



# Register config with the global DEFAULT_CONFIG
add_config("self_evolving_multi_agent", SelfEvolvingMultiAgentConfig)
