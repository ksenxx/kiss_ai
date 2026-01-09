# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration Pydantic models for KISS agent settings with CLI support."""

import os

from pydantic import BaseModel, Field


class APIKeysConfig(BaseModel):
    GEMINI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Gemini API key (can also be set via GEMINI_API_KEY env var)",
    )
    OPENAI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key (can also be set via OPENAI_API_KEY env var)",
    )
    ANTHROPIC_API_KEY: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""),
        description="Anthropic API key (can also be set via ANTHROPIC_API_KEY env var)",
    )
    TOGETHER_API_KEY: str = Field(
        default_factory=lambda: os.getenv("TOGETHER_API_KEY", ""),
        description="Together API key (can also be set via TOGETHER_API_KEY env var)",
    )
    OPENROUTER_API_KEY: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""),
        description="OpenRouter API key (can also be set via OPENROUTER_API_KEY env var)",
    )


class AgentConfig(BaseModel):
    api_keys: APIKeysConfig = Field(
        default_factory=APIKeysConfig, description="API keys configuration"
    )
    max_steps: int = Field(default=100, description="Maximum iterations in the ReAct loop")
    verbose: bool = Field(default=True, description="Enable verbose output")
    debug: bool = Field(default=False, description="Enable debug mode")
    artifact_dir: str = Field(default="artifacts", description="Directory to save artifacts")
    max_agent_budget: float = Field(default=1.0, description="Maximum budget for an agent")
    global_max_budget: float = Field(
        default=10.0, description="Maximum budget for the global agent"
    )
    use_google_search: bool = Field(default=True, description="Use Google search")


class DockerConfig(BaseModel):
    client_shared_path: str = Field(
        default="/testbed", description="Path inside Docker container for shared volume"
    )


class SelfEvolvingMultiAgentConfig(BaseModel):
    model: str = Field(default="gpt-4", description="Model name for the agent")
    docker_image: str = Field(default="python:3.11", description="Docker image to use")
    workdir: str = Field(default="/workspace", description="Working directory in container")
    max_steps: int = Field(default=30, description="Maximum steps for orchestrator")
    max_budget: float = Field(default=1.5, description="Maximum budget for orchestrator")
    enable_planning: bool = Field(default=True, description="Enable task planning")
    enable_error_recovery: bool = Field(default=True, description="Enable error recovery")
    enable_dynamic_tools: bool = Field(default=True, description="Enable dynamic tool creation")
    sub_agent_max_steps: int = Field(default=10, description="Max steps for sub-agents")
    sub_agent_max_budget: float = Field(default=0.5, description="Max budget for sub-agents")
    max_retries: int = Field(default=3, description="Maximum retries on error")
    max_dynamic_tools: int = Field(default=5, description="Maximum number of dynamic tools")
    max_plan_items: int = Field(default=10, description="Maximum items in a plan")
    # Evolver settings
    evolver_test_only: bool = Field(default=False, description="Run evolver in test mode only")
    evolver_model: str = Field(default="gpt-4", description="Model for agent evolution")
    evolver_population_size: int = Field(default=5, description="Population size for evolution")
    evolver_max_generations: int = Field(default=10, description="Max generations for evolution")
    evolver_output: str = Field(
        default="evolved_agent.py", description="Output file for evolved agent"
    )


class Config(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig, description="Agent configuration")
    docker: DockerConfig = Field(default_factory=DockerConfig, description="Docker configuration")
    self_evolving_multi_agent: SelfEvolvingMultiAgentConfig = Field(
        default_factory=SelfEvolvingMultiAgentConfig,
        description="Self-evolving multi-agent configuration"
    )


DEFAULT_CONFIG = Config()
