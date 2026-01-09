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
    max_agent_budget: float = Field(default=5.0, description="Maximum budget for an agent")
    global_max_budget: float = Field(
        default=10.0, description="Maximum budget for the global agent"
    )


class DockerConfig(BaseModel):
    client_shared_path: str = Field(
        default="/testbed", description="Path inside Docker container for shared volume"
    )


class Config(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig, description="Agent configuration")
    docker: DockerConfig = Field(default_factory=DockerConfig, description="Docker configuration")


DEFAULT_CONFIG = Config()
