"""Configuration for the Autoresearch Agent."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class AutoresearchAgentConfig(BaseModel):
    model_name: str = Field(
        default="claude-opus-4-6",
        description="LLM model to use",
    )
    max_steps: int = Field(
        default=100,
        description="Maximum steps per sub-session",
    )
    max_budget: float = Field(
        default=200.0,
        description="Maximum budget in USD",
    )
    max_sub_sessions: int = Field(
        default=10000,
        description="Maximum number of sub-sessions for auto-continuation",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )


class AutoresearchConfig(BaseModel):
    autoresearch_agent: AutoresearchAgentConfig = Field(
        default_factory=AutoresearchAgentConfig,
        description="Configuration for Autoresearch Agent",
    )


add_config("autoresearch", AutoresearchConfig)
