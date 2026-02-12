"""Configuration Pydantic models for coding agent settings."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class RelentlessCodingAgentConfig(BaseModel):
    model_name: str = Field(
        default="claude-opus-4-6",
        description="LLM model to use",
    )
    max_steps: int = Field(
        default=200,
        description="Maximum steps for the Relentless Coding Agent",
    )
    max_budget: float = Field(
        default=200.0,
        description="Maximum budget in USD for the Relentless Coding Agent",
    )
    max_sub_sessions: int = Field(
        default=200,
        description="Maximum number of sub-sessions for auto-continuation",
    )


class CodingAgentConfig(BaseModel):
    relentless_coding_agent: RelentlessCodingAgentConfig = Field(
        default_factory=RelentlessCodingAgentConfig,
        description="Configuration for Relentless Coding Agent",
    )


# Register config with the global DEFAULT_CONFIG
add_config("coding_agent", CodingAgentConfig)
