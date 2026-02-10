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
    trials: int = Field(
        default=200,
        description="Number of trials for failed subtasks",
    )


class KISSCodingAgentConfig(BaseModel):
    orchestrator_model_name: str = Field(
        default="claude-opus-4-6",
        description="LLM model to use for KISS Coding Agent",
    )
    subtasker_model_name: str = Field(
        default="claude-opus-4-6",
        description="LLM model to use for subtask generation and execution",
    )
    refiner_model_name: str = Field(
        default="claude-sonnet-4-5",
        description="LLM model to use for prompt refinement of failed tasks",
    )
    max_steps: int = Field(
        default=200,
        description="Maximum steps for the KISS Coding Agent",
    )
    max_budget: float = Field(
        default=100.0,
        description="Maximum budget in USD for the KISS Coding Agent",
    )
    trials: int = Field(
        default=200,
        description="Number of trials for failed subtasks",
    )


class CodingAgentConfig(BaseModel):
    relentless_coding_agent: RelentlessCodingAgentConfig = Field(
        default_factory=RelentlessCodingAgentConfig,
        description="Configuration for Relentless Coding Agent",
    )
    kiss_coding_agent: KISSCodingAgentConfig = Field(
        default_factory=KISSCodingAgentConfig,
        description="Configuration for KISS Coding Agent",
    )


# Register config with the global DEFAULT_CONFIG
add_config("coding_agent", CodingAgentConfig)
