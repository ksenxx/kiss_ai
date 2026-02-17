"""Configuration for the Assistant Agent."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class AssistantAgentConfig(BaseModel):
    model_name: str = Field(
        default="claude-opus-4-6",
        description="LLM model to use",
    )
    max_steps: int = Field(
        default=25,
        description="Maximum steps per sub-session",
    )
    max_budget: float = Field(
        default=200.0,
        description="Maximum budget in USD",
    )
    max_sub_sessions: int = Field(
        default=200,
        description="Maximum number of sub-sessions for auto-continuation",
    )
    headless: bool = Field(
        default=False,
        description="Run browser in headless mode",
    )


class AssistantConfig(BaseModel):
    assistant_agent: AssistantAgentConfig = Field(
        default_factory=AssistantAgentConfig,
        description="Configuration for Assistant Agent",
    )


add_config("assistant", AssistantConfig)
