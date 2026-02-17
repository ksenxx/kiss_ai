"""Configuration for IMO Agent."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class IMOAgentConfig(BaseModel):
    solver_model: str = Field(
        default="o3",
        description="Model for solving IMO problems (should be a powerful reasoning model)",
    )
    verifier_model: str = Field(
        default="gemini-2.5-pro",
        description="Model for verifying solutions",
    )
    validator_model: str = Field(
        default="gemini-2.5-pro",
        description="Model for independent validation against known answers",
    )
    max_refinement_rounds: int = Field(
        default=2,
        description="Max verification-refinement iterations per attempt",
    )
    num_verify_passes: int = Field(
        default=1,
        description="Number of verification passes required to accept a solution",
    )
    max_attempts: int = Field(
        default=1,
        description="Max independent attempts per problem",
    )
    max_budget: float = Field(
        default=50.0,
        description="Maximum budget in USD per problem",
    )


add_config("imo_agent", IMOAgentConfig)
