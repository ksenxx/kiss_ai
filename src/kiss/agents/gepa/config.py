# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""GEPA-specific configuration that extends the main KISS config."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class GEPAConfig(BaseModel):
    """GEPA-specific configuration settings."""

    reflection_model: str = Field(
        default="gemini-3-flash-preview",
        description="Model to use for reflection (e.g., 'gemini-3-flash-preview')",
    )
    max_generations: int = Field(
        default=10, description="Maximum number of evolutionary generations"
    )
    population_size: int = Field(
        default=8, description="Number of candidates to maintain in population"
    )
    pareto_size: int = Field(default=4, description="Maximum size of Pareto frontier")
    mutation_rate: float = Field(
        default=0.5,
        description="Probability of mutating a prompt template in each generation",
    )
    crossover_probability: float = Field(
        default=0.3,
        description="Probability of combining with lessons from Pareto frontier using crossover",
    )
    rollouts_per_generation: int = Field(default=1, description="Number of rollouts per generation")


add_config("gepa", GEPAConfig)
