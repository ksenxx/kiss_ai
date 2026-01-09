# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""KISSEvolve-specific configuration that extends the main KISS config."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class KISSEvolveConfig(BaseModel):
    """KISSEvolve-specific configuration settings."""

    max_generations: int = Field(
        default=10, description="Maximum number of evolutionary generations"
    )
    population_size: int = Field(
        default=8, description="Number of variants to maintain in population"
    )
    mutation_rate: float = Field(
        default=0.7,
        description="Probability of mutating a variant",
    )
    elite_size: int = Field(
        default=2, description="Number of best variants to preserve each generation"
    )
    # Island-based evolution parameters
    num_islands: int = Field(
        default=1, description="Number of islands for island-based evolution (1 = disabled)"
    )
    migration_frequency: int = Field(
        default=5, description="Number of generations between migrations"
    )
    migration_size: int = Field(
        default=1, description="Number of individuals to migrate between islands"
    )
    migration_topology: str = Field(
        default="ring", description="Migration topology: 'ring', 'fully_connected', or 'random'"
    )
    # Code novelty rejection sampling parameters
    enable_novelty_rejection: bool = Field(
        default=False,
        description="Enable code novelty rejection sampling to filter redundant code variants",
    )
    novelty_threshold: float = Field(
        default=0.95,
        description="Cosine similarity threshold for rejecting code "
        "(0.0-1.0, higher = more strict)",
    )
    max_rejection_attempts: int = Field(
        default=5,
        description="Maximum number of rejection attempts before accepting a variant anyway",
    )
    parent_sampling_method: str = Field(
        default="tournament",
        description="Parent sampling method: 'tournament', 'power_law', or 'performance_novelty'",
    )
    power_law_alpha: float = Field(
        default=1.0,
        description=(
            "Power-law sampling parameter (α): lower = more exploration, higher = more exploitation"
        ),
    )
    performance_novelty_lambda: float = Field(
        default=1.0,
        description=(
            "Performance-novelty sampling parameter (λ): controls selection pressure in sigmoid"
        ),
    )


add_config("kiss_evolve", KISSEvolveConfig)
