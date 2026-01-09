# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""AlgoTune-specific configuration that extends the main KISS config."""

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config

# Default path for AlgoTune repository
_DEFAULT_ALGOTUNE_PATH = str(Path(tempfile.gettempdir()) / "AlgoTune")


class AlgoTuneConfig(BaseModel):
    """AlgoTune-specific configuration settings."""

    # Task settings
    task: str = Field(
        default="matrix_multiplication",
        description="AlgoTune task name to optimize",
    )
    all_tasks: bool = Field(
        default=False,
        description="Solve all tasks in AlgoTuneTasks directory",
    )
    algotune_path: str = Field(
        default=_DEFAULT_ALGOTUNE_PATH,
        description="Path to the AlgoTune repository",
    )

    # Problem generation settings
    num_test_problems: int = Field(
        default=3,
        description="Number of test problems to generate for evaluation",
    )
    problem_size: int = Field(
        default=100,
        description="Size parameter for problem generation (task-specific)",
    )
    num_timing_runs: int = Field(
        default=5,
        description="Number of timing runs for performance measurement",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    # Evolution settings
    population_size: int = Field(
        default=4,
        description="Population size for evolutionary optimization",
    )
    max_generations: int = Field(
        default=3,
        description="Maximum number of generations to evolve",
    )
    mutation_rate: float = Field(
        default=0.8,
        description="Probability of mutation vs crossover",
    )
    elite_size: int = Field(
        default=1,
        description="Number of best variants to preserve each generation",
    )

    # Model settings
    model: str = Field(
        default="gemini-2.5-pro",
        description="Model to use for code generation",
    )


add_config("algotune", AlgoTuneConfig)
