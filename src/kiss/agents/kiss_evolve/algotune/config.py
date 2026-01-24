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
    algotune_repo_url: str = Field(
        default="https://github.com/oripress/AlgoTune.git",
        description="Git URL for cloning the AlgoTune repository",
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

    # Model settings
    model: str = Field(
        default="gemini-3-flash-preview",
        description="Model to use for code generation",
    )


add_config("algotune", AlgoTuneConfig)
