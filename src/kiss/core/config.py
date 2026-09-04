# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration Pydantic models for KISS agent settings."""

import os
import random
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

_PROJECT_DIR = Path(__file__).resolve().parents[3]
_ARTIFACTS_DIR_NAME = ".kiss.artifacts"

DEFAULT_MAX_BUDGET = 100.0
"""The product default spend cap, in USD, for one agent run.

This is the single source of truth for that number.  It backs both
:attr:`Config.max_budget` (the default for channel-agent command-line
runs) and ``vscode_config.DEFAULTS['max_budget']`` (what the settings
panel shows and what daemon-launched tasks read), which used to carry
two different literals — 200.0 here and 100 there — so a fresh install
disagreed with itself depending on which entry point the user reached
first.
"""

_artifact_dir: str | None = None
_artifact_dir_lock = threading.Lock()


def _artifact_root(base_dir: str | Path | None = None) -> Path:
    """Return the root directory for generated KISS artifacts."""
    root = Path(base_dir) if base_dir is not None else _PROJECT_DIR
    return root.resolve() / _ARTIFACTS_DIR_NAME


def _generate_artifact_dir() -> str:
    """Generate a unique artifact job directory under the project root.

    Returns:
        The absolute path to the newly created artifact directory.
    """
    artifact_subdir_name = (
        f"{time.strftime('job_%Y_%m_%d_%H_%M_%S')}_{random.randint(0, 1000000)}"
    )
    artifact_path = _artifact_root() / "jobs" / artifact_subdir_name
    artifact_path.mkdir(parents=True, exist_ok=True)
    return str(artifact_path)


def get_jobs_root(base_dir: str | Path | None = None) -> Path:
    """Return the directory that contains all per-job artifact subdirectories.

    Job artifacts (including saved trajectories) live under
    ``<base>/.kiss.artifacts/jobs/job_*``.  This returns the parent
    ``jobs`` directory, which is what the trajectory visualizer treats as
    its artifact directory.

    Args:
        base_dir: Optional base directory for the ``.kiss.artifacts`` root.
            Defaults to the project root when ``None``.

    Returns:
        The absolute path to the ``jobs`` directory.
    """
    return _artifact_root(base_dir) / "jobs"


def kiss_home() -> Path:
    """Return the KISS home directory ($KISS_HOME or ~/.kiss).

    Resolved lazily on every call so that ``KISS_HOME`` set after
    module import (as the test suite's conftest does) is honored.
    """
    env = os.environ.get("KISS_HOME")
    return Path(env) if env else Path.home() / ".kiss"


def get_artifact_dir() -> str:
    """Return this process's artifact directory, creating it lazily if needed.

    The directory is chosen once and never changes for the lifetime of
    the process.  It used to be replaceable at runtime, but
    ``Base.get_trajectory_path`` resolves it at *save* time rather than
    at run start, so swapping it mid-flight sent a running agent's
    trajectory to a different root than the one it started under — a
    hazard no lock can close, because it straddles the agent's lifetime
    rather than the assignment.
    """
    global _artifact_dir
    if _artifact_dir is None:
        with _artifact_dir_lock:
            if _artifact_dir is None:
                _artifact_dir = _generate_artifact_dir()
    return _artifact_dir


class _ArtifactDirProxy:
    def __fspath__(self) -> str:
        return get_artifact_dir()

    def __str__(self) -> str:
        return get_artifact_dir()

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


artifact_dir = _ArtifactDirProxy()


class Config(BaseModel):
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
    ZAI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("ZAI_API_KEY", ""),
        description="Z.AI (Zhipu/GLM) API key (can also be set via ZAI_API_KEY env var)",
    )
    MOONSHOT_API_KEY: str = Field(
        default_factory=lambda: os.getenv("MOONSHOT_API_KEY", ""),
        description="Moonshot AI (Kimi) API key (can also be set via MOONSHOT_API_KEY env var)",
    )
    max_budget: float = Field(
        default=DEFAULT_MAX_BUDGET,
        description=(
            "Maximum budget in USD for a single agent run, defaulting to "
            "DEFAULT_MAX_BUDGET. Only consumed as the "
            "default for channel-agent command-line runs (and kept in sync from "
            "the VS Code settings; daemon-launched tasks read max_budget from "
            "the VS Code config directly); KISSAgent and RelentlessAgent do NOT "
            "consult this field — they use their own defaults (10.0 USD and "
            "kiss.agents.sorcar.relentless_agent.DEFAULT_MAX_BUDGET, 200.0 USD, "
            "respectively; that is a different constant from this module's "
            "DEFAULT_MAX_BUDGET) unless max_budget is passed to run() explicitly."
        ),
    )


DEFAULT_CONFIG = Config()
