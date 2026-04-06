# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration Pydantic models for KISS agent settings with CLI support."""

import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_PROJECT_DIR = Path(__file__).resolve().parents[3]
_ARTIFACTS_DIR_NAME = ".kiss.artifacts"
_artifact_dir: str | None = None
_artifact_dir_lock = threading.Lock()


def _artifact_root(base_dir: str | Path | None = None) -> Path:
    """Return the root directory for generated KISS artifacts."""
    root = Path(base_dir) if base_dir is not None else _PROJECT_DIR
    return root.resolve() / _ARTIFACTS_DIR_NAME


def set_artifact_base_dir(base_dir: str | Path | None) -> str:
    """Set the base directory used to resolve ``artifact_dir``.

    Args:
        base_dir: Directory whose ``.kiss.artifacts`` child should contain
            generated job artifacts. ``None`` resets to the project root.

    Returns:
        The resolved artifact job directory.
    """
    global _artifact_dir
    _artifact_dir = _generate_artifact_dir(base_dir)
    return _artifact_dir


def _generate_artifact_dir(base_dir: str | Path | None = None) -> str:
    """Generate a unique artifact job directory under the configured base directory.

    Args:
        base_dir: Optional base directory for the ``.kiss.artifacts`` root.

    Returns:
        The absolute path to the newly created artifact directory.
    """
    artifact_subdir_name = (
        f"{time.strftime('job_%Y_%m_%d_%H_%M_%S')}_{random.randint(0, 1000000)}"
    )
    artifact_path = _artifact_root(base_dir) / "jobs" / artifact_subdir_name
    artifact_path.mkdir(parents=True, exist_ok=True)
    return str(artifact_path)


def get_artifact_dir() -> str:
    """Return the active artifact directory, creating it lazily if needed."""
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
        if isinstance(other, _ArtifactDirProxy):
            return get_artifact_dir() == other.__fspath__()
        if isinstance(other, str):
            return get_artifact_dir() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(get_artifact_dir())


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
    MINIMAX_API_KEY: str = Field(
        default_factory=lambda: os.getenv("MINIMAX_API_KEY", ""),
        description="MiniMax API key (can also be set via MINIMAX_API_KEY env var)",
    )


DEFAULT_CONFIG: Any = Config()
