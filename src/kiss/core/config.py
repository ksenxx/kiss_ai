# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration Pydantic models for KISS agent settings with CLI support."""

import os
import random
import time
from typing import Any

from pydantic import BaseModel, Field


def _generate_artifact_dir() -> str:
    """Generate a unique artifact subdirectory name based on timestamp and random number.

    Returns:
        str: The absolute path to the newly created artifact directory.
    """
    from pathlib import Path

    artifact_subdir_name = f"{time.strftime('job_%Y_%m_%d_%H_%M_%S')}_{random.randint(0, 1000000)}"
    artifact_path = Path(".kiss.artifacts").resolve() / "jobs" / artifact_subdir_name
    artifact_path.mkdir(parents=True, exist_ok=True)
    return str(artifact_path)


artifact_dir = _generate_artifact_dir()


class APIKeysConfig(BaseModel):
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
    NOVITA_API_KEY: str = Field(
        default_factory=lambda: os.getenv("NOVITA_API_KEY", ""),
        description="Novita API key (can also be set via NOVITA_API_KEY env var)",
    )


class Config(BaseModel):
    api_keys: APIKeysConfig = Field(
        default_factory=APIKeysConfig, description="API keys configuration"
    )


DEFAULT_CONFIG: Any = Config()
