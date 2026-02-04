# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""SWE-bench Verified configuration that extends the main KISS config."""

from pydantic import BaseModel, Field

from kiss.core.config_builder import add_config


class SWEBenchVerifiedConfig(BaseModel):
    """SWE-bench Verified benchmark configuration settings."""

    # Dataset settings
    dataset_name: str = Field(
        default="princeton-nlp/SWE-bench_Verified",
        description="HuggingFace dataset name for SWE-bench Verified",
    )
    split: str = Field(
        default="test",
        description="Dataset split to use (test is the only split for Verified)",
    )
    instance_id: str = Field(
        default="",
        description="Single instance ID to run (takes precedence over instance_ids)",
    )
    instance_ids: list[str] = Field(
        default_factory=list,
        description="Specific instance IDs to run (empty = all instances)",
    )
    max_instances: int = Field(
        default=0,
        description="Maximum number of instances to run (0 = all instances)",
    )

    # Docker settings
    docker_image_base: str = Field(
        default="slimshetty/swebench-verified:sweb.eval.x86_64.",
        description="Base Docker image name prefix for SWE-bench instances",
    )
    workdir: str = Field(
        default="/testbed",
        description="Working directory inside the Docker container",
    )

    # Agent settings
    model: str = Field(
        default="gemini-3-pro-preview",
        description="Model to use for the SWE agent",
    )
    max_steps: int = Field(
        default=100,
        description="Maximum number of steps per agent run",
    )
    max_budget: float = Field(
        default=5.0,
        description="Maximum budget per instance in USD",
    )

    # Sampling settings
    num_samples: int = Field(
        default=1,
        description="Number of solution samples per instance (for pass@k evaluation)",
    )

    # Evaluation settings
    run_evaluation: bool = Field(
        default=True,
        description="Whether to run official SWE-bench evaluation after solving",
    )
    max_workers: int = Field(
        default=8,
        description="Maximum number of parallel workers for evaluation",
    )
    run_id: str = Field(
        default="kiss_swebench_verified",
        description="Run ID for evaluation results",
    )

    # Output settings
    save_patches: bool = Field(
        default=True,
        description="Whether to save generated patches to disk",
    )
    save_trajectories: bool = Field(
        default=True,
        description="Whether to save agent trajectories",
    )


add_config("swebench_verified", SWEBenchVerifiedConfig)
