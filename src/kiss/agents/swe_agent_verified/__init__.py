# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""SWE-bench Verified benchmark integration module for KISS agents."""

from kiss.agents.swe_agent_verified.config import SWEBenchVerifiedConfig
from kiss.agents.swe_agent_verified.run_swebench import (
    download_swebench_verified,
    get_all_instance_ids,
    get_docker_image_name,
    get_instance_by_id,
    main,
    run_swebench,
    solve_instance,
)

__all__ = [
    "SWEBenchVerifiedConfig",
    "download_swebench_verified",
    "get_all_instance_ids",
    "get_docker_image_name",
    "get_instance_by_id",
    "main",
    "run_swebench",
    "solve_instance",
]
