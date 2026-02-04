# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""KISS evals package with benchmark integrations."""

from kiss.evals.algotune import AlgoTuneConfig, run_algotune
from kiss.evals.arvo_agent import find_vulnerability, get_all_arvo_tags
from kiss.evals.hotpotqa import HotPotQABenchmark, evaluate_hotpotqa_result
from kiss.evals.swe_agent_verified import (
    SWEBenchVerifiedConfig,
    download_swebench_verified,
    get_all_instance_ids,
    get_docker_image_name,
    get_instance_by_id,
    main,
    run_swebench,
    solve_instance,
)

__all__ = [
    "AlgoTuneConfig",
    "HotPotQABenchmark",
    "SWEBenchVerifiedConfig",
    "download_swebench_verified",
    "evaluate_hotpotqa_result",
    "find_vulnerability",
    "get_all_arvo_tags",
    "get_all_instance_ids",
    "get_docker_image_name",
    "get_instance_by_id",
    "main",
    "run_algotune",
    "run_swebench",
    "solve_instance",
]
