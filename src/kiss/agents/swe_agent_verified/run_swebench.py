#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""SWE-bench Verified runner using KISS agents to solve GitHub issues."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

import kiss.agents.swe_agent_verified.config  # noqa: F401
import kiss.core.utils as utils
from kiss.agents.kiss import dynamic_gepa_agent
from kiss.agents.swe_agent_verified.config import SWEBenchVerifiedConfig
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.docker.docker_manager import DockerManager

# SWE-bench agent prompt template
# Inspired by: https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/mini.yaml
SWE_PROMPT_TEMPLATE = """
## Role ##

You are an expert software engineer tasked with resolving GitHub issues.

## SWE Issue ##

Please solve the following GitHub issue.
{issue}

## End of SWE Issue ##

## Issue-Specific Instructions ##

  - You can execute bash commands and edit files to implement the necessary changes.
  - The repository is located in the '/testbed' directory.
  - You must NOT view or edit any files outside of the '/testbed' directory.
  - Create a todo list of sub-tasks to solve the issue systematically.
  - Use the 'do_sub_task' tool to execute each sub-task and update the todo list.

## End of Issue-Specific Instructions ##


## Recommended Workflow ##

Follow this workflow step-by-step, iterating as needed:

1. **Understand the Issue**: Read the issue description carefully and identify the root cause.
2. **Explore the Codebase**: Find and read relevant files to understand the code structure.
3. **Reproduce the Issue**: Create a script to reproduce the bug (if applicable).
4. **Implement the Fix**: Edit the source code to resolve the issue.
5. **Verify the Fix**: Run your reproduction script to confirm the fix works.
6. **Test Edge Cases**: Ensure your fix is robust and doesn't break other functionality.
7. **Generate the Patch**: When done, call 'finish' with status 'success' and the git diff.

If you cannot make progress or get stuck, you **MUST** call the 'finish' tool with
status 'failure' and provide an analysis of what went wrong.


## Useful Bash Command Examples ##

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'

# View entire file with line numbers
cat -n filename.py

# Search for patterns
grep -rn "pattern" /testbed/
```

### Git operations:

```bash
# Check current status
git status

# View changes
git diff

# Generate patch for submission
git diff > patch.diff
```

### Any other command you need can be run with the 'run_bash_command' tool.
"""


def download_swebench_verified(
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> list[dict[str, Any]]:
    """Download the SWE-bench Verified dataset from HuggingFace.

    The SWE-bench Verified dataset contains 500 human-validated samples from
    the SWE-bench test set. Each instance includes a problem statement,
    repository info, and gold patch for evaluation.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split to load.

    Returns:
        List of dictionaries representing dataset instances.

    Raises:
        KISSError: If the dataset cannot be loaded.
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        error_msg = f"Failed to load SWE-bench Verified dataset '{dataset_name}': {e}"
        raise KISSError(error_msg) from e

    instances = [dict(instance) for instance in dataset]
    return instances


def get_docker_image_name(instance: dict[str, Any], base: str) -> str:
    """Get the Docker image name for a SWE-bench instance.

    Args:
        instance: SWE-bench instance dictionary.
        base: Base Docker image name prefix.

    Returns:
        Full Docker image name for this instance.

    Raises:
        KISSError: If instance_id is not found.
    """
    instance_id = instance.get("instance_id")
    if not instance_id:
        raise KISSError("Instance does not contain a valid 'instance_id' key.")
    return f"{base}{instance_id}".lower()


def solve_instance(
    instance: dict[str, Any],
    config: SWEBenchVerifiedConfig,
    sample_idx: int = 0,
    prompt_template: str = SWE_PROMPT_TEMPLATE,
) -> dict[str, Any]:
    """Solve a single SWE-bench instance using a KISS agent.

    Args:
        instance: SWE-bench instance dictionary.
        config: Configuration settings.
        sample_idx: Sample index for pass@k evaluation.
        prompt_template: Prompt template for the agent.

    Returns:
        Dictionary containing:
        - instance_id: The instance ID
        - model_patch: The generated patch (if successful)
        - model_name_or_path: Identifier for this solution
        - status: 'success' or 'failure'
        - error: Error message (if failed)
        - trajectory: Agent trajectory (if save_trajectories is True)
    """
    instance_id = instance["instance_id"]
    image_name = get_docker_image_name(instance, config.docker_image_base)

    result: dict[str, Any] = {
        "instance_id": instance_id,
        "model_name_or_path": f"{config.model}_sample_{sample_idx}",
        "status": "failure",
    }

    try:
        with DockerManager(image_name, workdir=config.workdir, mount_shared_volume=False) as env:

            def do_sub_task(prompt: str) -> str:
                """Execute a sub-task using a separate agent.

                Args:
                    prompt: The prompt describing the sub-task. Must contain
                        all necessary information to complete the sub-task.

                Returns:
                    The result of the sub-task execution.
                """
                sub_agent = KISSAgent(name="Sub SWE Agent")
                sub_result = sub_agent.run(
                    model_name=config.model,
                    prompt_template=prompt,
                    tools=[env.run_bash_command, utils.finish],
                    max_steps=config.max_steps // 2,  # Half steps for sub-tasks
                    max_budget=config.max_budget / 4,  # Quarter budget for sub-tasks
                )
                return sub_result

            swe_agent = KISSAgent(name="KISS SWE Agent")
            issue_statement = instance["problem_statement"]

            agent_result_str = swe_agent.run(
                model_name=config.model,
                prompt_template=prompt_template,
                arguments={"issue": issue_statement},
                tools=[env.run_bash_command, do_sub_task, utils.finish],
                max_steps=config.max_steps,
                max_budget=config.max_budget,
            )

            agent_result = yaml.safe_load(agent_result_str)

            if agent_result.get("status") == "success":
                result["model_patch"] = agent_result.get("result", "")
                result["status"] = "success"
            else:
                result["error"] = agent_result.get("analysis", "Unknown failure")

            if config.save_trajectories:
                result["trajectory"] = swe_agent.get_trajectory()

    except KISSError as e:
        result["error"] = f"KISSError: {e}"
    except Exception as e:
        result["error"] = f"Exception: {e}"

    return result


def evaluate_results(
    results_path: Path,
    config: SWEBenchVerifiedConfig,
) -> dict[str, Any]:
    """Run the official SWE-bench evaluator on generated results.

    Args:
        results_path: Path to the JSONL file containing predictions.
        config: Configuration settings.

    Returns:
        Dictionary containing evaluation statistics.
    """
    print("\nRunning official SWE-bench evaluator...")

    eval_cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--predictions_path",
        str(results_path),
        "--max_workers",
        str(config.max_workers),
        "--run_id",
        config.run_id,
    ]

    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print("✅ Evaluation completed successfully")
            print(result.stdout)
        else:
            print(f"⚠️  Evaluator returned exit code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        # Try to read and return final stats
        results_dir = results_path.parent / "evaluation_results" / config.run_id
        results_json_path = results_dir / "results.json"

        if results_json_path.exists():
            with results_json_path.open() as f:
                return dict(json.load(f))

    except FileNotFoundError:
        print("\n⚠️  SWE-bench evaluator not found.")
        print("   The 'swebench' package is a dependency of KISS.")
        print("   Try reinstalling: pip install -e . or uv sync")
    except Exception as e:
        print(f"\n❌ Error running evaluator: {e}")

    return {}


def run_swebench(
    config: SWEBenchVerifiedConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the SWE-bench Verified benchmark.

    Args:
        config: Configuration settings. If None, uses defaults.
        **kwargs: Override configuration settings.

    Returns:
        Dictionary containing:
        - total_instances: Number of instances processed
        - successful: Number of successfully solved instances
        - failed: Number of failed instances
        - results: List of individual results
        - evaluation: Evaluation statistics (if run_evaluation is True)
    """
    if config is None:
        config = SWEBenchVerifiedConfig(**kwargs)
    elif kwargs:
        config = config.model_copy(update=kwargs)

    print("=" * 80)
    print("SWE-bench Verified Benchmark")
    print("=" * 80)

    # Download dataset
    print(f"\nLoading dataset: {config.dataset_name}...")
    instances = download_swebench_verified(config.dataset_name, config.split)
    print(f"Loaded {len(instances)} instances")

    # Filter instances if specified
    # Single instance_id takes precedence
    if config.instance_id:
        instances = [i for i in instances if i["instance_id"] == config.instance_id]
        print(f"Filtered to instance: {config.instance_id}")
    elif config.instance_ids:
        instances = [i for i in instances if i["instance_id"] in config.instance_ids]
        print(f"Filtered to {len(instances)} specified instances")

    if config.max_instances > 0:
        instances = instances[: config.max_instances]
        print(f"Limited to first {len(instances)} instances")

    # Setup output directory
    output_dir = Path(tempfile.mkdtemp())
    results_path = output_dir / "swebench_results.jsonl"

    print(f"\nOutput directory: {output_dir}")
    print(f"Model: {config.model}")
    print(f"Samples per instance: {config.num_samples}")
    print()

    # Process instances
    all_results: list[dict[str, Any]] = []
    successful = 0
    failed = 0
    prompt_template = SWE_PROMPT_TEMPLATE

    for i, instance in enumerate(instances, 1):
        instance_id = instance["instance_id"]
        print(f"\n[{i}/{len(instances)}] Processing: {instance_id}")
        print("-" * 40)

        for sample_idx in range(config.num_samples):
            if config.num_samples > 1:
                print(f"  Sample {sample_idx + 1}/{config.num_samples}")

            result = solve_instance(instance, config, sample_idx, prompt_template)
            all_results.append(result)

            if result["status"] == "success":
                successful += 1
                print("  ✅ Success")

                # Save result to JSONL
                if config.save_patches:
                    result_line = json.dumps(
                        {
                            "instance_id": result["instance_id"],
                            "model_patch": result.get("model_patch", ""),
                            "model_name_or_path": result["model_name_or_path"],
                        }
                    )
                    with results_path.open("a") as f:
                        f.write(result_line + "\n")
            else:
                failed += 1
                error_msg = result.get("error", "Unknown error")[:100]
                print(f"  ❌ Failed: {error_msg}")

                # Refine prompt template on failure
                if result.get("trajectory"):
                    try:
                        prompt_template = dynamic_gepa_agent(
                            original_prompt_template=SWE_PROMPT_TEMPLATE,
                            previous_prompt_template=prompt_template,
                            agent_trajectory_summary=result.get("error", ""),
                            model_name=config.model,
                        )
                    except Exception:
                        pass  # Keep current template if refinement fails

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total instances: {len(instances)}")
    print(f"Total samples: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if len(all_results) > 0:
        success_rate = (successful / len(all_results)) * 100
        print(f"Success rate: {success_rate:.2f}%")

    # Run evaluation
    evaluation_results: dict[str, Any] = {}
    if config.run_evaluation and results_path.exists():
        evaluation_results = evaluate_results(results_path, config)

        if evaluation_results:
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"Total instances: {evaluation_results.get('total', 'N/A')}")
            print(f"Submitted: {evaluation_results.get('submitted', 'N/A')}")
            print(f"Completed: {evaluation_results.get('completed', 'N/A')}")
            print(f"Resolved: {evaluation_results.get('resolved', 'N/A')}")

            submitted = evaluation_results.get("submitted", 0)
            resolved = evaluation_results.get("resolved", 0)
            if submitted > 0:
                resolution_rate = (resolved / submitted) * 100
                print(f"Resolution rate: {resolution_rate:.2f}%")
            print("=" * 60)

    return {
        "total_instances": len(instances),
        "total_samples": len(all_results),
        "successful": successful,
        "failed": failed,
        "results": all_results,
        "evaluation": evaluation_results,
        "output_dir": str(output_dir),
    }


def get_all_instance_ids(
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
) -> list[str]:
    """Get all instance IDs from the SWE-bench Verified dataset.

    Args:
        dataset_name: HuggingFace dataset name.

    Returns:
        List of instance IDs.
    """
    instances = download_swebench_verified(dataset_name)
    return [i["instance_id"] for i in instances]


def get_instance_by_id(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
) -> dict[str, Any] | None:
    """Get a specific instance by ID.

    Args:
        instance_id: The instance ID to retrieve.
        dataset_name: HuggingFace dataset name.

    Returns:
        Instance dictionary or None if not found.
    """
    instances = download_swebench_verified(dataset_name)
    for instance in instances:
        if instance["instance_id"] == instance_id:
            return instance
    return None


def main() -> None:
    """Main entry point."""
    config = DEFAULT_CONFIG.swebench_verified  # type: ignore[attr-defined]

    result = run_swebench(config=config)

    if result["successful"] > 0:
        print(f"\n✅ Completed! {result['successful']}/{result['total_samples']} successful")
    else:
        print(f"\n❌ No successful solutions out of {result['total_samples']} attempts")


if __name__ == "__main__":
    main()
