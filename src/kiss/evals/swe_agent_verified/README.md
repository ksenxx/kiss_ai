# SWE-bench Verified Benchmark Integration

This module provides integration with the [SWE-bench Verified](https://www.swebench.com/) benchmark for evaluating AI agents on real-world GitHub issues. SWE-bench Verified is a curated subset of 500 human-validated samples from the original SWE-bench test set.

## Overview

SWE-bench (Software Engineering Benchmark) evaluates language models on their ability to resolve real-world GitHub issues. Given a codebase and an issue description, the model must generate a patch that resolves the described problem.

**Key Features:**

- **500 Verified Instances**: Human-validated samples ensuring solvability
- **Real-World Issues**: Actual GitHub issues from popular Python repositories
- **Docker-Based Evaluation**: Consistent, reproducible evaluation environment
- **Official Evaluation Harness**: Integration with SWE-bench's official evaluator

## Installation

This module is part of the KISS Agent Framework. See the main [README.md](../../../../README.md) for installation instructions.

The SWE-bench package is included as a dependency of KISS. For Docker-based evaluation, ensure Docker is installed and running:

```bash
docker --version
```

## Quick Start

### Run on a Single Instance

```bash
# Run on a specific instance with a specific model
uv run src/kiss/evals/swe_agent_verified/run_swebench.py \
    --swebench_verified.model gemini-3-flash-preview \
    --swebench_verified.instance_id "django__django-11099"
```

### Run on All Instances

```bash
uv run src/kiss/evals/swe_agent_verified/run_swebench.py
```

### Run on Specific Instances

```python
from kiss.evals.swe_agent_verified import run_swebench, SWEBenchVerifiedConfig

# Run on specific instances
config = SWEBenchVerifiedConfig(
    instance_ids=["django__django-11099", "requests__requests-4718"],
    model="gemini-3-pro-preview",
    num_samples=1,
)
result = run_swebench(config=config)
print(f"Success rate: {result['successful']}/{result['total_samples']}")
```

### Run with Limited Instances

```python
from kiss.evals.swe_agent_verified import run_swebench

# Run on first 10 instances
result = run_swebench(max_instances=10, model="gpt-4o")
```

## API Reference

### `run_swebench`

```python
run_swebench(
    config: SWEBenchVerifiedConfig | None = None,
    **kwargs
) -> dict[str, Any]
```

Main entry point for running the SWE-bench Verified benchmark.

**Parameters:**

- `config`: Configuration settings. If None, uses defaults.
- `**kwargs`: Override any configuration setting.

**Returns:**

- `total_instances`: Number of instances processed
- `total_samples`: Total number of samples (instances × num_samples)
- `successful`: Number of successfully solved samples
- `failed`: Number of failed samples
- `results`: List of individual result dictionaries
- `evaluation`: Evaluation statistics from official evaluator
- `output_dir`: Path to output directory

### `solve_instance`

```python
solve_instance(
    instance: dict[str, Any],
    config: SWEBenchVerifiedConfig,
    sample_idx: int = 0,
    prompt_template: str = SWE_PROMPT_TEMPLATE,
) -> dict[str, Any]
```

Solve a single SWE-bench instance using a KISS agent.

**Parameters:**

- `instance`: SWE-bench instance dictionary
- `config`: Configuration settings
- `sample_idx`: Sample index for pass@k evaluation
- `prompt_template`: Custom prompt template (optional)

**Returns:**

- `instance_id`: The instance ID
- `model_patch`: Generated patch (if successful)
- `model_name_or_path`: Solution identifier
- `status`: 'success' or 'failure'
- `error`: Error message (if failed)
- `trajectory`: Agent trajectory (if enabled)

### `download_swebench_verified`

```python
download_swebench_verified(
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> list[dict[str, Any]]
```

Download the SWE-bench Verified dataset from HuggingFace.

### `get_all_instance_ids`

```python
get_all_instance_ids(
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
) -> list[str]
```

Get all instance IDs from the dataset.

### `get_instance_by_id`

```python
get_instance_by_id(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
) -> dict[str, Any] | None
```

Get a specific instance by its ID.

### `get_docker_image_name`

```python
get_docker_image_name(
    instance: dict[str, Any],
    base: str,
) -> str
```

Get the Docker image name for a SWE-bench instance.

## Configuration

Configuration is managed through `SWEBenchVerifiedConfig`. All settings can be overridden via command-line arguments or programmatically.

### Dataset Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | `"princeton-nlp/SWE-bench_Verified"` | HuggingFace dataset name |
| `split` | `"test"` | Dataset split to use |
| `instance_id` | `""` | Single instance ID to run via CLI (takes precedence over `instance_ids`) |
| `instance_ids` | `[]` | List of instance IDs for programmatic use (empty = all) |
| `max_instances` | `0` | Maximum instances to run (0 = all) |

### Docker Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `docker_image_base` | `"slimshetty/swebench-verified:sweb.eval.x86_64."` | Docker image prefix |
| `workdir` | `"/testbed"` | Working directory in container |

> **Note**: The SWE-bench runner uses `mount_shared_volume=False` when creating Docker containers to preserve the pre-existing repository content in `/testbed`. This is important because SWE-bench Docker images come with the repository already cloned and set up at the correct commit.

### Agent Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gemini-3-pro-preview"` | Model for the SWE agent |
| `max_steps` | `100` | Maximum steps per agent run |
| `max_budget` | `5.0` | Maximum budget per instance (USD) |

### Sampling Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | `1` | Samples per instance (for pass@k) |

### Evaluation Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `run_evaluation` | `True` | Run official SWE-bench evaluator |
| `max_workers` | `8` | Parallel workers for evaluation |
| `run_id` | `"kiss_swebench_verified"` | Run ID for results |

### Output Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_patches` | `True` | Save generated patches to disk |
| `save_trajectories` | `True` | Save agent trajectories |

### Command-Line Usage

```bash
# Run on a specific instance with a specific model
uv run src/kiss/evals/swe_agent_verified/run_swebench.py \
    --swebench_verified.model gemini-3-flash-preview \
    --swebench_verified.instance_id "django__django-11099"

# Run with custom model and budget
uv run src/kiss/evals/swe_agent_verified/run_swebench.py \
    --swebench_verified.model gpt-4o \
    --swebench_verified.max_steps 60 \
    --swebench_verified.max_budget 2.0

# Run on first N instances
uv run src/kiss/evals/swe_agent_verified/run_swebench.py \
    --swebench_verified.max_instances 5 \
    --swebench_verified.num_samples 3

# Disable evaluation
uv run src/kiss/evals/swe_agent_verified/run_swebench.py --no-swebench_verified.run_evaluation
```

## Dataset Structure

Each SWE-bench Verified instance contains:

| Field | Description |
|-------|-------------|
| `instance_id` | Unique identifier (e.g., `"django__django-11099"`) |
| `repo` | Repository name (e.g., `"django/django"`) |
| `base_commit` | Base commit SHA for the issue |
| `problem_statement` | The GitHub issue description |
| `hints_text` | Optional hints for solving |
| `created_at` | Issue creation timestamp |
| `patch` | Gold patch (ground truth solution) |
| `test_patch` | Test patch for validation |
| `version` | Repository version |
| `FAIL_TO_PASS` | Tests that should pass after fix |
| `PASS_TO_PASS` | Tests that should continue passing |

## Evaluation

The module integrates with the official SWE-bench evaluation harness. After generating patches, it:

1. Applies the patch to the repository
1. Runs the test suite
1. Checks if `FAIL_TO_PASS` tests now pass
1. Verifies `PASS_TO_PASS` tests still pass

### Evaluation Metrics

- **Resolved**: Patch passes all required tests
- **Submitted**: Patch was generated and submitted
- **Completed**: Evaluation completed without errors
- **Resolution Rate**: Percentage of submitted patches that resolve the issue

## Example Output

```
================================================================================
SWE-bench Verified Benchmark
================================================================================

Loading dataset: princeton-nlp/SWE-bench_Verified...
Loaded 500 instances
Limited to first 5 instances

Output directory: /tmp/tmpXXXXXX
Model: gemini-3-pro-preview
Samples per instance: 1

[1/5] Processing: django__django-11099
----------------------------------------
  ✅ Success

[2/5] Processing: requests__requests-4718
----------------------------------------
  ✅ Success

[3/5] Processing: matplotlib__matplotlib-23299
----------------------------------------
  ❌ Failed: Unable to locate the relevant source file

...

================================================================================
SUMMARY
================================================================================
Total instances: 5
Total samples: 5
Successful: 3
Failed: 2
Success rate: 60.00%

============================================================
EVALUATION RESULTS
============================================================
Total instances: 5
Submitted: 3
Completed: 3
Resolved: 2
Resolution rate: 66.67%
============================================================
```

## References

- [SWE-bench Website](https://www.swebench.com/)
- [SWE-bench GitHub](https://github.com/SWE-bench/SWE-bench)
- [SWE-bench Paper](https://openreview.net/forum?id=VTF8yNQM66) (ICLR 2024)
- [SWE-bench Verified Dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
- [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) (prompt inspiration)

## Citation

If you use this benchmark integration, please cite the original SWE-bench paper:

```bibtex
@inproceedings{
    jimenez2024swebench,
    title={{SWE}-bench: Can Language Models Resolve Real-world Github Issues?},
    author={Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=VTF8yNQM66}
}
```

## Authors

- Koushik Sen (ksen@berkeley.edu)
