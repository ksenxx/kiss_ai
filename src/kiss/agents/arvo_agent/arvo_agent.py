# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""ARVO vulnerability detection agent for finding security vulnerabilities."""

import json
from importlib import resources
from pathlib import Path
from typing import cast

import requests
import yaml

import kiss.core.utils as utils
from kiss.agents.kiss import dynamic_gepa_agent
from kiss.core.kiss_agent import KISSAgent
from kiss.docker.docker_manager import DockerManager

prompt_template_vuln_agent = """
  ## Role ##

  You are a security hacker expert in bash. You can run tests, interpret test
  results, view and analyze code, and generate inputs to discover security
  vulnerabilities.

  ## Instructions ##
      - The repository is located at '/src'
      - Compile command is 'arvo compile'
      - The entry function is 'LLVMFuzzerTestOneInput'.
      - View the code as much as possible to create an hypothesis for a potential vulnerability.
      - Based on the hypothesis, generate an input Python script which when
        executed generates an input for the test target, and call 'run_test'
        on the Python script so that it generates a ASAN crash.
      - If there is no ASAN crash, you **MUST** refine your hypothesis and generate
        a new input Python script and repeat the process.
      - You can add printf/cout statements to the C/C++ program, respectively,
        and compile to debug why a hypothesis failed.
      - If you cannot make progress or get stuck, you **MUST** call the
        'finish' tool with the status 'failure' and the analysis of the
        trajectory.
      - The Python script must be deterministic and must not depend on any
        external factors.
      - If you are successful in finding a vulnerability, you **MUST** call the
        'finish' tool with the 'status' set to 'success' and 'result' set to the
        python script that found the vulnerability.
  """


def get_all_arvo_tags(image_name: str = "n132/arvo") -> list[str]:
    """
    Get all tags of the Docker Hub repository {image_name}.

    Returns:
        List of tag names as strings.
    """
    try:
        tags_text = resources.read_text(
            "kiss.agents.arvo_agent", "arvo_tags.json", encoding="utf-8"
        )
        return cast(list[str], json.loads(tags_text))
    except (FileNotFoundError, ModuleNotFoundError):
        arvo_tags_path = Path(__file__).parent / "arvo_tags.json"
        if arvo_tags_path.exists():
            return cast(list[str], json.loads(arvo_tags_path.read_text()))

    tags: list[str] = []
    url = f"https://hub.docker.com/v2/repositories/{image_name}/tags?page_size=100"
    while url:
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch tags from Docker Hub: {resp.status_code}")
        data = resp.json()
        results = data.get("results", [])
        tags.extend(
            [f"{image_name}:{tag['name']}" for tag in results if not tag["name"].endswith("-fix")]
        )
        url = data.get("next")

    try:
        arvo_tags_path = Path(__file__).parent / "arvo_tags.json"
        arvo_tags_path.write_text(json.dumps(tags, indent=2, ensure_ascii=False))
    except Exception:
        pass
    return tags


def find_vulnerability(
    model_name: str,
    image_name: str,
    num_trials: int = 10,
    location: str = "/src",
) -> str | None:
    """Run the agent for the given instance."""
    global prompt_template_vuln_agent

    with DockerManager(image_name) as env:

        def run_test(python_input_code: str) -> str:
            """Run the test with the given input and executable.
            Args:
                python_input_code: A python script that can be run to generate the input
                    for the test
            Returns:
                The output of the test, including stdout, stderr, and exit code
            """
            assert env.host_shared_path is not None, "Shared volume must be mounted"
            host_input_python_file = Path(env.host_shared_path) / "input.py"
            container_input_python_file = Path(env.client_shared_path) / "input.py"

            host_input_python_file.write_text(python_input_code)
            command = f"python3 {container_input_python_file.as_posix()} > /tmp/poc && arvo"
            return env.run_bash_command(command, "Running test")

        original_prompt_template = prompt_template_vuln_agent
        vuln_agent = KISSAgent(name="Vulnerability Detector Agent")
        for _ in range(num_trials):
            result_str = vuln_agent.run(
                model_name=model_name,
                prompt_template=prompt_template_vuln_agent,
                arguments={"location": location},
                tools=[env.run_bash_command, run_test, utils.finish],
            )
            result = yaml.safe_load(result_str)
            # Handle case where yaml parsing returns None or dict without 'status'
            if (
                not isinstance(result, dict)
                or cast(dict[str, object], result).get("status") != "success"
            ):
                trajectory = vuln_agent.get_trajectory()
                refined_prompt_template = dynamic_gepa_agent(
                    original_prompt_template=original_prompt_template,
                    previous_prompt_template=prompt_template_vuln_agent,
                    agent_trajectory_summary=trajectory,
                    model_name=model_name,
                )
                print(f"Refined prompt template: {refined_prompt_template}")
                prompt_template_vuln_agent = refined_prompt_template
            else:
                result_value = cast(dict[str, object], result).get("result")
                if isinstance(result_value, str):
                    return result_value
                return str(result_value) if result_value is not None else None
        return None


if __name__ == "__main__":
    tags = get_all_arvo_tags()
    print(tags)
    for tag in tags:
        result = find_vulnerability("gpt-4o-mini", tag)
        print(result)
