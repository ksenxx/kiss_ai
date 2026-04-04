# Author: Koushik Sen (ksen@berkeley.edu)

"""WebArena agent adapter that delegates to KISS Sorcar.

Runs sorcar on WebArena tasks using its built-in Playwright browser tools.
Sorcar navigates the live web application, completes the task autonomously,
and outputs a final answer that is scored against WebArena's reference answers.

Usage:
    from kiss.benchmarks.webarena.agent import SorcarWebArenaAgent
    agent = SorcarWebArenaAgent(model="claude-opus-4-6")
    result = agent.run_task(Path("config_files/0.json"))
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

from kiss.benchmarks.webarena.webarena_prompt import WEBARENA_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_API_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
)


def _build_task_prompt(config: dict) -> str:
    """Build a sorcar task prompt from a WebArena task config.

    Args:
        config: Parsed WebArena task config dict.

    Returns:
        Task instruction string for sorcar.
    """
    intent = config["intent"]
    start_url = config.get("start_url", "")
    sites = config.get("sites", [])

    prompt = f"Task: {intent}\n\nStart URL: {start_url}\n"
    if sites:
        prompt += f"Site(s): {', '.join(sites)}\n"
    prompt += (
        "\nComplete the task using the browser. When finished, output your "
        "final answer on a line starting with 'ANSWER:'. Be exact — the "
        "evaluation checks your answer against reference values."
    )
    return prompt


def _extract_answer(stdout: str) -> str:
    """Extract the final answer from sorcar stdout.

    Looks for a line starting with 'ANSWER:' and returns the remainder.
    Falls back to the last non-empty line of stdout.

    Args:
        stdout: Captured stdout from sorcar.

    Returns:
        Extracted answer string.
    """
    for line in stdout.splitlines():
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    lines = [line for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _score(answer: str, config: dict) -> float:
    """Score the agent's answer against WebArena reference answers.

    Handles string_match evaluation type. Returns -1.0 for eval types
    that require a live browser page (url_match, program_html).

    Args:
        answer: Sorcar's extracted answer text.
        config: WebArena task config dict.

    Returns:
        1.0 on pass, 0.0 on fail, -1.0 if eval type is not scoreable
        from text output alone.
    """
    eval_cfg = config.get("eval", {})
    eval_types = eval_cfg.get("eval_types", [])

    if "string_match" not in eval_types:
        return -1.0

    ref = eval_cfg.get("reference_answers", {})
    must_include = ref.get("must_include", [])
    exact_match = ref.get("exact_match", "")
    fuzzy_match = ref.get("fuzzy_match", "")

    answer_lower = answer.lower()

    if exact_match:
        return 1.0 if exact_match.lower() in answer_lower else 0.0
    if must_include:
        return 1.0 if all(r.lower() in answer_lower for r in must_include) else 0.0
    if fuzzy_match:
        return 1.0 if fuzzy_match.lower() in answer_lower else 0.0
    return -1.0


class SorcarWebArenaAgent:
    """Agent that runs KISS Sorcar on WebArena tasks.

    Delegates to the sorcar CLI with browser tools enabled. Sorcar
    navigates the live WebArena web applications and outputs a final
    answer scored against WebArena's reference answers.
    """

    def __init__(self, model: str | None = None, timeout: int = 600) -> None:
        """Initialize the agent.

        Args:
            model: LLM model name (e.g. "claude-opus-4-6").
            timeout: Max seconds per task before killing sorcar.
        """
        self.model = model
        self.timeout = timeout

    def run_task(self, config_file: Path) -> dict:
        """Run sorcar on a single WebArena task.

        Writes a SYSTEM.md with WebArena-specific instructions before
        invoking sorcar, then scores the result against reference answers.

        Args:
            config_file: Path to the WebArena task JSON config file.

        Returns:
            Dict with task_id, answer, score, stdout, stderr, return_code.
        """
        config = json.loads(config_file.read_text())
        task_id = config.get("task_id", config_file.stem)
        task_prompt = _build_task_prompt(config)

        model_flag = f"-m {self.model}" if self.model else ""
        env = {k: v for k in _API_KEY_VARS if (v := os.environ.get(k, ""))}

        # Write WebArena system prompt so sorcar picks it up.
        system_md = Path.cwd() / "SYSTEM.md"
        system_md.write_text(WEBARENA_SYSTEM_PROMPT)

        cmd = (
            f"sorcar -t {shlex.quote(task_prompt)} -n {model_flag}"
        )
        logger.info("Running task %s: %s", task_id, config.get("intent", "")[:80])

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, **env},
            )
        except subprocess.TimeoutExpired:
            logger.warning("Task %s timed out after %ds", task_id, self.timeout)
            return {
                "task_id": task_id,
                "answer": "",
                "score": 0.0,
                "stdout": "",
                "stderr": "timeout",
                "return_code": -1,
            }

        answer = _extract_answer(result.stdout)
        score = _score(answer, config)

        return {
            "task_id": task_id,
            "answer": answer,
            "score": score,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
