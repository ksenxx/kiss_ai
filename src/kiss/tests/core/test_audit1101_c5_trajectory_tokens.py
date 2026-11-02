# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Saved trajectories must not carry the subscription tokens (audit 2026-11-01, C5 F1).

``config_to_dict`` strips the API keys from the ``config`` block that
``Base._save`` writes into every trajectory.  ``CLAUDE_CODE_OAUTH_TOKEN``
and ``CODEX_ACCESS_TOKEN`` are credentials too (both sit in
``API_KEY_ENV_VARS`` next to the keys) and must be stripped the same way.
The agent runs in a subprocess so that the synthetic tokens reach
``DEFAULT_CONFIG`` through its real environment defaults, not through a
patched object.
"""

import os
import sys
from pathlib import Path

import yaml

from kiss.core.processes import run_process_group

SAVE_ONE_TRAJECTORY = """
from kiss.core import config as config_module
from kiss.core.base import Base

config_module._artifact_dir = {artifact_dir!r}
agent = Base("audit1101 c5 tokens")
agent._add_message("user", "hello")
agent._save()
print(agent.get_trajectory_path())
"""

SYNTHETIC = {
    "CLAUDE_CODE_OAUTH_TOKEN": "synthetic-claude-oauth-token-c5",
    "CODEX_ACCESS_TOKEN": "synthetic-codex-access-token-c5",
    "OPENAI_API_KEY": "synthetic-openai-api-key-c5",
}


def test_subscription_tokens_are_not_written_to_trajectories(tmp_path: Path) -> None:
    proc = run_process_group(
        [sys.executable, "-c", SAVE_ONE_TRAJECTORY.format(artifact_dir=str(tmp_path))],
        timeout=120,
        env={**os.environ, **SYNTHETIC},
    )
    assert proc.returncode == 0, proc.stderr
    path = Path(proc.stdout.strip().splitlines()[-1])
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)["config"]
    # Ordinary settings, including the subscription switches, are still recorded.
    assert "max_budget" in config and "claude_code_use_subscription" in config
    for name, value in SYNTHETIC.items():
        assert name not in config, name
        assert value not in text, name
