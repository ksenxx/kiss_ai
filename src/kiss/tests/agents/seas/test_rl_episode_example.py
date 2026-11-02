# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the RL-style example SEA ``examples/rl_episode/rl_episode_sea.py``.

The SEA is the environment of a guess-the-number episode: a declared-schema
action (``@tool_schema``), hooks that end and score the episode
(``end_run`` from ``tool_call_hook`` and ``llm_response_hook``), plain
model I/O.  The tests run it the two ways a trainer does — through
``scripts/run_sea.py --json`` as a subprocess (the exact command line of
the headless path) and through :func:`~kiss.agents.sorcar.headless.run_sea`
in process — against the scripted local OpenAI-compatible server that
plays the model, reached through ``RL_POLICY_URL`` / ``RL_POLICY_MODEL``,
the variables a trainer sets to point an episode at its policy server.
What the model reads is asserted on the wire, never in memory.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.headless import run_sea
from kiss.agents.sorcar.sea_commands import evaluate_sea, sea_check, sea_layers
from kiss.agents.sorcar.sea_settings import execute_python_file
from kiss.tests.agents.seas.sea_contract import system_message, tool_names, tool_results
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve, tool_call_body

ROOT = Path(__file__).resolve().parents[5]
SEA_PATH = ROOT / "examples" / "rl_episode" / "rl_episode_sea.py"
SCRIPT = ROOT / "scripts" / "run_sea.py"
SEA = execute_python_file(str(SEA_PATH))
"""The SEA module's namespace: its constants and :func:`episode_outcome`."""

EPISODE = "episode-7"
SECRET = 45  # random.Random(EPISODE).randint(1, 100): str seeds hash the same everywhere
LOW, HIGH = SEA["LOW"], SEA["HIGH"]
OUTPUT_TOKENS_PER_REPLY = 100  # what tool_call_body reports by default


def _guess(value: int) -> bytes:
    return tool_call_body("guess", {"value": value}, 500)


def _finish() -> bytes:
    return tool_call_body(
        "finish", {"success": True, "is_continue": False, "summary_in_html": "<p>I quit.</p>"}, 500
    )


def _outcome(result: str) -> dict[str, Any]:
    outcome: dict[str, Any] = SEA["episode_outcome"](result)
    return outcome


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Name the scripted server's model as the policy; the URL is set per server."""
    monkeypatch.setenv(SEA["POLICY_MODEL_ENV"], MODEL)
    return monkeypatch


class TestCommandLine:
    """``scripts/run_sea.py <sea file> <episode> --json``: the headless path end to end."""

    def test_json_run_solves_the_episode_and_the_model_reads_the_declared_schema(
        self, tmp_path: Path, policy: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        with serve([_guess(50), _guess(25), _guess(37), _guess(SECRET)]) as (url, requests):
            env = os.environ.copy() | {SEA["POLICY_URL_ENV"]: url, "KISS_HOME": str(home)}
            env.pop("KISS_SORCAR_LOCAL", None)
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(SEA_PATH), EPISODE, "--json"],
                env=env, cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False,
            )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["success"] is True
        assert payload["error"] == ""
        assert payload["total_steps"] == 4
        assert _outcome(payload["result"]) == {
            "solved": True, "reward": 0.7, "attempts": 4, "secret": SECRET,
            "guesses": [50, 25, 37, SECRET], "turns": 4,
            "output_tokens": 4 * OUTPUT_TOKENS_PER_REPLY,
        }
        paths = [Path(p) for p in payload["trajectory_paths"]]
        assert paths and all(p.is_file() for p in paths), payload["trajectory_paths"]
        for path in paths:
            path.unlink()
        # On the wire: the declared schema (bounds the derived one cannot state), the
        # environment's own system prompt, plain I/O, and the observations as tool results.
        first = requests[0]
        guess = next(t["function"] for t in first["tools"] if t["function"]["name"] == "guess")
        assert guess["parameters"] == SEA["GUESS_SCHEMA"]
        assert guess["parameters"]["properties"]["value"]["minimum"] == 1
        assert guess["description"].startswith("Guess the secret number;")
        assert tool_names(first) == {"finish", "guess"}
        # Under plain_io the environment's prompt IS the system message: no
        # restricted-profile note, no MOST IMPORTANT INSTRUCTIONS, no Task Settings.
        assert system_message(first) == SEA["SYSTEM_PROMPT"]
        assert f"Episode {EPISODE!r}: find the secret number." in json.dumps(first["messages"])
        assert tool_results(requests[1]) == ["The secret is lower than 50. Guesses used: 1 of 10."]
        assert tool_results(requests[3])[-1] == (
            "The secret is higher than 37. Guesses used: 3 of 10."
        )
        assert "Steps:" not in json.dumps(requests[3]["messages"])
        assert len(requests) == 4, "the winning guess ended the run: no further model call"


class TestInProcess:
    """``run_sea()`` with the trainer's environment variables: caps, refusals, settings."""

    def test_turn_cap_ends_an_unsolved_episode_with_success_false(
        self, tmp_path: Path, policy: pytest.MonkeyPatch
    ) -> None:
        with serve([_guess(1)]) as (url, requests):  # the last body repeats: never right
            policy.setenv(SEA["POLICY_URL_ENV"], url)
            outcome = run_sea(SEA_PATH, EPISODE, work_dir=str(tmp_path), save_trajectories=False)
        assert outcome.error == ""
        assert outcome.success is False
        assert "is_continue: false" in outcome.result
        assert _outcome(outcome.result) == {
            "solved": False, "reward": 0.0, "attempts": 9, "secret": SECRET,
            "guesses": [1] * 9, "turns": 10,
            "output_tokens": 10 * OUTPUT_TOKENS_PER_REPLY,
        }, "the capped reply's action never runs: 9 observations, 10 turns"
        assert len(requests) == outcome.total_steps == SEA["MAX_TURNS"]
        assert outcome.trajectory_paths == ()

    def test_a_winning_guess_on_the_last_turn_still_counts(
        self, tmp_path: Path, policy: pytest.MonkeyPatch
    ) -> None:
        with serve([_guess(1)] * 9 + [_guess(SECRET)]) as (url, requests):
            policy.setenv(SEA["POLICY_URL_ENV"], url)
            outcome = run_sea(SEA_PATH, EPISODE, work_dir=str(tmp_path), save_trajectories=False)
        assert outcome.success is True
        episode = _outcome(outcome.result)
        assert (episode["solved"], episode["attempts"], episode["turns"]) == (True, 10, 10)
        assert episode["reward"] == 0.1
        assert len(requests) == 10

    def test_finish_is_refused_while_the_episode_is_open(
        self, tmp_path: Path, policy: pytest.MonkeyPatch
    ) -> None:
        with serve([_finish(), _guess(SECRET)]) as (url, requests):
            policy.setenv(SEA["POLICY_URL_ENV"], url)
            outcome = run_sea(SEA_PATH, EPISODE, work_dir=str(tmp_path), save_trajectories=False)
        assert outcome.success is True
        assert _outcome(outcome.result) == {
            "solved": True, "reward": 1.0, "attempts": 1, "secret": SECRET,
            "guesses": [SECRET], "turns": 2,
            "output_tokens": 2 * OUTPUT_TOKENS_PER_REPLY,
        }
        refusal = tool_results(requests[1])[0]
        assert refusal.startswith("REFUSED: the finish call did NOT run and had no effect. ")
        assert "The episode is not over: it ends by itself when a guess is right." in refusal

    def test_settings_follow_the_policy_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(SEA["POLICY_URL_ENV"], raising=False)
        monkeypatch.delenv(SEA["POLICY_MODEL_ENV"], raising=False)
        settings = evaluate_sea(sea_layers(SEA_PATH), EPISODE).settings
        assert settings["model_config"] == {"plain_io": True}
        assert settings["tool_profile"] == "none"
        assert "model" not in settings, "an empty model setting is no override"
        monkeypatch.setenv(SEA["POLICY_URL_ENV"], "http://127.0.0.1:1/v1")
        monkeypatch.setenv(SEA["POLICY_MODEL_ENV"], "policy-step-40")
        settings = evaluate_sea(sea_layers(SEA_PATH), EPISODE).settings
        assert settings["model_config"] == {"plain_io": True, "base_url": "http://127.0.0.1:1/v1"}
        assert settings["model"] == "policy-step-40"
        report = sea_check("rl_episode", SEA_PATH)
        assert "/rl_episode: Plays one episode of guess-the-number" in report
        assert "tools added: guess" in report
        assert "llm_response_hook" in report.split("methods defined:")[1].split("\n")[0]

    def test_invalid_actions_are_observations_not_guesses(
        self, tmp_path: Path, policy: pytest.MonkeyPatch
    ) -> None:
        """Values outside the declared schema never win, never count and never crash the tool."""
        invalid: list[Any] = [str(SECRET), HIGH + 1, SECRET + 0.5]
        bodies = [tool_call_body("guess", {"value": v}, 500) for v in invalid] + [_guess(SECRET)]
        with serve(bodies) as (url, requests):
            policy.setenv(SEA["POLICY_URL_ENV"], url)
            outcome = run_sea(SEA_PATH, EPISODE, work_dir=str(tmp_path), save_trajectories=False)
        assert outcome.success is True
        episode = _outcome(outcome.result)
        assert (episode["guesses"], episode["attempts"], episode["turns"]) == ([SECRET], 1, 4)
        assert episode["reward"] == 1.0
        rejection = f"Not a valid guess: value must be an integer from {LOW} to {HIGH}."
        assert tool_results(requests[3]) == [rejection] * 3

    def test_a_boolean_true_is_not_the_secret_one(
        self, tmp_path: Path, policy: pytest.MonkeyPatch
    ) -> None:
        """``True == 1`` in Python; the environment compares only integers the schema allows."""
        episodes = (f"episode-{n}" for n in range(1000))
        episode_of_one = next(e for e in episodes if random.Random(e).randint(LOW, HIGH) == 1)
        bodies = [tool_call_body("guess", {"value": True}, 500), _guess(1)]
        with serve(bodies) as (url, requests):
            policy.setenv(SEA["POLICY_URL_ENV"], url)
            outcome = run_sea(
                SEA_PATH, episode_of_one, work_dir=str(tmp_path), save_trajectories=False
            )
        episode = _outcome(outcome.result)
        assert (episode["secret"], episode["guesses"], episode["turns"]) == (1, [1], 2)
        assert tool_results(requests[1])[0].startswith("Not a valid guess:")

    def test_episode_outcome_rejects_a_result_without_one(self) -> None:
        for result in (
            "",
            "success: false\nsummary: '<p>KISSError: the model died</p>'\n",
            "summary: '<pre>[]</pre>'\n",
        ):
            with pytest.raises(ValueError, match="holds no episode outcome"):
                SEA["episode_outcome"](result)
        with pytest.raises(ValueError, match="not the finish YAML"):
            SEA["episode_outcome"]("summary: [broken")
