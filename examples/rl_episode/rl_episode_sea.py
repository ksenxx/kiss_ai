# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An RL-style SEA: one episode of guess-the-number, ended and scored by the environment.

The SEA is the environment.  It keeps the episode state (the secret, the
guesses so far), exposes the one action as a tool whose JSON schema it
declares (``@tool_schema``: an integer within the bounds, which the
schema derived from the signature could not state), returns every
observation as the tool's result, and ends the episode itself:

- ``tool_call_hook`` ends the run with the episode's outcome when a
  guess hits the secret (the terminal state), and refuses ``finish``
  while the episode is open, so neither the model nor the implicit
  finish can end it early;
- ``llm_response_hook`` counts the model's replies and output tokens
  and ends the run with ``success=False`` when the turn cap is reached
  without a winning guess (truncation).

The outcome is a JSON object (``solved``, ``reward``, ``attempts``,
``secret``, ``guesses``, ``turns``, ``output_tokens``) that
:func:`episode_outcome` reads back from the run's result.  ``turns``
is the number of model calls (the root layer's step count);
``output_tokens`` sums the replies ``llm_response_hook`` had seen when
the episode ended, which on a provider that runs tools natively inside
the turn (Codex) excludes the winning turn itself, so the run's own
``total_tokens_used`` stays the authoritative total::

    from kiss.agents.sorcar.headless import run_sea

    outcome = run_sea("examples/rl_episode/rl_episode_sea.py", "episode-7")
    episode = episode_outcome(outcome.result)   # {"solved": True, "reward": 0.7, ...}

The same file runs headless from the command line::

    uv run python scripts/run_sea.py examples/rl_episode/rl_episode_sea.py "episode-7" --json

or in a chat tab as ``/rl_episode episode-7`` once the ``examples``
folder is listed in ``$KISS_HOME/SEAS.md``.  The secret is derived from
the task text, so the same episode id replays the same secret.
``RL_POLICY_URL`` (an OpenAI-compatible API root) and ``RL_POLICY_MODEL``
point the episode at the policy server of a trainer; without them the
run uses the model given on the command line or the default model.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from typing import Any

import yaml

from kiss.agents.seas.base.base_sea import (
    ALLOW,
    LLMReply,
    Verdict,
    WorkerSea,
    end_run,
    refuse,
    tool_schema,
)

LOW, HIGH = 1, 100
"""The secret is an integer in this closed range."""

MAX_TURNS = 10
"""Model replies per episode; a binary search over 1..100 needs at most 7."""

POLICY_URL_ENV = "RL_POLICY_URL"
POLICY_MODEL_ENV = "RL_POLICY_MODEL"

GUESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "value": {
            "type": "integer",
            "minimum": LOW,
            "maximum": HIGH,
            "description": f"Your guess: an integer from {LOW} to {HIGH}.",
        },
    },
    "required": ["value"],
}

OUTCOME_TAG = "pre"
"""The HTML tag the outcome JSON is wrapped in: HTML passes through ``finish`` verbatim."""

SYSTEM_PROMPT = f"""You are playing guess-the-number.

A secret integer from {LOW} to {HIGH} has been chosen. On each turn call guess(value) \
exactly once; its result tells you whether the secret is higher or lower than your \
guess. The episode ends by itself when a guess is right or after {MAX_TURNS} turns, so \
never call finish and do not write explanations: just make the next guess."""


class RlEpisodeSea(WorkerSea):
    """One guess-the-number episode: the environment, its action, its termination and reward."""

    def __init__(self) -> None:
        super().__init__()
        self.secret = 0
        self.guesses: list[int] = []
        self.output_tokens = 0

    def description(self) -> str:
        return (
            "Plays one episode of guess-the-number as a reinforcement-learning environment: "
            "the model guesses through a declared-schema tool, the hooks end and score the "
            f"episode. `/rl_episode <episode id>` (the id seeds the secret, 1..{HIGH})."
        )

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        model_config: dict[str, Any] = {"plain_io": True}
        if os.environ.get(POLICY_URL_ENV):
            model_config["base_url"] = os.environ[POLICY_URL_ENV]
        return settings | {
            "model": os.environ.get(POLICY_MODEL_ENV, ""),
            "tool_profile": "none",
            "max_budget": 0.25,
            "model_config": model_config,
        }

    def prompt(self, task: str) -> str:
        self.secret = random.Random(task).randint(LOW, HIGH)
        return f"Episode {task!r}: find the secret number. Make your first guess."

    def system_prompt(self, system_prompt: str) -> str:
        return SYSTEM_PROMPT

    def tools(self, tools: list[Callable[..., Any]]) -> list[Callable[..., Any]]:
        return tools + [self.guess]

    @tool_schema(
        parameters=GUESS_SCHEMA,
        description="Guess the secret number; the result says whether it is higher or lower.",
    )
    def guess(self, value: int) -> str:
        """Record a wrong guess and return the observation (a right one ends the run first)."""
        if not _valid_guess(value):
            return f"Not a valid guess: value must be an integer from {LOW} to {HIGH}."
        self.guesses.append(value)
        hint = "higher" if self.secret > value else "lower"
        used = f"Guesses used: {len(self.guesses)} of {MAX_TURNS}."
        return f"The secret is {hint} than {value}. {used}"

    def tool_call_hook(self, name: str, args: dict[str, Any]) -> Verdict:
        if name == "guess" and self._wins(args):
            self.guesses.append(self.secret)
            return end_run(self._outcome(solved=True))
        if name == "finish":
            return refuse(
                "The episode is not over: it ends by itself when a guess is right. Call guess."
            )
        return ALLOW

    def llm_response_hook(self, reply: LLMReply) -> Verdict:
        self.output_tokens += reply.usage["output_tokens"]
        if reply.step >= MAX_TURNS and not self._solves(reply):
            return end_run(self._outcome(solved=False), success=False)
        return ALLOW

    def _wins(self, args: dict[str, Any]) -> bool:
        """Return whether a guess call's *args* name the secret (a bool or a string never does)."""
        value = args.get("value")
        return _valid_guess(value) and value == self.secret

    def _solves(self, reply: LLMReply) -> bool:
        """Return whether *reply* guesses the secret (the tool hook then ends the run)."""
        return any(
            call["name"] == "guess" and self._wins(call["arguments"]) for call in reply.tool_calls
        )

    def _outcome(self, solved: bool) -> str:
        """Return the episode's outcome: solved in N attempts, reward in [0, 1], as wrapped JSON."""
        attempts = len(self.guesses)
        reward = max(0.0, (MAX_TURNS - attempts + 1) / MAX_TURNS) if solved else 0.0
        outcome = json.dumps({
            "solved": solved,
            "reward": round(reward, 2),
            "attempts": attempts,
            "secret": self.secret,
            "guesses": self.guesses,
            "turns": self.step,
            "output_tokens": self.output_tokens,
        })
        return f"<{OUTCOME_TAG}>{outcome}</{OUTCOME_TAG}>"


def _valid_guess(value: Any) -> bool:
    """Return whether *value* is an action the schema allows: an integer from LOW to HIGH."""
    return isinstance(value, int) and not isinstance(value, bool) and LOW <= value <= HIGH


def episode_outcome(result: str) -> dict[str, Any]:
    """Return the episode's outcome from a run's *result* (the ``finish`` YAML).

    Args:
        result: ``SeaRunResult.result`` of :func:`kiss.agents.sorcar.headless.run_sea`,
            or the ``result`` of ``scripts/run_sea.py --json``.

    Raises:
        ValueError: When *result* holds no outcome (the run raised or the
            model ended it some other way).
    """
    try:
        payload = yaml.safe_load(result) if result else None
    except yaml.YAMLError as exc:
        raise ValueError(f"the result is not the finish YAML: {exc}") from exc
    summary = str(payload.get("summary", "")).strip() if isinstance(payload, dict) else ""
    start, end = f"<{OUTCOME_TAG}>", f"</{OUTCOME_TAG}>"
    outcome = None
    if summary.startswith(start) and summary.endswith(end):
        outcome = json.loads(summary[len(start):-len(end)])
    if not isinstance(outcome, dict):
        raise ValueError(f"the result holds no episode outcome: {summary!r}")
    return outcome
