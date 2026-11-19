# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""What the memory maintenance SEAs share: the options of the task text and the model calls.

``/memory_consolidate`` and ``/memory_quiz`` are jobs run by code: one
tool does the whole job and makes its model calls itself, each a prompt
of its own that holds nothing but what the call is about (no agent loop
reads one page family while it merges another), charged to the task as a
side call.  What a job may write is decided by words of the task text
(``--apply``, ``--write``), which the tool reads from the running task:
it is no argument of the tool, so the model cannot switch it on, and it
is read from the user's own words alone (:func:`own_words`), so neither
can the report of an earlier run that a scheduled run is shown.
"""

from __future__ import annotations

import json
from typing import Any

from kiss.agents.sorcar.agent_dispatch import _parent_model_config
from kiss.agents.sorcar.cron_agent import RUN_CONTEXT_END, _current_task_text
from kiss.agents.sorcar.sorcar_agent import _attribute_side_call_usage, side_call_binding
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import ModelRefusalError, OutputCapError
from kiss.core.models.model_families import other_family_model
from kiss.core.models.model_info import custom_model_config, is_known_model

# Stop reasons of a reply the output cap cut off (OpenAI, Anthropic, Gemini).
OUTPUT_LIMIT_STOPS = frozenset({"length", "max_tokens", "max_output_tokens", "MAX_TOKENS"})

# Share of the task's budget a job leaves unspent, so the run can still report.
BUDGET_RESERVE = 0.1


class BudgetSpent(Exception):  # noqa: N818 - a stop signal, not an error
    """The task's budget is down to its reserve: the job makes no further model call."""


def own_words(caller: Any, request: str) -> list[str]:
    """Return the words of the user's own text of the task *caller* runs.

    The text is cut on both sides.  A scheduled run puts a run context in
    front of it (:func:`~kiss.agents.sorcar.cron_agent.run_context`),
    which quotes the end of the previous run's report, where "the option
    --apply merges" is an ordinary sentence; and the daemon may add
    instructions after *request*, the sentence the SEA's ``prompt``
    appended.  Each line's words are followed by ``""``: an option's value
    is the word after it on its line.

    Args:
        caller: The agent of the running task.
        request: What the SEA's ``prompt`` appended to the task text.
    """
    text = _current_task_text(caller).rsplit(RUN_CONTEXT_END, 1)[-1].split(request, 1)[0]
    return [word for line in text.splitlines() for word in (*line.split(), "")]


def option(words: list[str], name: str) -> str:
    """Return the value of option *name* among *words*, ``""`` when the option is not given.

    Args:
        words: The words of the task text (:func:`own_words`).
        name: The option, written ``--name value``.

    Raises:
        ValueError: If the option is given without a value.
    """
    if name not in words:
        return ""
    value = words[words.index(name) + 1]
    if not value or value.startswith("--"):
        raise ValueError(f"{name} needs a value")
    return value


def count_option(words: list[str], name: str, default: int) -> int:
    """Return option *name* as a positive number, *default* when it is not given.

    Raises:
        ValueError: If the value is missing or not a whole number above zero.
    """
    value = option(words, name)
    if not value:
        return default
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"{name} takes a whole number above zero, not {value!r}")
    return int(value)


def json_objects(reply: str) -> list[dict[str, Any]] | None:
    """Return the objects of the JSON list in *reply*; ``None`` when it holds no list of them.

    An empty list is an answer ("there are none") and is returned for
    nothing else; ``None`` is a reply that could not be used: an empty
    one, one without a JSON list, one whose list holds no object.
    """
    start, end = reply.find("["), reply.rfind("]")
    try:
        items: list[Any] | None = json.loads(reply[start : end + 1]) if 0 <= start < end else None
    except ValueError:
        return None
    if not items:
        return items
    return [item for item in items if isinstance(item, dict)] or None


class ModelCalls:
    """The model calls of one job run, made for the task that runs it and charged to it.

    Attributes:
        words: The words of the user's own text (:func:`own_words`), where
            the job's options are.
        model: The model the task runs on.
        judge: The model that checks its work: ``--judge <model>`` of the
            task text, else the strongest configured model of another
            family, else *model* itself.
        calls: Model calls made so far.
        cost: Their cost in USD.
    """

    def __init__(self, caller: Any, request: str) -> None:
        """Bind the calls to *caller*, the agent of the running task.

        Args:
            caller: The agent of the running task.
            request: What the SEA's ``prompt`` appended to the task text.

        Raises:
            ValueError: If the task text gives ``--judge`` without a model
                or with one that is not known.
        """
        self.caller = caller
        self.words = own_words(caller, request)
        self.model = str(caller.model_name)
        named = option(self.words, "--judge")
        if named and not is_known_model(named):
            raise ValueError(f"--judge names no known model: {named}")
        self.judge = named or other_family_model(self.model) or self.model
        self.calls = 0
        self.cost = 0.0

    def ask(self, model: str, prompt: str) -> str:
        """Return *model*'s reply to *prompt*; ``""`` for one that was refused or cut short.

        Refused or cut short is what the model layer reports as such (an
        error, a stop reason).  A refusal an adapter returns as the text
        of the turn is returned as text: a caller reads a reply for its
        JSON list (:func:`json_objects`) and cannot use one without.

        Args:
            model: The model to ask.
            prompt: The whole prompt; the call has no system prompt and no tools.

        Raises:
            BudgetSpent: If the task's budget is down to the reserve its
                final report needs (:data:`BUDGET_RESERVE`); no call is made.
        """
        spent, limit = self.caller._budget_status()
        if limit - spent <= BUDGET_RESERVE * limit:
            raise BudgetSpent
        # The task's own endpoint for its model, a registered endpoint model's for another.
        config = _parent_model_config(self.caller, model, False) or custom_model_config(model)
        config = dict(config or {})
        config.pop("system_instruction", None)
        agent = KISSAgent("memory job")
        # Bound now, like a ``decide`` call: the spend is charged to this task.
        binding = side_call_binding(self.caller)
        try:
            reply = agent.run(
                model_name=model,
                prompt_template="{prompt}",
                arguments={"prompt": prompt},
                is_agentic=False,
                verbose=False,
                model_config=config or None,
            )
            # Anthropic and Gemini hand a tool-less caller the text of a reply the cap cut.
            if (agent.last_call_usage or {}).get("stop_reason") in OUTPUT_LIMIT_STOPS:
                reply = ""
        except (ModelRefusalError, OutputCapError):
            reply = ""
        finally:
            _attribute_side_call_usage(
                self.caller, model, agent.budget_used, agent.total_tokens_used, binding
            )
            self.calls += 1
            self.cost += agent.budget_used
        return reply

    def spend(self) -> str:
        """Return the calls and cost so far as a report line."""
        judge = "the same model" if self.judge == self.model else self.judge
        return (
            f"Model calls: {self.calls}, ${self.cost:.2f}; "
            f"written by {self.model}, checked by {judge}."
        )
