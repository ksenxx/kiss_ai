# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The ``decide`` tool: typed classification / routing / scoring inside a task.

The tool wraps :meth:`kiss.core.models.decisions_model.DecisionsModel.decide`
so a running Sorcar agent can ask OpenRouter's ``~typesafe/jev-latest`` a
set of typed questions about a piece of text and get calibrated
probabilities back, in one non-generative call costing a fraction of an
LLM round trip.  Typical uses: classify a support message, pick which of
several routes a request should take, or grade a candidate answer on a
rubric.

The question set travels as a JSON string (the same convention as
``run_parallel``'s ``tasks``): ``{"<name>": {"type": "noul" | "choice" |
"score", "instructions": "...", "criteria": ...}}``.  ``noul`` takes no
criteria; ``choice`` takes ``{"option_id": "description"}`` or a plain list
of option ids; ``score`` takes a list of rubric levels in ascending order.

The decisions call's tokens and USD are folded into the running task's
accounting exactly like ``talk``'s speech synthesis, so the per-task cost
shown to the user stays honest.
"""

import json
from collections.abc import Callable
from typing import Any, TypeGuard

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.decisions_model import (
    QUESTION_TYPES,
    DecisionsModel,
    choice,
    noul,
    reported_cost,
    score,
)
from kiss.core.models.model_info import MODEL_INFO, calculate_cost, model

DEFAULT_DECISIONS_MODEL = "openrouter/~typesafe/jev-latest"

_EXAMPLE_QUESTIONS = (
    '{"is_bug": {"type": "noul", "instructions": "Does the message report a software bug?"}, '
    '"route": {"type": "choice", "instructions": "Which team should handle this?", '
    '"criteria": {"billing": "payments and invoices", "support": "product help", '
    '"sales": "pricing and upgrades"}}, '
    '"urgency": {"type": "score", "instructions": "How urgent is the message?", '
    '"criteria": ["can wait a week", "this week", "today", "right now"]}}'
)


def decisions_tool_available() -> bool:
    """Report whether the ``decide`` tool can work in this process.

    The tool needs an OpenRouter key and the default decisions model in the
    catalog; without either every call would fail, so the tool is not
    offered to the agent at all (an unusable tool only costs prompt tokens).

    Returns:
        True when ``OPENROUTER_API_KEY`` is configured and
        :data:`DEFAULT_DECISIONS_MODEL` is a ``"dec": true`` catalog entry.
    """
    info = MODEL_INFO.get(DEFAULT_DECISIONS_MODEL)
    return bool(config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY) and (
        info is not None and info.is_decisions_supported
    )


def parse_questions(questions_json: str) -> dict[str, dict[str, Any]]:
    """Parse and normalise the tool's ``questions`` JSON argument.

    Each entry is rebuilt through the :func:`noul` / :func:`choice` /
    :func:`score` builders so the wire shape is exactly what the endpoint
    accepts (``choice`` option lists become ``{id: id}`` mappings, ``score``
    criteria are forced to a list).

    Args:
        questions_json: A JSON object ``{name: {"type", "instructions",
            "criteria"?}}``.

    Returns:
        The normalised ``{name: question}`` mapping for
        :meth:`DecisionsModel.decide`.

    Raises:
        KISSError: When the JSON does not parse, is not a non-empty object,
            or an entry has a missing/unknown ``type``, a non-string
            ``instructions``, or criteria of the wrong shape for its type.
    """
    try:
        raw = json.loads(questions_json)
    except (TypeError, ValueError) as e:
        raise KISSError(f"questions is not valid JSON: {e}") from e
    if not isinstance(raw, dict) or not raw:
        raise KISSError("questions must be a non-empty JSON object {name: question}")
    parsed: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        if not name.strip():
            raise KISSError("Question names must be non-empty strings")
        if not isinstance(spec, dict):
            raise KISSError(f"Question {name!r} must be a JSON object, got {type(spec).__name__}")
        kind = spec.get("type")
        if kind not in QUESTION_TYPES:
            raise KISSError(
                f"Question {name!r} has type {kind!r}; expected one of {', '.join(QUESTION_TYPES)}"
            )
        instructions = spec.get("instructions")
        if not _is_text(instructions):
            raise KISSError(f"Question {name!r} needs a non-empty 'instructions' string")
        criteria = spec.get("criteria")
        if kind == "noul":
            parsed[name] = noul(instructions)
        elif kind == "choice":
            if not _is_choice_criteria(criteria):
                raise KISSError(
                    f"choice question {name!r} needs non-empty 'criteria': "
                    "{option_id: description} or [option_id, ...] of non-empty strings"
                )
            parsed[name] = choice(instructions, criteria)
        else:
            if not isinstance(criteria, list) or len(criteria) < 2 or not all(
                _is_text(level) for level in criteria
            ):
                raise KISSError(
                    f"score question {name!r} needs 'criteria': a list of at least two "
                    "non-empty rubric level strings in ascending order"
                )
            parsed[name] = score(instructions, criteria)
    return parsed


def _is_text(value: Any) -> TypeGuard[str]:
    """Return True when *value* is a non-blank string (what the endpoint accepts)."""
    return isinstance(value, str) and bool(value.strip())


def _is_choice_criteria(criteria: Any) -> TypeGuard[dict[str, str] | list[str]]:
    """Return True when *criteria* is a valid non-empty ``choice`` option set.

    Args:
        criteria: ``{option_id: description}`` or ``[option_id, ...]``; every
            id and description must be a non-blank string, which is what the
            endpoint enforces (a number as an option id is an HTTP 400).
    """
    if isinstance(criteria, dict):
        return bool(criteria) and all(
            _is_text(key) and _is_text(value) for key, value in criteria.items()
        )
    return isinstance(criteria, list) and bool(criteria) and all(_is_text(c) for c in criteria)


def format_decision(model_name: str, response: dict[str, Any]) -> tuple[str, float, int]:
    """Render a decisions response for the agent and price it.

    Args:
        model_name: The catalog name the request was made with (its catalog
            prices are used only when the response carries no ``usage.cost``).
        response: The dict returned by :meth:`DecisionsModel.decide`.

    Returns:
        ``(text, cost_usd, total_tokens)`` where *text* is the JSON the tool
        returns: ``{"answers": ..., "model": <served id>, "usage":
        {"input_tokens", "output_tokens", "cost_usd"}}``.  *cost_usd* is
        OpenRouter's reported ``usage.cost`` when present (the served
        model's actual bill), else the catalog estimate.
    """
    usage: dict[str, Any] = {}
    if isinstance(response.get("usage"), dict):
        usage = response["usage"]
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cost = reported_cost(response)
    if cost is None:
        cost = calculate_cost(model_name, input_tokens, output_tokens)
    text = json.dumps(
        {
            "answers": response["answers"],
            "model": response.get("model", model_name),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
            },
        },
        indent=2,
    )
    return text, cost, input_tokens + output_tokens


def make_decide_tool(
    agent: Any,
    model_name: str = DEFAULT_DECISIONS_MODEL,
    model_config: dict[str, Any] | None = None,
) -> Callable[[str, str], str]:
    """Build the ``decide`` tool bound to *agent*'s task accounting.

    Args:
        agent: The Sorcar agent whose task receives the call's cost and
            tokens (via ``_attribute_sub_usage``); ``None`` skips
            attribution, for library use outside an agent.
        model_name: The ``"dec": true`` catalog model to call.
        model_config: Optional :class:`DecisionsModel` settings; ``base_url``
            and ``api_key`` override the OpenRouter defaults.

    Returns:
        The ``decide(state, questions)`` tool callable.
    """
    decisions_model = model(model_name, model_config=model_config)
    if not isinstance(decisions_model, DecisionsModel):
        raise KISSError(f"{model_name} is not a decisions model (catalog flag 'dec' is not set)")

    def decide(state: str, questions: str) -> str:
        try:
            parsed = parse_questions(questions)
            response = decisions_model.decide(state, parsed)
        except KISSError as e:
            return f"Error: {e.args[0]}"
        text, cost, tokens = format_decision(model_name, response)
        if agent is not None and (cost > 0 or tokens > 0):
            from kiss.agents.sorcar.sorcar_agent import _attribute_sub_usage

            _attribute_sub_usage(agent, cost, tokens, 0)
        return text

    decide.__name__ = "decide"
    decide.__doc__ = (
        f"Classify, route or score text with the {model_name} decisions model: "
        "ask typed questions about a piece of text and get calibrated probabilities "
        "back in one cheap, non-generative call (no LLM reasoning). "
        "Use it to triage or label text, choose between fixed routes, or grade a "
        "candidate against a rubric. Question types: 'noul' (is this true? -> "
        "probability 0-1), 'choice' (pick one option -> chosen id + per-option "
        "probabilities + confidence), 'score' (place on an ordered rubric -> fractional "
        "level + legend + per-level probabilities + confidence).\n\n"
        "Args:\n"
        "    state: The text to judge (plain text, or a JSON document of related context).\n"
        "    questions: JSON object {name: question}; each question is "
        '{"type": "noul"|"choice"|"score", "instructions": "<what to decide>", '
        '"criteria": ...} where noul has no criteria, choice criteria is '
        '{"option_id": "description"} or ["option_id", ...], and score criteria is a '
        f"list of rubric levels in ascending order. Example: {_EXAMPLE_QUESTIONS}\n\n"
        "Returns:\n"
        '    JSON {"answers": {name: answer}, "model": served model id, "usage": '
        '{"input_tokens", "output_tokens", "cost_usd"}}, or "Error: ..." when the '
        "questions are malformed or the request fails."
    )
    return decide
