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

The tool is offered only while :func:`decisions_tool_available` holds: the
``classify_with_decisions`` config key is on, an OpenRouter key is
configured and the catalog lists the model.  The same gate governs the
pre-run task classifier's decisions route, so turning the key off stops
every use of Jev.
"""

import json
import threading
from collections.abc import Callable
from typing import Any, TypeGuard

from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.locked import locked_call
from kiss.core.models.decisions_model import (
    QUESTION_TYPES,
    DecisionsModel,
    choice,
    noul,
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
    """Report whether Jev (the decisions model) may be used in this process.

    This is the ONE gate for every use of the decisions model: the
    ``decide`` tool offered to the agent and the pre-run task classifier's
    decisions route (:func:`kiss.agents.sorcar.task_classifier.classify_task`).
    The user's switch is the ``classify_with_decisions`` key of
    ``$KISS_HOME/config.json`` (on by default), read on every call so a
    config change applies to the next run without a daemon restart.
    Even when the switch is on, the model needs an OpenRouter key and a
    ``"dec": true`` catalog entry; without
    either every call would fail, so the tool is not offered at all (an
    unusable tool only costs prompt tokens).

    Returns:
        True when ``classify_with_decisions`` is on, ``OPENROUTER_API_KEY``
        is configured and :data:`DEFAULT_DECISIONS_MODEL` is a decisions
        catalog entry.
    """
    from kiss.core.user_config import load_config

    if not load_config().get("classify_with_decisions", True):
        return False
    info = MODEL_INFO.get(DEFAULT_DECISIONS_MODEL)
    return bool(config_module.DEFAULT_CONFIG.OPENROUTER_API_KEY) and (
        info is not None and info.is_decisions_supported
    )


#: The fields of a question; :func:`parse_questions` refuses any other.
_QUESTION_FIELDS = ("type", "instructions", "criteria")


def _decoding_stop(error: json.JSONDecodeError) -> str:
    """Return the sentence of a ``questions`` refusal that shows where decoding stopped.

    Three objects nest (the questions, a question, its criteria) and
    models close two: each of the 33 calls the task history shows
    refused (2026-11-11) lacked one closing brace at its end.  Such a
    text is not completed for the model: a brace missing between two
    questions leaves the text open at its end as well, and closing it
    there nests the second question in the first.

    Args:
        error: The decoder's error for the ``questions`` text.

    Returns:
        ``""`` for a fault at the start of the text (the decoder's
        message says it all), the text before a fault further on, or,
        when the text ends before its JSON does, the brace rule.
    """
    text, pos = error.doc, error.pos
    if not text[:pos].strip():
        return ""
    if text[pos:].strip():
        return f". Decoding stopped after {text[max(0, pos - 40):pos]!r}."
    return (
        ". The text ends there, before its JSON does: a closing brace is probably missing "
        "(every question, every criteria object and the questions object itself needs its own)."
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
        KISSError: When the JSON does not parse (the message shows where
            decoding stopped, :func:`_decoding_stop`), is not a non-empty
            object, or an entry has a field other than those three (a
            question nested in another one, which would be dropped
            unasked), a missing/unknown ``type``, a non-string
            ``instructions``, or criteria of the wrong shape for its type.
    """
    try:
        raw = json.loads(questions_json)
    except json.JSONDecodeError as e:
        raise KISSError(f"questions is not valid JSON: {e}{_decoding_stop(e)}") from e
    except (TypeError, ValueError, RecursionError) as e:  # RecursionError: nested too deep
        raise KISSError(f"questions is not valid JSON: {e}") from e
    if not isinstance(raw, dict) or not raw:
        raise KISSError("questions must be a non-empty JSON object {name: question}")
    parsed: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        if not name.strip():
            raise KISSError("Question names must be non-empty strings")
        if not isinstance(spec, dict):
            raise KISSError(f"Question {name!r} must be a JSON object, got {type(spec).__name__}")
        unexpected = [repr(field) for field in spec if field not in _QUESTION_FIELDS]
        if unexpected:
            raise KISSError(
                f"Question {name!r} takes only 'type', 'instructions' and 'criteria', not "
                f"{', '.join(unexpected)} (another question nested in it? Then the closing "
                f"brace of {name!r} is missing before it)"
            )
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


def decision_usage(
    decisions_model: DecisionsModel, response: dict[str, Any]
) -> tuple[float, int, int]:
    """Price a decisions response.

    Args:
        decisions_model: The model the request was made with: its usage
            and cost readers parse *response*, and the catalog prices of
            its name are used only when the response carries no
            ``usage.cost``.
        response: The dict returned by :meth:`DecisionsModel.decide`, or a
            charged reply it rejected (see ``take_partial_usage_response``).

    Returns:
        ``(cost_usd, input_tokens, output_tokens)``.  *cost_usd* is
        OpenRouter's reported ``usage.cost`` when present (the served
        model's actual bill), else the catalog estimate.
    """
    input_tokens, output_tokens = (
        decisions_model.extract_input_output_token_counts_from_response(response)[:2]
    )
    cost = decisions_model.extract_cost_from_response(response)
    if cost is None:
        cost = calculate_cost(decisions_model.model_name, input_tokens, output_tokens)
    return cost, input_tokens, output_tokens


def format_decision(
    decisions_model: DecisionsModel, response: dict[str, Any]
) -> tuple[str, float, int]:
    """Render a decisions response for the agent and price it.

    Args:
        decisions_model: The model the request was made with (see
            :func:`decision_usage`).
        response: The dict returned by :meth:`DecisionsModel.decide`.

    Returns:
        ``(text, cost_usd, total_tokens)`` where *text* is the JSON the tool
        returns: ``{"answers": ..., "model": <served id>, "usage":
        {"input_tokens", "output_tokens", "cost_usd"}}``.
    """
    cost, input_tokens, output_tokens = decision_usage(decisions_model, response)
    text = json.dumps(
        {
            "answers": response["answers"],
            "model": response.get("model", decisions_model.model_name),
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
    on_usage: Callable[[float, int], None] | None = None,
) -> Callable[[str, str], str]:
    """Build the ``decide`` tool bound to *agent*'s task accounting.

    Args:
        agent: The Sorcar agent whose task receives the call's cost and
            tokens (via ``_attribute_sub_usage``); ``None`` skips
            attribution, for library use outside an agent.
        model_name: The ``"dec": true`` catalog model to call.
        model_config: Optional :class:`DecisionsModel` settings; ``base_url``
            and ``api_key`` override the OpenRouter defaults.
        on_usage: Called with each call's ``(cost_usd, tokens)``, a charged
            reply the tool rejected included, for a caller without an
            agent that accounts for the spend itself (the task classifier).

    Returns:
        The ``decide(state, questions)`` tool callable.
    """
    decisions_model = model(model_name, model_config=model_config)
    if not isinstance(decisions_model, DecisionsModel):
        raise KISSError(f"{model_name} is not a decisions model (catalog flag 'dec' is not set)")
    # The calls of one tool take turns with its model: it keeps a single charged-but-rejected
    # reply, which a second call running at once would overwrite or take for its own.
    lock = threading.RLock()

    def attribute(cost: float, tokens: int, binding: tuple[Any, str] | None) -> None:
        """Fold one call's spend into the accounting of the task it was made for (no step)."""
        if on_usage is not None:
            on_usage(cost, tokens)
        if agent is not None:
            from kiss.agents.sorcar.sorcar_agent import _attribute_side_call_usage

            _attribute_side_call_usage(agent, model_name, cost, tokens, binding)

    def request(
        state: str, parsed: dict[str, dict[str, Any]]
    ) -> tuple[str, tuple[float, int] | None]:
        """Make one request; return the tool's reply and the ``(cost, tokens)`` it was charged."""
        try:
            response = decisions_model.decide(state, parsed)
        except KISSError as e:
            # A 2xx reply without ``answers`` was still charged for.
            charged = decisions_model.take_partial_usage_response()
            if not isinstance(charged, dict):
                return f"Error: {e.args[0]}", None
            cost, input_tokens, output_tokens = decision_usage(decisions_model, charged)
            return f"Error: {e.args[0]}", (cost, input_tokens + output_tokens)
        text, cost, tokens = format_decision(decisions_model, response)
        return text, (cost, tokens)

    def decide(state: str, questions: str) -> str:
        binding = None
        if agent is not None:
            from kiss.agents.sorcar.sorcar_agent import side_call_binding

            # Bound now: a reply that comes back after the task ended
            # (a handler a Stop abandoned) still charges this task.
            binding = side_call_binding(agent)
        try:
            parsed = parse_questions(questions)
        except KISSError as e:
            return f"Error: {e.args[0]}"
        text, usage = locked_call(lock, request, state, parsed)
        if usage is not None:
            attribute(*usage, binding)
        return text

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
