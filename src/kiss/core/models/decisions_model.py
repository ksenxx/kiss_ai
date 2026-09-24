# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Backend for OpenRouter's Decisions API (``POST /api/alpha/decisions``).

Decisions models — TypeSafe's ``~typesafe/jev-latest`` / ``typesafe/jev-1.13``
are the first — are not text generators.  A request carries a ``state``
(the text or JSON to judge) plus a set of typed ``questions``; the answer
is a typed value with calibrated probabilities per question, produced in
one pass with no autoregressive decoding:

* ``noul`` — "is this statement true?" → a probability in ``[0, 1]``.
* ``choice`` — pick one option from a labelled set → the chosen key plus
  a probability per option and a confidence.
* ``score`` — place the state on an ordered rubric → a fractional score,
  the rubric legend, a probability per level and a confidence.

OpenRouter lists these models only under
``GET /api/v1/models?output_modalities=decisions`` and rejects them on
``/chat/completions`` with HTTP 400, so they need this dedicated adapter.
The catalog marks them with ``"dec": true`` (``gen``/``fc``/``emb`` all
false) and the :func:`kiss.core.models.model_info.model` factory routes
such names here.

The primary entry point is :meth:`DecisionsModel.decide`.  The
:class:`~kiss.core.models.model.Model` contract is also honoured so the
adapter can be driven by generic code: :meth:`DecisionsModel.generate`
judges the conversation text against a fixed question set supplied as
``model_config["questions"]``.

Verified against the live endpoint on 2026-09-17; request/response shapes
follow the OpenRouter SDK reference
(https://openrouter.ai/docs/client-sdks/typescript/sdks/decisions/README).
"""

import json
import logging
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import requests

from kiss.core.kiss_error import KISSError
from kiss.core.models.model import (
    Attachment,
    Model,
    ThinkingCallback,
    TokenCallback,
    flatten_content_to_text,
)

logger = logging.getLogger(__name__)

OPENROUTER_DECISIONS_BASE_URL = "https://openrouter.ai/api"
"""API root the OpenRouter decisions endpoint hangs off (note: not ``/api/v1``)."""

DECISIONS_PATH = "/alpha/decisions"

QUESTION_TYPES = ("noul", "choice", "score")

DEFAULT_TIMEOUT_SECONDS = 60.0


def noul(instructions: str) -> dict[str, Any]:
    """Build a ``noul`` question: is *instructions* true of the state?

    Args:
        instructions: The statement or yes/no question to evaluate.

    Returns:
        A question dict for :meth:`DecisionsModel.decide`.  The answer is
        ``{"type": "noul", "noul": <probability the statement is true>}``.
    """
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: dict[str, str] | list[str]) -> dict[str, Any]:
    """Build a ``choice`` question: pick one option from a labelled set.

    Args:
        instructions: What to decide (e.g. ``"What does the user want?"``).
        criteria: The options, as ``{option_id: description}``; a plain
            list of option ids is accepted and used as its own descriptions.

    Returns:
        A question dict for :meth:`DecisionsModel.decide`.  The answer is
        ``{"type": "choice", "choice": <option_id>, "probabilities":
        {option_id: p, ...}, "confidence": <float>}``.
    """
    if isinstance(criteria, list):
        criteria = {option: option for option in criteria}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
    """Build a ``score`` question: place the state on an ordered rubric.

    Args:
        instructions: What to rate (e.g. ``"How urgent is this message?"``).
        criteria: Rubric levels in ascending order; level ``i`` is the
            description at index ``i``.  The endpoint requires a list here
            (a mapping is rejected with HTTP 400).

    Returns:
        A question dict for :meth:`DecisionsModel.decide`.  The answer is
        ``{"type": "score", "score": <float index on the rubric>, "legend":
        {"0": level0, ...}, "probabilities": {"0": p, ...}, "confidence":
        <float>}``.
    """
    return {"type": "score", "instructions": instructions, "criteria": list(criteria)}


def reported_cost(response: Any) -> float | None:
    """Return the ``usage.cost`` (USD) OpenRouter attached to a decisions response.

    Args:
        response: The parsed JSON body of a ``/decisions`` call.

    Returns:
        The reported cost, or ``None`` when the body has no numeric
        ``usage.cost``.
    """
    if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
        return None
    cost = response["usage"].get("cost")
    if isinstance(cost, bool) or not isinstance(cost, int | float):
        return None
    return float(cost)


def api_model_id(model_name: str) -> str:
    """Return the id OpenRouter expects for a catalog name.

    The catalog key carries KISS's ``openrouter/`` routing prefix
    (``openrouter/~typesafe/jev-latest``); the wire id does not.

    Args:
        model_name: The catalog model name.

    Returns:
        The model id to send in the request body.
    """
    return model_name.removeprefix("openrouter/")


class DecisionsModel(Model):
    """Adapter for models served through OpenRouter's decisions endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str = OPENROUTER_DECISIONS_BASE_URL,
        api_key: str = "",
        model_config: dict[str, Any] | None = None,
        token_callback: TokenCallback | None = None,
        thinking_callback: ThinkingCallback | None = None,
    ):
        """Initialize a decisions-model adapter.

        Args:
            model_name: Catalog name, e.g. ``"openrouter/~typesafe/jev-latest"``.
            base_url: API root; requests go to ``{base_url}/alpha/decisions``.
            api_key: OpenRouter API key (Bearer token).
            model_config: Optional settings.  ``"questions"`` (a
                ``{name: question}`` mapping built with :func:`noul`,
                :func:`choice`, :func:`score`) is the question set
                :meth:`generate` asks; ``"timeout"`` is the HTTP timeout in
                seconds (default 60); ``"extra_headers"`` (``{name: value}``,
                as produced by ``model_info.custom_model_config`` for a
                ``MY_MODELS.json`` endpoint) is sent on every request.
            token_callback: Called once with the JSON answers text by
                :meth:`generate` (the endpoint does not stream).
            thinking_callback: Accepted for interface parity; never invoked.
        """
        super().__init__(model_name, model_config, token_callback, thinking_callback)
        self.endpoint_url = base_url.rstrip("/") + DECISIONS_PATH
        self.api_key = api_key
        self.timeout = float(self.model_config.get("timeout", DEFAULT_TIMEOUT_SECONDS))

    def decide(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Answer typed *questions* about *state* in one request.

        Args:
            state: The content to judge: plain text, or a JSON object or
                array of related context.
            questions: ``{name: question}`` where each question comes from
                :func:`noul`, :func:`choice` or :func:`score` (or is an
                equivalent dict with a ``"type"`` key).

        Returns:
            The parsed response: ``{"model": <served model id>, "answers":
            {name: answer, ...}, "usage": {"input_tokens", "output_tokens",
            "cost"}, ...}``.  Answer shapes are documented on the three
            question builders.

        Raises:
            KISSError: On an unknown question type, a non-2xx response, an
                unreachable endpoint, or a response without ``answers``.
        """
        if not questions:
            raise KISSError("decide() needs at least one question")
        for name, question in questions.items():
            if question.get("type") not in QUESTION_TYPES:
                raise KISSError(
                    f"Question {name!r} has type {question.get('type')!r}; "
                    f"expected one of {', '.join(QUESTION_TYPES)}"
                )
        body = {"model": api_model_id(self.model_name), "state": state, "questions": questions}
        headers = {
            **(self.model_config.get("extra_headers") or {}),
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                self.endpoint_url, json=body, headers=headers, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise KISSError(f"Decisions request to {self.endpoint_url} failed: {e}") from e
        if response.status_code >= 400:
            raise KISSError(_http_error_message(response))
        try:
            parsed = response.json()
        except ValueError as e:
            raise KISSError(
                f"Decisions endpoint returned non-JSON body: {response.text[:200]!r}"
            ) from e
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), dict):
            raise KISSError(f"Decisions response has no 'answers' object: {parsed!r}"[:500])
        return parsed

    def initialize(self, prompt: str, attachments: list[Attachment] | None = None) -> None:
        """Start a conversation whose text :meth:`generate` will judge.

        Args:
            prompt: The state text.
            attachments: Not supported by decisions models (text input
                only); any given are dropped with a warning.
        """
        if attachments:
            logger.warning(
                "%s accepts text only; dropping %d attachment(s).",
                self.model_name,
                len(attachments),
            )
        self.conversation = [{"role": "user", "content": prompt}]

    def generate(self) -> tuple[str, Any]:
        """Judge the conversation text against ``model_config["questions"]``.

        The state is the concatenated text of every conversation message;
        the answers are appended to the conversation as an assistant
        message so the :class:`Model` contract holds.

        Returns:
            ``(json.dumps(answers), raw_response)``.

        Raises:
            KISSError: When no ``"questions"`` were configured, or on any
                error from :meth:`decide`.
        """
        questions = self.model_config.get("questions")
        if not questions:
            raise KISSError(
                f"{self.model_name} is a decisions model: it answers typed "
                "questions rather than generating text.  Either call "
                "decide(state, questions) directly or pass the question set as "
                'model_config={"questions": {...}} so generate() can ask them.'
            )
        state = "\n\n".join(
            flatten_content_to_text(message.get("content", "")) for message in self.conversation
        )
        response = self.decide(state, questions)
        text = json.dumps(response["answers"])
        self.conversation.append({"role": "assistant", "content": text})
        self._invoke_token_callback(text)
        return text, response

    def generate_and_process_with_tools(
        self,
        function_map: dict[str, Callable[..., Any]],
        tools_schema: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str, Any]:
        """Decisions models cannot call tools.

        Raises:
            KISSError: Always.
        """
        raise KISSError(
            f"{self.model_name} is a decisions model and cannot call tools; "
            "use decide() for typed questions."
        )

    def extract_input_output_token_counts_from_response(
        self, response: Any
    ) -> tuple[int, int, int, int]:
        """Read ``usage.input_tokens`` / ``usage.output_tokens`` from a response.

        Args:
            response: The dict returned by :meth:`decide` / :meth:`generate`.

        Returns:
            ``(input_tokens, output_tokens, 0, 0)``; the endpoint has no
            prompt cache.
        """
        usage: dict[str, Any] = {}
        if isinstance(response, dict) and isinstance(response.get("usage"), dict):
            usage = response["usage"]
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0), 0, 0

    def extract_cost_from_response(self, response: Any) -> float | None:
        """Return the USD cost OpenRouter reports in ``usage.cost``.

        Args:
            response: The dict returned by :meth:`decide` / :meth:`generate`.

        Returns:
            The reported cost, or ``None`` when the response lacks a numeric
            ``usage.cost`` (the caller then prices the tokens from the catalog).
        """
        return reported_cost(response)

    def get_embedding(self, text: str, embedding_model: str | None = None) -> list[float]:
        """Decisions models do not produce embeddings.

        Raises:
            KISSError: Always.
        """
        raise KISSError(f"{self.model_name} is a decisions model and cannot embed text.")


def _http_error_message(response: requests.Response) -> str:
    """Format a non-2xx decisions response as one line.

    The HTTP reason phrase is included because OpenRouter's own message
    often does not name the failure class (``"Missing Authentication
    header"`` for a 401), so the log line reads ``HTTP 401 Unauthorized``
    without a status-code lookup.

    Args:
        response: The failed HTTP response.

    Returns:
        ``"Decisions request failed (HTTP <code> <phrase>): <server message>"``.
    """
    try:
        detail = response.json().get("error", {}).get("message") or response.text
    except (ValueError, AttributeError):
        detail = response.text
    try:
        phrase = HTTPStatus(response.status_code).phrase
    except ValueError:
        phrase = ""
    return f"Decisions request failed (HTTP {response.status_code} {phrase}): {detail}".strip()
