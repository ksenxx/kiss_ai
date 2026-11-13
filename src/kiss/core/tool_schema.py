# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``@tool_schema``: the JSON schema a tool declares for itself, instead of a derived one.

By default the framework derives a tool's definition from its Python
signature and docstring (:func:`~kiss.core.models.model.Model._function_to_openai_tool`):
parameter names, Python types mapped to JSON-schema types, and the
docstring as the description.  That mapping loses the constraints a
schema can state — an enumeration of allowed values, an array of
objects with required fields — so a tool whose arguments are
constrained can declare its own schema::

    @tool_schema(
        parameters={
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["low", "high"]},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"file": {"type": "string"}, "line": {"type": "integer"}},
                        "required": ["file", "line"],
                    },
                },
            },
            "required": ["severity", "findings"],
        },
    )
    def report(severity: str, findings: list[dict]) -> str:
        '''File the findings of the review.'''

The declared schema is what every model sees, whether the provider
takes native tool definitions (OpenAI, Anthropic, Gemini, Codex) or
reads the text catalog of tools; the model's arguments reach the
function as keyword arguments exactly as it sent them.  The decorator
returns the function itself, so its name (the name the model calls),
signature and behaviour are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

TOOL_SCHEMA_ATTR = "__kiss_tool_schema__"
"""The function attribute :func:`tool_schema` stores the declaration on."""

UNLISTED_ATTR = "__kiss_unlisted__"
"""The function attribute :func:`unlisted` sets."""


def tool_schema[F: Callable[..., Any]](
    parameters: dict[str, Any], *, description: str = ""
) -> Callable[[F], F]:
    """Return a decorator declaring the JSON schema of a tool's parameters.

    Args:
        parameters: The JSON schema of the tool's arguments: an object
            schema (``{"type": "object", "properties": {...},
            "required": [...]}``), as the providers' tool definitions
            require.
        description: The tool's description shown to the model; the
            first line of the docstring when empty.

    Raises:
        ValueError: When *parameters* is not an object schema.
    """
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError(
            'tool_schema(parameters=...) needs an object schema: {"type": "object", ...}'
        )

    def declare(func: F) -> F:
        setattr(func, TOOL_SCHEMA_ATTR, {"parameters": parameters, "description": description})
        return func

    return declare


def declared_schema(func: Callable[..., Any]) -> dict[str, Any] | None:
    """Return the ``{"parameters", "description"}`` *func* declared with :func:`tool_schema`.

    A bound method reads its function's attribute, so a decorated method
    of a SEA class declares the schema of the bound tool it contributes.

    Returns:
        The declaration, or ``None`` for an undecorated function.
    """
    declared = getattr(func, TOOL_SCHEMA_ATTR, None)
    return declared if isinstance(declared, dict) else None


def unlisted[F: Callable[..., Any]](func: F) -> F:
    """Mark *func* as a tool the agent runs when called but does not list for the model.

    An unlisted tool's definition is left out of every model request,
    the native tool definitions and the text catalog alike, so it costs
    no context.  A model that learned its name and arguments another
    way (from the result of a search tool, say) can still call it.

    Returns:
        *func* itself.
    """
    setattr(func, UNLISTED_ATTR, True)
    return func


def is_unlisted(func: Callable[..., Any]) -> bool:
    """Return whether *func* was marked :func:`unlisted`."""
    return getattr(func, UNLISTED_ATTR, False) is True
