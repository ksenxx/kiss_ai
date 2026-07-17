# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Client-tool proxying for the synchronous ``kiss.server.sorcar.run`` API.

The caller of :func:`kiss.server.sorcar.run` lives in a different
process than the daemon that executes the agent, so its Python
callables cannot be handed to the agent directly.  Instead the client
serializes each tool into a JSON *spec* (:func:`tool_specs`) that
travels on the ``run`` command, and the daemon rebuilds each spec into
a *proxy* function (:func:`build_proxy_tools`) whose ``__name__``,
docstring, and ``__signature__`` match the original — so the agent's
OpenAI tool-schema builder registers it exactly like a native tool.
When the agent invokes a proxy, the daemon broadcasts a ``toolRequest``
event to the launching client, which executes the real callable locally
and replies with a ``toolResponse`` command carrying the result.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger("kiss-vscode")

# JSON-safe parameter annotations a client tool may declare.  Anything
# else (or a missing annotation) degrades to ``str`` — the same
# fallback the model-side schema builder applies to unknown types.
_TYPE_NAMES: dict[Any, str] = {str: "str", int: "int", float: "float", bool: "bool"}
_NAME_TYPES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}
_JSON_PRIMITIVES = (str, int, float, bool)


def tool_specs(tools: Sequence[Callable[..., Any]]) -> list[dict[str, Any]]:
    """Serialize client tool callables into JSON specs for the daemon.

    Each spec carries the tool's name, its full docstring (whose
    Google-style ``Args:`` section supplies per-parameter descriptions
    to the model schema), and its parameters with their JSON type names
    and defaults.  A ``"default"`` key is present exactly when the
    parameter is optional; a non-JSON-primitive default is transmitted
    as ``None`` (the client applies the real default locally when the
    argument is omitted).  A ``"kind": "keyword_only"`` key marks
    keyword-only parameters so the daemon rebuilds the exact signature
    (a required keyword-only parameter may legally follow an optional
    one).

    Args:
        tools: The tool callables to serialize.

    Returns:
        One JSON-serializable spec dictionary per tool, in order.

    Raises:
        ValueError: When an entry is not callable, has no valid
            identifier ``__name__`` (e.g. a lambda or ``functools.partial``),
            duplicates another tool's name, or has a parameter kind
            that cannot be expressed in a tool schema (``*args``,
            ``**kwargs``, or positional-only).
    """
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for func in tools:
        if not callable(func):
            raise ValueError(f"tool {func!r} is not callable")
        name = getattr(func, "__name__", "")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(
                f"tool {func!r} has no valid __name__ (got {name!r}); "
                "pass a plain named function"
            )
        if name in seen:
            raise ValueError(f"duplicate tool name {name!r}")
        seen.add(name)
        params: list[dict[str, Any]] = []
        for param in inspect.signature(func).parameters.values():
            if param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise ValueError(
                    f"tool {name!r} parameter {param.name!r} has "
                    f"unsupported kind {param.kind.description!r}; only "
                    "plain (keyword-bindable) parameters are supported"
                )
            # Under ``from __future__ import annotations`` (PEP 563)
            # the annotation arrives as the STRING ``"int"`` rather
            # than the ``int`` class — accept both spellings.
            annotation = param.annotation
            if isinstance(annotation, str):
                type_name = annotation if annotation in _NAME_TYPES else "str"
            else:
                type_name = _TYPE_NAMES.get(annotation, "str")
            entry: dict[str, Any] = {"name": param.name, "type": type_name}
            if param.kind is inspect.Parameter.KEYWORD_ONLY:
                entry["kind"] = "keyword_only"
            if param.default is not inspect.Parameter.empty:
                entry["default"] = (
                    param.default
                    if param.default is None
                    or isinstance(param.default, _JSON_PRIMITIVES)
                    else None
                )
            params.append(entry)
        specs.append(
            {
                "name": name,
                "description": inspect.getdoc(func) or "",
                "params": params,
            }
        )
    return specs


def build_proxy_tools(
    raw_specs: Any,
    call_remote: Callable[[str, dict[str, Any]], str],
) -> list[Callable[..., Any]]:
    """Rebuild daemon-side proxy functions from client tool specs.

    Each returned proxy mirrors the client tool's name, docstring, and
    signature (via ``__signature__``) so the agent registers and
    schemas it like a native tool, and forwards every invocation to
    *call_remote* with only the arguments the model actually supplied
    (omitted optional arguments fall back to the client function's own
    defaults when it runs).

    Malformed specs (non-dict entries, invalid names, broken parameter
    lists) come from outside the daemon process, so they are logged and
    skipped rather than allowed to kill the task thread.

    Args:
        raw_specs: The ``tools`` field of a ``run`` command — expected
            to be a list of spec dictionaries produced by
            :func:`tool_specs`, but treated as untrusted input.
        call_remote: Callback ``(tool_name, arguments) -> result`` that
            performs the round trip to the client.

    Returns:
        The proxy callables for every well-formed spec, in order.
    """
    proxies: list[Callable[..., Any]] = []
    if not isinstance(raw_specs, list):
        if raw_specs is not None:
            logger.warning(
                "Ignoring malformed tools field of type %s",
                type(raw_specs).__name__,
            )
        return proxies
    for spec in raw_specs:
        proxy = _build_proxy(spec, call_remote)
        if proxy is not None:
            proxies.append(proxy)
    return proxies


def _build_proxy(
    spec: Any,
    call_remote: Callable[[str, dict[str, Any]], str],
) -> Callable[..., Any] | None:
    """Build one proxy function from one client tool spec.

    Args:
        spec: One entry of a ``run`` command's ``tools`` list.
        call_remote: Callback ``(tool_name, arguments) -> result``.

    Returns:
        The proxy callable, or ``None`` when the spec is malformed.
    """
    if not isinstance(spec, dict):
        logger.warning("Skipping malformed tool spec %r", spec)
        return None
    name = spec.get("name")
    if not isinstance(name, str) or not name.isidentifier():
        logger.warning("Skipping tool spec with invalid name %r", name)
        return None
    raw_params = spec.get("params")
    parameters: list[inspect.Parameter] = []
    try:
        for raw in raw_params if isinstance(raw_params, list) else []:
            param_name = raw.get("name") if isinstance(raw, dict) else None
            if not isinstance(param_name, str) or not param_name.isidentifier():
                logger.warning(
                    "Skipping tool %r: invalid parameter entry %r", name, raw,
                )
                return None
            type_name = raw.get("type")
            parameters.append(
                inspect.Parameter(
                    param_name,
                    (
                        inspect.Parameter.KEYWORD_ONLY
                        if raw.get("kind") == "keyword_only"
                        else inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    default=(
                        raw["default"]
                        if "default" in raw
                        else inspect.Parameter.empty
                    ),
                    annotation=(
                        _NAME_TYPES.get(type_name, str)
                        if isinstance(type_name, str)
                        else str
                    ),
                )
            )
        signature = inspect.Signature(parameters)
    except ValueError:
        # Reserved-word parameter names (``isidentifier`` passes but
        # ``inspect.Parameter`` rejects them), duplicate names, or a
        # required positional parameter after an optional one —
        # impossible from ``tool_specs`` but possible from a
        # hand-crafted client.
        logger.warning("Skipping tool %r: invalid parameter list", name)
        return None

    def proxy(*args: Any, **kwargs: Any) -> str:
        bound = signature.bind_partial(*args, **kwargs)
        return call_remote(name, dict(bound.arguments))

    proxy.__name__ = name
    proxy.__qualname__ = name
    proxy.__doc__ = str(spec.get("description") or "") or f"Client tool {name}."
    proxy.__signature__ = signature  # type: ignore[attr-defined]
    return proxy
