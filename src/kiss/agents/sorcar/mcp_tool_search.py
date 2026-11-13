# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Deferred MCP tools: their names up front, their definitions on demand.

Every MCP tool definition an agent loads (name, description, argument
schema) is part of every model request.  A few large servers cost tens
of thousands of tokens that way, most of them for tools the task never
calls.  This module keeps the definitions of chosen servers out of the
request:

* :func:`split_by_budget` decides which servers load upfront;
* :func:`make_deferred_tools` builds the tools that stand in for the
  rest: a search tool whose description lists the deferred tools' names
  and which returns the definitions asked for, and a call tool that
  runs a deferred tool by name.

The agent's toolset does not change during a run: a definition reaches
the model as a tool result, at the end of the conversation, so the
providers' prompt caches, whose prefix starts with the tool
definitions, stay valid.

No server is deferred unless the user asks for it (:func:`upfront_tool_chars`,
``alwaysLoad``).  Deferral trades tokens for accuracy: a model that
sees only the names sometimes picks the wrong tool, where the same
model with every definition in view does not
(``reports/mcp-tool-deferral.html`` has the measurements).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from kiss.core.tool_interrupt import current_tool_call
from kiss.core.tool_schema import declared_schema, tool_schema, unlisted
from kiss.core.user_config import load_config

#: The definitions a keyword search returns unless the model asks for more.
DEFAULT_LIMIT = 5

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[^\W_]+")
_NAME_SEPARATOR = re.compile(r"[\s,]+")


@dataclass(frozen=True)
class ServerTools:
    """The tools of one connected MCP server.

    Attributes:
        name: The configured server name.
        always_load: The server's ``alwaysLoad`` setting: ``True`` loads
            its tools upfront, ``False`` defers them, ``None`` leaves the
            choice to the budget, if one is set.
        tools: The wrappers of its permitted tools
            (:func:`~kiss.agents.sorcar.mcp_servers.make_mcp_tool_wrapper`).
    """

    name: str
    always_load: bool | None
    tools: list[Any]


def upfront_tool_chars() -> int | None:
    """Return the budget of MCP tool definitions loaded upfront, in characters, if one is set.

    Returns:
        The ``mcp_upfront_tool_chars`` key of ``$KISS_HOME/config.json``
        (``0`` defers every server not marked ``alwaysLoad``); ``None``,
        which defers no server for its size, when the key is absent or
        is not a non-negative integer.
    """
    value = load_config().get("mcp_upfront_tool_chars")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _declared(tool: Any) -> dict[str, Any]:
    """The ``{"parameters", "description"}`` an MCP tool wrapper declares (every one does)."""
    return cast(dict[str, Any], declared_schema(tool))


def definition_chars(tool: Any) -> int:
    """Return the size of an MCP tool's definition as a model request carries it.

    Args:
        tool: A wrapper built by ``make_mcp_tool_wrapper``.

    Returns:
        The characters of its name, its description and its argument schema as JSON.
    """
    declared = _declared(tool)
    schema = json.dumps(declared["parameters"])
    return len(tool.__name__) + len(declared["description"]) + len(schema)


def _server_chars(server: ServerTools) -> int:
    """The size of *server*'s tool definitions, in characters."""
    return sum(definition_chars(tool) for tool in server.tools)


def split_by_budget(
    servers: list[ServerTools], budget: int | None,
) -> tuple[list[Any], list[ServerTools]]:
    """Decide which servers' tools load upfront and which are deferred.

    A server marked ``alwaysLoad`` loads upfront and one marked
    ``"alwaysLoad": false`` is deferred, whatever their sizes.  The
    unmarked ones all load upfront when *budget* is ``None``; otherwise
    they load upfront from the smallest up while their definitions
    together fit *budget* (servers of one size in the order given), so
    the deferred ones are the largest.

    Args:
        servers: Every connected server with its permitted tools.
        budget: Characters of definitions (:func:`definition_chars`)
            the unmarked servers may load upfront; ``None`` for no limit.

    Returns:
        The upfront tools and the deferred servers, both in the order given.
    """
    deferred = [server for server in servers if server.always_load is False]
    spent = 0
    for server in sorted((s for s in servers if s.always_load is None), key=_server_chars):
        spent += _server_chars(server)
        if budget is not None and spent > budget:
            deferred.append(server)
    upfront = [tool for server in servers if server not in deferred for tool in server.tools]
    return upfront, [server for server in servers if server in deferred]


def _stem(word: str) -> str:
    """Cut a plural ending, so that ``branches`` finds ``branch`` and ``queries`` ``query``."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _words(text: str) -> set[str]:
    """The stemmed lower-case words of *text*; camelCase, snake_case and kebab-case split."""
    return {_stem(word) for word in _WORD.findall(_CAMEL_BOUNDARY.sub(" ", text).lower())}


def _word_weights(tool: Any) -> dict[str, int]:
    """Map each word of *tool*'s definition to the weight of the best place it occurs in.

    A word of the tool's name weighs 3, of its description 2, and of
    its arguments' names and descriptions 1.
    """
    declared = _declared(tool)
    arguments = " ".join(
        f"{name} {schema.get('description', '')}"
        for name, schema in declared["parameters"]["properties"].items()
    )
    weights: dict[str, int] = {}
    for weight, text in ((1, arguments), (2, declared["description"]), (3, tool.__name__)):
        weights.update(dict.fromkeys(_words(text), weight))
    return weights


def _response() -> tuple[int | None, int | None]:
    """The ``(agent, step)`` of the model response the running tool call came in.

    See :attr:`~kiss.core.tool_interrupt.ToolCallToken.response`;
    ``(None, None)`` outside an agent's tool call.
    """
    token = current_tool_call()
    return token.response if token is not None else (None, None)


class DeferredTools:
    """The deferred MCP tools of one agent run: their definitions on demand, and their calls.

    A tool runs only when the model wrote the call with the tool's
    definition before it: a model that knows a tool by its name alone
    does call it with arguments it guessed.  So the definition must
    have been given to the agent that makes the call (each session of a
    task is an agent of its own, with a conversation of its own), in a
    model response before the call's: the calls of one response were
    all written before any of their results.  Not seen from here: a
    definition the agent compacted out of a long conversation since,
    and the responses of a model that runs the tools inside its own
    turn (Codex), where only the first of the two conditions is checked.

    Attributes:
        tools: The deferred tool wrappers by name.
        loaded: For each tool whose definition the model received, the
            ``(agent, step)`` of the response whose results first
            carried it (:func:`_response`).
    """

    def __init__(self, servers: list[ServerTools], search_name: str, call_name: str) -> None:
        """Index the tools of the deferred *servers*.

        Args:
            servers: The deferred servers with their tool wrappers.
            search_name: The name the search tool is registered under.
            call_name: The name the call tool is registered under.
        """
        self.search_name = search_name
        self.call_name = call_name
        self.tools = {tool.__name__: tool for server in servers for tool in server.tools}
        self.loaded: dict[str, tuple[int | None, int | None]] = {}
        self._weights = {name: _word_weights(tool) for name, tool in self.tools.items()}
        counts = Counter(word for weights in self._weights.values() for word in weights)
        self._rarity = {
            word: math.log(1 + len(self.tools) / count) for word, count in counts.items()
        }

    def _ranked(self, query: str) -> list[str]:
        """Return the names of the tools sharing a word with *query*, the best match first.

        A tool scores, for each word of the query it has, the weight of
        the place the word occurs in (:func:`_word_weights`) times the
        word's rarity among the deferred tools; equal scores go by name.
        """
        words = _words(query)
        scored = sorted(
            (-score, name)
            for name, weights in self._weights.items()
            if (score := sum(weights.get(word, 0) * self._rarity.get(word, 0.0) for word in words))
        )
        return [name for _, name in scored]

    def _given_to(self, agent: int | None, name: str) -> bool:
        """Whether *agent*'s conversation carries the definition of the tool *name*."""
        return name in self.loaded and self.loaded[name][0] == agent

    def _definition(self, name: str) -> str:
        """Return the definition of the tool *name* for the model, and record it as loaded."""
        declared = _declared(self.tools[name])
        agent, step = _response()
        if not self._given_to(agent, name):
            self.loaded[name] = (agent, step)
        return f"{name}: {declared['description']}\narguments: {json.dumps(declared['parameters'])}"

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> str:
        """Return the definitions of the deferred tools *query* names or describes.

        Args:
            query: Exact tool names (any number, separated by spaces or
                commas), each returned in full; otherwise keywords,
                matched against the tools' names, descriptions and
                arguments (:meth:`_ranked`).
            limit: The most definitions a keyword query returns.

        Returns:
            The definitions, the best match first; or a line saying
            that nothing matched, or that *limit* is no number.
        """
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            return f"Error: limit must be a positive integer, got {limit!r}."
        names = [name for name in dict.fromkeys(_NAME_SEPARATOR.split(query)) if name in self.tools]
        matches = len(names)
        if not names:
            ranked = self._ranked(query)
            matches, names = len(ranked), ranked[:limit]
        if not names:
            return (
                f"No deferred MCP tool matches {query!r}. A tool that is not deferred needs "
                "no lookup: call it directly, under its own name. The deferred tools are the "
                f"ones listed in the description of {self.search_name}; to find one of those, "
                "try other keywords."
            )
        note = (
            f"Showing {len(names)} of {matches} matches; name a tool exactly or raise "
            "limit for the others.\n\n"
        ) if matches > len(names) else ""
        definitions = "\n\n".join(self._definition(name) for name in names)
        return f"{note}{definitions}\n\nRun one with {self.call_name}(name, arguments)."

    def call(self, name: str, arguments: Any = None) -> str:
        """Run the deferred tool *name* with *arguments*, once its definition is loaded.

        Args:
            name: The tool's name, as the search tool's description lists it.
            arguments: The tool's arguments: a JSON object or its text;
                ``None`` for a call without arguments.

        Returns:
            The tool's result.  Without running the tool: its definition
            when the model had not received it yet, a line saying so
            when the definition is a result of the very response the
            call came in, or an ``Error:`` line for a name that is no
            deferred tool and for arguments that are no JSON object.
        """
        if name not in self.tools:
            closest = ", ".join(self._ranked(name)[:3])
            return (
                f"Error: {name!r} is not a deferred MCP tool."
                f"{f' Closest: {closest}.' if closest else ''} A tool that is not deferred "
                f"does not go through {self.call_name}: call it directly, under its own name. "
                f"The deferred tools are the ones listed in the description of {self.search_name}."
            )
        agent, step = _response()
        if not self._given_to(agent, name):
            return (
                f"Not run: the definition of {name} was not loaded. Here it is; call "
                f"{self.call_name} again with arguments that match it.\n\n{self._definition(name)}"
            )
        if step is not None and self.loaded[name][1] == step:
            return (
                f"Not run: the definition of {name} is in another result of this response, "
                f"so this call was written without it. Call {self.call_name} again with "
                "arguments that match it."
            )
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                return f"Error: arguments is not valid JSON ({exc})."
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return f"Error: arguments must be a JSON object, got {type(arguments).__name__}."
        return str(self.tools[name](**arguments))


def _direct_call_tool(deferred: DeferredTools, name: str) -> Any:
    """Return the tool that answers a call of the deferred tool *name* under its own name.

    Some models call a deferred tool by the name the search tool lists
    instead of through the call tool, and the agent would answer that
    there is no such tool.  This tool answers as the call tool does.
    It declares the deferred tool's schema, which types the arguments
    of a call a model wrote as text, and is
    :func:`~kiss.core.tool_schema.unlisted`, so no model request
    carries that definition.  Codex dispatches its model's native calls
    itself and answers one to a name it was not given a definition of
    ("unsupported call"), so such a call does not arrive here; the
    search and call tools, which it is given, work there as elsewhere.
    """

    @tool_schema(**_declared(deferred.tools[name]))
    def direct(**arguments: Any) -> str:
        return deferred.call(name, arguments)

    direct.__name__ = direct.__qualname__ = name
    return unlisted(direct)


def make_deferred_tools(
    servers: list[ServerTools], search_name: str, call_name: str,
) -> list[Any]:
    """Build the tools that stand in for the deferred *servers*.

    Args:
        servers: The deferred servers with their tool wrappers.
        search_name: The name of the search tool, free in the run's toolset.
        call_name: The name of the call tool, free in the run's toolset.

    Returns:
        The search tool, whose description lists the deferred tools'
        names by server; the call tool; and, listed in no model
        request, one tool per deferred tool under that tool's name
        (:func:`_direct_call_tool`).
    """
    deferred = DeferredTools(servers, search_name, call_name)
    index = "\n".join(
        f"{server.name} ({len(server.tools)}): {', '.join(tool.__name__ for tool in server.tools)}"
        for server in servers
    )

    @tool_schema(
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Exact tool names from the list (several separated by spaces) to get "
                        'exactly those, or keywords for what you need, e.g. "create pull request".'
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": f"Most definitions a keyword query returns ({DEFAULT_LIMIT}).",
                },
            },
            "required": ["query"],
        },
        description=(
            "Load the definitions of deferred MCP tools. The tools listed below are available, "
            "but their definitions (description and arguments) are not loaded: load a tool's "
            f"definition here before its first call, then run it with {call_name}.\n"
            f"Deferred tools by server:\n{index}"
        ),
    )
    def search(query: str, limit: int = DEFAULT_LIMIT) -> str:
        """Load the definitions of deferred MCP tools."""
        return deferred.search(query, limit)

    @tool_schema(
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The tool's name, as listed."},
                "arguments": {
                    "type": "object",
                    "description": "The tool's arguments, as its definition describes them.",
                },
            },
            "required": ["name"],
        },
        description=(
            "Run a deferred MCP tool: one of the tools listed in the description of "
            f"{search_name}, whose definition you loaded there."
        ),
    )
    def call(name: str, arguments: Any = None) -> str:
        """Run a deferred MCP tool."""
        return deferred.call(name, arguments)

    search.__name__ = search.__qualname__ = search_name
    call.__name__ = call.__qualname__ = call_name
    return [search, call, *(_direct_call_tool(deferred, name) for name in deferred.tools)]
