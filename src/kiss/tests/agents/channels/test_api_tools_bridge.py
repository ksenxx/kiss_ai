# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the live-callable → tools-file bridge.

``kiss.server.sorcar.run`` takes extra agent tools as a *file path*
whose top-level public functions the daemon imports.  Third-party
agents, however, hold LIVE callables (bound backend methods, per-
message ``reply`` closures).  ``_api_tools_bridge`` closes the gap: it
registers the live callables in a process-global registry and
generates a real tools file whose genuine top-level ``def`` wrappers
mirror each callable's name / signature / docstring and dispatch back
to the registry.  These tests drive the FULL production round trip:
``register_tools`` → the daemon-side loader
:func:`kiss.server.tools_file.load_tools_file` importing the generated
file → wrapper invocation → the live callable.
"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.third_party_agents import _api_tools_bridge as bridge
from kiss.server.tools_file import load_tools_file

_SENTINEL_DEFAULT = object()


def module_tool(city: str, unit: str = "C") -> str:
    """Return the temperature of a city.

    Args:
        city: Name of the city.
        unit: Temperature unit.
    """
    return f"{city}:{unit}"


class _Backend:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text: str, *, urgent: bool = False) -> str:
        """Send a message.

        Args:
            text: Message text.
            urgent: Whether the message is urgent.
        """
        self.sent.append(text)
        return f"sent:{text}:{urgent}"


class ApiToolsBridgeTest(unittest.TestCase):
    """Full register → load_tools_file → dispatch round trips."""

    def setUp(self) -> None:
        self._tokens: list[str] = []

    def tearDown(self) -> None:
        for token in self._tokens:
            bridge.release_tools(token)

    def _register(self, tools: list) -> tuple[str, str]:
        token, path = bridge.register_tools(tools)
        self._tokens.append(token)
        return token, path

    def test_generated_file_exists_and_is_python(self) -> None:
        token, path = self._register([module_tool])
        assert token
        assert Path(path).is_file() and path.endswith(".py")

    def test_public_function_wrapper_fidelity_and_dispatch(self) -> None:
        calls: list[tuple[str, int]] = []

        def greet(name: str, times: int = 2) -> str:
            """Greet someone.

            Args:
                name: Who to greet.
                times: How many times.
            """
            calls.append((name, times))
            return "hi " * times + name

        _token, path = self._register([greet])
        (wrapper,) = load_tools_file(path)
        assert wrapper.__name__ == "greet"
        assert "Who to greet." in (wrapper.__doc__ or "")
        params = inspect.signature(wrapper).parameters
        assert list(params) == ["name", "times"]
        # Annotation fidelity: the wrapper's PEP 563 annotation string
        # matches a native postponed-annotations tool exactly.
        assert params["times"].annotation == "int"
        assert params["name"].annotation == "str"
        # The wrapper's default is the ORIGINAL default object.
        assert params["times"].default == 2
        # Dispatch reaches the live closure; omitted params use the
        # original default.
        assert wrapper(name="ada") == "hi hi ada"
        assert wrapper("bob", times=1) == "hi bob"
        assert calls == [("ada", 2), ("bob", 1)]

    def test_bound_method_and_closure_share_live_state(self) -> None:
        backend = _Backend()
        replied: list[str] = []

        def reply(message: str) -> str:
            """Reply to the current conversation.

            Args:
                message: Text to send.
            """
            replied.append(message)
            return "ok"

        _token, path = self._register([backend.send, reply])
        loaded = {t.__name__: t for t in load_tools_file(path)}
        assert set(loaded) == {"send", "reply"}
        assert loaded["send"]("hello", urgent=True) == "sent:hello:True"
        assert loaded["send"](text="lo") == "sent:lo:False"
        assert loaded["reply"](message="pong") == "ok"
        assert backend.sent == ["hello", "lo"]
        assert replied == ["pong"]

    def test_keyword_only_parameters_preserved(self) -> None:
        def tag(prefix: str = ">", *, label: str) -> str:
            """Tag a label.

            Args:
                prefix: Prefix text.
                label: The label.
            """
            return prefix + label

        _token, path = self._register([tag])
        (wrapper,) = load_tools_file(path)
        params = inspect.signature(wrapper).parameters
        assert params["label"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["prefix"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert wrapper(label="x") == ">x"

    def test_non_literal_default_preserved_by_identity(self) -> None:
        seen: list[Any] = []

        def probe(value: Any = _SENTINEL_DEFAULT) -> str:
            """Probe a value.

            Args:
                value: Anything.
            """
            seen.append(value)
            return "probed"

        _token, path = self._register([probe])
        (wrapper,) = load_tools_file(path)
        assert (
            inspect.signature(wrapper).parameters["value"].default
            is _SENTINEL_DEFAULT
        ), "non-literal defaults must be hoisted by identity, not repr"
        wrapper()
        assert seen == [_SENTINEL_DEFAULT]

    def test_unannotated_params_supported(self) -> None:
        def loose(a, b=5):  # type: ignore[no-untyped-def]
            """Add two things.

            Args:
                a: First.
                b: Second.
            """
            return a + b

        _token, path = self._register([loose])
        (wrapper,) = load_tools_file(path)
        assert wrapper(a=1) == 6
        assert wrapper(a=1, b=2) == 3

    def test_multiple_tools_in_one_file(self) -> None:
        def one() -> str:
            """Return one."""
            return "1"

        def two() -> str:
            """Return two."""
            return "2"

        _token, path = self._register([one, two])
        loaded = load_tools_file(path)
        assert [t.__name__ for t in loaded] == ["one", "two"]
        assert [t() for t in loaded] == ["1", "2"]

    def test_exception_propagates_from_live_callable(self) -> None:
        def kaboom() -> str:
            """Always fail."""
            raise RuntimeError("kaboom-inner")

        _token, path = self._register([kaboom])
        (wrapper,) = load_tools_file(path)
        with self.assertRaisesRegex(RuntimeError, "kaboom-inner"):
            wrapper()

    def test_release_tools_invalidates_dispatch_and_removes_file(self) -> None:
        def gone() -> str:
            """Return gone."""
            return "gone"

        token, path = bridge.register_tools([gone])
        (wrapper,) = load_tools_file(path)
        bridge.release_tools(token)
        assert not Path(path).exists(), "release must delete the tools file"
        with self.assertRaises(RuntimeError):
            wrapper()
        # Releasing twice is a no-op.
        bridge.release_tools(token)

    def test_dispatch_unknown_token_or_tool_raises(self) -> None:
        def known() -> str:
            """Return known."""
            return "known"

        token, _path = self._register([known])
        with self.assertRaises(RuntimeError):
            bridge.dispatch("no-such-token", "known", {})
        with self.assertRaises(RuntimeError):
            bridge.dispatch(token, "unknown", {})
        with self.assertRaises(RuntimeError):
            bridge.tool_default("no-such-token", "known", "x")
        with self.assertRaises(RuntimeError):
            bridge.tool_doc("no-such-token", "known")

    def test_empty_tools_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bridge.register_tools([])

    def test_invalid_tools_raise_value_error(self) -> None:
        def ok() -> str:
            """Return ok."""
            return "ok"

        def star_args(*args: str) -> str:
            """Star args.

            Args:
                *args: Anything.
            """
            return ",".join(args)

        def star_kwargs(**kwargs: str) -> str:
            """Star kwargs."""
            return str(kwargs)

        def underscore_param(_hidden: str) -> str:
            """Underscore param.

            Args:
                _hidden: Hidden.
            """
            return _hidden

        cases: list[Any] = [
            [ok, ok],  # duplicate names
            ["not callable"],  # not callable
            [lambda x: x],  # lambda → invalid tool name
            [star_args],
            [star_kwargs],
            [underscore_param],
        ]
        for tools in cases:
            with self.assertRaises(ValueError, msg=repr(tools)):
                bridge.register_tools(tools)
        # Positional-only parameters cannot be bound by keyword.
        with self.assertRaises(ValueError):
            bridge.register_tools([divmod])

    def test_private_named_tool_rejected(self) -> None:
        def _private() -> str:
            """Return private."""
            return "private"

        with self.assertRaises(ValueError):
            bridge.register_tools([_private])

    def test_unparseable_annotation_falls_back_to_string_literal(
        self,
    ) -> None:
        def odd(x: str) -> str:
            """Odd tool.

            Args:
                x: X.
            """
            return x

        odd.__annotations__["x"] = "not valid ("
        _token, path = self._register([odd])
        (wrapper,) = load_tools_file(path)
        # PEP 563 stores the fallback string literal's SOURCE text
        # (quotes included); the point is that the generated module
        # still compiles and the tool still works.
        assert (
            inspect.signature(wrapper).parameters["x"].annotation
            == repr("not valid (")
        ), "unparseable annotation text must survive via a string literal"
        assert wrapper(x="ok") == "ok"

    def test_broken_signature_rejected(self) -> None:
        def weird() -> str:
            """Return weird."""
            return "weird"

        weird.__signature__ = "not a signature"  # type: ignore[attr-defined]
        with self.assertRaises(ValueError):
            bridge.register_tools([weird])


if __name__ == "__main__":
    unittest.main()
