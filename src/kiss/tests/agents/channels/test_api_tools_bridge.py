# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the live-callable → tools-file bridge.

``kiss.server.sorcar.run`` takes extra agent tools as a *file path*
whose ``get_tools()`` function the daemon imports and calls.
Third-party agents, however, hold LIVE callables (bound backend
methods, per-message ``reply`` closures).  ``_api_tools_bridge``
closes the gap: it registers the live callables in a process-global
registry and generates a tiny real tools file whose ``get_tools()``
returns the registered callables THEMSELVES — no wrapper code, so the
agent receives the original functions with their exact names,
signatures, docstrings, and default objects.  These tests drive the
FULL production round trip: ``register_tools`` → the daemon-side
loader :func:`kiss.server.tools_file.load_tools_file` importing the
generated file and calling its ``get_tools()`` → the live callables.
"""

from __future__ import annotations

import inspect
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from kiss.agents.third_party_agents import _api_tools_bridge as bridge
from kiss.server.tools_file import ToolsFileError, load_tools_file


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
    """Full register → load_tools_file → live-callable round trips."""

    def setUp(self) -> None:
        self._tokens: list[str] = []

    def tearDown(self) -> None:
        for token in self._tokens:
            bridge.release_tools(token)

    def _register(self, tools: list) -> tuple[str, str]:
        token, path = bridge.register_tools(tools)
        self._tokens.append(token)
        return token, path

    def test_loaded_tools_are_the_live_callables_themselves(self) -> None:
        calls: list[tuple[str, int]] = []

        def greet(name: str, times: int = 2, *, shout: bool = False) -> str:
            """Greet someone.

            Args:
                name: Who to greet.
                times: How many times.
                shout: Whether to shout.
            """
            calls.append((name, times))
            text = "hi " * times + name
            return text.upper() if shout else text

        _token, path = self._register([greet])
        (loaded,) = load_tools_file(path)
        assert loaded is greet, "the live callable must be handed over AS-IS"
        assert loaded.__name__ == "greet"
        assert "Who to greet." in (loaded.__doc__ or "")
        params = inspect.signature(loaded).parameters
        assert list(params) == ["name", "times", "shout"]
        assert params["times"].default == 2
        assert params["shout"].kind is inspect.Parameter.KEYWORD_ONLY
        assert loaded(name="ada") == "hi hi ada"
        assert loaded("bob", times=1) == "hi bob"
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

    def test_release_tools_invalidates_token_and_removes_file(self) -> None:
        def gone() -> str:
            """Return gone."""
            return "gone"

        token, path = bridge.register_tools([gone])
        (loaded,) = load_tools_file(path)
        assert loaded is gone
        bridge.release_tools(token)
        assert not Path(path).exists(), "release must delete the tools file"
        with self.assertRaises(RuntimeError):
            bridge.live_tools(token)
        bridge.release_tools(token)  # idempotent

    def test_released_token_fails_a_late_load_with_diagnostic(self) -> None:
        # A daemon loading a copy of the generated file AFTER release
        # must fail with the loader's diagnostic error, not silently
        # yield no tools.
        def late() -> str:
            """Return late."""
            return "late"

        token, path = bridge.register_tools([late])
        source = Path(path).read_text(encoding="utf-8")
        bridge.release_tools(token)
        directory = tempfile.mkdtemp(prefix="kiss_tp_tools_test_")
        self.addCleanup(shutil.rmtree, directory, True)
        copy = Path(directory) / "copy_tools.py"
        copy.write_text(source, encoding="utf-8")
        with self.assertRaisesRegex(ToolsFileError, "not registered"):
            load_tools_file(str(copy))

    def test_empty_tools_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bridge.register_tools([])

    def test_invalid_tools_raise_value_error(self) -> None:
        def ok() -> str:
            """Return ok."""
            return "ok"

        cases: list[Any] = [
            [ok, ok],
            ["not callable"],
            [lambda x: x],
        ]
        for tools in cases:
            with self.assertRaises(ValueError, msg=repr(tools)):
                bridge.register_tools(tools)

    def test_private_named_tool_rejected(self) -> None:
        def _private() -> str:
            """Return private."""
            return "private"

        with self.assertRaises(ValueError):
            bridge.register_tools([_private])

    def test_tool_named_get_tools_is_bridged_like_any_other(self) -> None:
        # The generated file's own ``get_tools`` selector lives at
        # module level; a LIVE tool that happens to be named
        # ``get_tools`` is just an entry in the returned list and must
        # not collide with it.
        def get_tools(query: str) -> str:
            """Homonymous live tool.

            Args:
                query: Anything.
            """
            return f"live:{query}"

        _token, path = self._register([get_tools])
        (loaded,) = load_tools_file(path)
        assert loaded is get_tools
        assert loaded(query="x") == "live:x"


if __name__ == "__main__":
    unittest.main()
