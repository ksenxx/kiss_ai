# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixer-5 SEA / user-asset bugs (findings F5-07, F5-08).

F5-07 — ``apply_sea`` treats the SEA (the only
loader of caller-supplied tool code) as untrusted; a broken script
must stop the task with a DIAGNOSTIC error.  Every import-time
failure — including ``KeyboardInterrupt`` and ``SystemExit``, which
are not ``Exception`` subclasses — must surface as
:exc:`~kiss.agents.sorcar.sea_apply.SeaError`: the loader's production
caller sits inside an ``except KeyboardInterrupt`` branch that cancels
the whole agent task, so letting either escape unwrapped would report
a broken script as a task cancellation (or kill the thread) instead of
a task error carrying the diagnostic.

F5-08 — the user-asset seeder issued a single ``os.write`` and
ignored its return count; the buffered file-object path now
guarantees the whole default is written before the hard link
publishes the file.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from kiss.agents.sorcar.sea_apply import apply_sea
from kiss.agents.sorcar.sea_settings import SeaError
from kiss.server.user_assets import ensure_user_asset_from_default


def _load_tools(sea_path: Any) -> list:
    """Apply *sea_path*'s overrides and return what its staged ``tools`` hook yields.

    The hook (the SEA's ``tools`` method) is applied to an empty
    toolset, as the daemon applies it to the run's own; a SEA without
    the method stages no hook and yields no tools.
    """
    cmd: dict[str, Any] = {"seaPath": sea_path}
    apply_sea(cmd)
    hook = cmd.get("toolsHook")
    return list(hook([])) if hook is not None else []


class TestBrokenAgentScriptRaisesDiagnostic(unittest.TestCase):
    """F5-07: every broken SEA raises a diagnostic error, only that.

    Loading and staging the script raises ``SeaError``; the
    staged ``tools`` hook, run later by the agent, raises
    ``SeaError`` (an ``Exception``, never the raw
    ``BaseException`` the script threw).
    """

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, body: str) -> str:
        path = self.root / "tools.py"
        path.write_text(body)
        return str(path)

    def test_keyboard_interrupt_in_script_raises_agent_file_error(self) -> None:
        path = self._write("raise KeyboardInterrupt('module interrupt')\n")
        try:
            _load_tools(path)
        except SeaError as err:
            self.assertIn("failed to import", str(err))
            self.assertIn("KeyboardInterrupt", str(err))
            # The loader's SeaError sits between the diagnostic and
            # the original raise; ``task_runner._stop_interrupt_wrapped``
            # walks the whole cause chain, so the interrupt must be in it.
            causes: list[BaseException] = []
            cause: BaseException | None = err.__cause__
            while cause is not None:
                causes.append(cause)
                cause = cause.__cause__
            self.assertTrue(
                any(isinstance(c, KeyboardInterrupt) for c in causes),
                f"KeyboardInterrupt missing from the cause chain: {causes!r}",
            )
        except BaseException as err:  # noqa: BLE001 — the bug under test
            self.fail(
                f"apply_sea let {type(err).__name__} escape "
                "unwrapped; the task runner would treat it as a task "
                "cancellation instead of a diagnostic task error",
            )
        else:
            self.fail("broken SEA must raise SeaError")

    def test_system_exit_in_script_raises_agent_file_error(self) -> None:
        path = self._write("raise SystemExit(3)\n")
        with self.assertRaisesRegex(SeaError, "SystemExit"):
            _load_tools(path)

    def test_plain_exception_in_script_raises_agent_file_error(self) -> None:
        path = self._write("raise RuntimeError('boom')\n")
        with self.assertRaisesRegex(SeaError, "RuntimeError: boom"):
            _load_tools(path)

    def test_syntax_error_in_script_raises_agent_file_error(self) -> None:
        path = self._write("def broken(:\n")
        with self.assertRaisesRegex(SeaError, "SyntaxError"):
            _load_tools(path)

    def test_missing_script_raises_agent_file_error(self) -> None:
        with self.assertRaisesRegex(SeaError, "not an existing"):
            _load_tools(str(self.root / "nowhere.py"))

    def test_non_string_sea_path_field_raises_agent_file_error(self) -> None:
        with self.assertRaisesRegex(SeaError, "path string"):
            _load_tools(42)

    def test_empty_sea_path_field_yields_no_tools(self) -> None:
        self.assertEqual(_load_tools(None), [])
        self.assertEqual(_load_tools(""), [])

    def test_healthy_script_stages_its_tools(self) -> None:
        path = self._write(
            '''
from kiss.agents.seas.base.base_sea import BaseSea

def greet(name: str) -> str:
    """Say hi."""
    return f'hi {name}'

class Sea(BaseSea):
    def tools(self, tools):
        """Return the tools."""
        return tools + [greet]
'''
        )
        # ``tools()`` ADDS to the built-in toolset through the staged
        # ``toolsHook`` callable (hooks are always staged, so they are not
        # listed as overrides); the caller's tool profile is left alone (no
        # ``toolProfile`` override, no legacy ``appendBasicTools`` field).
        cmd: dict[str, Any] = {"seaPath": path}
        self.assertEqual(apply_sea(cmd), set())
        staged = cmd["toolsHook"]([])
        self.assertEqual([t.__name__ for t in staged], ["greet"])
        self.assertEqual(staged[0](name="bob"), "hi bob")
        self.assertNotIn("tools", cmd)
        self.assertNotIn("toolProfile", cmd)
        self.assertNotIn("appendBasicTools", cmd)
        self.assertNotIn("toolsFile", cmd)

    def test_script_without_tools_method_stages_identity_hook(self) -> None:
        path = self._write(
            "from kiss.agents.seas.base.base_sea import BaseSea\n"
            "\n"
            "class Sea(BaseSea):\n"
            "    pass\n"
            "\n"
            "def greet(name: str) -> str:\n"
            "    \"\"\"Say hi.\"\"\"\n"
            "    return f'hi {name}'\n"
        )
        cmd: dict[str, Any] = {"seaPath": path}
        self.assertEqual(apply_sea(cmd), set())
        # No ``tools()`` method: the staged hook hands the toolset back
        # unchanged (the module-level ``greet`` is not picked up).
        self.assertEqual(cmd["toolsHook"]([]), [])
        self.assertEqual([t.__name__ for t in cmd["toolsHook"]([print])], ["print"])
        self.assertNotIn("tools", cmd)
        self.assertNotIn("appendBasicTools", cmd)
        self.assertEqual(_load_tools(path), [])

    def test_raising_repr_in_tools_result_raises_sea_script_error(
        self,
    ) -> None:
        # Validating the returned entries must never run user code
        # (e.g. a raising ``__repr__``) unguarded: any escape from the
        # validation must surface as SeaError, not as the raw
        # BaseException (which the task runner may misread as a
        # cancellation).
        path = self._write(
            '''
from kiss.agents.seas.base.base_sea import BaseSea

class _EvilRepr:
    def __repr__(self):
        raise KeyboardInterrupt('evil repr')

class Sea(BaseSea):
    def tools(self, tools):
        """Return a broken entry."""
        return tools + [_EvilRepr()]
'''
        )
        with self.assertRaisesRegex(SeaError, "list of tool callables"):
            _load_tools(path)

    def test_raising_exception_str_still_yields_agent_file_error(self) -> None:
        # Building the diagnostic itself must not run raising untrusted
        # code: an exception whose ``__str__`` raises (here a
        # KeyboardInterrupt) must still surface as SeaError with
        # the type-name-only fallback message.
        path = self._write(
            "class _EvilStr(Exception):\n"
            "    def __str__(self):\n"
            "        raise KeyboardInterrupt('str bomb')\n"
            "\n"
            "raise _EvilStr()\n"
        )
        with self.assertRaisesRegex(SeaError, "_EvilStr"):
            _load_tools(path)

    def test_nul_byte_path_raises_agent_file_error(self) -> None:
        # ``Path.is_file`` raises ValueError on an embedded NUL byte;
        # the loader must report the standard diagnostic instead of
        # leaking the ValueError.
        with self.assertRaisesRegex(SeaError, "not an existing"):
            _load_tools("bad\x00tools.py")

    def test_raising_iter_in_tools_result_raises_sea_script_error(
        self,
    ) -> None:
        path = self._write(
            '''
from kiss.agents.seas.base.base_sea import BaseSea

class _EvilList(list):
    def __iter__(self):
        raise SystemExit(9)

def ok() -> str:
    """Return ok."""
    return 'ok'

class Sea(BaseSea):
    def tools(self, tools):
        """Return a list whose iteration raises."""
        return _EvilList(tools + [ok])
'''
        )
        with self.assertRaisesRegex(SeaError, r"tools\(\).*broken list.*SystemExit"):
            _load_tools(path)


class TestUserAssetSeedIsComplete(unittest.TestCase):
    """F5-08: the seeded asset always holds the complete default."""

    def test_large_default_content_is_seeded_exactly(self) -> None:
        # Large enough (~5 MB) that a partial write would be visible;
        # the buffered writer must publish every byte before linking.
        content = ("# Trick line ...\n" * 64 + "unique-tail-marker\n") * 4800
        with TemporaryDirectory() as home:
            import os

            old = os.environ.get("KISS_HOME")
            os.environ["KISS_HOME"] = home
            try:
                path = ensure_user_asset_from_default(
                    "FIXER5_ASSET.md", content,
                )
            finally:
                if old is None:
                    os.environ.pop("KISS_HOME", None)
                else:
                    os.environ["KISS_HOME"] = old
            self.assertIsNotNone(path)
            assert path is not None
            on_disk = path.read_text(encoding="utf-8")
            self.assertEqual(len(on_disk), len(content))
            self.assertEqual(on_disk, content)
            # The staging temp file must not linger next to the asset.
            leftovers = [
                p.name for p in path.parent.iterdir()
                if p.name.startswith(".FIXER5_ASSET.md-")
            ]
            self.assertEqual(leftovers, [])

    def test_existing_asset_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as home:
            import os

            old = os.environ.get("KISS_HOME")
            os.environ["KISS_HOME"] = home
            try:
                first = ensure_user_asset_from_default("A.md", "original\n")
                assert first is not None
                second = ensure_user_asset_from_default("A.md", "replacement\n")
            finally:
                if old is None:
                    os.environ.pop("KISS_HOME", None)
                else:
                    os.environ["KISS_HOME"] = old
            assert second is not None
            self.assertEqual(second.read_text(), "original\n")


if __name__ == "__main__":
    unittest.main()
