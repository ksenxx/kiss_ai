# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixer-5 tools-file / user-asset bugs (findings F5-07, F5-08).

F5-07 — ``load_tools_file`` treats the tools file as untrusted; a
broken file must stop the task with a DIAGNOSTIC error.  Every
import-time failure — including ``KeyboardInterrupt`` and
``SystemExit``, which are not ``Exception`` subclasses — must surface
as :exc:`~kiss.server.tools_file.ToolsFileError`: the loader's
production caller sits inside an ``except KeyboardInterrupt`` branch
that cancels the whole agent task, so letting either escape unwrapped
would report a broken tools file as a task cancellation (or kill the
thread) instead of a task error carrying the diagnostic.

F5-08 — the user-asset seeder issued a single ``os.write`` and
ignored its return count; the buffered file-object path now
guarantees the whole default is written before the hard link
publishes the file.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kiss.server.tools_file import ToolsFileError, load_tools_file
from kiss.server.user_assets import ensure_user_asset_from_default


class TestBrokenToolsFileRaisesDiagnostic(unittest.TestCase):
    """F5-07: every broken tools file raises ToolsFileError, only that."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, body: str) -> str:
        path = self.root / "tools.py"
        path.write_text(body)
        return str(path)

    def test_keyboard_interrupt_in_tools_file_raises_tools_file_error(self) -> None:
        path = self._write("raise KeyboardInterrupt('module interrupt')\n")
        try:
            load_tools_file(path)
        except ToolsFileError as err:
            self.assertIn("failed to import", str(err))
            self.assertIn("KeyboardInterrupt", str(err))
            self.assertIsInstance(err.__cause__, KeyboardInterrupt)
        except BaseException as err:  # noqa: BLE001 — the bug under test
            self.fail(
                f"load_tools_file let {type(err).__name__} escape "
                "unwrapped; the task runner would treat it as a task "
                "cancellation instead of a diagnostic task error",
            )
        else:
            self.fail("broken tools file must raise ToolsFileError")

    def test_system_exit_in_tools_file_raises_tools_file_error(self) -> None:
        path = self._write("raise SystemExit(3)\n")
        with self.assertRaisesRegex(ToolsFileError, "SystemExit"):
            load_tools_file(path)

    def test_plain_exception_in_tools_file_raises_tools_file_error(self) -> None:
        path = self._write("raise RuntimeError('boom')\n")
        with self.assertRaisesRegex(ToolsFileError, "RuntimeError: boom"):
            load_tools_file(path)

    def test_syntax_error_in_tools_file_raises_tools_file_error(self) -> None:
        path = self._write("def broken(:\n")
        with self.assertRaisesRegex(ToolsFileError, "SyntaxError"):
            load_tools_file(path)

    def test_missing_tools_file_raises_tools_file_error(self) -> None:
        with self.assertRaisesRegex(ToolsFileError, "not an existing"):
            load_tools_file(str(self.root / "nowhere.py"))

    def test_non_string_tools_file_field_raises_tools_file_error(self) -> None:
        with self.assertRaisesRegex(ToolsFileError, "path string"):
            load_tools_file(42)

    def test_empty_tools_file_field_yields_no_tools(self) -> None:
        self.assertEqual(load_tools_file(None), [])
        self.assertEqual(load_tools_file(""), [])

    def test_healthy_tools_file_still_loads(self) -> None:
        path = self._write(
            "def greet(name: str) -> str:\n"
            "    \"\"\"Say hi.\"\"\"\n"
            "    return f'hi {name}'\n"
            "\n"
            "def get_tools():\n"
            "    \"\"\"Return the tools.\"\"\"\n"
            "    return [greet]\n"
        )
        tools = load_tools_file(path)
        self.assertEqual([t.__name__ for t in tools], ["greet"])
        self.assertEqual(tools[0](name="bob"), "hi bob")

    def test_missing_get_tools_raises_tools_file_error(self) -> None:
        path = self._write(
            "def greet(name: str) -> str:\n"
            "    \"\"\"Say hi.\"\"\"\n"
            "    return f'hi {name}'\n"
        )
        with self.assertRaisesRegex(ToolsFileError, "get_tools"):
            load_tools_file(path)

    def test_raising_repr_in_get_tools_result_raises_tools_file_error(
        self,
    ) -> None:
        # Validating the returned entries must never run user code
        # (e.g. a raising ``__repr__``) unguarded: any escape from the
        # validation must surface as ToolsFileError, not as the raw
        # BaseException (which the task runner may misread as a
        # cancellation).
        path = self._write(
            "class _EvilRepr:\n"
            "    def __repr__(self):\n"
            "        raise KeyboardInterrupt('evil repr')\n"
            "\n"
            "def get_tools():\n"
            "    \"\"\"Return a broken entry.\"\"\"\n"
            "    return [_EvilRepr()]\n"
        )
        with self.assertRaisesRegex(ToolsFileError, "non-callable entry"):
            load_tools_file(path)

    def test_raising_exception_str_still_yields_tools_file_error(self) -> None:
        # Building the diagnostic itself must not run raising untrusted
        # code: an exception whose ``__str__`` raises (here a
        # KeyboardInterrupt) must still surface as ToolsFileError with
        # the type-name-only fallback message.
        path = self._write(
            "class _EvilStr(Exception):\n"
            "    def __str__(self):\n"
            "        raise KeyboardInterrupt('str bomb')\n"
            "\n"
            "raise _EvilStr()\n"
        )
        with self.assertRaisesRegex(ToolsFileError, "_EvilStr"):
            load_tools_file(path)

    def test_nul_byte_path_raises_tools_file_error(self) -> None:
        # ``Path.is_file`` raises ValueError on an embedded NUL byte;
        # the loader must report the standard diagnostic instead of
        # leaking the ValueError.
        with self.assertRaisesRegex(ToolsFileError, "not an existing"):
            load_tools_file("bad\x00tools.py")

    def test_raising_iter_in_get_tools_result_raises_tools_file_error(
        self,
    ) -> None:
        path = self._write(
            "class _EvilList(list):\n"
            "    def __iter__(self):\n"
            "        raise SystemExit(9)\n"
            "\n"
            "def ok() -> str:\n"
            "    \"\"\"Return ok.\"\"\"\n"
            "    return 'ok'\n"
            "\n"
            "def get_tools():\n"
            "    \"\"\"Return a list whose iteration raises.\"\"\"\n"
            "    return _EvilList([ok])\n"
        )
        with self.assertRaisesRegex(ToolsFileError, "SystemExit"):
            load_tools_file(path)


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
