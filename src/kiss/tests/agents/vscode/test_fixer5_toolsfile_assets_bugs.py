# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixer-5 tools-file / user-asset bugs (findings F5-07, F5-08).

F5-07 — ``load_tools_file`` treats the tools file as untrusted and
caught ``Exception`` and ``SystemExit``, but a file raising
``KeyboardInterrupt`` at import time escaped the loader; its
production caller sits inside an ``except KeyboardInterrupt`` branch
that cancels the whole agent task, so a broken tools file cancelled
the task instead of degrading to "no extra tools".

F5-08 — the user-asset seeder issued a single ``os.write`` and
ignored its return count; the buffered file-object path now
guarantees the whole default is written before the hard link
publishes the file.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kiss.server.tools_file import load_tools_file
from kiss.server.user_assets import ensure_user_asset_from_default


class TestToolsFileNonExceptionEscapes(unittest.TestCase):
    """F5-07: no BaseException from a tools file escapes the loader."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, body: str) -> str:
        path = self.root / "tools.py"
        path.write_text(body)
        return str(path)

    def test_keyboard_interrupt_in_tools_file_yields_no_tools(self) -> None:
        path = self._write("raise KeyboardInterrupt('module interrupt')\n")
        try:
            tools = load_tools_file(path)
        except BaseException as err:  # noqa: BLE001 — the bug under test
            self.fail(
                f"load_tools_file let {type(err).__name__} escape; the "
                "task runner would treat it as a task cancellation",
            )
        self.assertEqual(tools, [])

    def test_system_exit_in_tools_file_yields_no_tools(self) -> None:
        path = self._write("raise SystemExit(3)\n")
        self.assertEqual(load_tools_file(path), [])

    def test_plain_exception_in_tools_file_yields_no_tools(self) -> None:
        path = self._write("raise RuntimeError('boom')\n")
        self.assertEqual(load_tools_file(path), [])

    def test_healthy_tools_file_still_loads(self) -> None:
        path = self._write(
            "def greet(name: str) -> str:\n"
            "    \"\"\"Say hi.\"\"\"\n"
            "    return f'hi {name}'\n"
        )
        tools = load_tools_file(path)
        self.assertEqual([t.__name__ for t in tools], ["greet"])
        self.assertEqual(tools[0](name="bob"), "hi bob")


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
