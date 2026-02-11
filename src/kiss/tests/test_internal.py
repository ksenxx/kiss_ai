"""Test suite for internal KISS components that don't require API calls."""

import os
import tempfile
import unittest

from kiss.core.utils import (
    add_prefix_to_each_line,
    fc,
)


class TestUtilsFunctions(unittest.TestCase):
    def test_add_prefix_to_each_line(self):
        self.assertEqual(
            add_prefix_to_each_line("line1\nline2\nline3", "> "),
            "> line1\n> line2\n> line3",
        )
        self.assertEqual(add_prefix_to_each_line("single line", ">> "), ">> single line")

    def test_fc_reads_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for fc function")
            temp_path = f.name
        try:
            self.assertEqual(fc(temp_path), "Test content for fc function")
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
