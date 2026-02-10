# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Integration tests for AgentEvolver Edit and MultiEdit tools.

This module tests that the Edit and MultiEdit tools:
1. Are correctly invoked by the KISSCodingAgent
2. Properly modify file contents
3. Respect writable path restrictions
4. Handle various edge cases
"""

import os
import shutil
import tempfile
import unittest

from kiss.core.useful_tools import UsefulTools


class TestEditToolEdgeCases(unittest.TestCase):
    """Edge case tests for Edit and MultiEdit tools."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.writable_dir = os.path.join(self.test_dir, "writable")
        os.makedirs(self.writable_dir)

        self.tools = UsefulTools(
            base_dir=self.test_dir,
            readable_paths=[self.test_dir],
            writable_paths=[self.writable_dir],
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_multiedit_replaces_overlapping_patterns(self) -> None:
        """Test MultiEdit with patterns that could overlap."""
        test_file = os.path.join(self.writable_dir, "test.py")
        original_content = "aaa"
        with open(test_file, "w") as f:
            f.write(original_content)

        result = self.tools.MultiEdit(
            file_path=test_file,
            old_string="a",
            new_string="bb",
            replace_all=True,
        )

        self.assertIn("Successfully replaced", result)

        with open(test_file) as f:
            content = f.read()

        # Each 'a' should be replaced with 'bb'
        self.assertEqual(content, "bbbbbb")


if __name__ == "__main__":
    unittest.main()
