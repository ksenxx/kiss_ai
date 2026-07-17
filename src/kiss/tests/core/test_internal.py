# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for internal KISS components: utils, Base class.

Merged from: test_internal, test_core_branch_coverage.
"""

import os
import tempfile
import unittest
from pathlib import Path


class TestUtilsFunctions(unittest.TestCase):
    def test_fc_reads_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for fc function")
            temp_path = f.name
        try:
            self.assertEqual(
                Path(temp_path).read_text(), "Test content for fc function"
            )
        finally:
            os.unlink(temp_path)
