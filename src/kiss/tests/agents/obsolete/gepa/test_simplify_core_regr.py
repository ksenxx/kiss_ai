# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end regression tests for the obsolete-gepa template utilities.

These five ``escape_invalid_template_field_names`` methods were split out
of ``kiss.tests.core.test_simplify_core_regr`` (whose remaining tests are
pure ``kiss.core``) because they depend on
``kiss.agents.obsolete.gepa.template_utils``, which lives outside
``src/kiss/core/``.
"""

import unittest

from kiss.agents.obsolete.gepa.template_utils import escape_invalid_template_field_names


class UtilsRegression(unittest.TestCase):
    def test_escape_keeps_valid_and_escapes_invalid(self) -> None:
        out = escape_invalid_template_field_names("hi {bad} {good}", {"good"})
        self.assertEqual(out.format(good="G"), "hi {bad} G")

    def test_escape_nested_spec_invalid(self) -> None:
        out = escape_invalid_template_field_names("{a:{b}}", {"a"})
        self.assertEqual(out.format(), "{a:{b}}")

    def test_escape_nested_spec_all_valid(self) -> None:
        out = escape_invalid_template_field_names("{a:{b}}", {"a", "b"})
        self.assertEqual(out.format(a=3, b=5), "    3")

    def test_escape_conversion_preserved(self) -> None:
        out = escape_invalid_template_field_names("{good!r}", {"good"})
        self.assertEqual(out.format(good="x"), "'x'")

    def test_escape_doubles_literal_braces(self) -> None:
        out = escape_invalid_template_field_names("a {{lit}} {good}", {"good"})
        self.assertEqual(out.format(good="G"), "a {lit} G")


if __name__ == "__main__":
    unittest.main()
