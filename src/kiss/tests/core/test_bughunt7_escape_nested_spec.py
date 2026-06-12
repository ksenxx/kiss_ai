# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 7: nested-format-spec handling in escape_invalid_template_field_names.

The function's contract is that formatting the escaped template with the
valid fields succeeds and leaves invalid placeholders verbatim.  Two
violations existed for placeholders with *nested* format specs:

1. ``{a:{b}}`` with valid ``a`` but invalid ``b`` produced
   ``"{a:{{b}}}"`` — ``str.format`` does not unescape doubled braces
   inside a format spec, so ``.format(a=...)`` raised
   ``ValueError: Invalid format specifier`` (the very crash the
   function exists to prevent).
2. ``{a:{b}}`` with *both* fields invalid double-escaped the inner
   braces (sanitised once, then the whole-placeholder escape doubled
   them again), so the formatted output corrupted the user's text to
   ``"{a:{{b}}}"`` instead of round-tripping ``"{a:{b}}"``.
"""

from kiss.core.utils import escape_invalid_template_field_names


def test_valid_outer_invalid_nested_spec_field_formats_safely() -> None:
    """{a:{b}} with only ``a`` valid must survive .format(a=...)."""
    escaped = escape_invalid_template_field_names("{a:{b}}", {"a"})
    # Must not raise, and the unfillable placeholder stays verbatim.
    assert escaped.format(a=5) == "{a:{b}}"


def test_invalid_outer_with_nested_spec_round_trips() -> None:
    """{a:{b}} with no valid fields must round-trip verbatim."""
    escaped = escape_invalid_template_field_names("{a:{b}}", set())
    assert escaped.format() == "{a:{b}}"


def test_all_valid_nested_spec_still_substitutes() -> None:
    """{a:{b}} with both valid keeps full nested-spec substitution."""
    escaped = escape_invalid_template_field_names("{a:{b}}", {"a", "b"})
    assert escaped.format(a=5, b=3) == "  5"


def test_plain_fields_unchanged() -> None:
    """Non-nested behaviour is preserved."""
    escaped = escape_invalid_template_field_names("hi {bad} {good}", {"good"})
    assert escaped.format(good="G") == "hi {bad} G"


def test_simple_spec_with_valid_field_unchanged() -> None:
    """A literal (field-free) format spec on a valid field is kept."""
    escaped = escape_invalid_template_field_names("{a:>4}", {"a"})
    assert escaped.format(a=7) == "   7"


def test_conversion_with_invalid_field_round_trips() -> None:
    """Invalid field with conversion and spec stays verbatim."""
    escaped = escape_invalid_template_field_names("{bad!r:>10}", set())
    assert escaped.format() == "{bad!r:>10}"
