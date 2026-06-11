# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 3: ``_coerce_tasks`` mishandles edge-case JSON list strings.

Two malformed-but-recoverable shapes from LLM tool calls were handled
wrongly:

* ``"[]"`` (a JSON empty list) fell through the ``and parsed`` guard and
  was wrapped as ``["[]"]`` — spawning one bogus sub-agent whose task is
  the literal string ``"[]"`` (a full LLM run wasted on garbage).
* ``"[1, 2]"`` (a JSON list with non-string elements) was treated as ONE
  task string ``"[1, 2]"`` instead of two tasks ``"1"`` and ``"2"``.
"""

from kiss.agents.sorcar.sorcar_agent import _coerce_tasks


def test_json_empty_list_returns_no_tasks() -> None:
    """'[]' means zero tasks, not one task whose text is '[]'."""
    assert _coerce_tasks("[]") == []


def test_json_list_with_non_string_elements_is_coerced() -> None:
    """JSON lists of non-strings become one task per element (str-coerced)."""
    assert _coerce_tasks("[1, 2]") == ["1", "2"]


def test_json_list_mixed_elements_is_coerced() -> None:
    assert _coerce_tasks('["a", 2]') == ["a", "2"]


def test_existing_shapes_still_work() -> None:
    """Regression guard: previously-supported shapes are unchanged."""
    assert _coerce_tasks("hello") == ["hello"]
    assert _coerce_tasks(["task A", "task B"]) == ["task A", "task B"]
    assert _coerce_tasks('["task A", "task B"]') == ["task A", "task B"]
    bad_json = '["task A", "task B"'
    assert _coerce_tasks(bad_json) == [bad_json]
