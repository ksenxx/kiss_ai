# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking down behavior before simplification.

Second simplification pass over the sorcar-agents area.  Covers the
exact code paths touched:

* ``ChatSorcarAgent.run()``'s inline result-summary extraction (nested
  try/except handling YAML dict/str/list/None/unparseable shapes) —
  extracted as the named module-level function
  ``chat_sorcar_agent._extract_result_summary``.

No mocks, patches, or fakes: every test drives the real functions with
real data.
"""

from __future__ import annotations

import yaml


class TestExtractResultSummary:
    """Lock down the result-summary extraction semantics of ``run()``.

    Encodes the exact behavior of the (previously inline) block in
    ``ChatSorcarAgent.run()`` around ``yaml.safe_load(result)``.
    """

    def _extract(self, result: str) -> str:
        from kiss.agents.sorcar.chat_sorcar_agent import _extract_result_summary

        return _extract_result_summary(result)

    def test_dict_with_string_summary(self) -> None:
        result = yaml.safe_dump({"success": True, "summary": "All done"})
        assert self._extract(result) == "All done"

    def test_dict_with_none_summary(self) -> None:
        assert self._extract("summary:\nsuccess: true") == ""

    def test_dict_without_summary_key(self) -> None:
        assert self._extract("success: true") == ""

    def test_dict_with_list_summary_dumped(self) -> None:
        result = yaml.safe_dump({"summary": ["step one", "step two"]})
        expected = yaml.safe_dump(["step one", "step two"], sort_keys=False).strip()
        assert self._extract(result) == expected

    def test_dict_with_mapping_summary_dumped_unsorted(self) -> None:
        result = "summary:\n  zebra: 1\n  apple: 2\n"
        expected = yaml.safe_dump({"zebra": 1, "apple": 2}, sort_keys=False).strip()
        assert self._extract(result) == expected
        assert self._extract(result).startswith("zebra")

    def test_non_dict_yaml_returns_raw_prefix(self) -> None:
        assert self._extract("just a plain sentence") == "just a plain sentence"
        assert self._extract("- a\n- b") == "- a\n- b"

    def test_unparseable_yaml_returns_raw_prefix(self) -> None:
        raw = "key: [unclosed"
        assert self._extract(raw) == raw

    def test_raw_prefix_capped_at_500_chars(self) -> None:
        raw = "word " * 300
        assert self._extract(raw) == raw[:500]

    def test_empty_string_returns_empty(self) -> None:
        assert self._extract("") == ""

    def test_dict_with_numeric_summary_dumped(self) -> None:
        expected = yaml.safe_dump(42, sort_keys=False).strip()
        assert self._extract("summary: 42") == expected
