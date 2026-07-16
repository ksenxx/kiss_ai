# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking down behavior before simplification.

Second simplification pass over the sorcar-agents area.  Covers the
exact code paths touched:

* ``format_skill_listing`` / ``format_command_listing`` description
  truncation at 100 characters (``desc[:97] + "..."``), ``(source)``
  prefixes, alignment, and empty-collection hints — the duplicated
  truncation is hoisted into ``skills.truncate_listing_description``.
* ``ChatSorcarAgent.run()``'s inline result-summary extraction (nested
  try/except handling YAML dict/str/list/None/unparseable shapes) —
  extracted as the named module-level function
  ``chat_sorcar_agent._extract_result_summary``.

No mocks, patches, or fakes: every test drives the real functions with
real data.
"""

from __future__ import annotations

import yaml

from kiss.agents.sorcar.custom_commands import CustomCommand, format_command_listing
from kiss.agents.sorcar.skills import Skill, format_skill_listing


def _skill(name: str, description: str, source: str = "user") -> Skill:
    return Skill(
        name=name,
        description=description,
        path=f"/tmp/{name}/SKILL.md",
        source=source,
    )


def _command(
    name: str,
    description: str,
    source: str = "user",
    argument_hint: str = "",
    path: str = "",
) -> CustomCommand:
    return CustomCommand(
        name=name,
        description=description,
        argument_hint=argument_hint,
        template="body",
        source=source,
        path=path or f"/tmp/{name}.md",
    )


class TestSkillListingTruncation:
    """Lock down ``format_skill_listing`` truncation and layout."""

    def test_long_description_truncated_to_100_chars(self) -> None:
        desc = "x" * 150
        listing = format_skill_listing({"a": _skill("a", desc)})
        assert "x" * 97 + "..." in listing
        assert "x" * 98 not in listing

    def test_exactly_100_chars_not_truncated(self) -> None:
        desc = "y" * 100
        listing = format_skill_listing({"a": _skill("a", desc)})
        assert desc in listing
        assert "..." not in listing

    def test_101_chars_truncated(self) -> None:
        desc = "z" * 101
        listing = format_skill_listing({"a": _skill("a", desc)})
        assert "z" * 97 + "..." in listing

    def test_source_prefix_and_alignment(self) -> None:
        skills = {
            "short": _skill("short", "one", source="bundled"),
            "muchlongername": _skill("muchlongername", "two", source="project"),
        }
        listing = format_skill_listing(skills)
        lines = listing.split("\n")
        assert lines[0] == "  muchlongername  (project) two"
        assert lines[1] == "  short           (bundled) one"

    def test_empty_skills_hint(self) -> None:
        listing = format_skill_listing({})
        assert listing.startswith("No skills found.")
        assert ".kiss/skills" in listing


class TestCommandListingTruncation:
    """Lock down ``format_command_listing`` truncation and layout."""

    def test_long_description_truncated_to_100_chars(self) -> None:
        desc = "x" * 150
        listing = format_command_listing({"a": _command("a", desc)})
        assert "x" * 97 + "..." in listing
        assert "x" * 98 not in listing

    def test_exactly_100_chars_not_truncated(self) -> None:
        desc = "y" * 100
        listing = format_command_listing({"a": _command("a", desc)})
        assert desc in listing
        assert "..." not in listing

    def test_empty_description_falls_back_to_filename(self) -> None:
        listing = format_command_listing(
            {"a": _command("a", "", path="/tmp/dir/mycmd.md")}
        )
        assert "mycmd.md" in listing

    def test_argument_hint_and_source(self) -> None:
        listing = format_command_listing(
            {"fix": _command("fix", "Fix a bug", argument_hint="[issue]")}
        )
        assert "  /fix [issue]  (user) Fix a bug" == listing

    def test_empty_commands_hint(self) -> None:
        listing = format_command_listing({})
        assert listing.startswith("No custom commands found.")
        assert ".kiss/commands" in listing


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
        raw = "word " * 300  # non-dict YAML, 1500 chars
        assert self._extract(raw) == raw[:500]

    def test_empty_string_returns_empty(self) -> None:
        assert self._extract("") == ""

    def test_dict_with_numeric_summary_dumped(self) -> None:
        # Non-str, non-None scalar summary goes through safe_dump.
        expected = yaml.safe_dump(42, sort_keys=False).strip()
        assert self._extract("summary: 42") == expected


class TestTruncateListingDescription:
    """Direct tests of the shared truncation helper."""

    def _truncate(self, desc: str) -> str:
        from kiss.agents.sorcar.skills import truncate_listing_description

        return truncate_listing_description(desc)

    def test_short_unchanged(self) -> None:
        assert self._truncate("hi") == "hi"

    def test_exactly_100_unchanged(self) -> None:
        assert self._truncate("a" * 100) == "a" * 100

    def test_101_truncated_to_100(self) -> None:
        out = self._truncate("a" * 101)
        assert out == "a" * 97 + "..."
        assert len(out) == 100

    def test_empty_unchanged(self) -> None:
        assert self._truncate("") == ""
