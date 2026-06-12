# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 7: argument re-injection in KISSAgent prompt substitution.

``KISSAgent._set_prompt`` substituted template arguments with one
sequential ``str.replace`` per key, so a placeholder-looking token
inside an earlier argument's VALUE (e.g. a ``task`` string that
literally contains ``{result}`` — common when tasks quote code or
template text) was re-expanded by a later key's replacement pass.  The
substituted prompt then leaked the other argument's value into the
middle of the first one, and the outcome depended on dict insertion
order.  Real-world trigger: ``generate_followup_text`` passes
``{"task": ..., "result": ...}`` where both values are arbitrary
user/LLM text.

The fix substitutes all ``{key}`` tokens in a single pass over the
template, so replacement values are never rescanned.
"""

from kiss.core.kiss_agent import KISSAgent

TEST_MODEL = "gemini-3-flash-preview"


def _substituted_prompt(template: str, arguments: dict[str, str]) -> str:
    """Build a real agent, set its prompt, and return the user message."""
    agent = KISSAgent("bughunt7-prompt")
    agent._reset(TEST_MODEL, False, None, None, None, verbose=False)
    agent._set_prompt(template, arguments)
    content: str = agent.messages[0]["content"]
    return content


def test_value_containing_other_placeholder_is_not_reexpanded() -> None:
    """A value holding the literal ``{result}`` must stay verbatim."""
    out = _substituted_prompt(
        "Task: {task}\nResult: {result}",
        {"task": "fix the {result} placeholder bug", "result": "SECRET"},
    )
    assert out == "Task: fix the {result} placeholder bug\nResult: SECRET"


def test_substitution_is_insertion_order_independent() -> None:
    """Reversed argument order must produce the identical prompt."""
    template = "A: {a}\nB: {b}"
    args_fwd = {"a": "uses {b} literally", "b": "BEE"}
    args_rev = {"b": "BEE", "a": "uses {b} literally"}
    assert (
        _substituted_prompt(template, args_fwd)
        == _substituted_prompt(template, args_rev)
        == "A: uses {b} literally\nB: BEE"
    )


def test_value_containing_own_placeholder_is_not_recursed() -> None:
    """A value holding its own ``{key}`` token must not recurse."""
    out = _substituted_prompt("X: {x}", {"x": "self {x} ref"})
    assert out == "X: self {x} ref"


def test_plain_substitution_and_unknown_placeholders_unchanged() -> None:
    """Normal substitution still works; unknown placeholders survive."""
    out = _substituted_prompt(
        "Hello {name}, keep {unknown}.", {"name": "World"}
    )
    assert out == "Hello World, keep {unknown}."


def test_repeated_placeholder_substituted_everywhere() -> None:
    """Every occurrence of a key is substituted."""
    out = _substituted_prompt("{x} and {x}", {"x": "V"})
    assert out == "V and V"
