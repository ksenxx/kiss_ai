# Breaking the Guardians verifier

Concrete counterexamples to the soundness claims of Erik Meijer's
"generate → verify → execute" pipeline for AI-agent workflows:

* Erik Meijer, *Guardians of the Agents: Formal Verification of AI
  Workflows*, ACM Queue **23**(4), Sep 2025.
  <https://queue.acm.org/detail.cfm?id=3762990>
* Erik Meijer, *In Code They Think; In Proof We Trust*, ACM Queue
  **24**(2), Jun 2026.
  <https://queue.acm.org/detail.cfm?id=3806226>

The reference implementation targeted here is the Universalis-style
Python library published by Nada Amin at
<https://github.com/metareflection/guardians> (MIT-licensed, revision
current as of mid-2026).  The `guardians/` sub-directory of this
project is a clean clone of that repo, kept unmodified so results can
be reproduced against the authors' own code.

## What each attack breaks

All attacks start from *the paper's own canonical email-agent setup*
(`examples/email_agent.py`; the tool specs, automaton, and base taint rule
are reproduced in `attacks/common.py`).  Attacks 1, 2, and 4 use that
policy unchanged; Attack 3 deliberately adds `summarize_emails` to the
rule's sanitizer list to test the framework's sanitizer trust model.  In
every case:

* `verify(workflow, policy, registry).ok` is `True`.
* The workflow executes to completion.
* The sensitive mail contents leave via the mail sink at runtime.

| # | Attack file                                  | Verifier weakness                                                                                     |
|---|----------------------------------------------|--------------------------------------------------------------------------------------------------------|
| 1 | `attacks/attack_subject_exfil.py`            | `TaintRule.sink_param="body"` ignores `subject`; `is_taint_sink` is not set on `subject`.              |
| 2 | `attacks/attack_type_confusion.py`           | Passing `to=[]` skips the Z3 precondition (only a warning) and lets `[] not in […]` evaluate to False. |
| 3 | `attacks/attack_sanitizer_confusion.py`      | `TaintRule.sanitizers` is a name-based allow-list; no proof required that the tool actually sanitizes. |
| 4 | `attacks/attack_multi_recipient_domain.py`   | `domain_of(to)` is a single suffix/last-`@` check, so `attacker@evil.com, friend@company.com` passes.  |

Each attack is a first-class module with a `build(scenario)` entry
point.  A test in `tests/` executes the attack end-to-end and
asserts that (a) the verifier accepts and (b) an eye-catching secret
marker still appears in the runtime exfiltration log.

## Running

```bash
# From the repo root:
cd meijer_break/guardians
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install pytest
cd ..

python demo.py                # narrative walkthrough
pytest -v tests               # 4 tests, one per attack
```

## Where the paper's soundness argument fails

The Amin / Meijer pipeline advertises:

1. **Taint analysis** with may-information over-approximation.
2. **Security automata** whose guards are evaluated statically.
3. **Z3-checked pre/postconditions**.

The attacks target the *joints* between these layers:

* Attack 1 exposes the *policy surface* joint: taint rules operate on
  policy-declared sink params, but sinks have *implicit* side-channels
  (subject, cc, bcc, headers) that no `is_taint_sink` flag defends
  unless the policy author enumerates them by hand.
* Attack 2 exposes the *abstraction joint* between Z3 (typed) and
  `safe_eval` (untyped Python): a value that Z3 refuses to reason about
  is downgraded to a warning, while the runtime evaluates the same
  expression happily.
* Attack 3 exposes the *label-vs-semantics joint*: sanitization is
  advertised as a proven property but implemented as an unchecked
  string-based label.
* Attack 4 exposes the *parser-model joint*: the verifier proves a
  property of a simplified last-`@`/suffix abstraction of an email
  address field, while real address fields can contain recipient lists.

Each is a fundamental gap: none can be closed by "tighten the policy
in this one place".  They all require *the framework itself* to grow
new invariants (mandatory `subject`/header sink coverage, an unambiguous
runtime type discipline, proof-carrying sanitizers, and recipient-list
parsing that checks every mailbox rather than a single suffix).
