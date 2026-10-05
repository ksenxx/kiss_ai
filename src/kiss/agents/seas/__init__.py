# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bundled Sorcar Extension Agents (SEAs) that extend Sorcar itself.

Unlike :mod:`kiss.agents.third_party_agents`, which wrap external
services, the SEAs here drive Sorcar's own workflows — for example
:mod:`kiss.agents.seas.merge.merge_sea`, the merge-conflict resolver the
auto-commit worktree merge runs when a squash merge conflicts, or
:mod:`kiss.agents.seas.sh.sh_sea`, which runs the shell command in its
prompt with the ``Bash`` tool alone and returns the output, or
:mod:`kiss.agents.seas.autorouter.autorouter_sea`, which finishes a task at the lowest
cost per accepted result by routing each unit of work to the cheapest
model tier that passes its acceptance check, or
:mod:`kiss.agents.seas.bestrouter.bestrouter_sea`, which runs every task on
one fixed frontier model and has a second model review its work, or
:mod:`kiss.agents.seas.skillopt.skillopt_sea`, which optimizes the prompt text of
a skill or of another SEA against an eval set (SkillOpt), or
:mod:`kiss.agents.seas.write.write_sea`, which adds a writing protocol to
the system prompt so the prose it produces is concise, professional
American English for a general audience that reads as if a person wrote
it, or
:mod:`kiss.agents.seas.write_paper.write_paper_sea`, which writes or revises a
research paper under the rules of ``templates/write_paper_prompt.md``
with tools for the AI-slop gates and the LaTeX build, or
:mod:`kiss.agents.seas.review_paper.review_paper_sea`, which reviews a paper for any
venue with tools that read the PDF page by page and check the review's
structure and AI-slop gates, or
:mod:`kiss.agents.seas.revise_and_review_paper.revise_and_review_paper_sea`, which
runs the two as a loop (write, fresh review, revise with experiments or
ablations where the review asks for evidence) until the review says
strong accept or the paper cannot improve further, or
:mod:`kiss.agents.seas.git_extract_knowledge.git_extract_knowledge_sea`, which builds the
durable memory of a git repository (a full-text block store of every
file, chunk, symbol, commit, change, tag, branch, contributor and
directory plus curated pages in the repository's domain memory) and
schedules its daily refresh, or :mod:`kiss.agents.seas.remember.remember_sea`
and :mod:`kiss.agents.seas.forget.forget_sea`, which add the prompt to, or
remove it from, the standing instructions in ``~/.kiss/AGENTS.md``
(the file appended to every task's system prompt; storage in
:mod:`kiss.agents.seas.agents_md`), or :mod:`kiss.agents.seas.ask.ask_sea`,
which answers a question about a running task from a digest of its
persisted events (the ``/ask`` command; typed into a running task's tab
the server dispatches it as a side channel next to the agent), or
:mod:`kiss.agents.seas.coding.coding_sea`, the :class:`ContainerHarness` that runs
Sorcar unattended inside a Docker container (the trial runners in
``benchmarkings/harnesstax`` generate a per-trial SEA file that binds to it).
Every SEA here is a sub-package ``<name>/`` holding ``<name>_sea.py`` plus
its helper modules and data files (``sh/sh_sea.py`` and ``sh/evals/``,
``coding/coding_sea.py`` and ``coding/coding_test_context.py``, ...);
shared helpers such as :mod:`kiss.agents.seas.agents_md` stay at the
package top level.  :mod:`kiss.agents.sorcar.sea_commands` exposes every
such folder as the chat slash command ``/<name>`` (``/merge``, ``/sh``,
``/autorouter``, ``/bestrouter``, ``/skillopt``, ``/write``, ``/write_paper``, ``/review_paper``,
``/revise_and_review_paper``, ``/git_extract_knowledge``, ``/remember``, ``/forget``,
...); a script
placed directly in the package, outside its own folder, is not
registered.  Every SEA file
defines one subclass of :class:`kiss.agents.seas.base.base_sea.BaseSea`
whose ``description()`` returns one sentence on what it does and how
to use it, which ``/<name> help`` prints without running the SEA.  A
SEA whose ``register_as_model()`` returns ``True`` is also listed in
the model picker under its name (``autorouter``, ``bestrouter``):
picking it runs every task of the tab through the SEA, with the model
routing protocol its ``system_prompt`` method appends to the system
prompt.
"""
