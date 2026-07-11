# Author: Koushik Sen (ksen@berkeley.edu)

"""Prompt templates for the DeepScholar-Bench Sorcar harness.

DeepScholar-Bench evaluates the ``Related Works`` section that a
system produces for a target arXiv paper.  We use the
``openai_deepresearch`` parser mode from the upstream evaluator, which
expects a single markdown file per task with inline citations in the
``[number](https://arxiv.org/abs/<arxiv_id>)`` form.  Numeric link labels are
required by the evaluator's claim- and citation-level metrics.  The prompt below
directs Sorcar to search the web, ground its claims in real arXiv
papers, and write exactly one output file with citations in that
format.
"""

from __future__ import annotations

SORCAR_TASK_TEMPLATE = """\
You are writing the Related Works section for an academic paper. \
Your response MUST be saved as a single markdown file at exactly \
this absolute path: {output_md_path}

STRICT REQUIREMENTS:
1. Output must be a well-structured Related Works section covering \
prior work on the topic of the paper whose abstract is provided \
below.
2. Only cite papers that are on arXiv AND were published on or before \
{cutoff_date}.
3. Every citation MUST be a numbered inline markdown link of the exact \
form [<number>](https://arxiv.org/abs/<arxiv_id>) — for example \
[1](https://arxiv.org/abs/1706.03762). Number distinct sources consecutively \
from 1 and reuse the same number when citing a source again. Use only this \
citation format. Do NOT put author names or titles inside the square brackets. \
Do NOT invent arXiv IDs; verify each ID via web search on arxiv.org and confirm \
the paper's title matches.
4. Group related papers thematically. Aim for ~400-600 words. Include \
at least 8 distinct arXiv citations (more is better).
5. Do not include content unrelated to the paper's topic. Do not fabricate \
citations or reference IDs. If you cannot verify an arXiv paper's ID, \
drop that citation rather than guess.
6. Do not include a "References" section — inline markdown links are \
the only reference format expected by the grader.

PAPER ABSTRACT:
{abstract}

When you have finished, write the complete Related Works section to \
{output_md_path} and exit. Do not print the result to stdout; only the \
saved file will be graded.
"""


def build_task(output_md_path: str, abstract: str, cutoff_date: str) -> str:
    """Return the Sorcar task string for a single DeepScholar-Bench query.

    Args:
        output_md_path: Absolute path where Sorcar must write the
            generated markdown.
        abstract: Abstract of the target paper (used to seed the topic).
        cutoff_date: Latest allowed publication date for cited papers
            (ISO-8601 date string).

    Returns:
        A fully-formatted task prompt ready to be passed to
        ``sorcar -t``.
    """
    return SORCAR_TASK_TEMPLATE.format(
        output_md_path=output_md_path,
        cutoff_date=cutoff_date,
        abstract=abstract.strip(),
    )
