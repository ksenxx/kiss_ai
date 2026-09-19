# Prompt template: write a research paper that reads as if a human wrote it

Distilled from about 250 paper-writing tasks in `~/.kiss/sorcar.db` (April to September 2026:
kiss_sorcar.tex, ks_assistant.tex, se_kiss_sorcar.tex, swedefend.tex, hydra_kv.tex,
cleverest_plus.tex, plus letters, posts and rebuttal comments). Every rule below traces to a
task the user had to issue after a first draft came back wrong: fabricated author names,
numbers that differed between abstract and evaluation, bug-fix chronology where the design
should have been, marketing words, references to "the previous draft", and four rounds of
AI-slop review on one paper alone.

How to use: fill in the `<<placeholders>>`, delete the parts that do not apply, and paste the
whole block as one task. The follow-up prompts at the end are for the revision rounds; one
pass was never enough in the history.

---

## The prompt

You are William Strunk Jr. and E. B. White and a senior computer scientist. You write short
natural sentences.

Write a paper for <<VENUE, e.g. NeurIPS 2027 / ICSE 2027>> at `<<PATH/paper.tex>>` on
<<TOPIC, one sentence>>. Follow the venue's submission guidelines strictly: page limit
<<N>> pages until the references, style file <<neurips_2026.sty / IEEEtran conference>>,
<<double-blind: anonymous authors, repository link withheld, our own prior work cited in the
third person / single-author preprint with the acknowledgments copied from
./papers/kisssorcar/kiss_sorcar.tex>>, and the venue checklist if there is one. Use the same
house style as the papers in <<./papers/kisssorcar/ and ./papers/swedefend/>>.

Use '<<claude-fable-5>>' model for all tasks, including software development. Use
'<<gpt-5.6-sol>>' (not codex) for a thorough read-only review of the other model's work.
Use at most 20% of the task budget for the review. Use the model names literally without
hallucinating new model names. If switching your own model resets your context, run the
writer and the reviewer as sub-agents instead.

### Sources of truth

The paper describes <<the system / the study>> as it exists in `<<./projects/x/>>`. The
results are in `<<./projects/x/results/*.csv, ./jobs/<run>/result.json>>`. The development
history is in `<<~/.kiss/sorcar.db and git log>>`. Notes: `<<./projects/x/*.md>>`. Read the
code and the raw results before writing anything. Do not write a number from memory.

### What the paper must contain, in this order

1. Abstract, 150 to 250 words: the problem, the approach, the main result with its number,
   and what is not claimed. No per-run numbers.
2. Introduction: motivate the problem; the state of the art and what it cannot do; the
   contributions as a short list; a scope paragraph stating what we do not claim.
3. Background or problem setting: the workload, the threat model, the hardware, the harness,
   whatever the reader needs to interpret the numbers.
4. The system (or method) as it is today, in the present tense. Describe each mechanism and
   the reason it is designed that way. Do not narrate bugs found and fixed, audits, or how the
   system got better over time. Readers want to know how it works.
5. Evaluation: setup, baselines with exact versions and where their numbers come from,
   metrics, results tables, negative results, and threats to validity.
6. <<Optional: one section on how the artifact was developed with KISS Sorcar. Quote the
   user's prompts verbatim, typos included, in a promptbox. Improvement steps, rejected ideas,
   and reviewer findings live only in this section.>>
7. Related work: the well known work plus the closest recent work (after <<DATE>>). Read each
   paper before you cite it. Say in a sentence or two how each relates to ours; do not list.
8. Limitations, Conclusion, Acknowledgments. The conclusion restates the main result with its
   number and nothing that the body does not support.

### Facts and numbers

- Define every reported number once as a `\newcommand` macro at the top of the .tex and use
  the macro everywhere. Recompute each number from the raw results and put the command that
  recomputes it in a comment next to the macro.
- A number that appears in the abstract, introduction, evaluation, and conclusion is the same
  number from the same configuration. Two metrics reported together (catch rate and false
  positive rate, speed and accuracy) come from the same run.
- Baseline numbers come from the baseline's own paper or leaderboard; cite it and state the
  configuration. If we reran a baseline and got a different number, report both.
- Arithmetic in the text (ratios, percentages, sums of table rows) must reconcile; check it.
- Every claim stays inside what the experiments support. If an experiment was not run, say so
  in Limitations. Keep negative results and unmet goals; do not round up.
- Line counts, model counts, version numbers and similar facts are recomputed from the
  current code with the counting rule stated in the paper.

### Citations

- Verify each bib entry at a primary source (DBLP bibtex page, ACM DL, USENIX page, arXiv
  abstract page) before adding it: title, every author with the correct first name, venue,
  year, pages or DOI. Do not invent a venue for an arXiv-only paper. Do not cite a paper you
  could not open. Log each verification with its URL in `./tmp/information-<<paper>>.md`.
- Cite what the text names. If the text says "Bitcask-style", cite Bitcask.
- Every `\cite` key has a bib entry and every bib entry is cited.
- Prefer well known work; for recent work, prefer papers that are already cited by others.

### Style (Strunk and White)

- Active voice, first person plural: "We measure", not "It was measured".
- Short declarative sentences, one idea each. Vary sentence length. Some paragraphs are short.
- Omit needless words. No "remarkably", "dramatically", "particularly", "fundamentally",
  "essentially", "quite"; "significantly" only with a statistical test.
- No marketing: no "powerful", "seamless", "elegant", "unique", "novel" as praise,
  "state-of-the-art" about our own work, "compares favorably", "outperforms" without the
  number, "to our knowledge the first".
- Definite, specific, concrete language: numbers, names, file names.
- Say each thing once. The architecture list appears in full once; other places refer back to
  it. Do not restate a quoted prompt in the prose that follows it.
- Present tense for the system, past tense for what we did.
- Each paragraph is a single line in the .tex source, with a blank line between paragraphs.
- `\paragraph{}` labels are used sparingly, and sections do not all share one micro-structure
  (mechanism, why it is safe, punchy last sentence).
- No mention of earlier drafts, reviewers, rebuttals, audits of "this submission", or
  "republishing". The paper is a standalone document.
- British or American spelling, one of them, throughout the prose (verbatim code excluded).

### AI slop

The text must not read as machine written. Each item below is fine once; in quantity they
are the tells that got flagged in the history.

1. Em dashes: zero in prose (`---` or `—`). Use commas, parentheses, a colon, or a new
   sentence.
2. Antithesis: "not X but Y", "X, not Y", "rather than", "instead of". At most <<5>> in the
   whole paper, only where the contrast is technical.
3. Performative honesty: "we state it plainly", "honest", "honestly", "we say so", "worth
   stating", "to be clear", "we do not claim", "we name them rather than claim their
   absence". State the caveat and drop the commentary about stating it.
4. Aphoristic closers and zingers at the end of paragraphs ("This is the overfitting the
   loop exists to catch", "for agents and humans alike", "earned its keep"), chiasmus, "the
   lesson is", "worth copying".
5. Setup question then answer ("The design question is therefore: ...? Section 4 is the
   answer."), "The picture is this:", "Here is the deal".
6. Colon-pivot sentences ("The verdict was blunt: ..."), one-line verdict sentences after a
   long sentence ("The same arithmetic held."), cleft sentences ("X is what makes Y", "What
   made this work was").
7. Single-word rhetorical italics (`\emph{not}`, `\emph{every}`, `\emph{before}`). Italics
   only when introducing a term.
8. Bold-label lists ("\textbf{An artifact.} ... \textbf{A methodology.} ..."), italic
   aphorism openers in observation lists, rule-of-three lists, a repeated intro phrase such
   as "in full:" before every quoted prompt.
9. Slop vocabulary: delve, leverage, pivotal, crucial, testament, landscape, tapestry,
   showcase, underscore, intricate, meticulous, seamless, vibrant, realm, myriad, foster,
   comprehensive, ever-evolving, fast-paced, game-changer, harness (as a verb), notably,
   moreover, furthermore, "it is worth noting", "in conclusion", "serves as", "plays a
   crucial role", "aligns with", "aims to explore". Also "precisely", "exactly", "genuine"
   used as intensifiers.
10. Anthropomorphized agent drama: "the agent declined", "said so with arithmetic", "waits
    politely", "the audit's verdict".
11. Coined capitalized concept names used as if established. If you coin a term, define it
    once and write it in lower case afterwards.
12. A mega-abstract with per-run numbers; a paper with no figures; every paragraph the same
    dense rectangle; flawless prose with a rhythmic cadence.
13. Duplicated sentences anywhere in the paper.
14. Markdown or Unicode artifacts in LaTeX: curly quotes, Unicode ellipsis, `**bold**`.

### Figures and tables

At least one architecture figure and one results figure (TikZ, or matplotlib to PDF with the
script committed next to the paper). Every table caption states the configuration. Every
number in a table comes from a macro or a script. Nothing spills into the margin.

### Process

1. Read the sources. Write `./tmp/PLAN.md`: the section outline and the list of numbers to
   compute, each with the command that computes it.
2. Search the internet extensively for the state of the art and the related work. Visit at
   least <<20>> distinct sources and log them in `./tmp/information-<<paper>>.md`.
3. Write the paper.
4. Build: pdflatex, bibtex, pdflatex, pdflatex (TeX Live at
   `<</usr/local/texlive/2026/bin/universal-darwin>>`). Zero errors, zero undefined
   references or citations, no overfull box over 10 pt. Render the pages to PNG and look at
   them. Fix tables that spill, prompt boxes that split badly, orphan headings, and `??`.
5. Run the gates below and fix everything they flag.
6. Independent review: run the reviewer model strictly read-only. Ask it to check, with line
   numbers, (a) every number against the raw results and the code, (b) every citation against
   a primary source, (c) internal consistency, (d) overclaims, (e) the AI-slop list above.
   Fix each finding you confirm. If a finding needs a new experiment, state the limitation
   instead of papering over it.
7. Open the venue's website and check the paper against the current guidelines: page limit,
   anonymity, checklist, fonts, margins, LLM-usage disclosure.
8. Rebuild, rerun the gates, delete the `./tmp` files, and git add the .tex, .bib, figures,
   scripts, and .pdf. Do not add .aux, .bbl, .blg, .log, or .out files.

### Gates (run on the prose only: exclude comments, verbatim prompts, and table cells)

```sh
grep -c -- '---' paper.tex                                              # em dashes: 0
grep -c -i 'rather than\|instead of\|, not ' paper.tex                  # <= 5
grep -c -i -w 'honest\|honestly\|plainly\|genuine\|precisely\|exactly' paper.tex   # <= 3
grep -c -i 'delve\|leverage\|pivotal\|crucial\|testament\|landscape\|tapestry\|showcase\|underscore\|seamless\|meticulous\|notably\|moreover\|furthermore\|worth noting\|serves as' paper.tex   # 0
grep -o '\\emph{[A-Za-z]*}' paper.tex | sort | uniq -c                   # single-word italics: term introductions only
grep -c -i 'draft\|preliminary\|this submission\|reviewer\|rebuttal' paper.tex   # 0
grep -o '[^.]*\.' paper.tex | sed 's/^ *//' | sort | uniq -d             # duplicated sentences: none
perl -ne 'print "$.: $_" if /[^\x00-\x7F]/' paper.tex                   # non-ASCII: none unless intended
```

Also check: the abstract is at most 250 words; every number in the abstract appears in the
evaluation; every `\ref` has a `\label`; every `\cite` has a bib entry and every bib entry is
cited; no bib entry has an author whose first name you did not see on the source page.

### Report back

File paths, page count, the gate counts before and after, the list of citations verified
with their source URLs, each reviewer finding and what you did with it, and anything the
paper claims that you could not verify.

---

## Follow-up prompts for the revision rounds

The history shows roughly a hundred tasks per paper after the first draft. These are the ones
that recurred, in the wording that worked.

- "Can you review the paper at <<path>> for AI slop or other issues that make it look AI
  generated? Search the internet for current lists of AI-writing tells, run pattern counts on
  the source, and list the findings with line numbers. Do not edit."
- "Apply fixes 1 to <<N>> directly to <<paper.tex>>, then rebuild the PDF and re-run the
  quantitative pattern counts to confirm the tells dropped."
- "Tone down the paper by removing marketing and sales words and phrases."
- "Rewrite the paper by strictly following the Strunk and White style guide."
- "Instead of the passive voice, write it as 'we'."
- "Check the consistency and correctness of the paper against itself and the latest code.
  Fix the paper and build it."
- "Check all citations and bib entries for correctness and existence precisely." (This found
  13 entries with wrong author first names in one paper and a missing author in another.)
- "Find more closely related work, read the papers before citing them, and make sure the
  citations are not hallucinated."
- "Check for repeated text and explanations and remove them. Make the text less verbose."
- "There is no need to describe the bugs you found and fixed. Readers only want to know how
  the system works. Keep the improvement steps only in Section <<N>>."
- "Write the paper so that it does not refer to the previous draft."
- "Make each paragraph single-lined in the .tex. Do not change any content."
- "Build the paper and take screenshots to check and fix formatting."
- "Check the venue website carefully and see if the paper meets all submission requirements."
- "Write the abstract using the points emphasized in the introduction." and "Update the
  conclusion with the evaluation results."
- "Include the table from <<source>> in the paper and reference it."
- "Remove all date, time, and commit numbers from the paper."
- "You MUST always build papers that you have modified."
