# PROGRESS — Critique of Vislesy blog post "We Forgot to Teach AI Agents to Be Wrong on Purpose"

## Task
Critique https://vislesy.com/2026/06/18/we-forgot-to-teach-ai-agents-to-be-wrong-on-purpose/
with fact check, no AI slop. Build (claude-fable-5) → review/debug (gpt-5.6-sol).
Deliver a self-contained HTML report with diagrams/illustrations in ./reports and open it. DONE.

## Final outcome
Deliverable: reports/vislesy-critique.html (~52.7 KB, self-contained: inline CSS + 5 inline SVG
diagrams, no external assets; external URLs are citation links only). Opened in default browser
via `open`. Git-added.

## Chronological steps
1. Read SORCAR.md; confirmed prior draft report existed at reports/vislesy-critique.html.
2. Switched to gpt-5.6-sol for an adversarial review of the existing draft + independent fact re-check.
3. Re-fetched the blog and extracted every source link (found the actual Lakatos link target =
   Proofs_and_Refutations; Biggs&Wilson link = philpapers BIGTIO-4; bioRxiv 2026.01.05.697809;
   Nature s41586-026-10644-y; arXiv 2502.14297; AAAI-1983 inheritance paper).
4. Verified primary sources directly:
   - Biggs & Wilson chapter (author PDF): confirmed year 2025 AND discovered the essay MISREPRESENTS
     their deduction argument — essay uses reductio (premise defeat), but B&W's criterion requires a
     conclusion defeated with premises UNDEFEATED (their Dina example). This is a 3rd substantive error.
   - Sakana arXiv 2502.14297 PDF: both quoted sentences verbatim.
   - bioRxiv preprint (Crossref abstract + Google full-text): "Every framework produced sophisticated
     hallucinations"; 8 open-source frameworks, 2 tasks — VERIFIED, with preprint/scope caveat.
   - Nature Co-Scientist: confirmed generate/reflect/rank/evolve choreography BUT it has explicit
     hypothesis state + deep-verification reviews → essay's "discarded all bookkeeping" is too strong.
   - AAAI-1983 paper is Etherington & Reiter → contradicts essay's "oldest is 1987".
5. gpt-5.6-sol review notes (20 items) written to tmp; key fixes: add bioRxiv + Nature rows, promote
   the B&W misrepresentation to a 3rd error, fix SVG wiring (platypus is-a arrows now point up;
   Pollock undercutter uses gold marker; Lakatos progressive/degenerating redefined via novel
   prediction not anomaly depth; positive heuristic uses purple marker), correct AI-slop section
   (typos do NOT prove authorship), add a real linked Sources section, responsive table wrapper,
   focus-visible + reduced-motion.
6. Switched back to claude-fable-5 and rewrote reports/vislesy-critique.html incorporating all fixes.
7. Validated structure (balanced tags, no dup ids, no missing anchors), rendered in browser + screenshot,
   opened in default browser, git-added, deleted all tmp scratch files.

## Verdict delivered in report
Strong, useful thesis; 3 substantive errors (2004→2025 date; Lakatos link→wrong book; misdrawn B&W
deduction argument); several imprecisions; no generic AI-slop signature but authorship not
determinable from text.
