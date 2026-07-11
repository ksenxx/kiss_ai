# Cleverest+ consistency review

## Scope

This review checked the manuscript against:

- `cleverest_plus/cleverest_plus/core.py` and `tests/test_core.py`;
- `sorcar_reported_frauds/evidence/analysis_summary.json` and `ablation_summary.csv`;
- the deterministic analysis script that produced both paper figures;
- the released-configuration definitions retained in that script.

## Material corrections

1. **Result identity:** Replaced the claim that 15/36 BIC and 24/36 BFC were measured for an integrated Cleverest+ portfolio. These numbers are the post-hoc union of all released LLM configurations and use more than one default budget. They are now labeled a retrospective upper bound throughout.
2. **Budget accounting:** Removed the ceiling allocation that assigned 12 trials when `T=10` and there were three configurations. The proposed allocation is now `4/3/3`, with an explicit distinction between trial, call, token, time, and dollar budgets.
3. **Baseline configuration:** Corrected the released GPT-4o default temperature from 0 to 0.5.
4. **Ablation semantics:** Replaced the additive-component table with the actual released, standalone configurations. The 10-iteration row is now identified as higher-compute rather than budget-matched.
5. **Ordinal results:** Removed the invented portfolio means (which had copied the DeepSeek-R1 means). Means are now described as averages of ordered codes, and no mean is assigned to the union.
6. **Statistics:** Tied Wilcoxon values only to DeepSeek-R1, marked them exploratory and unadjusted, and added paired binary gain/loss counts and exact p-values.
7. **Implementation status:** Distinguished implemented components from the unimplemented Cleverest adapter, portfolio dispatcher, dedicated sanitizer channel, and end-to-end campaign.
8. **Parser accuracy:** Corrected “exactly three keys” and “canonical base64” to match the code: two required fields, one optional field, no unknown fields, and validated base64 syntax.
9. **Oracle accuracy:** Removed the assertion that stderr-only parsing is authenticated or cannot be forged. The paper now says stdout markers are ignored, ordinary target stderr remains writable, and expected signatures are optional.
10. **Artifact claims:** Removed claims of integration tests and bundled per-issue outcomes. The available package has 13 passing pytest cases; the retained machine-readable data are aggregate summaries.
11. **Unsupported performance claims:** Removed claims that duplicate suppression made successful runs faster and that the new system was a drop-in successor.
12. **Prose:** Removed promotional filler, repeated “mutually reinforcing” language, causal claims unsupported by the data, and contradictory uses of “doubling.”

## Validation performed

- `cd cleverest_plus && uv run --project . --with pytest pytest -q` — 13 passed.
- `cd cleverest_plus && uv run --project . --with ruff ruff check cleverest_plus tests` — passed.
- Citation keys were checked against the bibliography; no missing or unused keys remain.
- Both PNGs were checked byte-for-byte against outputs from the retained analysis.
- The paper was compiled through BibTeX and three LaTeX passes; references and citations resolve.
