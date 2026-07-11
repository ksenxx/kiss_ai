# Reproduction record

## Frozen inputs

- Paper: `tmp/fse26-llmtesting.pdf`, SHA-256 `b3f45602cd2a5a2fddcbc85f779b8d56ed606e0c9ce6a4c1d1b38e369961560e`.
- Artifact: `niMgnoeSeeL/cleverest` commit `bac4344ab3813a138608e86da8f1103dfe11ff62` (same commit as canonical `chinggg/Cleverest` tag `artifact-v1.1`).
- `repdata.tar.xz`: SHA-256 `6355a4a18659944f6226658729a9e7cdd0cfc3e53ae140036054cf457c498933`.
- Released aggregate CSV: SHA-256 `3780d09f479fff0d730f4a3a6c77c296be98d5e32807ef9208ea520464ecec37`.
- WAFLGo merged CSV: SHA-256 `49c428c6456d3e798a5aec02e1273105dc572d90a374a1c968a65168d969d476`.
- ClevFuzz merged CSV: SHA-256 `4f3885f8a9e73151e2224857b5f2ae76319c86e185fc4f306a2732d85c204119`.

## What was rerun

1. Extracted the replication archive: 123,995 tar members (122,486 regular files and 1,509 directories), no unsafe paths/links.
2. Ran the released `figure/gencsv.py` on the 1,500 subject-level `SUMMARY_*` files. It emitted 6,760 rows. The output was **byte-identical** to `figure/aggregated_results.csv` (same SHA-256 above).
3. Executed every cell of `figure/result.ipynb` using Jupyter, pandas, NumPy, seaborn, and matplotlib. It completed with exit 0 and reproduced Tables 2, 3, and 5; executed notebook SHA-256 `a3c5c1d48c2b42483fd7eac2fb1d900fa9e8e3b85be7a927ed35c54b9f7524cd`.
4. Independently reaggregated the CSV with `evidence/analyze_results.py` (not the authors' notebook), including issue-bootstrap intervals and paired exploratory tests.
5. Ran 13 tests, Ruff, and BasedPyright against the hardened `cleverest_plus` prototype; all passed.
6. Ran two harmless local vulnerability POCs; both reproduced.

## Independently reproduced default results

| Scenario | Rows | Mean ordinal code | Issues reached ≥1/10 | changed ≥1/10 | bugs ≥1/10 | successful trials | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| BIC / bug finding | 360 | 1.3667 | 31/36 | 20/36 | 7/36 | 60/360 | 79.253 |
| BFC / reproduction | 360 | 1.2694 | 25/36 | 17/36 | 13/36 | 107/360 | 52.567 |

These match Table 2. Note that averaging codes 0/1/2/3 is descriptively reproducible but assumes an interval scale the paper itself calls ordinal.

## Reproduced Table 5 computation

On the 50 commits where WAFLGo ran, the notebook gives:

- WAFLGo: 10/26 BIC and 8/24 BFC issues with ≥1 success;
- Cleverest: 7/26 and 11/24;
- Cleverest+ClevFuzz union: 20/26 and 18/24;
- mean WAFLGo times: 19:39:50 and 21:48:32;
- mean combined ClevFuzz times: 05:42:01 and 06:11:24.

The notebook explicitly overwrites every `Init=B` WAFLGo case with `success=10` **and** `time=24*3600`. Treating an already-crashing initial seed as time zero instead would reduce the reported WAFLGo means by about 4.62 h (BIC, five cases/26) and 5 h (BFC, five cases/24), to roughly 15.0 h and 16.8 h. The qualitative speed ordering remains, but the quoted factor is sensitive to this convention.

## Data-dependent aggregation transformations

Across all 6,760 rows, `gencsv.py`/its raw inputs cause these post-run treatments (counts independently recovered):

- 116 raw final `R` rows containing a `behave` status become score 2 (`D`);
- 84 `X` rows with different coarse crash strings become score 2;
- 4 `X` rows at two commits receive manually inspected reach score 1;
- one Poppler #1305 message-only `D` is set to `N` because it could not be reproduced.

They are commented in source, so they are not hidden from an artifact reader, but the paper does not disclose their frequency or a preregistered adjudication rule. They affect ordinal-effectiveness analyses; the final case is explicit post-hoc correction.

## What was not rerun, and why

- We did **not** repeat 1,500 live LLM experiments: no OpenAI/NVIDIA credentials were supplied; outputs are nondeterministic; the DeepSeek endpoint is unversioned; doing so would not bit-reproduce released generations.
- We did **not** repeat 24-hour fuzzing campaigns or rebuild eight historical projects: the local Docker daemon was unavailable. We did execute the shipped table-generation notebook over released fuzz CSVs.
- We did **not** validate every claimed fuzzer crash against a reference stack because the merged CSVs contain coarse classifications and paths, not all crash artifacts/stack traces.

Thus this is an exact **computational reproduction of released raw results into paper results**, plus independent reanalysis—not a fresh experimental replication of every campaign.
