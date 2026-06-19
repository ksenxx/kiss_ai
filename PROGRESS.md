# Progress

- Started task: run all tests without modifying code and report causes of failures.
- Read `SORCAR.md` first as required; it was empty.
- Checked CPU core count: 10 cores, so tests must be split into 8 parallel splits (cores - 2).
- Collected pytest node IDs with `uv run pytest --collect-only -q --no-cov`; default pytest configuration deselects slow tests, yielding 3677 collected runnable tests (out of 3749 total, 72 deselected).
- Recollected with `uv run pytest --collect-only -q --no-cov -m ''` to include slow tests as part of "all tests", yielding 3749 collected tests.
- Split the 3749 node IDs into 8 temporary split files: 469, 469, 469, 469, 469, 468, 468, and 468 tests.
- Ran all 8 splits in parallel using `run_parallel` as requested. Every split exited with code 1. Aggregate split summaries: split 1 = 1 failed / 463 passed / 5 skipped; split 2 = 4 failed / 461 passed / 4 skipped; split 3 = 3 failed / 461 passed / 5 skipped; split 4 = 2 failed / 461 passed / 6 skipped; split 5 = 2 failed / 463 passed / 4 skipped; split 6 = 2 failed / 461 passed / 5 skipped; split 7 = 3 failed / 460 passed / 5 skipped; split 8 = 2 failed / 463 passed / 3 skipped. Aggregate: 19 failed, 3693 passed, 37 skipped.
