# Task: Run all tests in parallel and report failure causes (no code modifications)

## Steps
1. Counted cores: 10 → used 10 - 2 = 8 parallel splits.
2. Collected 3098 test node IDs (`uv run pytest --collect-only -q`, 72 deselected) into tmp/all_tests.txt.
3. Split round-robin into 8 files tmp/split_0..7.txt (388/388/387×6 tests).
4. Ran 8 sub-agents in parallel via run_parallel; each ran `uv run pytest -q -p no:cacheprovider --timeout=300` on its split.

## Results
- Total: 3096 ran → 3094 passed equivalent (2 failed), ~30 skipped, 0 errors.
- Split 0: 384 passed, 4 skipped
- Split 1: 385 passed, 3 skipped
- Split 2: 381 passed, 1 FAILED, 5 skipped
- Split 3: 383 passed, 4 skipped
- Split 4: 383 passed, 1 FAILED, 3 skipped
- Split 5: 383 passed, 4 skipped
- Split 6: 383 passed, 4 skipped
- Split 7: 384 passed, 3 skipped

## Failures (both flaky parallel-run races, pass in isolation)
1. test_slack_agent.py::TestSlackAgent::test_authenticate_invalid_token
   - AssertionError: 'xoxb-invalid-test-token-for-methods' is None (line 347).
   - Cause: shared global token file ~/.kiss/third_party_agents/slack/token.json
     written by TestSlackChannelBackendMethods.setup_method in a concurrent split.
2. test_run_once.py::TestCLIOneShotMode::test_channel_with_invalid_token
   - SystemExit: 1 instead of RuntimeError("Failed to connect").
   - Cause: concurrent split's setup/teardown deleted the same shared token file
     between _save_token() and main()'s _load_token().
- Both re-ran in isolation and PASSED. No product-code bugs found.

## Cleanup
- Deleted tmp/all_tests.txt, tmp/split_*.txt, tmp/result_*.txt.
