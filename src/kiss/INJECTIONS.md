## Trick

Reproduce the issue by writing real end-to-end tests with 100% coverage. Then fix the issue. You can use screenshots to validate the implementation.

## Trick

Use 'claude-opus-5' model for all tasks, including software development. Use 'gpt-5.6-sol' (not codex) for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 20% of task budget in gpt-5.6-sol for reviewing and debugging, and ask the model to not invent new problems. Use the model names literally without hallucinating new model names.

## Trick

Can you run all tests? Use `run_parallel` tool to split and run tests in parallel. Determine which test failures are due to a bug in the project or a bug in the test. Fix them accordingly.

## Trick

Can you use the instructions in the file below to perform the task?

## Trick

Reproduce any violation of the invariant by writing end-to-end tests with 100% coverage. Then fix the issue.

## Trick

Can you check the following message for a merge conflict and help me fix it?

## Trick

Make sure that there is no reward hacking and cheating to fit data or tests when reviewing your implementation.

## Trick

Build the paper, then take screenshots to check and fix the formatting.

## Trick

Why did the last task fail? Thoroughly and precisely analyze the logs and the events of the task. Reproduce the issue by writing an integration test. Then fix the issue.
