## Trick

Reproduce the issue by writing real end-to-end tests with 100% coverage. Then fix the issue. You can use screenshots to validate the implementation.

## Trick

Use 'claude-opus-5' model for all tasks, including software development. Use 'gpt-5.6-sol' (not codex) for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 20% of task budget in gpt-5.6-sol for reviewing and debugging. Use the model names literally without hallucinating new model names.

## Trick

Can you run all tests? Use `run_parallel` tool to split and run tests in parallel. Determine which test failures are due to a bug in the project or a bug in the test. Fix them accordingly.

## Trick

Can you use the instructions in the file below to perform the task?

## Trick

Reproduce any violation of the invariant by writing end-to-end tests with 100% coverage. Then fix the issue.

## Trick

If ./ROUTING.md exists, use the instructions in the file for model routing. Otherwise, use the best model from ~/.kiss/MODEL_INFO.json for various subtasks. Search the internet extensively to figure out which model is best yet cheap for each sub-task. Here are some hints, but the internet has better knowledge: claude-opus-5 — best for SWE work, gpt-5.6-sol — best for reviewing, and openrouter/z-ai/glm-5.2 — for SWE tasks when budget is low, and gpt-5.6-luna for review when budget is low. Irrespective of whether ./ROUTING.md exists or not, after the task completes, based on your experience in completing the task, create or update the model routing strategy (as text) in ./ROUTING.md that reduces token cost while not degrading the quality of the work.

## Trick

Can you check the following message for a merge conflict and help me fix it?

## Trick

Make sure that there is no reward hacking and cheating to fit data or tests when reviewing your implementation.

## Trick

Build the paper, then take screenshots to check and fix the formatting.

## Trick

Why did the last task fail? Thoroughly and precisely analyze the logs and the events of the task. Reproduce the issue by writing an integration test. Then fix the issue.
