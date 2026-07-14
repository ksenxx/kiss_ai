## Trick

Search the internet extensively.

## Trick

Reproduce the issue by writing real end-to-end tests with 100% coverage. Then fix the issue.

## Trick

Use claude-fable-5 model for all tasks, including software development. Use gpt-5.6-sol (not codex) for a thorough review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. No need to check if the models exist.

## Trick

Can you run all tests? Running all tests, collecting information, and fixing them is time-consuming. So split the set of tests by the number of test methods into the number of cores - 2, and run all splits in parallel using `run_parallel` tool. Determine which test failures are due to a bug in the project or a bug in the test. Fix them accordingly.

## Trick

Can you use the instructions in the file below to perform the task?

## Trick

Create an HTML report with diagrams and illustrations (that do not look AI-generated) in ./reports, and open it in the user's default browser.

## Trick

Reproduce any violation of the invariant by writing end-to-end tests with 100% coverage. Then fix the issue.

## Trick

MAKE SURE THAT YOU DO NOT DO REWARD HACKING OR CHEATING IN THE MODELS OR AGENTS YOU ARE IMPLEMENTING TO FIT DATA. YOUR SOLUTION MUST GENERALIZE BEYOND THE DATA PROVIDED.

## Trick

Use openrouter/z-ai/glm-5.2 model for all tasks including coding, bug fixing, and test creation. ALWAYS use claude-fable-5 to carefully and thoroughly review and debug the work done by openrouter/z-ai/glm-5.2 for bugs and missing code. 

## Trick

Use the best model from ~/.kiss/MODEL_INFO.json for various subtasks. Search the internet extensively to figure out which model is best yet cheap for each sub-task. Here are some hints, but the internet has better knowledge: claude-fable-5 — best for SWE work, gpt-5.6-sol — best for reviewing, and openrouter/z-ai/glm-5.2 — for SWE tasks when budget is low, and gpt-5.5 for review when budget is low.  After the task completes, based on your experience and internet research, add a model routing strategy (as text) to ./ROUTING.md that reduces token cost while not degrading the quality of the work. 

## Trick

Can you check the following message for a merge conflict and help me fix it?

## Trick

Make sure that there is no reward hacking and cheating when reviewing your implementation.

## Trick

Build the paper, then take screenshots to check and fix the formatting.

## Trick

Why did the last task fail? Thoroughly and precisely analyze the logs and the events of the task. Reproduce the issue by writing an integration test. Then fix the issue.


