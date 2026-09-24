## Trick

Use 'claude-fable-5-1' model for all tasks, including software development. Use 'gpt-6-sol' (not codex) using `run_parallel` tool for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 75% of the task budget in gpt-6-sol for reviewing and debugging, and ask the model not to invent new problems. Use the model names literally without hallucinating new model names.

## Trick

Reproduce any violation of the invariant by writing end-to-end tests with 100% coverage. Then fix the issue.

## Trick

Can you git pull origin/<current-branch>, merge with <current-branch>, and push? 

## Trick

Authenticate on my behalf using claude-fable-5-1 as the model. Check the channel's existing credentials first and stop if they are valid. The channel's authentication tools open the sign-in page or developer portal in my default browser when they can and return its URL (and code): always show me that URL and code with ask_user_question so I can finish in my OWN browser if no window appeared. Never drive sign-in pages or developer portals with your own browser tools, do not retry or relaunch the browser, and never ask for or type my password or 2FA code. When I paste back a token or redirect URL, finish the authentication with the channel's tools and verify with its check tool.

## Trick

Use 'claude-fable-5' model for all tasks, including software development. Use 'gpt-5.6-sol' (not codex) using `run_parallel` tool for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 50% of the task budget in gpt-5.6-sol for reviewing and debugging, and ask the model not to invent new problems. Use the model names literally without hallucinating new model names.

## Trick

Use 'claude-fable-5-1' model for all tasks, including software development. Use 'gpt-6-astra' (not codex) using `run_parallel` tool for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 100% of the task budget in gpt-6-astra for reviewing and debugging, and ask the model not to invent new problems. Use the model names literally without hallucinating new model names.

## Trick

Use 'openrouter/moonshotai/kimi-k3' model for all tasks, including software development. Use 'gpt-5.6-sol' (not codex) using `run_parallel` tool for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 50% of the task budget in gpt-5.6-sol for reviewing and debugging, and ask the model not to invent new problems. Use the model names literally without hallucinating new model names.

## Trick

If ./ROUTING.md exists, use the instructions in the file for model routing. Otherwise, use the best model from ~/.kiss/MODEL_INFO.json for various subtasks. Search the internet extensively to figure out which model is best yet cheap for each subtask. Here are some hints, but the internet has better knowledge: claude-fable-5 and openrouter/moonshotai/kimi-k3 — best for SWE work; gpt-5.6-sol — best for reviewing; openrouter/qwen/qwen3.8-max, openrouter/x-ai/grok-4.6, openrouter/z-ai/glm-5.3, openrouter/deepseek/deepseek-v4-pro-0813 — for SWE tasks when budget is low; and gpt-5.6-luna and openrouter/deepseek/deepseek-v4-pro-0813 for review when budget is low. Irrespective of whether ./ROUTING.md exists or not, after the task completes, based on your experience in completing the task, create or update the model routing strategy (as text) in ./ROUTING.md that reduces token cost while not degrading the quality of the work.
