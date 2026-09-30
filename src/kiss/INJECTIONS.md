## Trick

claude-fable-5-1 model be used for all tasks, including software development. Use gpt-6-astra (not codex) using `run_parallel` tool for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 75% of the task budget in gpt-6-astra for reviewing and debugging, and ask the model not to invent new problems. Use the model names literally without hallucinating new model names.

## Trick

Reproduce any violation of the invariant by writing end-to-end tests with 100% coverage. Then fix the issue.

## Trick

Can you git pull origin/<current-branch>, merge with <current-branch>, and push? 

## Trick

Authenticate on my behalf using claude-fable-5-1 as the model. Check the channel's existing credentials first and stop if they are valid. The channel's authentication tools open the sign-in page or developer portal for me by themselves and say where it went ('opened_in'): when it is in the Browser tab, tell me with ask_user_question to finish there (and the code, if any) and never ask me to open a URL; only when it opened in my default browser or nowhere, show me the URL and code with ask_user_question so I can finish in my OWN browser. Never drive sign-in pages or developer portals with your own browser tools, do not retry or relaunch the browser, and never ask for or type my password or 2FA code. When I paste back a token or redirect URL, finish the authentication with the channel's tools and verify with its check tool.

## Trick

Use 'claude-fable-5' model for all tasks, including software development. Use 'gpt-5.6-sol' (not codex) using `run_parallel` tool for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 50% of the task budget in gpt-5.6-sol for reviewing and debugging, and ask the model not to invent new problems. Use the model names literally without hallucinating new model names.

## Trick

Use 'openrouter/moonshotai/kimi-k3' model for all tasks, including software development. Use 'gpt-6-astra' (not codex) using `run_parallel` tool for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 50% of the task budget in gpt-6-astra for reviewing and debugging, and ask the model not to invent new problems. Use the model names literally without hallucinating new model names.
