# Prompt Tricks

> Reusable prompt snippets ("tricks") that boost KISS Sorcar result quality. In the VS Code extension these appear under the "Tricks" button; they are the concatenation of two `## Trick`-sectioned Markdown files: `~/.kiss/MY_INJECTION.md` (your personal tricks, auto-created on first read and never overwritten thereafter) and the bundled [`src/kiss/INJECTIONS.md`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/INJECTIONS.md), read directly from the package so every upgrade delivers the latest bundled tricks. Edit `~/.kiss/MY_INJECTION.md` to customize; your tricks are listed first. Append one or more to your task prompt.

## Multi-Model Quality

```text
Use 'claude-fable-5' model for all tasks, including software development. Use 'gpt-5.6-sol'
(not codex) using `run_parallel` tool for a thorough read-only review and debugging of the
other model's work. Thoroughly check whether the other model has missed any code or wiring or
introduced any bugs. Use at most 50% of the task budget in gpt-5.6-sol for reviewing and
debugging, and ask the model not to invent new problems. Use the model names literally without
hallucinating new model names.
```

## Bug Fixing

```text
Reproduce any violation of the invariant by writing end-to-end tests with 100% coverage. Then
fix the issue.
```

## Instructions from a File

```text
Can you use the instructions in the file @ to perform the task?
```

## Git

```text
Can you check the following message for a merge conflict and help me fix it?
```

## Autonomous Browser Authentication

```text
You MUST use the user's default browser and computer use to authenticate using claude-fable-5-1
as the model. Do all the steps on the user's behalf and ask the user's help ONLY if you are
stuck on login or captcha.
```

## Multi-Model Quality (GPT-6 Astra Review)

```text
Use 'claude-fable-5' model for all tasks, including software development. Use 'gpt-6-astra'
(not codex) using `run_parallel` tool for a thorough read-only review and debugging of the
other model's work. Thoroughly check whether the other model has missed any code or wiring or
introduced any bugs. Use at most 50% of the task budget in gpt-6-astra for reviewing and
debugging, and ask the model not to invent new problems. Use the model names literally without
hallucinating new model names.
```

## Multi-Model Quality (Kimi K3)

```text
Use 'openrouter/moonshotai/kimi-k3' model for all tasks, including software development. Use
'gpt-5.6-sol' (not codex) using `run_parallel` tool for a thorough read-only review and
debugging of the other model's work. Thoroughly check whether the other model has missed any
code or wiring or introduced any bugs. Use at most 50% of the task budget in gpt-5.6-sol for
reviewing and debugging, and ask the model not to invent new problems. Use the model names
literally without hallucinating new model names.
```

## Self-Improving Model Routing

```text
If ./ROUTING.md exists, use the instructions in the file for model routing. Otherwise, use the
best model from ~/.kiss/MODEL_INFO.json for various subtasks. Search the internet extensively
to figure out which model is best yet cheap for each subtask. Here are some hints, but the
internet has better knowledge: claude-fable-5 and openrouter/moonshotai/kimi-k3 — best for SWE
work; gpt-5.6-sol — best for reviewing; openrouter/qwen/qwen3.8-max, openrouter/x-ai/grok-4.6,
openrouter/z-ai/glm-5.3, openrouter/deepseek/deepseek-v4-pro-0813 — for SWE tasks when
budget is low; and gpt-5.6-luna and openrouter/deepseek/deepseek-v4-pro-0813 for review when
budget is low. Irrespective of whether ./ROUTING.md exists or not, after the task completes,
based on your experience in completing the task, create or update the model routing strategy
(as text) in ./ROUTING.md that reduces token cost while not degrading the quality of the work.
```

See also [Sample Tasks](sample-tasks.md) and [Tips](tips.md).
