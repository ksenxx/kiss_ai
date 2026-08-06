# Prompt Tricks

> Reusable prompt snippets ("tricks") that boost KISS Sorcar result quality. In the VS Code extension these appear under the "Tricks" button; they are the concatenation of `~/.kiss/MY_INJECTION.md` (your personal tricks, auto-seeded on first read) and the bundled `src/kiss/INJECTIONS.md` (read directly from the package so every upgrade delivers the latest defaults). Append one or more to your task prompt.

## Multi-Model Quality

```text
Use 'claude-fable-5' model for all tasks, including software development. Use 'gpt-5.6-sol'
(not codex) for a thorough read-only review and debugging of the other model's work. Thoroughly
check whether the other model has missed any code or wiring or introduced any bugs. Use at most
20% of task budget in gpt-5.6-sol for reviewing and debugging, and ask the model to not invent
new problems. Use the model names literally without hallucinating new model names.
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

See also [Sample Tasks](sample-tasks.md) and [Tips](tips.md).
