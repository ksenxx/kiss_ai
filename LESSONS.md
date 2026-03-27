# Lessons

- The VSCode extension has an embedded copy of the project at `src/kiss/agents/vscode/kiss_project/`. When modifying files like `server.py`, sync the embedded copy too.
- When a Python server communicates running state to a TypeScript extension via events, every exit path from the task execution method MUST send the "not running" status event. Use an outer try/finally wrapper to guarantee this.
- Tests that use `inspect.getsource()` to verify code structure will break when methods are refactored. When splitting a method into outer/inner, update source inspection tests to check the correct method.
