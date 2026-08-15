# KISS Sorcar Client Interfaces

> KISS Sorcar is used through three client interfaces, all served by one local daemon (`kiss-web`): the VS Code extension, the remote web/mobile app, and the Python client API `kiss.server.sorcar.run`.

## The `kiss-web` Daemon

The `kiss-web` daemon hosts all agents, chat sessions, and the web app. The VS Code extension starts it automatically; you can also manage it yourself:

```bash
# Start the daemon (serves the web app and the extension).
kiss-web

# Pin the daemon's working directory.
kiss-web --workdir "$HOME/projects/my-repo"

# Print the active remote (cloudflared) URL and exit.
kiss-web --url
```

| Flag | Description |
|------|-------------|
| `--workdir` | Working directory for the daemon |
| `--url` | Print the active remote URL and exit |

## VS Code Extension and Web/Mobile App

Open the KISS Sorcar sidebar in VS Code (or the remote web app in a browser) and type or speak your task. The chat interface provides:

- `@` file/folder mentions with ranked project-file completion.
- Per-task **git worktree isolation** with auto-commit and merge on success, or an interactive merge/discard prompt — toggle both in the Settings panel.
- A model picker, per-task budget caps, chat history with resume, and an agent dashboard (burger menu, bottom-left).
- Wake-word voice chat ("sorcar, …") via the mic button, including steering a running agent by voice.
- Live steering: inject a message into a running agent, or switch its model mid-run.
- Tab mirroring — every VS Code window and web client shows the same tabs with the same contents.
- API keys, a custom model endpoint, custom HTTP headers, budget limits, and the remote-access password, all set in the Settings panel.

The remote web app is the same interface served over a cloudflared tunnel: copy the URL and password from the Settings panel and open it on any device.

## Python Client API

Any Python process can run a task on the daemon with `kiss.server.sorcar.run` and block until it finishes:

```python
from kiss.server import sorcar

result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
print(result.text, result.success, result.cost, result.tokens, result.steps)

# Continue the same chat (the agent sees the prior task as context):
follow_up = sorcar.run("Now fix the typos you found", chat_id=result.chat_id)
```

Keyword options:

| Option | Description |
|--------|-------------|
| `work_dir` | Working directory for the task; the daemon's default when empty |
| `model` | Model name; the daemon's selected default when empty |
| `chat_id` | Existing chat session id to continue; a new chat when empty |
| `tools` | Path to a Python file whose `get_tools()` function returns the functions the daemon registers as extra agent tools |
| `use_worktree` | Run the task in an isolated git worktree (default `True`) |
| `auto_commit` | Auto-commit the task's changes on success (default `True`) |
| `max_budget` | Per-task budget override in USD |
| `model_config` | Per-task model configuration override (custom endpoint / headers) |
| `web_tools` | Per-task browser-tool enablement override |
| `is_parallel` | Whether the agent may spawn parallel sub-agents (default `True`) |
| `timeout` | Maximum seconds to wait for the task to finish (default `3600`) |
| `sock_path` | Daemon Unix-domain-socket path override |

The returned `TaskResult` carries `text`, `success`, `cost`, `tokens`, `steps`, `chat_id`, and `task_id`.

## Skills, MCP Servers, and Customization

- Agent Skills loaded from `~/.kiss/skills`, `<project>/.kiss/skills`, Claude skill directories, `.agents/skills`, and bundled Sorcar skills.
- MCP server discovery from `~/.kiss/mcp.json`, `<project>/.kiss/mcp.json`, and `<project>/.mcp.json`; OAuth tokens are persisted under `~/.kiss/mcp_auth/`.
- "Tricks" button entries read from `~/.kiss/INJECTIONS.md` (one per `## Trick` section), seeded on install from the bundled `src/kiss/INJECTIONS.md`.
- Welcome-screen sample-task chips are the concatenation of `~/.kiss/MY_TASK_TEMPLATES.md` (your personal tasks) and the bundled `src/kiss/SAMPLE_TASKS.md`.
