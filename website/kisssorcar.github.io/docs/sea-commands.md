# Slash Commands for Sorcar Extension Agents

> Every Sorcar Extension Agent (SEA) file named `xxx_sea.py` that the `kiss-web` daemon can see becomes a chat command `/xxx`. The bundled channel agents (`/slack`, `/gmail`, `/github`, ...) are registered automatically; you add your own by listing folders in `~/.kiss/SEAS.md`. This page describes the file format and what happens between typing `/xxx` and the SEA running, so you can plug in your own agents without reading the source.

## What a slash command does

Typing this in the chat box of the VS Code extension or the web app:

```
/slack tell #eng that the deploy is done
```

is the same as asking Sorcar, in plain language, to run the `slack_sea.py` agent on the task "tell #eng that the deploy is done", except that the routing is fixed: the session must call its `run_agent` tool first, with the SEA's absolute path and your text, and cannot pick a different agent or explore the code first. A slash command is therefore the predictable way to invoke a specific SEA.

Any file that is a valid SEA works: a bundled channel agent, or a file of your own whose top-level getters (`prompt()`, `model()`, `tools()`, `system_prompt()`, ...) configure the run. The SEA file format is described in [Client Interfaces: Sorcar Extension Agents](cli.md#sorcar-extension-agents-seas).

## Where commands come from

The daemon builds the command list from three sources:

1. **Bundled channel agents** in `src/kiss/agents/third_party_agents/` of the installed package. These always win a name clash.
2. **Bundled Sorcar-extending agents** in `src/kiss/agents/seas/` of the installed package, such as `/merge` (the merge-conflict resolver the auto-commit worktree merge also runs on its own) and `/sh` (runs the command you type — `/sh git status --short` — with the `Bash` tool alone, directly in the tab's working directory, and returns its output). Any `SEAS.md` folder may shadow these.
3. **Your folders** listed in `~/.kiss/SEAS.md` (or `$KISS_HOME/SEAS.md` when `KISS_HOME` is set).

The command name is the file name with `_sea.py` removed: `deploy_sea.py` is `/deploy`, `pr-review_sea.py` is `/pr-review`. A file is registered only when it is a regular file, ends in `_sea.py`, and the remaining stem uses ASCII letters, digits, `_`, or `-` only. `release.notes_sea.py` (a dot) or `my agent_sea.py` (a space) is skipped silently because the command could not be typed. A leading underscore is allowed: `_scratch_sea.py` is `/_scratch`.

## SEAS.md syntax

`~/.kiss/SEAS.md` is a plain text file with one folder per line. Despite the `.md` suffix, it is not rendered Markdown; the rules are:

| Rule | Example |
| --- | --- |
| One folder path per line | `/opt/team-agents` |
| Blank lines are ignored | |
| A line starting with `#` is a comment | `# personal agents` |
| ` #` (a space, then `#`) starts an inline comment | `~/my-seas   # personal` |
| `~` and `$VARIABLE` are expanded | `$WORK/agents` |
| A relative path is resolved against your home directory, not the current project | `my-seas` means `~/my-seas` |
| Spaces inside a path work verbatim; no quoting or escaping | `/Users/alice/team seas` |
| Missing or unreadable folders are skipped | `/does/not/exist` has no effect |

Do not shell-quote paths. `"~/my seas"` would look for a folder whose name starts with a literal quote. Backslashes are kept as typed, so Windows paths such as `C:\Users\alice\seas` need no escaping.

A complete example:

```
# ~/.kiss/SEAS.md
~/my-seas              # personal agents
$WORK/agents           # agents shared by the team
/opt/agents/experimental
```

### Precedence when two folders define the same name

Later lines override earlier ones: the folder at the bottom of `SEAS.md` beats the folder at the top. The bundled `third_party_agents/` folder beats every `SEAS.md` folder, and every `SEAS.md` folder beats the bundled `seas/` folder (the Sorcar-extending agents such as `/merge` and `/sh`). In the example above, if `~/my-seas` and `/opt/agents/experimental` both contain `deploy_sea.py`, `/deploy` runs the experimental one; if either contains `slack_sea.py`, `/slack` still runs the bundled Slack agent; if either contains `merge_sea.py`, `/merge` runs your copy instead of the bundled one.

### Changes take effect while the daemon runs

The daemon rescans `SEAS.md` and every listed folder every 2 seconds. Adding a line, dropping a new `xxx_sea.py` into a listed folder, or deleting one updates the command list without a restart, and every open chat receives the new list, so the autocomplete popup updates too. A command that has just appeared is also resolved on first use even if the scan has not run yet.

## The slash-command flow

```
  you type "/deploy ship v2.3" and press Enter
        |
        v
  chat client (VS Code extension / web app)
    - shows a "Commands" popup while you type the command word
    - submits the prompt unchanged
        |
        v
  kiss-web daemon
    - prompt starts with "/deploy" and "deploy" is registered?
        |                       |
        | yes                   | no  -> handled as an ordinary prompt
        v
    - rewrites the prompt into a directive:
        "Call run_agent IMMEDIATELY with
           agent = /abs/path/to/deploy_sea.py
           task  = ship v2.3"
        |
        v
  Sorcar session calls run_agent(agent="/abs/path/to/deploy_sea.py",
                                 task="ship v2.3")
        |
        v
  daemon imports deploy_sea.py, applies its getters, runs the sub-task
        |
        v
  result is relayed back into your chat
```

Details worth knowing:

- **Autocomplete.** As soon as the first character in the box is `/`, a popup lists the matching commands (substring match, case-insensitive). Pick one with the mouse, or move with the arrow keys and press Tab or Enter; the client inserts `/name ` with a trailing space and closes the popup. Once you type a space after the command word, the popup hides and normal `@file` mentions and ghost-text suggestions resume.
- **Only at position 0.** The command must be the first character of the prompt with no leading whitespace or blank line. `/deploy` on a later line of a multi-line prompt is ordinary text.
- **Exact word match.** `/deployx ship` looks up a command named `deployx`; it does not match `/deploy`.
- **Text is required.** `/deploy` alone, or a `/name` that is not registered, is not rewritten: it is sent to the model as a normal prompt and answered like any other question.
- **`<task>` blocks stay intact.** A prompt such as `/deploy <task>build</task><task>publish</task>` reaches the SEA as one sub-task with the tags in place; the daemon does not split it into two Sorcar tasks.
- **History keeps what you typed.** The task list, the chat history, and the frequent-task chips record `/deploy ship v2.3`, not the rewritten directive.
- **Where the sub-task runs.** A path-named SEA runs in the calling chat's working directory through the standard task lifecycle (worktree, auto-commit), unless the SEA's own `work_dir()`, `use_worktree()`, or `auto_commit()` getters say otherwise.

## Add your own command in three steps

1. Create a folder and a SEA file. The smallest useful SEA pins a system prompt; everything else keeps the daemon's defaults:

   ```python
   # ~/my-seas/standup_sea.py
   def system_prompt():
       return (
           "You write terse daily stand-up notes from the user's text. "
           "Three bullets: done, next, blocked. No preamble."
       )
   ```

   For a SEA that carries its own tools, return a list of callables from `tools()`; the bundled channel agents are written this way and are a good template (see [`src/kiss/agents/third_party_agents/`](https://github.com/ksenxx/kiss_ai/tree/main/src/kiss/agents/third_party_agents)).

2. Register the folder:

   ```bash
   echo '~/my-seas' >> ~/.kiss/SEAS.md
   ```

3. Within two seconds, type `/st` in the chat box: the popup offers `/standup`. Send `/standup finished the docs page, next is the release, blocked on review` and the note comes back in the chat.

If the command does not appear, check that the file name ends in `_sea.py`, that the stem uses only `A-Z a-z 0-9 _ -`, and that the folder line in `SEAS.md` resolves to the folder you expect (remember that relative paths are anchored at `~`, not at the project). A broken SEA (import error, getter raising, wrong return type) is still listed as a command; the failure surfaces as a diagnostic in the task result when you run it.
