# Slash Commands for Sorcar Extension Agents

> Every Sorcar Extension Agent (SEA) is a folder `xxx/` that holds the script `xxx_sea.py` plus the helper modules and data files the agent needs. Every such folder the `kiss-web` daemon can see becomes a chat command `/xxx`, named after the folder. The bundled channel agents (`/slack`, `/gmail`, `/github`, ...) are registered automatically; you add your own by listing the parent folders in `~/.kiss/SEAS.md`. `/xxx help` prints the SEA's one-sentence `description()`. This page describes the layout and what happens between typing `/xxx` and the SEA running, so you can plug in your own agents without reading the source.

## What a slash command does

Typing this in the chat box of the VS Code extension or the web app:

```
/slack tell #eng that the deploy is done
```

is the same as asking Sorcar, in plain language, to run the `slack/slack_sea.py` agent on the task "tell #eng that the deploy is done", except that the routing is fixed: the session must call its `run_agent` tool first, with the SEA's absolute path and your text, and cannot pick a different agent or explore the code first. A slash command is therefore the predictable way to invoke a specific SEA.

Any file that is a valid SEA works: a bundled channel agent, or a file of your own whose top-level getters (`prompt()`, `model()`, `tools()`, `system_prompt()`, ...) configure the run. The SEA file format is described in [Client Interfaces: Sorcar Extension Agents](cli.md#sorcar-extension-agents-seas).

## Where commands come from

The daemon builds the command list from three sources:

1. **Bundled channel agents** in `src/kiss/agents/third_party_agents/` of the installed package. These always win a name clash.
2. **Bundled Sorcar-extending agents** in `src/kiss/agents/seas/` of the installed package, such as `/merge` (the merge-conflict resolver the auto-commit worktree merge also runs on its own) and `/sh` (runs the command you type — `/sh git status --short` — with the `Bash` tool alone, directly in the tab's working directory, and returns its output), and `/task_update` (`/task_update <task_id>` reports what that task has done so far and its partial results from its persisted transcript; the task-info panel runs the same agent for the running task every 10 minutes and on its refresh button), and `/write_paper` (`/write_paper <instructions>` writes or revises a research paper under the rules embedded in `write_paper_sea.py`; the instructions name the venue, the `.tex` path, the topic, the sources of truth and optionally the writer and reviewer models, the agent reads the highly cited related work of the last two or three years before writing, and it gets `check_paper`, which runs the AI-slop and consistency gates on the prose and flags fabricated-looking references in the `.bib` file, and `build_paper`, which runs pdflatex and bibtex or biber and summarizes the log), and `/review_paper` (`/review_paper <instructions>` reviews a paper (PDF, .tex, .md or .txt) for any venue as a careful human reviewer would; the instructions name the paper, the venue, the output path and the cutoff date for related work, and optionally the word limit (700 by default) and a second model that checks the review; the agent searches at least 20 related-work sources and judges the novelty against recent papers of the venue, and it gets `read_paper`, which returns the paper's text page by page with the template's margin line numbers removed, and `check_review`, which checks the review's structure, word limit and AI-slop and reviewer-boilerplate gates with line numbers), and `/revise_and_review_paper` (`/revise_and_review_paper Writing: <...> Review: <...>` runs the two as a loop: the coordinator writes the paper with `/write_paper` from the writing instructions, stages a copy of the built PDF and has `/review_paper` review it as a fresh, memory-free session that sees no earlier review or notes and ends with a `Recommendation:` line, then revises against the review, running experiments, ablations or the AI-discovery loop on the benchmarks the writing instructions allow when the review asks for evidence; it stops at strong accept, at the round cap (default 6) or when two rounds in a row fail to raise the verdict, and keeps every review and a per-round log under `reports/`), and `/git_extract_knowledge` (`/git_extract_knowledge <repo>` builds the durable memory of a git repository given as a local path or a clone URL: `index_repo` indexes every tracked file, 80-line chunk, symbol definition, commit, per-file change with its patch, tag, branch, contributor and directory as full-text blocks in `~/.kiss/memories/<repo>/knowledge.sqlite3`, incrementally after the first run; the agent then writes curated pages — overview, domain glossary and concepts, architecture, conventions, history, one page per module, FAQ — into the repository's domain memory, where every Sorcar run inside that repository finds them with `memory_search`, plus a `knowledge-lookup` page with the shell command that queries the block store, `python -m kiss.agents.seas.git_extract_knowledge.git_extract_knowledge_sea search <repo> "<query>"`; the first run schedules a cron job that runs `update <repo>` every morning (04:00 America/Los_Angeles when scheduled; the cron hour is fixed in the daemon's local time, so it moves by one hour across daylight-saving changes unless that clock is on Pacific time), and `ask <repo> <question>` answers from the memory), and `/remember` and `/forget` (`/remember <instruction>` appends the instruction as a bullet line to `~/.kiss/AGENTS.md`, the file appended to the system prompt of every task, so every later task follows it, creating the file on first use and leaving hand-written text alone; `/forget <instruction>` removes the stored line that matches ignoring case and spacing, or lists the stored instructions and removes the one you meant). Any `SEAS.md` folder may shadow these.
3. **Your folders** listed in `~/.kiss/SEAS.md` (or `$KISS_HOME/SEAS.md` when `KISS_HOME` is set).

The command name is the SEA's folder name: `deploy/deploy_sea.py` is `/deploy`, `pr-review/pr-review_sea.py` is `/pr-review`. A folder is registered only when it directly contains a regular file named `<folder>_sea.py` and the folder name uses ASCII letters, digits, `_`, or `-` only. `release.notes/` (a dot) or `my agent/` (a space) is skipped silently because the command could not be typed, and so is a folder whose script has another name (`deploy/main_sea.py`). A leading underscore is allowed: `_scratch/_scratch_sea.py` is `/_scratch`. A loose `xxx_sea.py` placed directly in a listed folder is not a command: every SEA needs its own folder, which is also where its helper modules, prompts and data files live.

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

Later lines override earlier ones: the folder at the bottom of `SEAS.md` beats the folder at the top. The bundled `third_party_agents/` folder beats every `SEAS.md` folder, and every `SEAS.md` folder beats the bundled `seas/` folder (the Sorcar-extending agents such as `/merge`, `/sh` and `/task_update`). In the example above, if `~/my-seas` and `/opt/agents/experimental` both contain `deploy/deploy_sea.py`, `/deploy` runs the experimental one; if either contains `slack/slack_sea.py`, `/slack` still runs the bundled Slack agent; if either contains `merge/merge_sea.py`, `/merge` runs your copy instead of the bundled one.

### Changes take effect while the daemon runs

The daemon rescans `SEAS.md` and every listed folder every 2 seconds. Adding a line, dropping a new `xxx/xxx_sea.py` into a listed folder, or deleting one updates the command list without a restart, and every open chat receives the new list, so the autocomplete popup updates too. A command that has just appeared is also resolved on first use even if the scan has not run yet.

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
           agent = /abs/path/to/deploy/deploy_sea.py
           task  = ship v2.3"
        |
        v
  Sorcar session calls run_agent(agent="/abs/path/to/deploy/deploy_sea.py",
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
- **`/deploy help` shows the description.** The one reserved sub-task is `help` (any letter case, nothing after it): the daemon does not run the SEA or call a model; it imports `deploy_sea.py`, calls its `description()` and posts the returned sentence as the task result. A SEA without a callable `description()` returning a non-empty string gets a diagnostic instead.
- **`<task>` blocks stay intact.** A prompt such as `/deploy <task>build</task><task>publish</task>` reaches the SEA as one sub-task with the tags in place; the daemon does not split it into two Sorcar tasks.
- **History keeps what you typed.** The task list, the chat history, and the frequent-task chips record `/deploy ship v2.3`, not the rewritten directive.
- **Where the sub-task runs.** A path-named SEA runs in the calling chat's working directory through the standard task lifecycle (worktree, auto-commit), unless the SEA's own `work_dir()`, `use_worktree()`, or `auto_commit()` getters say otherwise.
- **How long it may run.** `run_agent` waits 300 seconds by default and stops the sub-task when the wait runs out. An SEA that works longer defines `dispatch_timeout()` returning the seconds it needs (`/write_paper` 6 h, `/review_paper` 2 h, `/revise_and_review_paper` 24 h, `/write` 1 h); the directive then adds `timeout = "<seconds>"`. A missing getter or a value that is not a positive number leaves the directive as before.

## Add your own command in three steps

1. Create a parent folder, a folder named after the command, and the SEA file inside it. Every SEA must define `description()`, one sentence that says what it does and how to use it (this is what `/standup help` prints). The smallest useful SEA adds a system prompt; everything else keeps the daemon's defaults:

   ```python
   # ~/my-seas/standup/standup_sea.py
   def description():
       return (
           "Turns your text into a three-bullet daily stand-up note "
           "(done, next, blocked); use it as /standup <what happened>."
       )


   def system_prompt():
       return (
           "You write terse daily stand-up notes from the user's text. "
           "Three bullets: done, next, blocked. No preamble."
       )
   ```

   Helper modules, prompt files and data the SEA needs go into the same `standup/` folder. For a SEA that carries its own tools, return a list of callables from `tools()`; the bundled channel agents are written this way and are a good template (see [`src/kiss/agents/third_party_agents/`](https://github.com/ksenxx/kiss_ai/tree/main/src/kiss/agents/third_party_agents), one folder per agent).

2. Register the folder:

   ```bash
   echo '~/my-seas' >> ~/.kiss/SEAS.md
   ```

3. Within two seconds, type `/st` in the chat box: the popup offers `/standup`. Send `/standup help` to see the description you wrote, then `/standup finished the docs page, next is the release, blocked on review` and the note comes back in the chat.

If the command does not appear, check that the script is `<folder>/<folder>_sea.py` (`standup/standup_sea.py`, not `standup_sea.py` at the top of `~/my-seas`), that the folder name uses only `A-Z a-z 0-9 _ -`, and that the folder line in `SEAS.md` resolves to the folder you expect (remember that relative paths are anchored at `~`, not at the project). A broken SEA (import error, getter raising, wrong return type) is still listed as a command; the failure surfaces as a diagnostic in the task result when you run it.
