# How to use the file?

Select a task below in sorcar editor and press cmd/ctrl-L to run the task in the chat window.

## increase test coverage

can you write integration tests with no mocks or test doubles to achieve 100% branch coverage of the files under [src/kiss/core/](src/kiss/core/), [src/kiss/core/models/](src/kiss/core/models/), and [src/kiss/agents/sorcar/](src/kiss/agents/sorcar/)? Please check the branch coverage first for the existing tests with the coverage tool by running 'uv run pytest -v'.  Then try to reach uncovered branches by crafting integration tests without any mocks, test doubles. You MUST repeat the task until you get 100% branch coverage or you cannot increase branch coverage after 10 tries.

## code review

find redundancy, duplication, AI slop, lack of elegant abstractions, and inconsistencies in the code of the project, and fix them. Make sure that you test every change by writing and running integration tests with no mocks or test doubles to achieve 100% branch coverage. Do not change any functionality or UI. Make that existing tests pass.

## documentation update

Can you carefully read all \*.md files, except API.md, in the project and check their consistency against the code in the project, grammar, and correctness? Fix them with precision.

## check

run 'uv run check --full' and fix

## test

run 'uv run pytest -v' with 900 seconds timeout and fix tests

## race detection

can you please work hard and carefully to precisly detect all actual race conditions in [src/kiss/agents/sorcar/sorcar.py](src/kiss/agents/sorcar/sorcar.py)? You can add random delays within 0.1 seconds before racing events to reliably trigger a race condition to confirm a race condition.

## test compaction

can you use [src/kiss/scripts/redundancy_analyzer.py](src/kiss/scripts/redundancy_analyzer.py) to get rid of redundant test methods in [src/kiss/tests/](src/kiss/tests/)?  Make sure that you don't decrease the overall branch coverage after removing the redundant test methods.

## slack setup and authentication

Set up Slack API authentication for sorcar so it can read/write messages to Slack channels. Follow these steps:

1. **Get the bot token**: Read the Slack bot token from the file [slack_token.txt](slack_token.txt) in the project root. If the file does not exist or is empty, launch the browser to `https://api.slack.com/apps` and ask the user to: (a) log in to Slack if needed, (b) select an existing Slack app or create a new one named "KISS Sorcar Bot", (c) go to "OAuth & Permissions", (d) ensure the bot has scopes: `channels:read`, `channels:history`, `chat:write`, `users:read`, (e) install the app to the workspace if not already installed, and (f) copy the "Bot User OAuth Token" (starts with `xoxb-`). Ask the user to paste the token, then save it to `slack_token.txt`.

1. **Register the workspace**: Use the `add_workspace` tool to register the workspace with a friendly name (ask the user for the workspace name) and the bot token from step 1. This validates the token and saves the configuration to `~/.kiss/slack/workspaces.json`.

1. **Verify the connection**: Use the `list_channels` tool to list all public channels in the workspace. Show the channel list to the user to confirm everything works.

1. **Test reading and writing**: Use `read_messages` to read the last 5 messages from a channel the user specifies (e.g. `#general`). Then use `send_message` to send a test message like "Hello from Sorcar!" to a channel the user specifies. Ask the user to confirm the message appeared in Slack.

If you need the user to do something on the browser that you cannot perform (like logging in or clicking OAuth allow), use `ask_user_browser_action` and wait for their response.

# Pending

When I click a recent item in the welcome window of the chat window in sorcar, it should behave similarly as clicking an item in the task history button in the chatbox of sorcar.

To validate that the code server creates a data directory for an instance of sorcar, can you launch sorcar in a task and validate that the chat window of the newly launched sorcar does not show the chat window events from the parent sorcar.

When I click a recent item in the welcome window of the sorcar (run with 'uv run sorcar'), you MSUT not open the list of task history in the UI.

When I launch KISS sorcar (using 'uv run sorcar') from inside a task run by sorcar, then whatever is printed in the chat window of the sorcar gets copied to the chat window of the newly launched sorcar. Can you validate this bugs by launching sorcar and fix it without breaking any other functionality or feature.

You have implemented a restart logic for code-server in case the code-server shuts down, but I want you to investigate the root cause of why the code-server is shutting down intermittently in the first place. See if you can fix the intermittent shutdown of the code server without changing any functionality in the project except for the fix.

can you check if [print_to_console.py](src/kiss/core/print_to_console.py) and [chatbot_ui.py](src/kiss/agents/sorcar/chatbot_ui.py) print exactly the same contents when an agent is executed on a task? Write a regression test for this.

Can you read [sorcar.py](src/kiss/agents/sorcar/sorcar.py) and carefully find all threads, timers, processes, and other forms of concurrency introduced by [sorcar.py](src/kiss/agents/sorcar/sorcar.py)? Then can you write a task in [PLAN.md](PLAN.md) which when given to the agent will reduce the amount of concurrency present in sorcar.

When I run sorcar on a task for very long time, the mac os x runs out of resources. Can you investigate the code for resource and memory hogs. For example, task_history.json could be very large. You may want to convert it into jsonl
format and read tasks on demand by sorcar. Find other memory and resource hogging issues in the prject.
