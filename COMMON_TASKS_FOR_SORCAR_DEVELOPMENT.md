# How to use the file?

Select a task below in sorcar editor and press cmd/ctrl-L to run the task in the chat window.

## increase test coverage

can you write integration tests with no mocks or test doubles to achieve 100% branch coverage of the files under src/kiss/core/, src/kiss/core/models/, and src/kiss/agents/sorcar/? Please check the branch coverage first for the existing tests with the coverage tool by running 'uv run pytest -v'.  Then try to reach uncovered branches by crafting integration tests without any mocks, test doubles. You MUST repeat the task until you get 100% branch coverage or you cannot increase branch coverage after 10 tries.

## code review

find redundancy, duplication, AI slop, lack of elegant abstractions, and inconsistencies in the code of the project, and fix them. Make sure that you test every change by writing and running integration tests with no mocks or test doubles to achieve 100% branch coverage. Do not change any functionality or UI. Make that existing tests pass.

## documentation update

Can you carefully read all \*.md files, except API.md, in the project and check their consistency against the code in the project, grammar, and correctness? Fix them with precision.

## check

run 'uv run check --full' and fix

## test

run 'uv run pytest -v' with 900 seconds timeout and fix tests

## race detection

can you please work hard and carefully to precisly detect all actual race conditions in src/kiss/agents/sorcar/sorcar.py? You can add random delays within 0.1 seconds before racing events to reliably trigger a race condition to confirm a race condition.

## test compaction

can you use src/kiss/scripts/redundancy_analyzer.py to get rid of redundant test methods in src/kiss/tests/?  Make sure that you don't decrease the overall branch coverage after removing the redundant test methods.

# Past tasks

When I click a recent item in the welcome window of the chat window in sorcar, it should behave similarly as clicking an item in the task history button in the chatbox of sorcar.

To validate that the code server creates a data directory for an instance of sorcar, can you launch sorcar in a task and validate that the chat window of the newly launched sorcar does not show the chat window events from the parent sorcar.

When I click a recent item in the welcome window of the sorcar (run with 'uv run sorcar'), you MSUT not open the list of task history in the UI.

When I launch KISS sorcar (using 'uv run sorcar') from inside a task run by sorcar, then whatever is printed in the chat window of the sorcar gets copied to the chat window of the newly launched sorcar. Can you validate this bugs by launching sorcar and fix it without breaking any other functionality or feature.

You have implemented a restart logic for code-server in case the code-server shuts down, but I want you to investigate the root cause of why the code-server is shutting down intermittently in the first place. See if you can fix the intermittent shutdown of the code server without changing any functionality in the project except for the fix.

can you check if src/kiss/core/print_to_console.py and src/kiss/agents/sorcar/chatbot_ui.py print exactly the same contents when an agent is executed on a task? Write a regression test for this.

Can you read src/kiss/agents/sorcar/sorcar.py and carefully find all threads, timers, processes, and other forms of concurrency introduced by src/kiss/agents/sorcar/sorcar.py? Then can you write a task in PLAN.md which when given to the agent will reduce the amount of concurrency present in sorcar.

When I run sorcar on a task for very long time, the mac os x runs out of resources. Can you investigate the code for resource and memory hogs. For example, task_history.json could be very large. You may want to convert it into jsonl
format and read tasks on demand by sorcar. Find other memory and resource hogging issues in the prject.

For the app, called gmail, create an {app}\_agent.py in src/kiss/channels/, an extension of SorcarAgent with a set of tools, which will help the user to get authenticated to the app via the browser if not authenticated yet, store the autentication token safely in the Path.home() / ".kiss/channels/{app}" dir, and use it along with tools to perform an app related task given to the app agent. Investigate the web for the app to indentify a small set of tools which will be given the agent total control over the app, implement them, and provide them as tools to the agent so that the agent can perform a given task on the app using the tools. write a main method in src/kiss/channels/{app}\_agent.py, so that it takes --task argument and executes the task using the agent.

Can you add the task results and the events file name as fields to each json object in task_history.jsonl. The file must update the result field once the sorcar agent finishes its task. If the task fails or is interruped by the user, then also update the result field with a suitable message. If the task is incomplete add the progress summary as result to the task.

Can you look at install.sh and installlib.sh and create a standalone macos x package for the project containing all dependencies such as code server, uv, git, brew if needed, Xcode develoeprs tools.  The package MUST be installable without internet. 