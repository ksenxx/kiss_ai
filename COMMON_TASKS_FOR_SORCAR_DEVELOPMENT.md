# How to use the file?

Select a task below in sorcar editor and press cmd/ctrl-L to run the task in the chat window.

## increase test coverage

can you write integration tests (possibly running 'uv run sorcar') with no mocks or test doubles to achieve 100% branch coverage of the files under src/kiss/agents/sorcar/? Please check the branch coverage first for the existing tests with the coverage tool.  Then try to reach uncovered branches by crafting integration tests without any mocks, test doubles. You MUST repeat the task until you get 100% branch coverage or you cannot increase branch coverage after 10 tries.

## code review

find redundancy, duplication, AI slop, lack of abstractions, and inconsistencies in the code of the project, and fix them. Make sure that you test every change by writing and running integration tests with no mocks or test doubles to achieve 100% branch coverage. Do not change any functionality. Make that existing tests pass.

## check

run 'uv run check --full' and fix

## test

run 'uv run pytest -v' with 900 seconds timeout and fix tests

## race detection

can you please work hard and carefully to precisly detect all actual race conditions in src/kiss/agents/sorcar/sorcar.py? You can add random delays within 0.1 seconds before racing events to reliably trigger a race condition to confirm a race condition.

## test compaction

can you use src/kiss/scripts/redundancy_analyzer.py to get rid of redundant test methods in src/kiss/tests/?  Make sure that you don't decrease the overall branch coverage after removing the redundant test methods.

# Pending

When I click a recent item in the welcome window of the chat window in sorcar, it should behave similarly as clicking an item in the task history button in the chatbox of sorcar.

To validate that the code server creates a data directory for an instance of sorcar, can you launch sorcar in a task and validate that the chat window of the newly launched sorcar does not show the chat window events from the parent sorcar.

When I click a recent item in the welcome window of the sorcar (run with 'uv run sorcar'), you MSUT not open the list of task history in the UI.

When I launch KISS sorcar (using 'uv run sorcar') from inside a task run by sorcar, then whatever is printed in the chat window of the sorcar gets copied to the chat window of the newly launched sorcar. Can you validate this bugs by launching sorcar and fix it without breaking any other functionality or feature.

You have implemented a restart logic for code-server in case the code-server shuts down, but I want you to investigate the root cause of why the code-server is shutting down intermittently in the first place. See if you can fix the intermittent shutdown of the code server without changing any functionality in the project except for the fix.

can you check if print_to_console.py and chatbot_ui.py print exactly the same contents when an agent is executed on a task?

Can you read sorcar.py and carefully find all threads, timers, processes, and other forms of concurrency introduced by sorcar.py?


