# How to use the file?

Select a task below in sorcar editor and press cmd/ctrl-L to run the task in the chat window.

## increase test coverage

can you write integration tests with no mocks or test doubles to achieve 100% branch coverage of the files under src/kiss/core/, src/kiss/core/models/, and src/kiss/agents/sorcar/? Please check the branch coverage first for the existing tests with the coverage tool by running 'uv run pytest -v' with a timeout of 900 second.  Then try to reach uncovered branches by crafting integration tests without any mocks, test doubles. You MUST repeat the task until you get 100% branch coverage or you cannot increase branch coverage after 10 tries.

## code review

find redundancy, duplication, AI slop, lack of elegant abstractions, and inconsistencies in the code of the project, and fix them. Make sure that you test every change by writing and running integration tests with no mocks or test doubles to achieve 100% branch coverage. Do not change any functionality or UI. Make sure that existing tests pass. Tests must be run
with a timeout of 900 seconds.

## documentation update

Can you carefully read all \*.md files, except API.md, in the project and check their consistency against the code in the project, grammar, and correctness? Fix them with precision.

## check

run 'uv run check --full' and fix

## test

run 'uv run pytest -v' with 900 seconds timeout and fix tests

## race detection

can you please work hard and carefully to precisly detect all actual race conditions in src/kiss/agents/sorcar/sorcar.py? You can add random delays within 0.1 seconds before racing events to reliably trigger a race condition to confirm a race condition. DO NOT FIX the race conditions.

## dead code elimination

can you carefully analyze all Python source files under src/kiss/ and identify any dead code — unused functions, unreachable branches, unused imports, and unused variables? Remove them, ensure existing tests still pass (run 'uv run pytest -v' with 900 seconds timeout), and verify that branch coverage does not decrease after your changes.

## error handling audit

can you carefully audit all Python source files under src/kiss/ for improper error handling? Look for bare except clauses, overly broad exception catching, swallowed exceptions (caught but silently ignored), missing error context in raised exceptions, and places where errors should be caught but aren't. Fix them to ensure exceptions are specific, informative, and properly propagated or logged. Make sure existing tests still pass (run 'uv run pytest -v' with 900 seconds timeout) and that branch coverage does not decrease.

## test compaction

can you use src/kiss/scripts/redundancy_analyzer.py to get rid of redundant test methods in src/kiss/tests/?  Make sure that you don't decrease the overall branch coverage after removing the redundant test methods. Run tests with 900 seconds
timeout.

# Past tasks

When I click a recent item in the welcome window of the chat window in sorcar, it should behave similarly as clicking an item in the task history button in the chatbox of sorcar.

To validate that the code server creates a data directory for an instance of sorcar, can you launch sorcar in a task and validate that the chat window of the newly launched sorcar does not show the chat window events from the parent sorcar.

When I click a recent item in the welcome window of the sorcar (run with 'uv run sorcar'), you MUST not open the list of task history in the UI.

When I launch KISS sorcar (using 'uv run sorcar') from inside a task run by sorcar, then whatever is printed in the chat window of the sorcar gets copied to the chat window of the newly launched sorcar. Can you validate this bugs by launching sorcar and fix it without breaking any other functionality or feature.

You have implemented a restart logic for code-server in case the code-server shuts down, but I want you to investigate the root cause of why the code-server is shutting down intermittently in the first place. See if you can fix the intermittent shutdown of the code server without changing any functionality in the project except for the fix.

can you check if src/kiss/core/print_to_console.py and src/kiss/agents/sorcar/chatbot_ui.py print exactly the same contents when an agent is executed on a task? Write a regression test for this.

Can you read src/kiss/agents/sorcar/sorcar.py and carefully find all threads, timers, processes, and other forms of concurrency introduced by src/kiss/agents/sorcar/sorcar.py? Then can you write a task in PLAN.md which when given to the agent will reduce the amount of concurrency present in sorcar.

When I run sorcar on a task for a very long time, the macOS runs out of resources. Can you investigate the code for resource and memory hogs. For example, task_history.json could be very large. You may want to convert it into jsonl
format and read tasks on demand by sorcar. Find other memory and resource hogging issues in the project.

For the app, called whatsapp, create an {app}\_agent.py in src/kiss/channels/, an extension of SorcarAgent with a set of tools, which will help the user to get authenticated to the app via the browser if not authenticated yet, store the authentication token safely in the Path.home() / ".kiss/channels/{app}" dir, and use it along with tools to perform an app related task given to the app agent. Investigate the web for the app to identify a small set of tools which will be given the agent total control over the app, implement them, and provide them as tools to the agent so that the agent can perform a given task on the app using the tools. write a main method in src/kiss/channels/{app}\_agent.py, so that it takes --task argument and executes the task using the agent.

Can you add the task results and the events file name as fields to each json object in task_history.jsonl. The file must update the result field once the sorcar agent finishes its task. If the task fails or is interrupted by the user, then also update the result field with a suitable message. If the task is incomplete add the progress summary as result to the task.

Can you look at install.sh and installlib.sh and create a standalone macOS package for the project containing all dependencies such as code server, uv, git, brew if needed, Xcode developers tools. The package MUST be installable without internet.

can you write a background_agent.py in src/kiss/agents/claw/ similar to sorcar except that you will wait for messages on the slack channel #sorcar from ksen, treat the message as a task, complete the task using sorcar_agent.py, and send the results back to the slack channel. There will be no GUI integration unlike sorcar. The call to the tools cli_wait_for_user and cli_ask_user_question should send a message to the slack for user feedback from ksen. Once user ksen responds, the agent must continue to finish the task. Use the src/kiss/channels/slack_agent.py for communication via slack. Test it by running it and interacting with it. You can ask the user during testing to authenticate and to send tasks and respond to cli actions. Make sure that the background_agent.py is not hogging CPU, memory, file resources while waiting for user
message.

Look at the code of sorcar. Search the internet and find out how the Claude code extension has been implemented. Can you write down a plan in PLAN.md tp implement an extension in src/kiss/agents/vscode to VS code exactly like the sorcar chat window with same layout buttons and their functionalities, but uses src/kiss/agents/sorcar/sorcar_agent.py. Moreover you MUST support the workflow of sorcar except the ones that you get free from Desktop VSCode. Test it by launching vscode and installing the extension and see it behaves as expected by taking screenshots. Do not hardcode the location of of the KISS project in the extension. You MUST be very precise and careful. You must search the internet if you do not know the answer for something.

You MUST check whether you can run a sample task in the VSCode extension. Launch Desktop VSCode and use screen capture to validate. Do the testing throroughly and fix all bugs.

in the vscode extension you have to add a major feature. You need to add a new column, called chat_id as a text field storing a unique hex code for an agent chat session, to task_history table of the history database. When the user clicks the Clear button in the chatbox, you should create a new chat_id which is not present in the table. Then when a task is added, add the task to the task_history table as before, but add the new chat_id to row of the task_history table. When a task is submitted to the agent, append the tasks and results with the same chat_id in chronological order to the prompt and then run the sorcar agent and telling the agent that "## Previous tasks and results from the chat session for reference"
