# How to use the file?

Select a task below in sorcar editor and press cmd/ctrl-L to run the task in the chat window.

## increase test coverage

can you write integration tests with no mocks or test doubles to achieve 100% branch coverage of the files under src/kiss/agents/sorcar/? Please check the branch coverage first for the existing tests with the coverage tool.  Then try to reach uncovered branches by crafting integration tests without any mocks, test doubles. You MUST repeat the task until you get 100% branch coverage or you cannot increase branch coverage after 10 tries.

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

# documentation update

Can you read all \*.md files, except API.md, in the project carefully and check and precisely fix any inconsistencies with the code in the project?


# porting 'autresearch' to src/kiss/agents/autoresearch/

can you implement the 'autoresearch' agent at https://github.com/karpathy/autoresearch in the folder src/kiss/agents/autoresearch/ using src/kiss/agents/sorcar/sorcar_agent.py and src/kiss/agents/sorcar/sorcar.py ? Please write integration tests with no mocks or test doubles to achieve 100% branch coverage.  Please do it precisely and do the the most intuitive design for the ambiguous parts.  Simplify code.  Use the browser tool if necessary.

can you write documentation on 'autoresearch' and how to use it, how it works, and what are the advantages of KISS based 'autresesearch' over original 'autoresearch' in src/kiss/agents/autoresearch/README.md?
