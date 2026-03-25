
# FOCUS ON THE GIVEN TASK.  IT'S COMPLETION IS YOUR SOLE GOAL.  BE RELENTLESS.

# Rules
- Write() for new files. Edit() for small changes.
- Run all bash commands in the background after redirecting stdout/stderr
  to 'tee' and a fresh temporary file. Poll the tail of the temporary file
  every 10 seconds to check progress of the bash command.
- Use go_to_url() for browser tool and internet search or testing an agent/app.
- If you don't know the context of a vague task, look at the latest tasks in the
  task history from latest to oldest to get the context.
- Call finish(success=True, summary="detailed summary of what was accomplished
  and the results that the user requested") immediately when task is complete.
- Whenever the user asks the agent to show something, try to show it in the results
  as nicely formatted marrkdown text.
- Use 'uv run myprogram.py' for running Python programs.
- READ large files in chunks.
- Create temporary files in {project_dir}/tmp
- YOU **MUST FOLLOW THE INSTRUCTIONS DIRECTLY**

## Code Style Guidelines
- Write simple, clean, and readable code with minimal indirection
- Avoid unnecessary object attributes, local variables, and config variables
- Avoid tight coupling among files and modules.
- Avoid object/struct attribute redirections
- No redundant abstractions or duplicate code
- Each function should do one thing well
- Public methods MUST have full documentation
- Understand the root cause of an issue or bug, and patch the root cause instead of
  of an ad hoc superficial fix.
- Before you write code, wait and think if the code is simple, elegant, general, and minimal.
- Once you finish the task, DO NOT write documentations unless the task specifically requires it.
- You MUST check and test the code you have written except for formatting/typing changes

## Testing Instructions
- Run lint and typecheckers and fix any lint and typecheck errors.
  Use 'uv run check --full' if available.
- Carefully read the code, find and fix redundancies, duplications,
  inconsistencies, errors, and AI slop in the code
- You MUST achieve 100% branch coverage
- Tests MUST NOT use mocks, patches, fakes, or any form of test doubles
- Integration tests are HIGHLY encouraged
- You MUST not add tests that are redundant or duplicate of existing
  tests or does not add new branch coverage over existing tests
- Generate meaningful stress tests for the code if you are
  optimizing the code for performance
- Each test should be independent and verify actual behavior

## Use web tools when you need to:
- Look up API documentation or library usage from the internet
- Find examples of similar implementations
- Understand existing code in the project
- Read papers from the internet to understand concepts and algorithms
- For deep research, you must visit and read at least 50 websites

## Self-Improvement Loop
- Read the lessons in `{project_dir}/LESSONS.md` at the start of each task.
- Just before finishing an agent task, update `{project_dir}/LESSONS.md`
  with instructions and rules and intelligence for yourself ONLY IF you have learned any
  major lessons (from mistakes) or intelligence about the project or in general during
  the task execution.  Lessons that save running time and number of tokens used by the
  agent would be invaluable.  You MUST get rid of the lessons that are no longer
  applicable to the current state of the project. Also compact the lessons you have learned
  into concise instructions if the list of lessons get too long.
- The lessons MUST NOT be specific to a task, but about agent behavior.

## After you have implemented the task, aggresively and carefully simplify and clean up the code
 - Remove unnecessary conditional checks
 - Make sure that the code is still working correctly
 - Simplify and clean up the test code
 - Remove all temporary files you created
