## Code Style Guidelines
- Write simple, clean, and readable code with minimal indirection
- Avoid unnecessary object attributes and local variables and config variables
- No redundant abstractions or duplicate code and config code
- Each function should do one thing well
- Use clear, descriptive names
- NO need to write documentations or comments unless absolutely necessary
- Public methods MUST have full documentation.
- Check and test the code you have written

## Testing Requirements
- Run lint and typecheckers and fix any lint and typecheck errors
- Generate comprehensive tests so that you achieve 100% branch coverage
- Tests MUST NOT use mocks, patches, or any form of test doubles
- Integration tests are HIGHLY encouraged
- You MUST not add tests that are redundant or duplicate of existing
  tests or does not add new coverage over existing tests
- Generate meaningful stress tests for the code if you are
  optimizing the code for performance
- Each test should be independent and verify actual behavior

## Use tools when you need to:
- Look up API documentation or library usage from the internet
- Find examples of similar implementations
- Understand existing code in the project

## After you have implemented the task, aggresively and carefully simplify and clean up the code
 - Remove unnessary object/struct attributes, variables, config variables
 - Avoid object/struct attribute redirections
 - Remove unnessary conditional checks
 - Remove redundant and duplicate code
 - Remove unnecessary comments
 - Make sure that the code is still working correctly
 - Simplify and clean up the test code
