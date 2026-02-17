# Common Cursor Prompts

Frequently used prompts extracted from 817 conversations, sorted by usage frequency.

## Tier 1: Daily Drivers

### Lint & Fix
```
run 'uv run check --clean' and fix
```

### Run Tests & Fix
```
run 'uv run pytest -v' and fix
```

### Update Documentation
```
can you update all *.md files in the project based on the latest changes to the code
```

### Paste Error & Fix
```
I am getting the following error: <paste traceback>. Can you fix it?
```

### Simplify & Clean Code
```
can you simplify and clean the code of <file> by removing unnecessary object attributes, variables, config variables, redundant and duplicate code, without changing functionality
```

## Tier 2: Regular Use

### Run a File & Fix
```
can you run 'uv run <file_path>' and fix
```

### Create a New Component (modeled after existing)
```
can you create <component> similar to <existing_file> but with <differences>. Test it exhaustively.
```

### Verify Testing
```
did you test it thoroughly?
```

### Find Bugs
```
carefully review all python files in the project and find bugs
```

### Modify Parameters/API
```
can you add/remove/rename parameter '<name>' in <file>
```

## Tier 3: Weekly Patterns

### Increase Test Coverage
```
can you increase the test coverage of <file> to 100%. DO NOT USE MOCKS. Write integration tests. Do not add redundant tests.
```

### Remove Redundant Tests
```
can you remove redundant and duplicate tests in <test_dir> while maintaining coverage
```

### Check Doc-Code Consistency
```
can you read <doc_file> line-by-line and carefully check its consistency with the latest code and fix
```

### Move/Restructure Files
```
can you safely and carefully move <file(s)> to <destination> so nothing breaks
```

### Optimize Performance
```
can you optimize <component> so that it finishes successfully (highest priority), runs faster, with lower cost, and lower token usage
```

### Onboard Models
```
can you onboard <model_name> from <provider> in model_info.py and test in both agentic and non-agentic mode
```

### Git Push
```
can you push the recent commits to origin
```

### Write Blog/Social Post
```
can you create a post for LinkedIn and X.com about <topic>
```

## Composite Workflow

### Full Cycle (lint + test + docs)
```
run 'uv run check --clean' and fix, then run 'uv run pytest -v' and fix, then update all *.md files based on the latest code changes
```
