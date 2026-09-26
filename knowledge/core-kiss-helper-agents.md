---
title: Ready-made helper agents in agents/kiss.py (prompt refiner, docker bash agent,
  simple coding agent)
uuid: 6f87ef9b-4d27-4464-ae96-098fbd94e7e4
summary: 'agents/kiss.py helpers: prompt_refiner_agent (placeholder-name bug), run_bash_task_in_sandboxed_ubuntu_latest
  (Docker Bash), get_run_simple_coding_agent (test tool + finish).'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Helper agents in `src/kiss/agents/kiss.py`

They are exported lazily from `kiss.agents` (`src/kiss/agents/__init__.py`). No other code in the repo calls
them. They are also short examples of how to use `KISSAgent.run`.

## `prompt_refiner_agent(original_prompt_template, previous_prompt_template, agent_trajectory_summary, model_name)`
- A non-agentic run (`is_agentic=False`, a single generation, no tools) of the `prompt_template_refiner` template.
  The template wraps each input in `<user_input>` tags and has a "Security Override" section telling the model
  to treat those inputs as untrusted data.
- **Known bug (verified in the code):** the function passes the argument key `agent_trajectory_summary`, but the
  template's placeholder is `{agent_trajectory}`. `substitute_prompt_args` substitutes only the keys it is given
  and leaves unknown braces alone, so no error is raised: the literal text `{agent_trajectory}` stays in the
  prompt and the trajectory is never sent. Renaming one side to match the other fixes it.

## `run_bash_task_in_sandboxed_ubuntu_latest(task, model_name)`
It opens `DockerManager("ubuntu:latest")` as a context manager and runs an agentic `KISSAgent` whose only
tool is `env.Bash`. The built-in `finish(result)` is added automatically, so the result is plain text.

## `get_run_simple_coding_agent(test_fn) -> run_simple_coding_agent(prompt_template, arguments, model_name)`
This is a closure factory. The agent gets `tools=[test_fn, utils.finish]`, and the prompt tells it to test
the code and call `finish(success=True, summary_in_html="<pre><code>...")`. The result is parsed with
`yaml.safe_load(...)["summary"]`. Because `utils.finish` passes the summary through `ensure_html`, the
returned code is HTML.

## Patterns shown
- Structured finish: register `kiss.core.utils.finish` and parse the YAML result.
- Plain finish: register nothing and take the result string.
- Non-agentic calls must not pass `tools` (otherwise `KISSError`).

## Sources
- `src/kiss/agents/kiss.py` (`prompt_template_refiner`, `prompt_refiner_agent`, `run_bash_task_in_sandboxed_ubuntu_latest`, `get_run_simple_coding_agent`)
- `src/kiss/agents/__init__.py`
- `src/kiss/core/utils.py` (`substitute_prompt_args`, `finish`)
