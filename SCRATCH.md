How should intelligence be measured in the era of AI?

My test lately: how much high-quality work you get done while you are away from the keyboard.

Can you write a task so well in a paragraph that an AI agent can run with it for 2-4 hours, sometimes days, without supervision, and come back with work you would actually ship?

I tried this on a hard systems problem. Over several long unattended sessions, my agent framework, KISS Sorcar, built a key-value store that runs 5.9x faster than Microsoft's FASTER on the exact workload FASTER was designed for. My entire hands-on contribution: six task prompts and two short steering messages. API cost under $400.

Writing those six short prompts took some thought. A task an agent can run with for hours needs exact goals, hard constraints, and tests the agent cannot game. That is the skill I am practicing now, and it has become my definition of a productive day: a few focused hours of task writing, while the agents do the rest (research, software development, paper writing, AI discovery).

I would genuinely like to compare notes. What is the longest task you have handed to an AI agent, and did the result hold up?  Do you spend all your working time in front of Claude Code or Codex, verifying their outcomes?

Paper: https://kisssorcar.github.io/assets/hydra_kv.pdf

Or you can use this promptlet with free OSS KISS Sorcar (https://github.com/ksenxx/kiss_ai) and see the magic happening dynamically: If ./ROUTING.md exists, use the instructions in the file for model routing. Otherwise, use the best model from ~/.kiss/MODEL_INFO.json for various subtasks. Search the internet extensively to figure out which model is best yet cheap for each sub-task. Here are some hints, but the internet has better knowledge: claude-fable-5 — best for SWE work, gpt-5.6-sol — best for reviewing, and openrouter/z-ai/glm-5.2 — for SWE tasks when budget is low, and gpt-5.5 for review when budget is low. Irrespective of whether ./ROUTING.md exists or not, after the task completes, based on your experience in completing the task, create or update the model routing strategy (as text) in ./ROUTING.md that reduces token cost while not degrading the quality of the work.


XSS, SQL injection, Authentication bypass, insecure cryptography 