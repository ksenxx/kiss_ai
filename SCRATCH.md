How should intelligence be measured in the era of AI?

My test lately: how much high-quality work you get done while you are away from the keyboard.

Can you write a task so well in a paragraph that an AI agent can run with it for 2-4 hours, sometimes days, without supervision, and come back with work you would actually ship?

I tried this on a hard systems problem. Over several long unattended sessions, my agent framework, KISS Sorcar, built a key-value store that runs 5.9x faster than Microsoft's FASTER on the exact workload FASTER was designed for. My entire hands-on contribution: six task prompts and two short steering messages. API cost under $400.

Writing those six short prompts took some thought. A task an agent can run with for hours needs exact goals, hard constraints, and tests the agent cannot game. That is the skill I am practicing now, and it has become my definition of a productive day: a few focused hours of task writing, while the agents do the rest (research, software development, paper writing, AI discovery).

I would genuinely like to compare notes. What is the longest task you have handed to an AI agent, and did the result hold up?  Do you spend all your working time in front of Claude Code or Codex, verifying their outcomes?

Paper: https://kisssorcar.github.io/assets/hydra_kv.pdf