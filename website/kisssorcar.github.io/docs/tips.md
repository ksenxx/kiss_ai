# Tips for KISS Sorcar

> Practical tips for getting the highest-quality work from KISS Sorcar.

## Get the Highest Quality Work

- Add both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in the Settings panel.
- Add the following text to your prompt:

```text
Use claude-fable-5 model for all tasks including software development. Use gpt-5.6-sol (not
codex) for thorough review and debugging of the work done by the other model. Thoroughly check
if the other model has missed some code or has introduced bugs. No need to check if the models
exist.
```

## Voice Chat

If you have an `OPENAI_API_KEY`, with the **sorcar** wake word, KISS Sorcar starts behaving like a super-intelligent Alexa:

```text
Speak 'sorcar', your task ...
```

Click the mic button below the chat input box if it is grey and wait for it to start pulsing blue. For the Sorcar CLI, type `/voice` and Enter to activate voice interaction. Speak "sorcar" followed by your task; KISS Sorcar will run the task and tell you the results using its own voice. The voice interface distinguishes among different speakers.

## Novel Features: set_model and Steering-on-the-Fly

You can **instantaneously inject a user message** into a running agent and make the agent take the message into account in the rest of its execution. While an agent is running, you can also ask it to **dynamically change its model** for the rest of the execution. These unique KISS Sorcar features make multi-model reasoning and dynamic steering of hours-to-days tasks possible — model-routing intelligence can be expressed in a few sentences.

## Prompt Like the Developer

Look at the commit messages at <https://github.com/ksenxx/kiss_ai/commits/main/> — each commit message contains the prompt the developer used — and see what changed in the commit.

**Always write precise 1–6 sentence prompts.** Long prompts confuse models. **Do not plan ahead of time.** Let KISS Sorcar plan dynamically, which is always better than AI-written static plans.

## Remote Web/Mobile App

Go to the Settings panel and copy the URL at the top (it points to the latest cloudflared URL for the KISS Sorcar webapp). Send it to your mobile device (SMS, Slack, or email), open it in a browser, and enter your remote password (also set on the Settings page).

## CLI REPL

Just run:

```bash
sorcar
```

A powerful Claude-Code-style interface with skills, MCP, commands, streamed trajectories, and syntax-highlighted scrolling output.

## Docker

```bash
~/kiss_ai/sorcar-docker
```

Runs KISS Sorcar in a Docker container and exposes a VS Code interface in the host browser.

## Implementing a Software Feature

Add this sentence to your prompt:

```text
Reproduce the issue by writing a real end-to-end test. Then fix the issue.
```

## Fresh Knowledge

The internet has fresher knowledge than a frontier model. Add:

```text
Search the internet extensively.
```

## Git Merge Conflicts

```text
Can you check the following merge conflict message and help me fix it?
<<copy_paste_the_conflict_message_from_the_chat>>
```

## No Need for a Shell

Just type or speak your shell command in the chat input textbox.

## AI Discovery and Optimization

Use the ready-made prompts in [Sample Tasks](sample-tasks.md) — AI discovery, software/AI-system optimization, and GEPA prompt optimization are all driven by a single prompt.

## UI Pointers

- **Promptlets:** click "Inject Promptlet" below the chat input to insert a useful promptlet.
- **Dashboard:** the burger menu at the bottom left shows all agents with stats and filters — an agent dashboard.
- **Settings:** the settings button at the top right configures the remote app URL/password, budget limit per task, working directory, API keys, and a custom model endpoint.
