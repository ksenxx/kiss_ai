# Tip

## Update button in the settings

If the Update button in settings fails, run the full installation command again.  It will not delete your history.

```
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

# Tip

You can ask a question about the current task by prefixing the question with the command /ask. 

# Tip

{{PRODUCT_NAME}} supports commands prefixed with `/`. Type `/` in the chat textbox to see all available commands. To build your own command say `/xyz`, write a Sorcar Extension Agent (or a SEA) in a folder `/path/to/seas` and append the folder to the file `~/.kiss/SEAS.md`.  Coammnd `/xyz` will then be availble to {{PRODUCT_NAME}} UI.  More information on Sorcar Extension Agents (SEAs) can be found at [https://github.com/ksenxx/kiss_ai/blob/main/README.md](https://github.com/ksenxx/kiss_ai/blob/main/README.md).

# Tip

## Run `/autorouter` and `/bestrouter` as Models

`autorouter` and `bestrouter` are two bundled SEAs that decide which model runs your task. Each can be used in three ways:

1. **As a model.** Open the model picker and select `autorouter` or `bestrouter` in place of a model name. Every task you send from that tab then runs through the chosen router, so you never have to prefix your prompts.
2. **As a command.** Prefix a single task: `/autorouter add a --json flag to the export command and cover it with tests`. Run `/autorouter help` or `/bestrouter help` to print what each does.
3. **From a running agent or a Python script.** `run_agent(agent="autorouter", task="...")` or `run_agent(agent="bestrouter", task="...")`.

**What `autorouter` does.** Frontier models cost 40 to 100 times more per token than small models, while most agent tokens go to exploration, file reads, test output, and mechanical edits that a small model handles as well. `autorouter` splits the task into units of work that each have a mechanical acceptance check, classifies every unit into a `small`, `medium`, or `frontier` tier with the cheap non-generative `decide` tool, dispatches each unit to the cheapest model of that tier that your installation can run, verifies the result through the acceptance check, and escalates one tier up on a verified failure. Its objective is cost per accepted task, not cost per token. Every routing decision is appended to the ledger `~/.kiss/MODEL_DECISIONS.md`, and `/rsi7d all` refreshes the router's per-model cost, speed, and reliability evidence from your own task history.

**What `bestrouter` does.** It runs every task, including software development, on `claude-fable-5-1`, then dispatches `gpt-6-astra` through `run_parallel` for a read-only review and debugging pass over that work, on at most 75% of the task budget. Use it when quality matters more than cost; it needs both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set in Settings.

Any SEA whose `register_as_model()` returns `True` appears in the model picker the same way, so you can write your own router.

# Tip

## Sorcar Extension Agents (SEAs)

A **Sorcar Extension Agent (SEA)** is a plain Python file that defines a complete custom agent: its top-level `X()` functions — named after `sorcar.run()`'s parameters — compute the run's task prompt, system prompt, model, budget, tools, and safety hooks. Pass the file's path as `extension_agent_path` to `sorcar.run()` and the daemon imports it on every run. All third-party agents, such as the Slack and Gmail agents, are implemented in {{PRODUCT_NAME}} as SEAs. See the "Sorcar Extension Agents (SEAs)" section in `README.md` for a full example, and the detailed SEA guide at <https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/server/README.md>.

# Tip

**Recursive self improvement (RSI)** of a SEA is enabled based on past trajectories of the SEA.  Run `/rsi7d <SEA_NAME> [<SEA_NAME> ...]` to self improve those SEAs from their trajectories of the last 7 days, `/rsi7d all` for every indexed SEA, or `/rsi7d --seas-dir <folder> [<SEA_NAME> ...]` for the SEAs of your own folder (one listed in `~/.kiss/SEAS.md`, for instance): the tools mine and patch only the SEAs in that scope.  Free-form instructions may follow the scope.  `/rsi7d all` also covers KISS Sorcar itself (its system prompt `src/kiss/SYSTEM.md`, your `~/.kiss/SORCAR.md`, and its code): changes to Sorcar itself are made only after it asks you for permission in the chat, unless your task text already grants it, e.g. `/rsi7d all, you may modify KISS Sorcar itself without asking`.

# Tip

Run `/git_extract_knowledge <repo-name>` to create a memory/context based on the 
repository <rep-name>.  While running a task, {{PRODUCT_NAME}} can quickly lookup 
the memory about the repository to perform complex tasks on the repository or to
quickly answer questions about the repository.


# Tip

In VS Code, you can run {{PRODUCT_NAME}} in two modes: full editor mode, where the chats open as editor tabs, and non-editor mode, where the chats open in the sidebar.  You can switch between the two modes by selecting/deselecting the "Chat in the editor" option on the {{PRODUCT_NAME}} settings page.

# Tip 

{{PRODUCT_NAME}} now uses a quick task classifier to determine whether the task should run with git worktree mode and whether the task is complex or simple.  You can toggle the task classifier in the settings by selecting/deselecting the option "Classify tasks before running". With an OpenRouter API key the classifier asks the `~typesafe/jev-latest` decisions model (about 0.2 s and $0.00003 per task); deselect "Classify with Jev" to pin the LLM classifier, one non-agentic call on the run's own model (skipped for `cc/*` and `codex/*` models).

# Tip

## Prompt {{PRODUCT_NAME}} like the Developer of {{PRODUCT_NAME}}

**Always write precise less than 10 sentence prompts.** Long prompts confuse models. **Do not plan ahead of time.** Let {{PRODUCT_NAME}} plan dynamically, which is always better than AI-written static plans. The waterfall model doesn't work well in contemporary times.

See the commit messages at https://github.com/ksenxx/kiss_ai which include the prompts used by the developer of {{PRODUCT_NAME}}.

**No need to use generic skills for debugging, code review, etc.** Frontier models have been trained on those skills.

# Tip

## To get the Highest Quality Work from {{PRODUCT_NAME}}

- Add both ANTHROPIC_API_KEY and OPENAI_API_KEY in the Settings panel
- Add the following text to your prompt:

```
claude-fable-5-1 model be used for all tasks, including software development. Use gpt-6-astra (not codex) using `run_parallel` tool for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs. Use at most 75% of the task budget in gpt-6-astra for reviewing and debugging, and ask the model not to invent new problems. Use the model names literally without hallucinating new model names.
```

# Tip

## Novel Features: set_model and Steering-on-the-Fly

You can **instantaneously inject a user message** into a running agent and make the agent take the message into account in the rest of its execution.

Moreover, while an agent is running, you can ask it to **dynamically change its model** for the rest of the agent's execution.

These are unique features of {{PRODUCT_NAME}}. These two **IPs (intellectual properties)** make {{PRODUCT_NAME}} super powerful for multi-model reasoning and dynamic steering of tasks running for hours to days. You can describe model-routing intelligence in a few sentences.

# Tip

## You Can Now Have Voice Chat with {{PRODUCT_NAME}}

If you have an **OPENAI_API_KEY**, with the __Hey Sorcar__ wake word, {{PRODUCT_NAME}} starts behaving like a super-intelligent **Alexa**.

```
Speak 'Hey Sorcar', your task ...
```

Click the **mic** button below the chat input box if it is grey and wait for it to start pulsing blue. Speak 'Hey Sorcar' followed by your task, and {{PRODUCT_NAME}} will automatically run the task and tell you the results using its own voice. The voice interface distinguishes among different speakers.

You can also steer the agent's execution and ask for status when an agent is running using voice.

# Tip

## To Use the {{PRODUCT_NAME}} Remote Web/Mobile App

Go to the Settings panel and copy the URL at the top. This URL contains a message showing the latest cloudflared URL where you can find the {{PRODUCT_NAME}} web app. Send the URL from the Settings page to your mobile device. Also view or set the remote password on the Settings page. You can SMS, Slack, or email the URL to the mobile device.

Open the URL in a browser on the mobile device and enter your remote password. You will see your familiar Codex-like chat interface.

# Tip

## To Run Tasks from Python Scripts

Any Python process can launch a task on the running {{PRODUCT_NAME}} daemon and block until it finishes:

```python
from kiss.server import sorcar

result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
print(result.text, result.success, result.cost)

# Continue the same chat with the prior task as context:
sorcar.run("Now fix the typos you found", chat_id=result.chat_id)
```

You can also pass a Python file of extra tools via `tools="/path/to/my_tools.py"`.

# Tip

## To Run {{PRODUCT_NAME}} in a Docker Container

Just run:

```bash
sorcar-docker
```

It runs {{PRODUCT_NAME}} in a Docker container and exposes a VS Code interface in the host machine's browser.

# Tip

## To Run {{PRODUCT_NAME}} on a server via ssh

Just run:

```bash
rsorcar username@ip_address
```

# Tip

## No Need to Use a Shell

Just type or speak your shell command prefixed with the command `/sh` in 
the chat input textbox.

# Tip

## AI Discovery and Auto Research

All you need to do is use a variant of the following prompt with {{PRODUCT_NAME}}:

```
Can you AI discover the lightest and fastest AI model that will give >95% accuracy and recall on the data at \<</path/to/data>> at the cost of $0.25 per query?  Use 'modal' CLI to train your models on GPUs and evaluate if needed. Your total budget for Modal.com is $1,000. Experiment with a smaller data subset and fewer model parameters to run experiments quickly, then extrapolate. Do not STOP until you reach the goals. Create a detailed report.
```

# Tip

## AI Optimization of Software and AI Systems

All you need to do is use a variant of the following prompt with {{PRODUCT_NAME}}:

```
Can you run the command \<<command>> in the background and monitor its output in real time to optimize the code at \<<folder_name_or_url>> for the following metrics: \<<speed, accuracy, recall, cost>>. Then use AI discovery to optimize.  You may add diagnostic code that prints metrics, such as running time, at a finer granularity.  Don't forget to remove the diagnostic code after optimization is complete. You MUST NOT STOP until the metrics achieve the following values: \<<give_concrete_values_for_metrics>>. Create a report.
```

# Tip

## Useful Promptlets

Click on the "Inject Promptlet" button below the chat input textbox to insert a useful promptlet into your prompt.

# Tip

## Agent Dashboard and History

Click the burger menu button in the bottom-left corner to see all agents in {{PRODUCT_NAME}}, along with various stats and filters, including running and failed tasks. It is an agent dashboard.

# Tip

## Settings

Click on the settings button in the "..." menu. Use the Settings interface to get the URL for the remote web/mobile app, set the remote web app access password, set the budget limit per task, set the working directory, and set various API keys and a custom model endpoint.
