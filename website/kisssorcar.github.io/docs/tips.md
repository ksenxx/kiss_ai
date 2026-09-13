# Tips for KISS Sorcar

> Practical tips for getting the highest-quality work from KISS Sorcar. These mirror the built-in tips shipped in [`src/kiss/TIPS.md`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/TIPS.md).

## Update Button in the Settings

If the Update button in settings fails, run the full installation command again. It will not delete your history.

```bash
curl -fsSL https://raw.githubusercontent.com/ksenxx/kiss_ai/main/scripts/install.sh | bash
```

## Chat in the Editor or in the Sidebar

In VS Code, you can run KISS Sorcar in two modes: full editor mode, where the chats open as editor tabs, and non-editor mode, where the chats open in the sidebar. You can switch between the two modes by selecting/deselecting the "Chat in the editor" option on the KISS Sorcar settings page.

## Task Classifier

KISS Sorcar now uses a quick task classifier to determine whether the task should run with git worktree mode and whether the task is complex or simple. You can toggle the task classifier in the settings by selecting/deselecting the option "Classify tasks before running".

## Sorcar Extension Agents (SEAs)

A **Sorcar Extension Agent (SEA)** is a plain Python file that defines a complete custom agent: its top-level `X()` functions — named after `sorcar.run()`'s parameters — compute the run's task prompt, system prompt, model, budget, tools, and safety hooks. Pass the file's path as `extension_agent_path` to `sorcar.run()` and the daemon imports it on every run. All third-party agents, such as the Slack and Gmail agents, are implemented in KISS Sorcar as SEAs. See the "Sorcar Extension Agents (SEAs)" section in the [README](https://github.com/ksenxx/kiss_ai#sorcar-extension-agents-seas) for a full example, and the detailed SEA guide at [`src/kiss/server/README.md`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/server/README.md).

## Prompt KISS Sorcar Like the Developer of KISS Sorcar

**Always write precise less than 10 sentence prompts.** Long prompts confuse models. **Do not plan ahead of time.** Let KISS Sorcar plan dynamically, which is always better than AI-written static plans. The waterfall model doesn't work well in contemporary times.

See the commit messages at <https://github.com/ksenxx/kiss_ai> which include the prompts used by the developer of KISS Sorcar.

**No need to use generic skills for debugging, code review, etc.** Frontier models have been trained on those skills.

## Get the Highest Quality Work

- Add both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in the Settings panel.
- Add the following text to your prompt:

```text
Use 'claude-fable-5-1' model for all tasks, including software development. Use 'gpt-5.6-sol'
(not codex) using `run_parallel` tool for a thorough read-only review and debugging of the
other model's work. Thoroughly check whether the other model has missed any code or wiring
or introduced any bugs. Use at most 50% of the task budget in gpt-5.6-sol for reviewing and
debugging, and ask the model not to invent new problems. Use model names literally; don't
hallucinate new model names.
```

## Ask for Task Status Anytime

To get the status of a task, open a new chat and ask KISS Sorcar what the status is of the task being done…

## Novel Features: `set_model` and Steering-on-the-Fly

You can **instantaneously inject a user message** into a running agent and make the agent take the message into account in the rest of its execution.

Moreover, while an agent is running, you can ask it to **dynamically change its model** for the rest of the agent's execution.

These are unique features of KISS Sorcar. These two **IPs (intellectual properties)** make KISS Sorcar super powerful for multi-model reasoning and dynamic steering of tasks running for hours to days. You can describe model-routing intelligence in a few sentences.

## Voice Chat with KISS Sorcar

If you have an **`OPENAI_API_KEY`**, with the **sorcar** wake word, KISS Sorcar starts behaving like a super-intelligent **Alexa**.

```text
Speak 'sorcar', your task ...
```

Click the **mic** button below the chat input box if it is grey and wait for it to start pulsing blue. Speak "sorcar" followed by your task, and KISS Sorcar will automatically run the task and tell you the results using its own voice. The voice interface distinguishes among different speakers.

You can also steer the agent's execution and ask for status when an agent is running using voice.

## Use the KISS Sorcar Remote Web/Mobile App

Go to the Settings panel and copy the URL at the top. This URL contains a message showing the latest cloudflared URL where you can find the KISS Sorcar web app. Send the URL from the Settings page to your mobile device. Also view or set the remote password on the Settings page. You can SMS, Slack, or email the URL to the mobile device.

Open the URL in a browser on the mobile device and enter your remote password. You will see your familiar Codex-like chat interface.

## Run Tasks from Python Scripts

Any Python process can launch a task on the running KISS Sorcar daemon and block until it finishes:

```python
from kiss.server import sorcar

result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
print(result.text, result.success, result.cost)

# Continue the same chat with the prior task as context:
sorcar.run("Now fix the typos you found", chat_id=result.chat_id)
```

You can also pass a Python file of extra tools via `tools="/path/to/my_tools.py"`.

## Run KISS Sorcar in a Docker Container

Just run:

```bash
sorcar-docker
```

It runs KISS Sorcar in a Docker container and exposes a VS Code interface in the host machine's browser.

## Run KISS Sorcar on a Server via SSH

Just run:

```bash
rsorcar username@ip_address
```

## Fix a git Merge Conflict

Then run the following task:

```text
Can you check the following merge conflict message and help me fix it? <<copy_paste_the_conflict_message_from_the_chat>>
```

## No Need to Use a Shell

Just type or speak your shell command in the chat input textbox.

## AI Discovery and Auto Research

All you need to do is use a variant of the following prompt with KISS Sorcar:

```text
Can you AI discover the lightest and fastest AI model that will give >95% accuracy and
recall on the data at <</path/to/data>> at the cost of $0.25 per query? Use 'modal' CLI to
train your models on GPUs and evaluate if needed. Your total budget for Modal.com is $1,000.
Experiment with a smaller data subset and fewer model parameters to run experiments quickly,
then extrapolate. Do not STOP until you reach the goals. Create a detailed report.
```

## AI Optimization of Software and AI Systems

All you need to do is use a variant of the following prompt with KISS Sorcar:

```text
Can you run the command <<command>> in the background and monitor its output in real time
to optimize the code at <<folder_name_or_url>> for the following metrics: <<speed, accuracy,
recall, cost>>. Then use AI discovery to optimize. You may add diagnostic code that prints
metrics, such as running time, at a finer granularity. Don't forget to remove the diagnostic
code after optimization is complete. You MUST NOT STOP until the metrics achieve the
following values: <<give_concrete_values_for_metrics>>. Create a report.
```

## More Prompt Examples for Connecting to Slack, SMS, Gmail…

See them on the welcome page when you create a new chat. Click on them to copy them to the chat input textbox.

## Useful Promptlets

Click on the "Inject Promptlet" button below the chat input textbox to insert a useful promptlet into your prompt.

## Agent Dashboard and History

Click the burger menu button in the bottom-left corner to see all agents in KISS Sorcar, along with various stats and filters, including running and failed tasks. It is an agent dashboard.

## Settings

Click on the settings button in the "..." menu. Use the Settings interface to get the URL for the remote web/mobile app, set the remote web app access password, set the budget limit per task, set the working directory, and set various API keys and a custom model endpoint.

## Use Optimized Multi-Model Routing to Save Cost or Improve Quality

Add the following text to your prompt:

```text
If ./ROUTING.md exists, use the instructions in the file for model routing. Otherwise, use
the best model from ~/.kiss/MODEL_INFO.json for various subtasks. Search the internet
extensively to figure out which model is best yet cheap for each subtask. Here are some
hints, but the internet has better knowledge: claude-fable-5 and
openrouter/moonshotai/kimi-k3 — best for SWE work; gpt-5.6-sol — best for reviewing;
openrouter/qwen/qwen3.8-max, openrouter/x-ai/grok-4.6, openrouter/z-ai/glm-5.3,
openrouter/deepseek/deepseek-v4-pro-0813 — for SWE tasks when budget is low; and
gpt-5.6-luna and openrouter/deepseek/deepseek-v4-pro-0813 for review when budget is low.
Irrespective of whether ./ROUTING.md exists or not, after the task completes, based on your
experience in completing the task, create or update the model routing strategy (as text) in
./ROUTING.md that reduces token cost while not degrading the quality of the work.
```
