# Tips for KISS Sorcar

> Practical tips for getting the highest-quality work from KISS Sorcar. These mirror the built-in tips shipped in [`src/kiss/TIPS.md`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/TIPS.md).

## Prompt KISS Sorcar Like the Developer

Look at the commit messages at <https://github.com/ksenxx/kiss_ai/commits/main/> — each commit message contains the prompt the developer used, and the diff shows exactly what changed. This will help you get started with KISS Sorcar on any task like a pro.

**Always write precise 1–6 sentence prompts.** Long prompts confuse models. **Do not plan ahead of time.** Let KISS Sorcar plan dynamically — that is always better than AI-written static plans. The waterfall model does not work that well in contemporary times.

**No need to use generic skills for debugging, code review, etc.** Frontier models have already been trained on those skills.

## Get the Highest Quality Work

- Add both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in the Settings panel.
- Add the following text to your prompt:

```text
Use 'claude-fable-5' for all tasks, including software development. Use 'gpt-5.6-sol' (not
codex) for a thorough read-only review and debugging of the other model's work. Thoroughly
check whether the other model has missed any code or wiring or introduced any bugs. Use at
most 20% of task budget in gpt-5.6-sol for reviewing and debugging. Use the model names
literally without hallucinating new model names.
```

## Ask for Task Status Anytime

If you want to know the status of a running task, open a new chat and simply ask KISS Sorcar what is the status of the task doing …

## Novel Features: `set_model` and Steering-on-the-Fly

You can **instantaneously inject a user message** into a running agent and make the agent take that message into account for the rest of its execution.

While an agent is running, you can also ask it to **dynamically change its model** for the remainder of the run.

These are unique features of KISS Sorcar. Together, these two **IPs (intellectual properties)** make KISS Sorcar exceptionally powerful for multi-model reasoning and dynamic steering of tasks that run for hours to days — model-routing intelligence can be expressed in just a few sentences.

## Use Optimized Multi-Model Routing to Save Cost or Improve Quality

Add the following text to your prompt:

```text
If ./ROUTING.md exists, use the instructions in the file for model routing. Otherwise, use
the best model from ~/.kiss/MODEL_INFO.json for various subtasks. Search the internet
extensively to figure out which model is best yet cheap for each subtask. Here are some
hints, but the internet has better knowledge: claude-opus-5 — best for SWE work;
gpt-5.6-sol — best for reviewing; openrouter/z-ai/glm-5.2 — for SWE tasks when budget is
low; and gpt-5.6-luna for review when budget is low. Irrespective of whether ./ROUTING.md
exists or not, after the task completes, based on your experience in completing the task,
create or update the model routing strategy (as text) in ./ROUTING.md that reduces token
cost while not degrading the quality of the work.
```

## Voice Chat with KISS Sorcar

If you have an `OPENAI_API_KEY`, with the **sorcar** wake word, KISS Sorcar starts behaving like a super-intelligent **Alexa**.

```text
Speak 'sorcar', your task ...
```

Click the mic button below the chat input box if it is grey and wait for it to start pulsing blue. Speak "sorcar" followed by your task; KISS Sorcar will run the task and tell you the results using its own voice. The voice interface distinguishes among different speakers.

You can also **steer the agent's execution and ask for status** while an agent is running — using voice.

## Use the KISS Sorcar Remote Web/Mobile App

Go to the Settings panel and copy the URL at the top. This URL points to the latest cloudflared URL where the KISS Sorcar web app is exposed. Send this URL from the Settings page to your mobile device — SMS, Slack, or email work equally well. Also see/set the remote password on the Settings page.

Open the URL in a browser on the mobile device and enter your remote password. You will see a familiar Codex-like chat interface.

**Always write precise 1–6 sentence prompts.** Long prompts confuse models. **Do not plan ahead of time.** Let KISS Sorcar plan dynamically, which is always better than AI-written static plans.

## Remote Web/Mobile App

Go to the Settings panel and copy the URL at the top (it points to the latest cloudflared URL for the KISS Sorcar webapp). Send it to your mobile device (SMS, Slack, or email), open it in a browser, and enter your remote password (also set on the Settings page).

## Run Tasks from Python Scripts

Any Python process can launch a task on the running KISS Sorcar daemon and block until it finishes:

```python
from kiss.server import sorcar

result = sorcar.run("Summarize README.md", work_dir="/path/to/repo")
print(result.text, result.success, result.cost)
```

You can also pass a Python file of extra tools via `tools="/path/to/my_tools.py"`, and continue a chat with `chat_id=result.chat_id`.

## Run KISS Sorcar in a Docker Container

Just run:

```bash
~/kiss_ai/sorcar-docker
```

It runs KISS Sorcar in a Docker container and exposes a VS Code interface in the host machine's browser.

## Implementing a Software Feature

Definitely add the following sentence to your KISS Sorcar prompt:

```text
Reproduce the issue by writing real end-to-end tests with 100% coverage. Then fix the issue.
```

## Fix a git Merge Conflict

Run the following task:

```text
Can you check the following merge conflict message and help me fix it?
<<copy_paste_the_conflict_message_from_the_chat>>
```

## No Need to Use a Shell

Just type or speak your shell command in the chat input textbox.

## AI Discovery and Auto Research

Use a variant of the following prompt with KISS Sorcar:

```text
Can you AI discover the lightest and fastest AI model that will give >95% accuracy and
recall on the data at <</path/to/data>> at the cost of $0.25 per query? Use 'modal' CLI
to train your models on GPUs and evaluate if needed. The total budget for Modal.com is
$1,000. Experiment with a smaller subset of data and fewer model parameters to run
experiments quickly, then extrapolate. Do not STOP until you reach the goals. Create a
detailed report.
```

## AI Optimization of Software and AI Systems

Use a variant of the following prompt with KISS Sorcar:

```text
Can you run the command <<command>> in the background and monitor its output in real time
to optimize the code at <<folder_name_or_url>> with respect to the following metrics:
<<speed, accuracy, recall, cost>>. Then use AI discovery to optimize. You can add
diagnostic code that prints metrics, such as running time, at a finer granularity. Do not
forget to remove the diagnostic code after the optimization is complete. You MUST NOT
STOP until the metrics achieve the following values: <<give_concrete_values_for_metrics>>.
Create a report.
```

## More Prompt Examples for Connecting to Slack, SMS, Gmail…

See them on the welcome page when a new chat is created. Click on them to copy them to the chat input textbox.

## Useful Promptlets

Click the **"Inject Promptlet"** button below the chat input textbox to insert a useful promptlet into your prompt.

## Agent Dashboard and History

Click the burger menu button in the bottom-left corner to see all agents in KISS Sorcar, along with various stats and filters, including running and failed tasks. It is an agent dashboard.

## Settings

Click the **Settings** button at the top-right corner. You can get the URL for the remote web/mobile app, set the remote web app access password, set the budget limit per task, set the working directory, and set various API keys and a custom model endpoint — all from the Settings interface.
