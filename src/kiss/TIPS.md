# Tip

## To get the Highest Quality Work from KISS Sorcar

- Add both ANTHROPIC_API_KEY and OPENAI_API_KEY in the Settings panel
- Add the following text to your prompt:

```
Use 'claude-fable-5 model' for all tasks, including software development. Use 'gpt-5.6-sol' (not codex) for a thorough read-only review and debugging of the other model's work. Thoroughly check whether the other model has missed any code or wiring or introduced any bugs.  Use at most 20% of task budget in gpt-5.6-sol for reviewing and debugging. Use the model names literally without hallucinating new model names.
```

# Tip

## Novel Features: set_model and Steering-on-the-Fly

You can **instantaneously inject a user message** into a running agent and make the agent take the message into account in the rest of its execution.

Moreover, while an agent is running, you can ask it to **dynamically change its model** for the rest of the execution of the agent.

These are unique features of KISS Sorcar. These two **IPs (intellectual properties)** make KISS Sorcar super powerful for multi-model reasoning and dynamic steering of tasks running for hours to days. Model routing intelligence can be expressed in a few sentences.

# Tip

## Prompt KISS Sorcar like the Developer of KISS Sorcar

Please look at the commit messages at [https://github.com/ksenxx/kiss_ai/commits/main/], find the prompt that the developer used for that commit in a commit message, and see what changed in the commit. This will help you get started with KISS Sorcar on any task like a pro.

**Always write precise 1-6 sentence prompts.** Long prompts confuse models. **Do not plan ahead of time.** Let KISS Sorcar plan dynamically, which is always better than AI-written static plans. The waterfall model does not work that well in contemporary times.

# Tip

## Use Optimized Multi-Model Routing to Save Cost or Improve Quality

**Add the following text to your prompt:**

```
If ./ROUTING.md exists, use the instructions in the file for model routing.  Otherwise,
Use the best model from ~/.kiss/MODEL_INFO.json for various subtasks. Search the internet extensively to figure out which model is best yet cheap for each sub-task. Here are some hints, but the internet has better knowledge: claude-fable-5 — best for SWE work, gpt-5.6-sol — best for reviewing, and openrouter/z-ai/glm-5.2 — for SWE tasks when budget is low, and gpt-5.5 for review when budget is low.  Irrespective of whether ./ROUTING.md exists or not, after the task completes, based on your experience in completing the task, create or update the model routing strategy (as text) in ./ROUTING.md that reduces token cost while not degrading the quality of the work.
```

# Tip

## You Can Now Have Voice Chat with KISS Sorcar

If you have an **OPENAI_API_KEY**, with the __sorcar__ wake word, KISS Sorcar starts behaving like a super-intelligent **Alexa**.

```
Speak 'sorcar', your task ...
```

Click the **mic** button below the chat input box if it is grey and wait for it to start pulsing blue. For Sorcar CLI, type /voice and enter to activate voice interaction. Speak 'sorcar' followed by your task, and KISS Sorcar will automatically run the task and tell you the results using its own voice. The voice interface distinguishes among different speakers.

You can also steer the agent's execution and ask for status when an agent is running using voice.

# Tip

## To Use the KISS Sorcar Remote Web/Mobile App

Go to the Settings panel and copy the URL at the top. This URL contains a message showing the latest cloudflared URL where you can find the KISS Sorcar webapp. Send the URL from the Settings page to your mobile device. Also see/set the remote password on the Settings page. You can SMS, Slack, or email the URL to the mobile device.

Open the URL in a browser on the mobile device and enter your remote password. You will see your familiar Codex-like chat interface.

# Tip

## To Use the KISS Sorcar CLI REPL Interface

Just run:

```bash
sorcar
```

It has a powerful Claude Code-style interface. It supports skills, MCP, commands, etc. The trajectories are streamed, and the output scrolls while being syntax-highlighted.

# Tip

## To Run KISS Sorcar in a Docker Container

Just run:

```bash
~/kiss_ai/sorcar-docker
```

It runs KISS Sorcar in a Docker container and exposes a VS Code interface in the host machine's browser.

# Tip

## If You Are Implementing a Software Feature

Definitely add the following sentence to your KISS Sorcar prompt:

```
Reproduce the issue by writing real end-to-end tests with 100% coverage. Then fix the issue.
```

# Tip

## The Internet Has Fresher Knowledge than a Frontier Model

Add the following sentence to your prompt to make KISS Sorcar search the internet extensively for information:

```
Search the internet extensively.
```

# Tip

## If You Get a git Merge Conflict

Then run the following task:

```
Can you check the following merge conflict message and help me fix it? <<copy_paste_the_conflict_message_from_the_chat>>
```

# Tip

## No Need to Use a Shell

Just type or speak your shell command in the chat input textbox.

# Tip

## AI Discovery and Auto Research

All you need to do is use a variant of the following prompt with KISS Sorcar:

```
Can you AI discover the lightest and fastest AI model that will give the best accuracy and recall on the data at \<</path/to/data>> at the cheapest price? Separate 20% of the data for evals, and your discovery strategy must not look at the evals data. Use 'modal' CLI to train your models on GPUs and evaluate if needed. The total budget for Modal.com is $ 1,000. Experiment with a smaller subset of data and fewer model parameters to run experiments quickly, then extrapolate. Do not STOP until accuracy/recall reaches 95% on evals and you can process each query in less than 600 seconds and under 50 USD per query, amortized over all queries. Create an HTML report with diagrams and illustrations (which does not look AI-generated) in ./reports and open it in the user's default browser?
```

# Tip

## AI Optimization of Software and AI Systems

All you need to do is use a variant of the following prompt with KISS Sorcar:

```
Can you run the command \<<command>> in the background and monitor its output in real time to optimize the code at \<<folder_name_or_url>> with respect to the following metrics: \<<speed, accuracy, recall, cost>>. Then use AI discovery to optimize.  You can add a diagnostic code that prints metrics, such as running time, at a finer granularity.  Do not forget to remove the diagnostic code after the optimization is complete. You MUST NOT STOP until the metrics achieve the following values: \<<give_concrete_values_for_metrics>>. Create an HTML report with diagrams and illustrations (that do not look AI-generated) in ./reports, and open it in the user's default browser.
```

# Tip

## More Prompt Examples for Connecting to Slack, SMS, Gmail ...

See them on the welcome page when a new chat is created. Click on them to copy them to the chat input textbox.

# Tip

## Useful Promptlets

Click on the "Inject Promptlet" button below the chat input textbox to insert a useful promptlet into your prompt.

# Tip

## Agent Dashboard and History

Click the burger menu button in the bottom-left corner to see all agents in KISS Sorcar, along with various stats and filters, including running and failed tasks. It is an agent dashboard.

# Tip

## Settings

Click on the settings button at the top right corner. You can get the URL for the remote web/mobile app, set the remote web app access password, set the budget limit per task, set the working directory, and set various API keys and a custom model endpoint using the Settings interface.
