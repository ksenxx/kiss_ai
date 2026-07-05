# Tip

## For the highest quality work from KISS Sorcar
- Add both ANTHROPIC_API_KEY and OPENAI_API_KEY in the Settings panel 
- Add the following text to your prompt:

```
Use claude-fable-5 model for all tasks including coding, bug fixing, and test creation. Use gpt-5.5-xhigh (not codex) for thorough review and debugging of the work done by the other model. Check if the other model has missed some code or has introduced bugs. No need to check if the models exist.
```

# Tip

## You Can Now have Coversation with KISS Sorcar via Voice

```
Speak 'sorcar', your task ...
```

Make sure that the microphone button below the chat input box is pulsing blue.  Speak 'sorcar' followed by your task, KISS Sorcar will automatically run the task and tell you the result using its own voice. You need the OPENAI_API_KEY to enable this feature.  The voice interface distinguishes among different speakers.

# Tip

## Prompt KISS Sorcar like the Developer of KISS Sorcar 

Look at the commit messages at [https://github.com/ksenxx/kiss_ai/commits/main/], find the prompt that the developer used for that commit in a commit message, and what changed in the commit.  This will help you get started with KISS Sorcar on any tasks like a pro.

# Tip


## To Use the Remote Web/Mobile App

Go to the Settings panel and copy the URL at the top.  This URL contains a message showing the latest cloudfared URL where you can find the KISS Sorcar webapp.  Send the URL from the Settings page to your mobile device.  Also see/set the remote password on the Settings page.  You can possibly SMS, slack, or email the URL to the mobile device.  Open the URL in the mobile device in a browser and enter your remote password.  You will see your faminiliar KISS Sorcar chat interface from VS Code.

# Tip 

## If You are Implementing a Software Feature

Definitely add the following sentence to the prompt of the KISS Sorcar:  Reproduce the issue by writing an end-to-end real test. Then fix the issue.

# Tip

## Internet has Fresher Knowledge than a Frontier Model

Add the following sentence to your prompt to make KISS Sorcar search the internet extensively for information: Search the internet extensively.

# Tip

## If You Get git Merge Conflict

Then run the following task:  Can you check the following message on merge conflict and help me fix it? <<copy_paste_theConflict_message_from_the_chat>>

# Tip

## No need to use shell

Just type or speak your shell command in the chat input textbox.

# Tip 

## AI Discover and Auto Research

All you need to do is to use a variant of the followng prompt:

```
Can you discover the lightest and fastest AI model that will give the best accuracy and recall on the data at \<</path/to/data>> at the cheapest price? Analyze the data and search the internet extensively to propose the first few models. Implement and experiment with each of your proposals. Note down the ideas you used to optimize the accuracy/recall and speed/cost metrics achieved in a file, so that you can use the file to not repeat ideas that have already been tried and/or failed. You can also use the file to combine ideas that have been successful in the past. Separate 20% of the data for evals, and your discovery strategy must not look at the evals data. Use 'modal' CLI to train your models on GPUs and evaluate if needed. Total budget for Modal.com is $1000. Experiment with a smaller subset of data and fewer parameters in a model to do experiments quickly, and then extrapolate. Use internet search extensively at every step. MAKE SURE THAT YOU DO NOT DO REWARD HACKING OR CHEATING IN THE MODELS OR AGENTS YOU ARE IMPLEMENTING TO FIT DATA. YOUR SOLUTION MUST GENERALIZE BEYOND THE DATA PROVIDED. Do not STOP until accuracy/recall reaches 95% on evals and you can process each query in less than 600 seconds and under 50 USD per query amortized over all queries. Create an html report with diagrams and illustrations in ./reports and open it in the user's default browser?
```

# Tip

## AI Optimization of Software and AI Systems

All you need to do is to use a variant of the followng prompt:

```
Can you run the command \<<command>> in the background and monitor its output in real time to optimize the code at \<<folder_name_or_url>> with respect to the following metrics: \<<speed,accuracy,recall,cost>>. You can add diagnostic code which will print the metrics, such as running time at a finer level of granularity. Check for opportunities to optimize the code on the basis of the metrics information. If you discover any opportunities to optimize the metric based on the code, logs, events, and the command output, optimize the code and run the command again. Note down the ideas you used to optimize the code and the metric you achieved in a file, so that you can use the file to not repeat ideas that have already been tried and failed. You can also use the file to combine ideas that have been successful in the past. Repeat the process. Do not forget to remove the diagnostic code after the optimization is complete. You MUST NOT STOP until the metrics achieve the following values:\<<give_concrete_values_for_metrics>>. Use the internet extensively to get new ideas for optimization. Create an html report with diagrams and illustrations in ./reports and open it in the user's default browser?
```

