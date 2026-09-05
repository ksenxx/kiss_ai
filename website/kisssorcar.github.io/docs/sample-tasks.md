# Sample Tasks for KISS Sorcar

> Ready-to-use example prompts shipped with KISS Sorcar. Replace the `<<...>>` placeholders with your own values. In the VS Code extension these appear as welcome-screen chips. The bundled tasks live in [`src/kiss/SAMPLE_TASKS.md`](https://github.com/ksenxx/kiss_ai/blob/main/src/kiss/SAMPLE_TASKS.md); you can add your own at `~/.kiss/MY_TASK_TEMPLATES.md`.

## Code Understanding & Editing

```text
Can you show me the detailed step-by-step workflow of <<your algorithm or feature>>
```

```text
Can you change the step <<specify step>> as follows: <<whatever way you want to change>>
```

## Messaging

```text
Can you authenticate me with the <<workspace name>> workspace on Slack?
```

```text
Can you authenticate me with Gmail?
```

```text
Can you check my gmail every hour and ping me on slack if there is any important email that
needs my immediate attention?
```

```text
Can you authenticate me with the iMessage agent?
```

```text
Can you send "Hello from Sorcar!" to 1-800-772-1213?
```

## Fact-Checking & Security Review

```text
Can you read <<url>>, and thoroughly and precisely check for **wrong assumptions**, **cheating**,
**irreproducibility issues**, **fraud**, **potential for cheating in evaluation**, **AI Slop**,
and **security vulnerabilities**? Use the internet extensively and do not believe what people
say -- verify it yourself. Do not hesitate to download code and run it to validate results. For
security vulnerabilities, create a POC and test it. Create a report.
```

## AI Discovery

```text
Sorcar for AI Discovery: Can you AI discover the lightest and fastest AI model that will give
the best accuracy and recall on the data at <</path/to/data>> at the cheapest price? Use 'modal'
CLI to train your models on GPUs and evaluate if needed. The total budget for Modal.com is
$ 1,000. Do not STOP until accuracy/recall reaches 99% and price per query on the model is less
than $0.50. Create a report.
```

## Software Optimization

```text
Sorcar for Optimization: Can you run the command <<command>> and optimize it with respect to the
following metrics: <<speed, accuracy, recall, cost>>. Then use AI discovery to optimize. You can
add a diagnostic code that prints metrics, such as running time, at a finer granularity. Do not
forget to remove the diagnostic code after the optimization is complete. You MUST NOT STOP until
the metrics achieve the following values: <<give_concrete_values_for_metrics>>. Create a report.
```

## GEPA Prompt Optimization

```text
Sorcar GEPA Prompt Optimizer: Can you optimize a prompt for a ChatSorcarAgent of the
kiss-agent-framework Python library using the following GEPA algorithm on the data at
<<url_or_db_file_of_data>> using claude-fable-5? You can find the trajectory events of an agent
execution in ~/.kiss/sorcar.db after the agent has finished its execution. Split the dataset
into a 50% dev set and a 50% val set.

RUN_GEPA: Sample 100 data points from the val set and call it the sval set. Maintain a Pareto
frontier in the folder ./pareto, with a sub-folder for each node in the frontier. A node
contains a prompt file (prompt.md) and a JSON file, say score.json, containing the list of data
points (ids) from the sval set that were correctly predicted by the prompt. When you add a node
to the Pareto frontier, make sure that the list of correctly predicted data points is not a
subset or equal to an existing list of data points in some node in the frontier. If such a node
exists, do not add the new node. After adding a node, remove all nodes whose list of datapoints
is a subset or equal to the list of datapoints in the added node. Then run the following
algorithm.

1. pick a node from the Pareto frontier with probability 0.5
   a. sample a minibatch of 5 datapoints from the dev set
   b. run the agent with the prompt from the node on the minibatch
   c. If the agent incorrectly predicts for some datapoints, analyze and reflect on the
      trajectory events of the agent on those datapoints available at ~/.kiss/sorcar.db and
      propose a new prompt that will fix the mistakes made by the agent on datapoints
      incorrectly predicted
   d. if the agent predicts correctly on the minibatch, then evaluate it on the sval set and
      create the list of datapoints on which the agent with the new prompt predicts correctly.
   e. Add the new prompt and the list of datapoints to the Pareto frontier

1. Pick two nodes from the Pareto frontier randomly with the remaining probability.
   a. Sample a minibatch of 5 datapoints from the dev set
   b. Merge the prompts from the two nodes into a new prompt.
   c. If the agent predicts correctly on the minibatch with the new prompt, then evaluate it on
      the sval set and create the list of datapoints on which the agent with the new prompt
      predicts correctly.
   d. Add the new prompt and the list of datapoints to the Pareto frontier

1. Repeat steps 1 and 2 until there is no change in the prompt after 3 iterations.
   END_RUN_GEPA
   Repeat RUN_GEPA until there is no change in the prompt after 3 iterations.

At each step, keep track of the best prompt, which has the maximum number of successfully
predicted data points, in ./pareto/optimal.md. MAKE SURE THAT YOU DO NOT DO REWARD HACKING OR
CHEATING IN THE AGENT YOU ARE IMPLEMENTING TO FIT DATA. YOUR SOLUTION MUST GENERALIZE BEYOND THE
DATA PROVIDED. Use internet search extensively at every step. Do not worry about budget. Create
an HTML report with diagrams and illustrations in ./reports and open it in the user's default
browser. Do NOT STOP until you cannot improve the accuracy and recall after three consecutive
rollouts. Use gpt-5.6-sol model (not codex) for a thorough review of the work done at every
step by the other model.
```

See also [Prompt Tricks](prompt-tricks.md) and [Tips](tips.md).
