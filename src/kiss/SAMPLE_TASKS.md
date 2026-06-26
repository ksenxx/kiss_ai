## Task

Can you show me the detailed step-by-step workflow of \<<your algorithm or feature>>

## Task

Can you change the step \<<specify step>> as follows: \<<whatever way you want to change>>

## Task

can you authenticate me with the \<<workspace name>> workspace on Slack using the Slack agent?

## Task

Can you create a cron job with a name prefixed with "kiss-" which will check every 3 seconds if there are latest unanswered messages from /\<<user name>> in the channel sorcar using the Slack agent, then it will run the messages as tasks one-by-one in the order of arrival and respond with the result suitably formatted for Slack.

## Task

can you authenticate me with the iMessage agent?

## Task

can you send "Hello" to 1-800-772-1213?

## Task

can you authenticate me with Gmail using the Gmail agent? Use the user's default browser to ask the user to log in and to get the authentication token.

## Task

Can you read \<<url>>, and thoroughly and precisely check for **wrong assumptions**, **cheating**, **irreproducibility issues**, **fraud**, **potential for cheating in evaluation**, **AI Slop**, and **security vulnerabilities**? Use internet search extensively and do not believe what people say -- verify them yourself. Do not hesitate to download code and run it to validate results. For security vulnerabilities, create a POC and test it. Generate an html report in ./sorcar_reported_frauds/ and open in the user's default browser. Thoroughly fact check everything you claim in the report.

## Task

Sorcar for AI Discovery: Can you discover the lightest and fastest AI model that will give the best accuracy and recall on the data at \<</path/to/data>> at the cheapest price? Analyze the data and search the internet extensively to propose the first few models. Implement and experiment with each of your proposals. Note down the ideas you used to optimize the accuracy/recall and speed/cost metrics achieved in a file, so that you can use the file to not repeat ideas that have already been tried and/or failed. You can also use the file to combine ideas that have been successful in the past. Separate 20% of the data for evals, and your discovery strategy must not look at the evals data. Use 'lambda' CLI to train your models on GPUs and evaluate if needed. Total budget for Lambda Labs is $1000. Experiment with a smaller subset of data and fewer parameters in a model to do experiments quickly, and then extrapolate. Use internet search extensively at every step. MAKE SURE THAT YOU DO NOT DO REWARD HACKING OR CHEATING IN THE MODELS OR AGENTS YOU ARE IMPLEMENTING TO FIT DATA. YOUR SOLUTION MUST GENERALIZE BEYOND THE DATA PROVIDED. Do not STOP until accuracy/recall reaches 95% on evals and you can process each query in less than 600 seconds and under 50 USD per query amortized over all queries. Create an html report with diagrams and illustrations in ./reports and open it in the user's default browser?

## Task

Sorcar for Optimization: Can you run the command \<<command>> in the background and monitor its output in real time to optimize the code at \<<folder_name_or_url>> with respect to the following metrics: \<<speed,accuracy,recall,cost>>. You can add diagnostic code which will print the metrics, such as running time at a finer level of granularity. Check for opportunities to optimize the code on the basis of the metrics information. If you discover any opportunities to optimize the metric based on the code, logs, events, and the command output, optimize the code and run the command again. Note down the ideas you used to optimize the code and the metric you achieved in a file, so that you can use the file to not repeat ideas that have already been tried and failed. You can also use the file to combine ideas that have been successful in the past. Repeat the process. Do not forget to remove the diagnostic code after the optimization is complete. You MUST NOT STOP until the metrics achieve the following values:\<<give_concrete_values_for_metrics>>. Use the internet extensively to get new ideas for optimization. Create an html report with diagrams and illustrations in ./reports and open it in the user's default browser?

## Task

Sorcar GEPA Prompt Optimizer: Can you optimize a prompt for a ChatSorcarAgent of the kiss-agent-framework Python library using the following GEPA algorithm on the data at \<<url_or_db_file_of_data>> using claude-opus-4-7? You can find the trajectory events of an agent execution in ~/.kiss/sorcar.db after the agent has finished its execution. Split the dataset into 50% dev set and 50% val set.

RUN_GEPA: Sample 100 data points from the val set and call it sval set. Maintain a pareto frontier in the folder ./pareto where we have a sub-folder for each node in the frontier. A node contains a prompt file (prompt.md) and a json file, say score.json, containing the list of sessions (ids) from the val set that were correctly predicted with the prompt. When you add a node to the pareto frontier make sure that the list of correctly predicted sessions is not a subset or equal to an existing list of sessions in some node in the frontier. If such a node exists, do not add the new node. After adding a node, remove all nodes whose list of sessions is a subset or equal to the list of sessions in the added node. Then run the following algorithm.

1. pick a node from the pareto frontier with probability 0.5
   a. sample a minibatch of 5 sessions from the dev set
   b. run the agent with the prompt from the node on the minibatch
   c. if the agent incorrectly predicts for some sessions, analyze and reflect on the trajectory events of the agent on those sessions available at ~/.kiss/sorcar.db and propose a new prompt which will fix the mistakes made by the agent on sessions incorrectly predicted.
   d. if the agent predicts correctly on the minibatch, then evaluate it on the sval set and create the list of sessions on which the agent with the new prompt predicts correctly.
   e. Add the new prompt and the list of sessions to the pareto frontier

1. pick two nodes from the pareto frontier randomly with the remaining probability.
   a. sample a minibatch of 5 sessions from the dev set
   b. merge the prompts from the two nodes into a new prompt.
   c. if the agent predicts correctly on the minibatch with the new prompt, then evaluate it on the sval set and create the list of sessions on which the agent with the new prompt predicts correctly.
   d. Add the new prompt and the list of sessions to the pareto frontier

1. Repeat steps 1 and 2 until there is no change in the prompt after 3 iterations.
   END_RUN_GEPA
   Repeat RUN_GEPA until there is no change in the prompt after 3 iterations.

In each step, keep track of the best prompt which has the maximum number of successfully predicted sessions in ./pareto/optimal.md. In the prompt always add the sentence "Use internet search extensively at every step." MAKE SURE THAT YOU DO NOT DO REWARD HACKING OR CHEATING IN THE AGENT YOU ARE IMPLEMENTING TO FIT DATA. YOUR SOLUTION MUST GENERALIZE BEYOND THE DATA PROVIDED. Use internet search extensively at every step. Do not worry about budget. Create an html report with diagrams and illustrations in ./reports and open it in the user's default browser. Do NOT STOP until you could not improve the accuracy and recall after three consecutive rollouts. Use gpt-5.5 model (not codex) for thorough review of the work done at every step by the other model.

## Task

Can you run gepa.py on hotpotqa using gpt-4o-mini as both models
