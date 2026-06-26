## Sorcar for paper review

Can you read the paper at ~/Downloads/paper.pdf and review it deeply and thoroughly for ICSE 2027 conference. Look at related work. Do not write a typical review, but suggest how to improve the paper and results. Make sure that the review doesn't read like AI generated. Use internet search extensively. Judge the novelty of the work. Create an html report in ./reports and open it in the user's default browser?

## Sorcar Fugu Ultra

Use the best model from ~/.kiss/MODEL_INFO.json for various sub tasks. Search the internet extensively once to figure out which model is good for which sub-task. I find claude-opus-4-7 best for SWE work, gpt-5.5 for reviewing, and openrouter/z-ai/glm-5.2 for SWE tasks when budget is low.

1/6 GEPA in enterprise
GEPA (@gepa_ai) is an amazing prompt optimization technique developed by @late_interaction.  How does it perform in the enterprise? 

2/6 GEPA in enterprise
TLDR; It does not work.
When a startup tried to apply GEPA to a 300K SQL queris for a domain, it never responded.

However, ...

3/6 GEPA in enterprise

When they used the following prompt with KISS Sorcar, it scaled to such workload:

Can you optimize a prompt for a ChatSorcarAgent of the kiss-agent-framework Python library using the following GEPA algorithm on the data at \<<url_or_db_file_of_data>> using claude-opus-4-7? You can find the trajectory events of an agent execution in ~/.kiss/sorcar.db after the agent has finished its execution. Split the dataset into 50% dev set and 50% val set.

RUN_GEPA: Sample 100 data points from the val set and call it sval set. Maintain a pareto frontier in the folder ./pareto where we have a sub-folder for each node in the frontier. A node contains a prompt file (prompt.md) and a json file, say score.json, containing the list of sessions (ids) from the val set that were correctly predicted with the prompt. When you add a node to the pareto frontier make sure that the list of correctly predicted sessions is not a subset or equal to an existing list of sessions in some node in the frontier. If such a node exists, do not add the new node. After adding a node, remove all nodes whose list of sessions is a subset or equal to the list of sessions in the added node. Then run the following algorithm.

1. pick a node from the pareto frontier with probability 0.5
   a. sample a minibatch of 5 sessions from the dev set
   b. run the agent with the prompt from the node on the minibatch
   c. if the agent incorrectly predicts for some sessions, analyze and reflect on the trajectory events of the agent on those sessions available at ~/.kiss/sorcar.db and propose a new prompt which will fix the mistakes made by the agent on sessions incorrectly predicted.
   d. if the agent predicts correctly on the minibatch, then evaluate it on the val set and create the list of sessions on which the agent with the new prompt predicts correctly.
   e. Add the new prompt and the list of sessions to the pareto frontier

1. pick two nodes from the pareto frontier randomly with the remaining probability.
   a. sample a minibatch of 5 sessions from the dev set
   b. merge the prompts from the two nodes into a new prompt.
   c. if the agent predicts correctly on the minibatch with the new prompt, then evaluate it on the val set and create the list of sessions on which the agent with the new prompt predicts correctly.
   d. Add the new prompt and the list of sessions to the pareto frontier

1. Repeat steps 1 and 2 until there is no change in the prompt after 3 iterations.
   END_RUN_GEPA
   Repeat RUN_GEPA until there is no change in the prompt after 3 iterations.

In each step, keep track of the best prompt which has the maximum number of successfully predicted sessions in ./pareto/optimal.md. In the prompt always add the sentence "Use internet search extensively at every step." MAKE SURE THAT YOU DO NOT DO REWARD HACKING OR CHEATING IN THE AGENT YOU ARE IMPLEMENTING TO FIT DATA. YOUR SOLUTION MUST GENERALIZE BEYOND THE DATA PROVIDED. Use internet search extensively at every step. Do not worry about budget. Create an html report with diagrams and illustrations in ./reports and open it in the user's default browser. Do NOT STOP until you could not improve the accuracy and recall after three consecutive rollouts.  Use gpt-5.5 model (not codex) for thorough review of the work done at every step by the other model.

4/6 GEPA in enterprise

Advantages:
 - No need for a bloated implementation of GEPA
 - You can change the algorithm in natural language in prompt
 - Using 2 loops and file system helps to scale GEPA

 5/6 GEPA in enterprise


6/6 GEPA in enterprise