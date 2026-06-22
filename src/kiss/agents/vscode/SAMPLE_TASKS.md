## Task

Can you show me the detailed step-by-step workflow of \<<your algorithm or feature>>

## Task

Can you change the step \<<specify step>> as follows: \<<whatever way you want to change>>

## Task

Can you run gepa.py on hotpotqa using gpt-4o-mini as both models

## Task

can you authenticate me with the learningsystems workspace on slack using the slack agent?

## Task

Can you create a cron job with name prefixed with "kiss-" which will check every 3 seconds if there are latest unresponsed messages from /<<user name>> in the channel sorcar using the slack agent, then it will run the messages as tasks one-by-one in the order of arrival and respond with the result suitably formatted for slack.

## Task

can you authenticate me with the imessages agent?

## Task

can you send "Hello" to 1-800-772-1213?

## Task

can you authenticate me with gmail using the gmail agent? Use the user's default browser to ask user login and to get authentication token.

## Task

Can you read \<<url>>, and thoroughly and precisely check for **wrong assumptions**, **cheating**, **irreproducibility issues**, **fraud**, **potential for cheating in evaluation**, **AI Slop**, and **security vulnerabilities**? Use internet search extensively and do not believe what people say--verify them yourself. Do not hesitate to download code and run them to validate results. For security vulnerabilities, create a POC and test it. Generate an html report in PWD/sorcar_reported_frauds/ and open in the user's default browser. Thoroughly fact check everything you claim in the report.

## Task

Sorcar for auto research: Can you discover the lightest and fastest AI model that will give the best accuracy and recall on the data at \<</path/to/data>> at the cheapest price? Analyze the data and search internet extensively to propose first few models. Implement and experiment each of your proposals. Note down the ideas you used to optimize the accuracy/recall and speed/cost metrics achieved in a file, so that you can use the file to not repeat ideas that have already been tried and/or failed. You can also use the file to combine ideas that have been successful in the past. Separate 20% data for evals and your discovery strategy must not look at the evals data. Use 'lambda' CLI to train your models on GPUs and evaluate if needed. Total budget for lambda labs is $1000. Experiment with smaller subset of data and fewer parameters in a model to do experiments quickly and then extrapolate. Use internet search extensively at every step. MAKE SURE THAT YOU DO NOT DO REWARD HACKING OR CHEATING IN THE MODELS OR AGENTS YOU ARE IMPLEMENTING TO FIT DATA. YOUR SOLUTION MUST GENERALIZE BEYOND THE DATA PROVIDED. Do not STOP until accuracy/recall reaches 95% on evals and you can process each query in less than 600 seconds and under 50 USD per query amortized over all queries. Create an html report with diagrams and illustrations in PWD/reports and open it in the user's default browser?

## Task

Sorcar for optimization: Can you run the command \<<command>> in the background and monitor its output in real time to optimize the code at \<<folder_name_or_url>> with respect to the following metrics: \<<speed,accuracy,recall,cost>>. You can add diagnostic code which will print the metrics, such as running time at finer level of granularity. Check for opportunities to optimize the code on the basis of the metrics information. If you discover any opportunities to optimize the metric based on the code, logs, events, and the command output, optimize the code and run the command again. Note down the ideas you used to optimize the code and the metric you achieved in a file, so that you can use the file to not repeat ideas that have already been tried and failed. You can also use the file to combine ideas that have been successful in the past. Repeat the process. Do not forget to remove the diagnostic code after the optimization is complete. You MUST NOT STOP until the metrics achieve the following values:\<<give_concrete_values_for_metrics>>. Use internet extensively to get new ideas for optimization. Create an html report with diagrams and illustrations in PWD/reports and open it in the user's default browser?

## Task

Sorcar GEPA Prompt Optimizer: Can you optimize a prompt for a ChatSorcarAgent of kiss-agent-framework Python library using the GEPA algorithm on the data at \<<url_or_db_file_of_data>> using claude-opus-4-8? You can find the trajectory events of an agent execution in ~/.kiss/sorcar.db after the agent has finished its execution. In the prompt always add the sentence "Use internet search extensively at every step." MAKE SURE THAT YOU DO NOT DO REWARD HACKING OR CHEATING IN THE AGENT YOU ARE IMPLEMENTING TO FIT DATA. YOUR SOLUTION MUST GENERALIZE BEYOND THE DATA PROVIDED. Use internet search extensively at every step. Do not worry about budget. Create an html report with diagrams and illustrations in PWD/reports and open it in the user's default browser. Do NOT STOP until you could not improve the accuracy and recall after three consecutive rollouts.

