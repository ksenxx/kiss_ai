from kiss.core.kiss_agent import KISSAgent

agent = KISSAgent("test_agent")

agent.run(
    model_name="claude-3-opus-20240229",
    prompt_template=(
        "Write the a file named 'test_output.txt' with the content 'Hello from the agent!'"
    ),
)
