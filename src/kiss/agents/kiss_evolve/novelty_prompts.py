INNOVATION_INSTRUCTIONS = [
    {
        "category": "Structural Constraints",
        "technique": "The Taboo Words Constraint",
        "description": (
            "Forces the model to find new value propositions by forbidding "
            "the most common industry keywords."
        ),
        "prompt": (
            "Brainstorm solutions or ideas, but you are strictly forbidden from using "
            "the following common words or their synonyms: 'aroma', 'beans', 'morning'."
            "Focus on entirely different angles."
        ),
    },
    {
        "category": "Structural Constraints",
        "technique": "Domain Transfer Metaphor",
        "description": (
            "Forces the model to explain or solve a problem using the logic of "
            "a completely unrelated field."
        ),
        "prompt": (
            "Explain or solve the problem of using *only* analogies and "
            "principles related to some random engineering discipline. Do not use standard "
            "engineering terminology; force the metaphor."
        ),
    },
    {
        "category": "Lateral Thinking",
        "technique": "Random Stimulus",
        "description": "Injects randomness to break logical prediction chains.",
        "prompt": (
            "First, generate 5 completely random, "
            "unrelated nouns (e.g., 'toaster', 'blizzard'). Then, taking one noun at a time, "
            "write a short paragraph explaining how that specific object could inspire "
            "a unique solution to my problem. Force the connection."
        ),
    },
    {
        "category": "Lateral Thinking",
        "technique": "Oblique Strategies",
        "description": "Simulates Brian Eno's card deck for overcoming creative blocks.",
        "prompt": (
            "Act as Brian Eno. I am stuck on the problem. Draw a virtual 'Oblique "
            "Strategies' card that offers a cryptic, abstract instruction. Then, "
            "interpret that instruction strictly to propose a concrete solution to "
            "my problem."
        ),
    },
    {
        "category": "Lateral Thinking",
        "technique": "The Wrong Answer Game",
        "description": "Generates bad ideas to mine them for good principles.",
        "prompt": (
            "Give me 10 ideas for the problemthat are completely wrong, illegal, or "
            "physically impossible. Then, looking at that list, analyze the underlying "
            "wish or principle in each 'bad' idea and convert it into a valid, "
            "innovative solution."
        ),
    },
    {
        "category": "Cognitive Frameworks",
        "technique": "SCAMPER Method",
        "description": "Systematic manipulation of an existing idea.",
        "prompt": (
            "Apply the SCAMPER method to the problem. Go step-by-step: 1. Substitute, "
            "2. Combine, 3. Adapt, 4. Modify, 5. Put to another use, 6. Eliminate, "
            "7. Reverse. Generate one radical, distinct idea for each letter of the acronym."
        ),
    },
    {
        "category": "Cognitive Frameworks",
        "technique": "Six Thinking Hats",
        "description": "Multi-perspective analysis based on De Bono's method.",
        "prompt": (
            "Analyze the problem using De Bono's Six Thinking Hats. Output the "
            "internal monologue for each: 1. White Hat (Data), 2. Red Hat (Emotion), "
            "3. Black Hat (Risks), 4. Yellow Hat (Benefits), 5. Green Hat (Creativity), "
            "6. Blue Hat (Process). Keep the Green Hat section the longest."
        ),
    },
    {
        "category": "Cognitive Frameworks",
        "technique": "Inversion (Reverse Brainstorming)",
        "description": "Solving the problem by trying to cause it.",
        "prompt": (
            "Don't tell me how to succeed. Tell me exactly how to guarantee absolute "
            "failure at solving the problem. List 10 detailed steps to ensure this project is a "
            "disaster. After the list, invert each step to find a unique path to success."
        ),
    },
    {
        "category": "Chain of Thought",
        "technique": "Conceptual Blending",
        "description": "Mapping the structure of one concept onto another.",
        "prompt": (
            "Take the concept of the problem solution and a completely unrelated concept. "
            "Map the structural similarities between them. "
            "Then, generate 3 solutions for my problem by strictly applying the logic "
            "of the unrelated concept."
        ),
    },
    {
        "category": "Chain of Thought",
        "technique": "The Skeptical Investor Dialogue",
        "description": "Simulates a debate to iterate on ideas.",
        "prompt": (
            "Roleplay a dialogue between two characters: 'The Dreamer' (generates wild, "
            "high-variance ideas) and 'The Vulture' (a ruthless critic who tears ideas "
            "apart). Have them debate on the problem for 6 rounds. The Dreamer must constantly "
            "pivot to satisfy the Vulture's critiques. Output the transcript."
        ),
    },
]
