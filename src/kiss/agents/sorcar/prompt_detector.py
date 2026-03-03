import re
from pathlib import Path


class PromptDetector:
    """
    An intelligent detector that analyzes Markdown content to determine if it
    represents an AI prompt or system instruction file.
    """

    THRESHOLD = 3.0

    STRONG_INDICATORS: dict[str, float] = {
        r"(?i)^(?:#\s*)?(?:system\s+)?prompt": 3.0,
        r"(?i)you\s+are\s+a(?:n)?\s+\w+": 2.0,
        r"(?i)act\s+as\s+a(?:n)?\s+\w+": 2.0,
        r"(?i){{\s*[\w_]+\s*}}": 1.5,
        r"(?i)<(system|user|assistant|instruction|context|example)>": 2.0,
    }

    MEDIUM_INDICATORS: dict[str, float] = {
        r"(?i)^#+\s*(?:role|persona|context|instruction|task|constraints|output\s+format)": 1.5,
        r"(?i)few-shot": 1.5,
        r"(?i)chain\s+of\s+thought": 1.5,
        r"(?i)step-by-step": 1.0,
        r"(?i)your\s+task\s+is": 1.5,
        r"(?i)do\s+not\s+(?:hallucinate|invent)": 1.5,
    }

    WEAK_INDICATORS: dict[str, float] = {
        r"(?i)temperature\s*:": 0.5,
        r"(?i)top_p\s*:": 0.5,
        r"(?i)json\s+mode": 1.0,
        r"```(?:json|xml|markdown)": 0.5,
    }

    def _extract_frontmatter(self, content: str) -> tuple[dict | None, str]:
        """Attempts to manually extract YAML-style frontmatter without external deps."""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1]
                remaining_content = parts[2]

                # Simple manual parsing of key: value
                data = {}
                for line in frontmatter_str.split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        data[key.strip()] = val.strip()
                return data, remaining_content
        return None, content

    def analyze(self, file_path: str) -> tuple[bool, float, list[str]]:
        """
        Analyzes a file to check if it is a prompt.
        Returns: (Is Prompt?, Confidence Score, Reasons)
        """
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() != ".md":
            return False, 0.0, ["File not found or not .md"]

        try:
            raw_content = path.read_text(encoding="utf-8")
        except Exception as e:
            return False, 0.0, [f"Error reading file: {str(e)}"]

        score = 0.0
        reasons = []

        # 1. Analyze Frontmatter
        frontmatter, content = self._extract_frontmatter(raw_content)
        if frontmatter:
            prompt_keys = ["model", "temperature", "top_p", "stop_sequences", "inputs"]
            found_keys = [k for k in prompt_keys if k in frontmatter]
            if found_keys:
                points = len(found_keys) * 0.5
                score += points
                reasons.append(f"Found prompt metadata: {found_keys} (+{points})")

        # 2. Regex Pattern Matching
        def check_patterns(indicators: dict[str, float], level: str) -> float:
            local_score = 0.0
            for pattern, points in indicators.items():
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    # Diminishing returns for repeated patterns
                    actual_points = points + (0.1 * (len(matches) - 1))
                    actual_points = min(actual_points, points * 2)
                    local_score += actual_points
                    reasons.append(f"Found {level} indicator '{pattern}' (+{actual_points:.1f})")
            return local_score

        score += check_patterns(self.STRONG_INDICATORS, "STRONG")
        score += check_patterns(self.MEDIUM_INDICATORS, "MEDIUM")
        score += check_patterns(self.WEAK_INDICATORS, "WEAK")

        # 3. Density Check (Prompts use high rates of imperative verbs)
        imperative_verbs = {
            "write",
            "explain",
            "summarize",
            "translate",
            "classify",
            "act",
            "ignore",
            "return",
            "output",
        }
        words = re.findall(r"\w+", content.lower())
        if words:
            verb_count = sum(1 for w in words if w in imperative_verbs)
            density = verb_count / len(words)
            if density > 0.05:  # If >5% of words are commands
                score += 1.0
                reasons.append("High density of imperative command verbs (+1.0)")

        is_prompt = score >= self.THRESHOLD
        return is_prompt, score, reasons


# ==========================================
# Run the test suite
# ==========================================
if __name__ == "__main__":
    import os
    import tempfile

    detector = PromptDetector()

    test_files = {
        "coding_agent.md": """
# System Prompt
You are an expert Python developer.
## Constraints
- Do not use classes unless necessary.
- Return only code.
""",
        "email_template.md": """
---
description: generate marketing email
model: gpt-4
---
# Task
Write a marketing email for {{ product_name }} targeting {{ audience }}.
Ensure the tone is professional.
""",
        "xml_prompt.md": """
<system>
You are a helpful assistant.
</system>
<instruction>
Analyze the following text.
</instruction>
""",
        "readme.md": """
# Project Documentation
This project is a web scraper.
## Installation
Run `pip install -r requirements.txt`.
## Usage
Run `python main.py`.
""",
        "blog_post.md": """
# My Trip to Japan
I went to Tokyo last week. It was amazing.
You should go there if you have the chance.
The food is great.
""",
    }

    print(f"{'File Name':<20} | {'Is Prompt?':<10} | {'Score':<6} | {'Reasoning'}")
    print("-" * 100)

    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in test_files.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

            is_prompt, score, reasons = detector.analyze(file_path)

            status = "YES" if is_prompt else "NO"
            print(f"{filename:<20} | {status:<10} | {score:<6.1f} | {str(reasons)[:60]}...")
