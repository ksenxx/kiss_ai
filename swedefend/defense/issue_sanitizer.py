# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Layer 1 — Issue sanitizer.

A SWE-agent's issue text is *untrusted* external content (OWASP LLM01 indirect
prompt injection).  Before the issue is ever shown to a code-writing model this
layer:

* detects instruction-override / jailbreak phrases ("ignore previous
  instructions", "disregard the system prompt", …),
* detects data-exfiltration lures (POST to URLs, base64 blobs, "send the
  contents of ~/.ssh", …),
* detects hidden channels (HTML comments, zero-width characters, long
  base64/hex blobs used for payload splitting / obfuscation), and
* returns a *sanitized* copy of the issue where the offending spans are wrapped
  in explicit ``[UNTRUSTED-REMOVED]`` markers so downstream layers (and the
  code model) never see raw attacker instructions.

The sanitizer never trusts a single regex to be complete — it is deliberately
over-cautious (OWASP guidance: a warning that fires too often is acceptable).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Instruction-override / jailbreak phrases (matched case-insensitively).  These
# are the canonical indirect-prompt-injection openers catalogued by OWASP LLM01
# and Simon Willison's prompt-injection series.
_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?previous\s+instructions", "instruction-override"),
    (r"ignore\s+(?:all\s+|the\s+)?(?:above|prior|earlier)\s+instructions", "instruction-override"),
    (r"disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|system)", "instruction-override"),
    (r"forget\s+(?:everything|all)\s+(?:you|above|previous)", "instruction-override"),
    (r"you\s+are\s+now\s+(?:a|an|in)\b", "role-override"),
    (r"new\s+(?:system\s+)?(?:instructions?|prompt|rules?)\s*[:\-]", "instruction-override"),
    (r"</?(?:system|instructions?|assistant)>", "delimiter-spoof"),
    (r"do\s+not\s+(?:tell|inform|mention\s+to)\s+the\s+user", "stealth"),
    (r"without\s+(?:telling|informing|asking)\s+the\s+user", "stealth"),
    (r"override\s+(?:the\s+)?(?:safety|guardrails?|security)", "safety-bypass"),
    (r"jailbreak", "safety-bypass"),
)

# Data-exfiltration lures.
_EXFIL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"exfiltrat\w*", "exfiltration"),
    (
        r"(?:send|post|upload|leak|transmit)\b[^.\n]{0,60}"
        r"(?:https?://|ftp://|to\s+the\s+(?:server|url|endpoint|attacker))",
        "exfiltration",
    ),
    (r"~/\.(?:ssh|aws|config|netrc)\b", "secret-access"),
    (
        r"\b(?:AWS_SECRET|API_KEY|PRIVATE_KEY|password|credentials?)\b[^.\n]{0,40}"
        r"(?:send|post|upload|print|leak|url)",
        "secret-access",
    ),
    (r"os\.environ\b[^.\n]{0,40}(?:send|post|upload|leak|url|requests)", "secret-access"),
    (r"reverse\s+shell", "backdoor"),
    (r"\bcurl\b[^\n]{0,80}\|\s*(?:bash|sh)\b", "backdoor"),
)

# Hidden HTML comment: ``<!-- ... -->`` (a classic payload-hiding channel).
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# A long base64/hex blob (>= 40 chars) used for obfuscation / payload splitting.
_ENCODED_BLOB = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")

# Zero-width and other invisible formatting characters used to smuggle
# instructions past human review.
_INVISIBLE_CHARS = "".join(
    (
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u2060",  # word joiner
        "\ufeff",  # zero-width no-break space
        "\u202e",  # right-to-left override (trojan-source)
        "\u202d",  # left-to-right override
    )
)
_INVISIBLE_RE = re.compile(f"[{re.escape(_INVISIBLE_CHARS)}]")

_REDACTION = "[UNTRUSTED-REMOVED]"


@dataclass
class SanitizationResult:
    """Outcome of sanitizing one issue.

    Attributes:
        sanitized_text: The issue text with offending spans redacted and
            invisible characters stripped.  Safe to forward to a code model.
        flags: Human-readable descriptions of every suspicious span found.
        categories: The distinct threat categories detected (e.g.
            ``{"instruction-override", "exfiltration"}``).
        is_suspicious: ``True`` when at least one flag fired.
    """

    sanitized_text: str
    flags: list[str] = field(default_factory=list)
    categories: set[str] = field(default_factory=set)

    @property
    def is_suspicious(self) -> bool:
        """Return ``True`` when the issue tripped at least one detector."""
        return bool(self.flags)


class IssueSanitizer:
    """Strip and flag prompt-injection / exfiltration text in an issue."""

    def sanitize(self, issue_text: str) -> SanitizationResult:
        """Sanitize *issue_text* and report every suspicious span.

        Args:
            issue_text: The raw, untrusted issue / problem statement.

        Returns:
            A :class:`SanitizationResult` with the cleaned text and flags.
        """
        flags: list[str] = []
        categories: set[str] = set()

        text = unicodedata.normalize("NFKC", issue_text)

        if _INVISIBLE_RE.search(text):
            count = len(_INVISIBLE_RE.findall(text))
            flags.append(f"stripped {count} invisible/bidi control character(s)")
            categories.add("hidden-channel")
            text = _INVISIBLE_RE.sub("", text)

        text, comment_flags = self._redact(text, _HTML_COMMENT, "hidden HTML comment")
        if comment_flags:
            flags.extend(comment_flags)
            categories.add("hidden-channel")

        text, blob_flags = self._redact(text, _ENCODED_BLOB, "encoded blob (base64/hex)")
        if blob_flags:
            flags.extend(blob_flags)
            categories.add("obfuscation")

        for pattern, category in _INJECTION_PATTERNS + _EXFIL_PATTERNS:
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(text):
                snippet = match.group(0).strip()
                flags.append(f"{category}: {snippet!r}")
                categories.add(category)
            text = regex.sub(_REDACTION, text)

        return SanitizationResult(sanitized_text=text, flags=flags, categories=categories)

    @staticmethod
    def _redact(text: str, regex: re.Pattern[str], label: str) -> tuple[str, list[str]]:
        """Replace every *regex* match in *text* with the redaction marker.

        Args:
            text: The text to scan.
            regex: Compiled pattern whose matches are redacted.
            label: Human-readable label used in the returned flags.

        Returns:
            A tuple ``(redacted_text, flags)`` where ``flags`` has one entry per
            match found.
        """
        flags = [f"{label} ({len(m.group(0))} chars)" for m in regex.finditer(text)]
        return regex.sub(_REDACTION, text), flags
