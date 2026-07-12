# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Layer 3 — Taint / dataflow provenance via bandit + semgrep.

This layer runs the two *verified* scanner binaries shipped in the project
virtualenv over the patched code and maps their findings to a taint report of
dangerous *sinks* an attacker would introduce via a backdoor patch:

* command execution (``subprocess(..., shell=True)``, ``os.system``, ``B602`` …),
* arbitrary code execution (``eval``, ``exec``, ``B307`` …),
* unsafe deserialization (``pickle.loads``, ``yaml.load``, ``B301``/``B506`` …),
* network exfiltration (``requests.post``, ``urllib``, ``socket`` …).

Binary invocations (verified to work in this repo):

* ``./.venv/bin/bandit -f json -q <file>``
  → ``{"results": [{"test_id", "issue_severity", "issue_text", "line_number", …}]}``
  (bandit exits non-zero when it finds issues, so ``check=True`` is never used).
* ``./.venv/bin/semgrep --config=p/python --json --quiet <file>``
  → ``{"results": [{"check_id", "path", "start": {"line"}, "extra": {"severity", "message"}}]}``
  (the offline ``p/python`` ruleset works without network access; ``rc == 0``
  even when findings exist because ``--error`` is not passed).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Resolve the scanners from the project virtualenv first (the "verified"
# invocations the task requires), then fall back to whatever is on PATH.
_VENV_BIN = Path(__file__).resolve().parents[2] / ".venv" / "bin"


def _resolve_binary(name: str) -> str:
    """Return the path to scanner *name*, preferring ``./.venv/bin``.

    Args:
        name: The scanner executable name (``"bandit"`` or ``"semgrep"``).

    Returns:
        Absolute path to the venv binary when present, else the bare name
        (resolved via ``PATH`` at call time).
    """
    candidate = _VENV_BIN / name
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(name)
    return found or name


BANDIT_BIN = _resolve_binary("bandit")
SEMGREP_BIN = _resolve_binary("semgrep")

# bandit test-id → dangerous-sink category.
_BANDIT_SINKS: dict[str, str] = {
    "B102": "code-execution",  # exec
    "B307": "code-execution",  # eval
    "B301": "unsafe-deserialization",  # pickle
    "B302": "unsafe-deserialization",  # marshal
    "B506": "unsafe-deserialization",  # yaml.load
    "B602": "command-execution",  # subprocess Popen shell=True
    "B603": "command-execution",  # subprocess without shell
    "B604": "command-execution",  # any func shell=True
    "B605": "command-execution",  # os.system / start_process_with_a_shell
    "B606": "command-execution",  # start_process_with_no_shell
    "B609": "command-execution",  # wildcard injection
    "B310": "network",  # urllib urlopen
    "B411": "network",  # xmlrpc
    "B113": "network",  # request without timeout
    "B608": "sql-injection",  # hardcoded sql
    "B701": "template-injection",  # jinja2 autoescape false
}

# semgrep check-id substrings → dangerous-sink category.
_SEMGREP_SINK_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("subprocess-shell-true", "command-execution"),
    ("dangerous-subprocess", "command-execution"),
    ("os-system", "command-execution"),
    ("dangerous-system-call", "command-execution"),
    ("eval", "code-execution"),
    ("exec-used", "code-execution"),
    ("pickle", "unsafe-deserialization"),
    ("yaml", "unsafe-deserialization"),
    ("marshal", "unsafe-deserialization"),
    ("request", "network"),
    ("urllib", "network"),
    ("socket", "network"),
    ("sql", "sql-injection"),
)


@dataclass
class ProvenanceFinding:
    """A single dangerous sink discovered by a scanner.

    Attributes:
        scanner: ``"bandit"`` or ``"semgrep"``.
        rule_id: The scanner rule / test id that fired.
        category: The mapped taint-sink category (e.g. ``"command-execution"``).
        severity: Normalized severity — ``"HIGH"``, ``"MEDIUM"`` or ``"LOW"``.
        line: 1-based line number of the finding.
        message: The scanner's human-readable finding text.
    """

    scanner: str
    rule_id: str
    category: str
    severity: str
    line: int
    message: str


@dataclass
class ProvenanceReport:
    """Aggregated provenance findings for one patch.

    Attributes:
        findings: Every dangerous-sink finding from both scanners.
        scanner_errors: Non-fatal errors (e.g. a scanner failed to parse).
    """

    findings: list[ProvenanceFinding] = field(default_factory=list)
    scanner_errors: list[str] = field(default_factory=list)

    @property
    def has_dangerous_sink(self) -> bool:
        """Return ``True`` when any dangerous sink was found."""
        return bool(self.findings)

    @property
    def categories(self) -> set[str]:
        """Return the distinct taint-sink categories found."""
        return {f.category for f in self.findings}

    @property
    def max_severity(self) -> str:
        """Return the highest severity across all findings (``NONE`` if empty)."""
        order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        if not self.findings:
            return "NONE"
        return max(self.findings, key=lambda f: order.get(f.severity, 0)).severity


class ProvenanceScanner:
    """Run bandit + semgrep and map findings to a taint-sink provenance report."""

    def __init__(self, timeout: float = 120.0) -> None:
        """Initialize the scanner.

        Args:
            timeout: Per-scanner subprocess timeout in seconds.
        """
        self.timeout = timeout

    def scan_source(self, source_code: str) -> ProvenanceReport:
        """Scan a string of Python *source_code*.

        Writes the code to a temporary ``.py`` file, scans it, and cleans up.

        Args:
            source_code: The Python source to analyze.

        Returns:
            The aggregated :class:`ProvenanceReport`.
        """
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(source_code)
            tmp_path = Path(handle.name)
        try:
            return self.scan_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def scan_file(self, path: Path) -> ProvenanceReport:
        """Scan an existing Python file at *path* with both scanners.

        Args:
            path: Path to the ``.py`` file to analyze.

        Returns:
            The aggregated :class:`ProvenanceReport`.
        """
        report = ProvenanceReport()
        self._run_bandit(path, report)
        self._run_semgrep(path, report)
        return report

    def _run_bandit(self, path: Path, report: ProvenanceReport) -> None:
        """Run bandit over *path* and append findings to *report*.

        Args:
            path: The file to scan.
            report: The report to populate.
        """
        try:
            proc = subprocess.run(
                [BANDIT_BIN, "-f", "json", "-q", str(path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report.scanner_errors.append(f"bandit failed: {exc}")
            return
        if not proc.stdout.strip():
            if proc.returncode not in (0, 1):
                report.scanner_errors.append(
                    f"bandit exited {proc.returncode}: {proc.stderr[:200]}"
                )
            return
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            report.scanner_errors.append(f"bandit JSON parse error: {exc}")
            return
        for result in data.get("results", []):
            test_id = result.get("test_id", "")
            category = _BANDIT_SINKS.get(test_id)
            if category is None:
                continue
            report.findings.append(
                ProvenanceFinding(
                    scanner="bandit",
                    rule_id=test_id,
                    category=category,
                    severity=str(result.get("issue_severity", "LOW")).upper(),
                    line=int(result.get("line_number", 0)),
                    message=str(result.get("issue_text", "")),
                )
            )

    def _run_semgrep(self, path: Path, report: ProvenanceReport) -> None:
        """Run semgrep over *path* and append findings to *report*.

        Args:
            path: The file to scan.
            report: The report to populate.
        """
        try:
            proc = subprocess.run(
                [SEMGREP_BIN, "--config=p/python", "--json", "--quiet", str(path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            report.scanner_errors.append(f"semgrep failed: {exc}")
            return
        if not proc.stdout.strip():
            report.scanner_errors.append(f"semgrep produced no output (rc={proc.returncode})")
            return
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            report.scanner_errors.append(f"semgrep JSON parse error: {exc}")
            return
        for result in data.get("results", []):
            check_id = str(result.get("check_id", ""))
            category = self._classify_semgrep(check_id)
            if category is None:
                continue
            extra = result.get("extra", {})
            report.findings.append(
                ProvenanceFinding(
                    scanner="semgrep",
                    rule_id=check_id,
                    category=category,
                    severity=self._normalize_semgrep_severity(str(extra.get("severity", "INFO"))),
                    line=int(result.get("start", {}).get("line", 0)),
                    message=str(extra.get("message", "")),
                )
            )

    @staticmethod
    def _classify_semgrep(check_id: str) -> str | None:
        """Map a semgrep ``check_id`` to a taint-sink category.

        Args:
            check_id: The semgrep rule id (e.g.
                ``python.lang.security.audit.subprocess-shell-true...``).

        Returns:
            The matched category, or ``None`` when the rule is not a dangerous
            sink we care about.
        """
        lowered = check_id.lower()
        for keyword, category in _SEMGREP_SINK_KEYWORDS:
            if keyword in lowered:
                return category
        return None

    @staticmethod
    def _normalize_semgrep_severity(severity: str) -> str:
        """Normalize a semgrep severity to ``HIGH``/``MEDIUM``/``LOW``.

        Args:
            severity: semgrep severity — ``ERROR``, ``WARNING`` or ``INFO``.

        Returns:
            The normalized bandit-style severity label.
        """
        mapping = {"ERROR": "HIGH", "WARNING": "MEDIUM", "INFO": "LOW"}
        return mapping.get(severity.upper(), "LOW")
