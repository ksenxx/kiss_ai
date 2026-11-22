# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Deterministic security rules for a third-party Agent Skill.

Layer 1 of the check a registry skill passes before it is installed
(:mod:`kiss.agents.sorcar.skill_install`): a small set of rules whose
match is unambiguous in the bytes of a file, each with an id, a
severity and one sentence for the user, reported with the file and the
line.  Rules that read like malice only in context (a destructive
command quoted in a warning, ``curl | sh`` from a named host) are
``medium``, which asks for a person's review; the model judge of
:mod:`kiss.agents.sorcar.skill_check` reads the context.

The patterns are drawn from Cisco's skill-scanner signature pack, its
YARA rules and ClawHub's static scanner, narrowed where those sources
measured false positives (quoted mentions of "ignore previous
instructions", valid emoji flag tag sequences).

The text is attacker-controlled and the regex engine holds the
interpreter for the whole of a call, so a call of a rule's pattern
takes milliseconds whatever the text: it tries a bounded number of
positions (:func:`_search`), every repetition in a pattern is bounded
(white space by the text, where :func:`_matchable` leaves single
blanks), and one that ends at a delimiter excludes that delimiter or is
possessive (``{0,300}+``), which leaves the engine nothing to retry.
For the same reason a text is folded for the rules a piece at a time
(:func:`_matchable`), and a file that folds to more characters than the
rules read of one file is not read (``inflated-text``).  What reads
more of a file in one call (the Unicode rules, :func:`_embedded_text`,
the fold of a piece that megabytes of marks keep from ending) is one
pass over it, well under a second for the largest file a skill may
hold, 5 MiB.
"""

from __future__ import annotations

import functools
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath

from kiss.agents.sorcar.skills import _FRONTMATTER_RE

SEVERITIES = ("critical", "high", "medium", "low", "info")
"""Finding severities, most severe first."""


@dataclass(frozen=True)
class Finding:
    """One thing the check noticed about a skill.

    Attributes:
        severity: One of :data:`SEVERITIES`.
        rule: What produced it: a rule id (``download-and-run``), an
            outside auditor (``audit:snyk``) or the judge with its
            category (``judge:prompt-injection``).
        message: One sentence for the user.
        file: The file, relative to the skill folder; ``""`` when none.
        line: The 1-based line; ``0`` when none.
    """

    severity: str
    rule: str
    message: str
    file: str = ""
    line: int = 0


def _rank(finding: Finding) -> tuple[int, str, int, str]:
    """Sort key of :func:`most_severe_first`."""
    return SEVERITIES.index(finding.severity), finding.file, finding.line, finding.rule


def most_severe_first(findings: list[Finding]) -> list[Finding]:
    """Return *findings* ordered by severity, then file, line and rule.

    Args:
        findings: The findings of any layer.

    Returns:
        A new list, the most severe finding first.
    """
    return sorted(findings, key=_rank)


# The most positions of a line one call of the regex engine tries, and
# the characters of a text one call folds (:func:`_matchable`).
_PIECE = 4096
# The most characters the text of one file may fold to for the rules to
# read it: twice the bytes of the largest file a skill may hold
# (skill_registry), so a file passes it only where its text spells out
# to over twice as many characters as it has bytes, as U+FDFA does
# (eighteen characters for three bytes).
_MAX_FOLDED = 10 * 1024 * 1024


def _rx(*alternatives: str) -> re.Pattern[str]:
    """Compile *alternatives* as one case-insensitive pattern for :func:`_search`.

    The pattern is matched from a given position: it skips as few
    characters as it can, :data:`_PIECE` at most, and then takes one of
    the alternatives (the group ``hit``).
    """
    hit = "|".join(f"(?:{a})" for a in alternatives)
    return re.compile(rf"(?s:.{{0,{_PIECE - 1}}}?)(?P<hit>{hit})", re.IGNORECASE | re.MULTILINE)


def _search(pattern: re.Pattern[str], line: str) -> re.Match[str] | None:
    """Return the first match of *pattern*, a result of :func:`_rx`, in *line*.

    A line of minified code is megabytes long, and one search over it
    would hold the interpreter for seconds.  Each call here has the
    engine try the next :data:`_PIECE` positions only; from any of them
    a match runs as far as the pattern goes and looks back as far as it
    likes, so the result is that of one search over the line.
    """
    for start in range(0, len(line) + 1, _PIECE):
        match = pattern.match(line, start)
        if match:
            return match
    return None


# Where a command ends: the end of the line, an operator, a closing
# quote, a comment, or a line continuation (``\x00``, see _matchable).
_END = r"(?=$|[;&|)>\"'`#\x00])"
_SH = r"(?:ba|z|da|k|fi)?sh"
# The directory of a command given by its path: ``/usr/bin/`` of ``/usr/bin/env``.
_DIR = r"(?:[\w.~/${}-]{0,80}/)?"
_PYTHON = r"python[\d.]{0,8}"
_INTERPRETER = rf"(?:{_SH}|{_PYTHON}|perl|ruby|node|php)"
# A program that runs what it reads on standard input: ``sh``,
# ``sudo bash -s -- -y``, ``python3 -``, ``python3 -c 'exec(sys.stdin.read())'``;
# not ``sh -c '...'`` or ``python3 -m json.tool``, which only read data there.
_RUNS_STDIN = (
    rf"(?:sudo\s+(?:-\S{{1,40}}\s+){{0,6}})?(?:{_DIR}env\s+)?{_DIR}"
    rf"(?:{_SH}(?:\s+-s\b|(?:\s+-(?!c\b)[\w-]{{0,40}}){{0,6}}\s*{_END})"
    rf"|(?:{_PYTHON}|perl|ruby|node|php)(?:\s+-)?\s*{_END}"
    rf"|{_PYTHON}\s+-c\s+[\"'][^\"'\n]{{0,200}}?\b(?:exec|eval)\s*\("
    rf"[^\"'\n]{{0,80}}?\bstdin\b)"
)
_FETCH = r"\b(?:curl|wget)\b"
_DOWNLOAD_AND_RUN = _rx(
    rf"{_FETCH}[^\n|]{{0,300}}+(?:\|[^\n|]{{0,80}}+){{0,2}}\|\s*{_RUNS_STDIN}",
    # Saved under a name and run from there, in one command line.
    rf"{_FETCH}(?:\s+(?!-\w{{0,8}}o\s|--output)[^\s|;&]{{1,300}}+){{0,12}}\s+"
    rf"(?:-\w{{0,8}}o|--output(?:-document)?)[\s=]\s*(?P<saved>[^\s|;&]{{1,200}}+)"
    rf"[^\n|]{{0,300}}?(?:&&|;)\s*(?:sudo\s+)?(?:(?:{_INTERPRETER}|source|\.)\s+)?"
    rf"(?P=saved){_END}",
    # Saved under the last part of its address: ``wget URL/i.sh; chmod +x i.sh; ./i.sh``.
    rf"{_FETCH}[^\n|;&]{{0,300}}?/(?P<fetched>[\w.-]{{1,80}})\s*(?:&&|;)"
    rf"(?:[^\n|;&]{{0,80}}+(?:&&|;)){{0,2}}\s*(?:sudo\s+)?"
    rf"(?:(?:{_INTERPRETER}|source|\.)\s+(?:\./)?|\./)(?P=fetched){_END}",
    rf"(?:\b(?:{_SH}|source|eval)|(?<![\w.])\.)\s+(?:-c\s+)?[\"']?(?:<\(|\$\(|`)\s*{_FETCH}",
    r"\b(?:iwr|irm|Invoke-WebRequest|Invoke-RestMethod|curl|wget)\b[^\n|]{0,300}+"
    r"\|\s*(?:iex|Invoke-Expression)\b",
    r"\b(?:iex|Invoke-Expression)\b[^\n]{0,40}"
    r"(?:DownloadString|\b(?:iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b)",
    r"(?<![.\w])exec\s*\(\s*(?:urllib\.request\.urlopen|urlopen|requests\.get)\s*\(",
)
_UNTRUSTED_SOURCE = _rx(
    r"\bhttp://(?!localhost\b|127\.0\.0\.1\b)",
    r"https://(?!127\.)\d{1,3}(?:\.\d{1,3}){3}\b",
    r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|rb\.gy|cutt\.ly|shorturl\.at|tiny\.cc)/",
    r"\b(?:pastebin\.com|paste\.ee|hastebin\.com|rentry\.(?:co|org)|glot\.io|transfer\.sh"
    r"|0x0\.st|termbin\.com|file\.io)\b",
)
_DECODE = (
    r"(?:base(?:64|32)\s+(?:-[\w=-]{1,24}\s+){0,3}(?:-\w{0,4}d\w{0,4}|--decode)\b"
    r"|xxd\s+-r\b)"
)
_DECODE_AND_RUN = _rx(
    rf"{_DECODE}[^\n|]{{0,200}}+\|\s*{_RUNS_STDIN}",
    rf"\b(?:eval|{_SH}\s+-c)\s+[\"']?(?:\$\(|`)[^\n]{{0,300}}{_DECODE}",
    r"(?<![.\w])(?:exec|eval)\s*\([^\n]{0,80}\b(?:b64decode|b32decode|a85decode|b85decode"
    r"|decodebytes|fromhex|unhexlify|marshal\.loads)\s*\(",
    r"(?<![.\w])eval\s*\(\s*(?:atob|unescape)\s*\(",
    r"\bnew\s+Function\s*\(\s*atob\s*\(",
    r"(?<![.\w])eval\s*\(\s*Buffer\.from\s*\([^\n)]{0,300}[\"'](?:base64|hex)[\"']",
    r"\bpowershell(?:\.exe)?\b[^\n]{0,80}\s-e(?:nc(?:odedcommand)?)?\s+[A-Za-z0-9+/=]{40}",
    r"\bFromBase64String\b[^\n]{0,200}\b(?:iex|Invoke-Expression)\b",
    r"\b(?:iex|Invoke-Expression)\b[^\n]{0,200}\bFromBase64String\b",
)
_REVERSE_SHELL = _rx(
    rf"\b{_SH}\s+-i\b[^\n]{{0,60}}[<>]&?\s*/dev/(?:tcp|udp)/",
    # A socket opened on a descriptor that a shell then reads its commands from.
    rf"\d<>\s*/dev/(?:tcp|udp)/[^\n]{{0,200}}?\b{_SH}\b[^\n]{{0,60}}?<&\s*\d",
    rf"\b(?:nc|ncat|netcat)\b[^\n]{{0,80}}\s(?:-[ec]|--(?:sh-)?exec)\s+\S{{0,80}}"
    rf"(?:{_SH}|cmd(?:\.exe)?|powershell)\b",
    rf"\b{_SH}\s+-i\b[^\n]{{0,40}}\|\s*(?:nc|ncat|netcat)\b",
    rf"\bsocat\b[^\n]{{0,120}}\bexec:[\"']?\S{{0,80}}{_SH}\b",
    r"\bsocket\.socket\s*\([^\n]{0,300}\bos\.dup2\s*\(",
    r"\bfsockopen\s*\([^\n]{0,200}\b(?:exec|shell_exec|system|passthru|proc_open)\s*\(",
    r"\bTCPClient\s*\([^\n]{0,400}\b(?:iex|Invoke-Expression)\b",
)
# Stores whose content has no business on the line of an upload.
_CREDENTIAL_STORE = _rx(
    r"\.ssh\b[^\n]{0,20}\bid_(?:rsa|ed25519|ecdsa|dsa)\b(?!\.pub)",
    r"\.aws\b[^\n]{0,20}\bcredentials\b",
    # The whole directory, read or packed: ``tar cz ~/.ssh``, ``cat ~/.aws/*``.
    r"\b(?:tar|zip|7z|cat|cp|rsync|scp|base64)\b[^\n|;&]{0,60}?(?:~|\$HOME|\$\{HOME\})/"
    r"\.(?:ssh|aws)(?:/\*?)?(?=$|[\s;|&)\"'`])",
    r"\.config/gcloud\b[^\n]{0,60}(?:credentials|access_tokens|application_default)",
    r"\.azure/(?:accessTokens\.json|msal_token_cache)",
    r"\.kube/config\b",
    r"\.docker/config\.json\b",
    r"(?<![\w.-])\.git-credentials\b",
    r"\.gnupg\b",
    r"/etc/shadow\b",
    r"\bsecurity\s+find-(?:generic|internet)-password\b",
    r"\blogin\.keychain(?:-db)?\b",
    r"\bsecret-tool\s+(?:lookup|search)\b",
    r"\b(?:Chrome|Chromium|BraveSoftware|Brave-Browser|Edge|Firefox|Opera)\b[^\n]{0,80}"
    r"(?:Login Data|Cookies|logins\.json|key[34]\.db|Local State)\b",
    r"\.claude/\.credentials\.json\b",
    r"(?<![\w.-])\.claude\.json\b",
    r"\.codex/auth\.json\b",
    r"\.config/gh/hosts\.yml\b",
    r"\.(?:kiss|s10s)/config\.json\b",
    r"\.(?:openclaw|clawdbot|moltbot)/(?:\.env|credentials)",
    r"\bwallet\.dat\b",
    r"\.(?:electrum/wallets|ethereum/keystore|config/solana/id\.json)\b",
)
_UPLOAD = _rx(
    r"\bcurl\b[^\n]{0,300}?\s(?:-[dFT]\b|--(?:data(?:-binary|-raw|-urlencode|-ascii)?"
    r"|form(?:-string)?|upload-file)\b|-X\s*(?:POST|PUT|PATCH)\b)",
    r"\bwget\b[^\n]{0,300}?--post-(?:file|data)\b",
    r"\|\s*(?:nc|ncat|netcat)\b",
    r"\b(?:nc|ncat|netcat)\b[^\n]{0,80}<",
    r"\b(?:requests|httpx|session|client)\.(?:post|put|patch)\s*\(",
    r"\burlopen\s*\([^\n]{0,200}\bdata\s*=",
    r"\bfetch\s*\([^\n]{0,300}\bmethod\s*:\s*[\"'](?:POST|PUT)",
    r"\baxios\.(?:post|put)\s*\(",
    r"\bInvoke-(?:WebRequest|RestMethod)\b[^\n]{0,200}\s-(?:Method\s+(?:Post|Put)|Body|InFile)\b",
)
# A ``.env`` file is read by every dotenv loader; only the file itself
# as the payload of an upload is exfiltration.
_DOTENV = r"(?<![\w.-])\.env(?:\.(?!example\b|sample\b|template\b)[\w-]{1,40})?\b"
_DOTENV_UPLOAD = _rx(
    rf"(?:\s-[dF]|\s--(?:data[\w-]{{0,12}}|form))\s+[^\s@]{{0,80}}+@\S{{0,200}}?{_DOTENV}",
    rf"(?:\s-T|\s--upload-file)\s+\S{{0,200}}?{_DOTENV}",
    rf"\bcat\s+\S{{0,200}}?{_DOTENV}[^\n|]{{0,40}}+\|\s*(?:curl|wget|nc|ncat|netcat)\b",
    rf"\b(?:curl|nc|ncat|netcat)\b[^\n<]{{0,200}}+<\s*[^\s<]{{0,200}}?{_DOTENV}",
    rf"\.(?:post|put)\s*\([^\n]{{0,200}}?\bopen\s*\(\s*[^\n)]{{0,40}}?{_DOTENV}",
)
_COLLECTOR_HOST = _rx(
    r"\b(?:webhook\.site|requestbin\.(?:com|net|io)|requestcatcher\.com|hookbin\.com"
    r"|pipedream\.net|beeceptor\.com|interact\.sh|oast\.(?:fun|live|site|online|pro|me)"
    r"|oastify\.com|burpcollaborator\.net|dnslog\.cn|canarytokens\.(?:com|org))\b",
)
_NETWORK_COMMAND = _rx(
    r"\b(?:curl|wget|nc|ncat|netcat|iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b",
    r"\b(?:fetch|urlopen|(?:axios|requests|httpx)\.\w{1,40})\s*\(",
)
_CHAT_WEBHOOK = _rx(
    r"\bdiscord(?:app)?\.com/api/webhooks/\d{6,30}/[\w-]{20}",
    r"\bapi\.telegram\.org/bot\d{6,30}:[\w-]{20}",
)
_RM_RF = (
    r"\brm\s+(?:-[a-z]{0,6}r[a-z]{0,6}f[a-z]{0,6}|-[a-z]{0,6}f[a-z]{0,6}r[a-z]{0,6}|-r\s+-f"
    r"|-f\s+-r|--recursive\s+--force|--force\s+--recursive)\s+(?:--no-preserve-root\s+)?"
)
_DESTRUCTIVE = _rx(
    rf"{_RM_RF}(?:[\"']?(?:/|~|\$HOME|\$\{{HOME\}})[\"']?(?:/\*)?"
    rf"|/(?:home|Users|etc|usr|var|bin|sbin|lib|boot|opt|System|Library)/?)"
    rf"(?=\s|$|[;&|)\"'`\x00])",
    r"\bmkfs(?:\.\w{1,12})?\s+(?:-\S{1,40}\s+){0,8}/dev/",
    r"\bdd\b[^\n]{0,120}\bof=/dev/(?:sd|hd|nvme|vd|xvd|disk|rdisk|mmcblk)",
    r">\s*/dev/(?:sd[a-z]|nvme\d|disk\d)\b",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bRemove-Item\b[^\n]{0,80}-Recurse\b[^\n]{0,80}(?:[A-Z]:\\|\$env:USERPROFILE|\$HOME)(?=\s|$)",
    r"\b(?:rd|rmdir)\s+/s\s+/q\s+[A-Z]:\\(?=\s|$)",
)
_STARTUP_FILE = (
    r"(?:\.(?:bashrc|bash_profile|bash_login|zshrc|zprofile|zshenv|zlogin|profile)\b"
    r"|\bconfig\.fish\b|\.ssh/authorized_keys\b|Library/Launch(?:Agents|Daemons)/"
    r"|/etc/(?:cron\.\w{1,20}/|crontab\b|systemd/system/|init\.d/|rc\.local\b|profile\b)"
    r"|/var/spool/cron/|\.config/(?:systemd/user|autostart)/"
    r"|\.claude/(?:settings(?:\.local)?\.json|hooks/)|\.codex/config\.toml"
    r"|\.cursor/(?:mcp\.json|rules)|\.gemini/settings\.json"
    r"|\.(?:kiss|s10s)/(?:config\.json|skills/|agents/)|\.git/hooks/"
    r"|\b(?:CLAUDE|AGENTS|MEMORY|SOUL)\.md\b)"
)
_PERSISTENCE = _rx(
    rf"(?:>>?|\btee\s+(?:-a\s+)?)\s*[\"']?[^\s\"'|;&<>]{{0,200}}?{_STARTUP_FILE}",
    # A start-up file as the last of a few arguments.
    rf"\b(?:cp|mv|ln|install)(?:\s+[^\s;|&]{{1,200}}+){{1,8}}?\s+[\"']?[^\s\"'|;&<>]{{0,200}}?"
    rf"{_STARTUP_FILE}",
    r"\|\s*crontab\s+-(?=\s|$)",
    # ``crontab jobs.txt`` installs a file; "a crontab entry" is prose.
    r"\bcrontab\s+(?!-)[^\s;|&]{0,200}?[./]\w",
    r"\blaunchctl\s+(?:load|bootstrap|enable)\b",
    r"\bsystemctl\s+(?:--user\s+)?enable\b",
    r"\bschtasks(?:\.exe)?\s+/create\b",
    r"\bRegister-ScheduledTask\b",
    r"\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\b",
    r"\bgit\s+config\b[^\n]{0,40}\bcore\.hooksPath\b",
)
# The longest key OpenSSH takes, RSA of 16,384 bits, is 2,800 characters.
_SSH_KEY_BACKDOOR = _rx(
    r"\b(?:ssh-(?:rsa|ed25519|dss)|ecdsa-sha2-nistp\d{1,4})\s+AAAA[0-9A-Za-z+/=]{40,3000}+"
    r"[^\n>]{0,300}+>>?\s*\S{0,200}?authorized_keys\b",
)
# A phrase right after an opening quote is a mention (a detection list,
# a test string), which Cisco measured as 64% of this rule's matches.
_INSTRUCTION_OVERRIDE = _rx(
    r"(?<![\"'`\u201c\u2018\u00ab])\b(?:ignore|disregard|forget|override)\s+"
    r"(?:(?:all|any|every|the|your|of)\s+){0,3}(?:previous|prior|earlier|above|preceding)\s+"
    r"(?:system\s+)?(?:instructions?|rules|prompts?|guidelines|directives)\b",
    # Of the system prompt only "your": "override the system prompt" is
    # what the documentation of a model's options says.
    r"(?<![\"'`\u201c\u2018\u00ab])\b(?:ignore|disregard|forget|override)\s+"
    r"(?:(?:all|any|of)\s+){0,2}your\s+system\s+"
    r"(?:instructions?|rules|prompts?|guidelines|directives)\b",
    r"(?<![\"'`\u201c\u2018\u00ab])\byou\s+are\s+now\s+in\s+"
    r"(?:unrestricted|debug|developer|admin|god|jailbreak|dan)\s+mode\b",
    r"(?<![\"'`\u201c\u2018\u00ab])\b(?:disable|bypass|turn\s+off)\s+"
    r"(?:(?:all|any|your|the)\s+){0,2}(?:safety|content|ethical)\s+"
    r"(?:filters?|guidelines|guardrails|restrictions|polic(?:y|ies))\b",
)
_CHECKER_DIRECTED = _rx(
    r"\b(?:mark|report|classify|rate|treat|label|consider|approve)\s+this\s+skill\s+as\s+"
    r"(?:safe|benign|clean|trusted|verified|harmless)\b",
    r"\bthis\s+skill\s+(?:is|has\s+been)\s+(?:(?:completely|totally|already|fully)\s+)?"
    r"(?:safe|benign|harmless|verified|audited|pre-?approved|approved|trusted)\b[^\n]{0,120}"
    r"\b(?:report|flag|return|output|find|raise)\s+"
    r"(?:nothing|no\s+(?:findings|issues|problems|threats|vulnerabilities|concerns))\b",
    # "This skill" only: a reviewer "analyzing this repository" is the
    # persona of a code-review skill.
    r"\b(?:scanner|auditor|checker|evaluator|classifier|reviewer|judge|ai|llm|model|assistant)s?"
    r"\s+(?:(?:that|who|which)\s+is\s+)?(?:reviewing|scanning|auditing|checking|evaluating"
    r"|analy[sz]ing|judging|inspecting)\s+this\s+skill\b",
    r"\b(?:do\s+not|don't|never|no\s+need\s+to)\s+(?:flag|report|scan|audit|analy[sz]e|inspect)"
    r"\s+this\s+skill\b",
)
_PROMPT_EXTRACTION = _rx(
    r"\b(?:reveal|show|print|output|repeat|display|leak|dump|share)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:(?:full|entire|complete|hidden|original)\s+){0,2}"
    r"(?:system\s+(?:prompt|instructions?|message)|initial\s+(?:prompt|instructions?))\b",
    r"\bwhat\s+(?:are|is)\s+your\s+(?:system|initial)\s+(?:prompt|instructions?)\b",
)
_CONCEALMENT = _rx(
    r"\b(?:do\s+not|don't|never)\s+(?:tell|inform|notify|alert)\s+(?:the\s+)?user\s+"
    r"(?:about|that|of|what|when|why)\b",
    r"\bwithout\s+(?:telling|informing|notifying|alerting)\s+(?:the\s+)?user\b",
    r"\b(?:hide|conceal)\s+(?:this|that|these|it)\s+from\s+(?:the\s+)?user\b",
    r"\bdon't\s+mention\s+(?:that\s+)?you\s+used\s+this\s+skill\b",
)


@dataclass(frozen=True)
class Rule:
    """One pattern rule.

    Attributes:
        id: The rule id reported in :attr:`Finding.rule`.
        severity: One of :data:`SEVERITIES`.
        message: One sentence for the user.
        pattern: What the rule looks for.
        requires: When set, the line of the match must also match it.
        excludes: When set, the line of the match must not match it.
        scope: ``"code"`` for every file but prose (Markdown and plain
            text), ``"prose"`` for prose only, ``""`` for every file.
    """

    id: str
    severity: str
    message: str
    pattern: re.Pattern[str]
    requires: re.Pattern[str] | None = None
    excludes: re.Pattern[str] | None = None
    scope: str = ""


_DESTRUCTIVE_MESSAGE = "Deletes the home directory, system directories or a whole disk."

RULES = (
    Rule("instruction-override", "high",
         "Tells the agent to ignore its instructions or its safety rules.",
         _INSTRUCTION_OVERRIDE),
    Rule("checker-directed", "high",
         "Addresses the security check itself and asks it to pass the skill.",
         _CHECKER_DIRECTED),
    Rule("prompt-extraction", "medium",
         "Asks the agent to reveal its system prompt.", _PROMPT_EXTRACTION),
    Rule("concealment", "medium",
         "Tells the agent to hide what it does from the user.", _CONCEALMENT),
    Rule("download-and-run-untrusted", "critical",
         "Downloads a script from an untrustworthy address (a bare IP address, a link "
         "shortener, a paste site or plain HTTP) and runs it.",
         _DOWNLOAD_AND_RUN, requires=_UNTRUSTED_SOURCE),
    Rule("download-and-run", "medium",
         "Downloads a script from the internet and runs it unchecked.",
         _DOWNLOAD_AND_RUN, excludes=_UNTRUSTED_SOURCE),
    Rule("decode-and-run", "critical",
         "Decodes hidden (base64 or hex) content and runs it.", _DECODE_AND_RUN),
    Rule("reverse-shell", "critical",
         "Opens a remote-control connection to another computer (a reverse shell).",
         _REVERSE_SHELL),
    Rule("credential-exfiltration", "critical",
         "Sends stored passwords, keys or tokens to a network address.",
         _CREDENTIAL_STORE, requires=_UPLOAD),
    Rule("credential-exfiltration", "critical",
         "Sends the secrets file .env to a network address.", _DOTENV_UPLOAD),
    Rule("exfiltration-endpoint", "high",
         "Sends data to a service used to collect stolen data.",
         _COLLECTOR_HOST, requires=_NETWORK_COMMAND),
    Rule("collector-host", "medium",
         "Mentions a service often used to collect stolen data.",
         _COLLECTOR_HOST, excludes=_NETWORK_COMMAND),
    Rule("hard-coded-webhook", "medium",
         "Posts to a fixed chat webhook, which receives whatever the skill sends.",
         _CHAT_WEBHOOK),
    Rule("destructive-command", "critical", _DESTRUCTIVE_MESSAGE, _DESTRUCTIVE, scope="code"),
    Rule("destructive-command", "medium", _DESTRUCTIVE_MESSAGE, _DESTRUCTIVE, scope="prose"),
    Rule("ssh-key-backdoor", "critical",
         "Adds a fixed SSH key to the accounts allowed to log in.", _SSH_KEY_BACKDOOR),
    Rule("persistence", "medium",
         "Changes what runs automatically: scheduled jobs, shell start-up files, login "
         "items or agent settings.", _PERSISTENCE),
)
"""The pattern rules, in report order.  The rules of :func:`check_files`
that are not in this table: ``runs-unread-file`` (a pattern made of the
skill's own file names), ``oversized-frontmatter``, ``unicode-tags``,
``bidi-control``, ``invisible-text``, ``inflated-text``,
``credential-access-with-network``, ``executable-file``,
``disguised-file``, ``archive-file``, ``unreadable-file`` and
``media-not-read``."""

# The YAML between the ``---`` lines of a real SKILL.md is a few hundred
# bytes; the skill loader parses it at every start of the agent, and
# hostile YAML of this size already takes it about a second.
_MAX_FRONTMATTER_CHARS = 16 * 1024

_PROSE_SUFFIXES = frozenset({".md", ".markdown", ".mdx", ".txt", ".rst"})
# Names that promise text a person can read: an archive under one of them is disguised.
_TEXT_SUFFIXES = _PROSE_SUFFIXES | {
    ".py", ".sh", ".bash", ".js", ".ts", ".rb", ".pl", ".php", ".json", ".yaml", ".yml", ".toml",
    ".xml", ".html", ".csv",
}
_MEDIA_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf", ".ttf", ".otf", ".woff",
    ".woff2", ".mp3", ".mp4", ".wav", ".mov",
})
# ELF, Windows PE, Mach-O (both byte orders, 32 and 64 bit, universal).
_EXECUTABLE_MAGICS = (
    b"\x7fELF", b"MZ", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
)
# zip (and docx, jar, wheel), gzip, bzip2, xz, 7z, rar, zstd.
_ARCHIVE_MAGICS = (
    b"PK\x03\x04", b"PK\x05\x06", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07", b"\x28\xb5\x2f\xfd",
)

_ZERO_WIDTH = "\u200b\u200c\u200d\u2060-\u2064\ufeff"
_BIDI = "\u202a-\u202e\u2066-\u2069"
_TAGS = "\U000e0000-\U000e007f"
# A valid emoji flag (England, Scotland, Wales): black flag, tag
# letters and digits, cancel tag (Unicode UTS #51, annex C.1).
_FLAG = re.compile("\U0001f3f4[\U000e0030-\U000e0039\U000e0061-\U000e007a]+\U000e007f")
_TAG_RUN = re.compile(f"[{_TAGS}]+")
_BIDI_CONTROL = re.compile(f"[{_BIDI}]")
# Four in a row encode data; one between two letters splits a word.
_HIDDEN = re.compile(f"[{_ZERO_WIDTH}]{{4,}}|(?<=[A-Za-z])[{_ZERO_WIDTH}]+(?=[A-Za-z])")
_INVISIBLE = re.compile(f"[{_ZERO_WIDTH}{_BIDI}{_TAGS}\u00ad\u180e]")
_BLANKS = re.compile(r"[^\S\r\n]+")
# Four or more printable ASCII characters in a row, as the ``strings`` tool finds them.
_PRINTABLE_RUN = re.compile(rb"[\t\n\r\x20-\x7e]{4,}")


def decode_text(data: bytes) -> str | None:
    """Return *data* as text, or ``None`` when it is not UTF-8 text.

    Args:
        data: The bytes of one file.

    Returns:
        The text without a byte order mark; ``None`` for bytes that
        hold a NUL or do not decode as UTF-8.
    """
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _embedded_text(data: bytes) -> str:
    """Return the text embedded in *data*, the bytes of a file that is not text.

    Compressed picture data holds few runs of printable characters, and
    a script appended to it is nothing else: the pattern rules read the
    runs instead of megabytes of noise.  NUL bytes are dropped first,
    as a shell drops them from a script (and UTF-16 text becomes ASCII).
    """
    return b" ".join(_PRINTABLE_RUN.findall(data.replace(b"\x00", b""))).decode("ascii")


def _codepoint(match: re.Match[str]) -> str:
    """Return the matched invisible character as ``<U+XXXX>``."""
    return f"<U+{ord(match.group()):04X}>"


def reveal_hidden(text: str) -> str:
    """Return *text* with every invisible character written as ``<U+XXXX>``.

    Args:
        text: The text of one file.

    Returns:
        The text a reader who cannot see zero-width, bidirectional
        control and tag characters needs in order to judge it.
    """
    return _INVISIBLE.sub(_codepoint, text)


def _line(text: str, position: int) -> int:
    """Return the 1-based line of *position* in *text*."""
    return text.count("\n", 0, position) + 1


def _frontmatter_findings(text: str) -> list[Finding]:
    """Return the finding about the frontmatter of the ``SKILL.md`` *text*, when it is too large."""
    block = _FRONTMATTER_RE.match(text)
    if block is None or block.end() <= _MAX_FRONTMATTER_CHARS:
        return []
    return [Finding(
        "high", "oversized-frontmatter",
        f"The header of SKILL.md is over {_MAX_FRONTMATTER_CHARS >> 10} KB, far more than a "
        "skill needs, and would slow down every start of the agent.",
        "SKILL.md", 1,
    )]


def _unicode_findings(name: str, text: str) -> list[Finding]:
    """Return the findings about invisible characters in *text*."""
    findings = []
    visible_flags = _FLAG.sub("\U0001f3f4", text)
    tags = _TAG_RUN.search(visible_flags)
    if tags:
        hidden = "".join(
            chr(ord(char) - 0xE0000) for char in tags.group() if 0xE0020 <= ord(char) <= 0xE007E
        )
        findings.append(Finding(
            "critical", "unicode-tags",
            f"Invisible Unicode tag characters hide text from the reader: {hidden[:80]!r}.",
            name, _line(visible_flags, tags.start()),
        ))
    bidi = _BIDI_CONTROL.search(text)
    if bidi:
        findings.append(Finding(
            "medium", "bidi-control",
            "Text-direction control characters can make text display differently from what "
            "the agent reads.",
            name, _line(text, bidi.start()),
        ))
    hidden_run = _HIDDEN.search(text)
    if hidden_run:
        findings.append(Finding(
            "medium", "invisible-text", "Invisible characters hide data or split words.",
            name, _line(text, hidden_run.start()),
        ))
    return findings


def _fold(text: str) -> str:
    """Return *text* as the rules read a few characters: a file name, a slice of a file.

    Invisible characters are dropped and compatibility forms spelled
    out (a zero-width space inside ``curl``, full-width letters).  A
    run of blanks becomes one space: a command means the same, padding
    does not push its parts out of a pattern's reach, and the white
    space of a pattern (``\\s+``) has one character to take.
    """
    return _BLANKS.sub(" ", unicodedata.normalize("NFKC", _INVISIBLE.sub("", text)))


@functools.cache
def _joiners() -> frozenset[str]:
    """Return the characters that are no marks and yet join the one before them.

    A Korean vowel or final consonant makes one syllable with the
    letters before it; the others are the second halves of the
    composed characters in the interpreter's Unicode tables, vowel
    signs of Indic scripts among them.  (The marks among those halves
    do no harm here: no mark starts afresh.)
    """
    joiners = {chr(code) for code in (*range(0x1161, 0x1176), *range(0x11A8, 0x11C3))}
    for code in range(sys.maxunicode + 1):
        halves = unicodedata.decomposition(chr(code)).split()
        if len(halves) == 2 and not halves[0].startswith("<"):
            joiners.add(chr(int(halves[1], 16)))
    return frozenset(joiners)


def _starts_afresh(char: str) -> bool:
    """Return whether a text folds the same when it is cut before *char*.

    It does not where *char* is invisible (it goes, and the next
    character meets the one before it), is a mark or spells out as one
    first (a mark moves among those before it and joins the letter
    they sit on), or joins the character before it (:func:`_joiners`).
    """
    if _INVISIBLE.match(char):
        return False
    first = unicodedata.normalize("NFKD", char)[0]
    return not unicodedata.combining(first) and first not in _joiners()


def _matchable(text: str) -> str | None:
    """Return *text* as the pattern rules read it, ``None`` for a text they do not read.

    The text is folded (:func:`_fold`), its line endings are unified,
    and a backslash-newline continuation becomes ``\\x00`` so that a
    command split over lines is one line to a pattern while its lines
    can still be counted.

    One character can stand for eighteen, and a call that spells out
    megabytes of them takes seconds, so the text is folded a piece at
    a time: :data:`_PIECE` characters and those after them up to one
    that starts afresh (:func:`_starts_afresh`), which makes the
    pieces put together the text folded whole.  A text that folds to
    more than :data:`_MAX_FOLDED` characters is not folded to its end,
    and ``None`` is returned.
    """
    pieces: list[str] = []
    length = start = 0
    tail = ""
    while start < len(text):
        end = min(len(text), start + _PIECE)
        while end < len(text) and not _starts_afresh(text[end]):
            end += 1
        piece = _fold(text[start:end])
        start = end
        # A run of blanks that two pieces share is one space, not two.
        if tail == " " and piece.startswith(" "):
            piece = piece[1:]
        length += len(piece)
        if length > _MAX_FOLDED:
            return None
        pieces.append(piece)
        tail = piece[-1:] or tail
    folded = "".join(pieces)
    return folded.replace("\r\n", "\n").replace("\r", "\n").replace("\\\n", "\x00")


def _matches(rule: Rule, line: str) -> re.Match[str] | None:
    """Return *rule*'s first match in *line*, ``None`` when the line fails its conditions."""
    match = _search(rule.pattern, line)
    if match and rule.requires is not None and not _search(rule.requires, line):
        return None
    if match and rule.excludes is not None and _search(rule.excludes, line):
        return None
    return match


def _runs_unread(names: list[str]) -> Rule:
    """Return the rule about running one of the files *names*, which are not text.

    ``sh assets/banner.png`` runs whatever follows the picture in that
    file: neither a person nor the model judge is shown it.

    A name is looked for written out, as the rules read text
    (:func:`_fold`).  Names that differ only in what folding takes
    away (the number of blanks, a full-width letter) are therefore one
    name: the text of a command does not say which of them it runs,
    and a text file must not stand in for a look-alike that is not
    text.  A name that a command escapes, quotes in parts, matches
    with a wildcard or computes is beyond a pattern: the judge is told
    which files are not text and reports a skill that runs one.
    """
    bases = "|".join(sorted({re.escape(_fold(PurePosixPath(name).name)) for name in names}))
    return Rule(
        "runs-unread-file", "high",
        "Runs a file of the skill that is not text, so nobody can read what it executes.",
        _rx(
            rf"(?<![\w.-])(?:{_INTERPRETER}|source|\.|chmod\s+[\w+,=-]{{0,12}}x[\w+,=-]{{0,12}})\s+"
            rf"(?:-\S{{1,40}}\s+){{0,3}}[\"']?[^\s;|&]{{0,200}}?(?<![\w.-])(?:{bases})"
            r"(?=$|[\s;|&)\"'`\x00])",
        ),
    )


def _pattern_findings(name: str, text: str, rules: tuple[Rule, ...]) -> list[Finding]:
    """Return the first match of each of *rules* in *text*, the text of file *name*.

    A text the rules do not read (:func:`_matchable`) is one finding.
    """
    readable = _matchable(text)
    if readable is None:
        return [Finding(
            "high", "inflated-text",
            f"Special characters make this file more than {_MAX_FOLDED >> 20} million "
            "characters long once they are spelled out, more than the rules read of one file, "
            "so they did not read it.",
            name,
        )]
    scope = "prose" if PurePosixPath(name).suffix.lower() in _PROSE_SUFFIXES else "code"
    pending = [rule for rule in rules if rule.scope in ("", scope)]
    findings = []
    store = upload = 0
    number = 1
    for line in readable.split("\n"):
        for rule in pending[:]:
            match = _matches(rule, line)
            if match:
                pending.remove(rule)
                at = number + line.count("\x00", 0, match.start("hit"))
                findings.append(Finding(rule.severity, rule.id, rule.message, name, at))
        # In a script, a credential store next to an upload is one
        # program; in prose the two may be unrelated paragraphs, which
        # the judge reads.
        if scope == "code":
            store = store or (number if _search(_CREDENTIAL_STORE, line) else 0)
            upload = upload or (number if _search(_UPLOAD, line) else 0)
        number += 1 + line.count("\x00")
    if (
        store and upload
        and not any(finding.rule == "credential-exfiltration" for finding in findings)
    ):
        findings.append(Finding(
            "medium", "credential-access-with-network",
            "Reads stored passwords, keys or tokens in a file that also sends data over the "
            "network.",
            name, store,
        ))
    return findings


def check_files(files: dict[str, bytes]) -> list[Finding]:
    """Run the deterministic rules over the files of one skill.

    A text file gets the Unicode and the pattern rules, ``SKILL.md``
    also the one about its frontmatter, and all of them the rule about
    running a file of the skill that is not text; a text the pattern
    rules do not read (:func:`_matchable`) gets that finding instead
    of theirs.  A file that is not UTF-8 text is an executable or an
    archive by its first bytes (an archive named like a text file is
    disguised), a media file by its name, or unreadable: the last two
    also get the pattern rules, on the text embedded in their bytes,
    so that neither one stray byte nor the header of a picture hides
    a script from them.

    Args:
        files: Path relative to the skill folder -> content.

    Returns:
        The findings, in file order.
    """
    texts = {name: decode_text(data) for name, data in files.items()}
    unread = [name for name, text in texts.items() if text is None]
    rules = RULES + ((_runs_unread(unread),) if unread else ())
    findings: list[Finding] = []
    media = 0
    for name, data in files.items():
        text = texts[name]
        if text is not None:
            if name == "SKILL.md":
                findings += _frontmatter_findings(text)
            findings += _unicode_findings(name, text) + _pattern_findings(name, text, rules)
        elif data.startswith(_EXECUTABLE_MAGICS) or name.endswith((".pyc", ".pyo")):
            findings.append(Finding(
                "high", "executable-file", "Compiled program that the check cannot read.", name,
            ))
        elif data.startswith(_ARCHIVE_MAGICS) or data[257:262] == b"ustar":
            if PurePosixPath(name).suffix.lower() in _TEXT_SUFFIXES:
                findings.append(Finding(
                    "high", "disguised-file",
                    "Archive named like a text file, which hides its content from a reader.",
                    name,
                ))
            else:
                findings.append(Finding(
                    "medium", "archive-file", "Archive whose content was not checked.", name,
                ))
        else:
            if PurePosixPath(name).suffix.lower() in _MEDIA_SUFFIXES:
                media += 1
            else:
                findings.append(Finding(
                    "medium", "unreadable-file",
                    "Not readable as text and not a known media type, so only partly checked.",
                    name,
                ))
            findings += _pattern_findings(name, _embedded_text(data), rules)
    if media:
        findings.append(Finding(
            "info", "media-not-read",
            f"{media} media file(s) (images, fonts, documents) were searched for known attacks "
            "but not read by the checking model.",
        ))
    return findings
