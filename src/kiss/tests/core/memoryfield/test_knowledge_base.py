"""End-to-end tests for the repo knowledge base in ``knowledge/`` and its search test script."""

import subprocess
import sys
from pathlib import Path

import yaml

from kiss.core.memoryfield import MemoryDir
from kiss.core.memoryfield.pages import MAX_PAGE_BYTES

REPO = Path(__file__).resolve().parents[5]
KNOWLEDGE = REPO / "knowledge"
SCRIPT = KNOWLEDGE / "eval" / "search_eval.py"


def run_eval(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the search test script offline and capture its output."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--hashed", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def write_kb(root: Path, gold: str) -> Path:
    """Create a three-page knowledge base and a question file whose gold page is *gold*."""
    memory = MemoryDir(root)
    memory.write("git-squash-merge", "Squash merge of the task branch.", summary="squash merge")
    memory.write("web-stealth", "Captcha and bot detection evasion.", summary="captcha stealth")
    memory.write("zebra-quartz", "Zebra quartz.", title="zebra", summary="quartz")
    questions = root / "q.yaml"
    questions.write_text(yaml.safe_dump([{"q": "captcha bot detection", "gold": [gold]}]))
    return questions


def test_every_page_is_valid_and_gold_pages_exist() -> None:
    memory = MemoryDir(KNOWLEDGE)
    names = memory.page_names()
    assert len(names) > 100
    for name in names:
        page = memory.read(name)
        assert page.title and page.summary and page.frontmatter.get("uuid"), name
        assert len(page.raw.encode()) <= MAX_PAGE_BYTES, name
    questions = yaml.safe_load((KNOWLEDGE / "eval" / "questions.yaml").read_text())
    assert {g for q in questions for g in q["gold"]} <= set(names)


def test_eval_reports_hit(tmp_path: Path) -> None:
    questions = write_kb(tmp_path, "web-stealth")
    result = run_eval("--dir", str(tmp_path), "--questions", str(questions), "--min-recall5", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "#1  captcha bot detection  (top: web-stealth)" in result.stdout
    assert "R@1 1.00" in result.stdout


def test_eval_fails_below_min_recall(tmp_path: Path) -> None:
    # The hashed embedder gives this page (no shared words) score 0, so search drops it.
    questions = write_kb(tmp_path, "zebra-quartz")
    result = run_eval("--dir", str(tmp_path), "--questions", str(questions), "--min-recall5", "1")
    assert result.returncode == 1
    assert "miss  captcha bot detection" in result.stdout


def test_eval_rejects_unknown_gold_page(tmp_path: Path) -> None:
    questions = write_kb(tmp_path, "no-such-page")
    result = run_eval("--dir", str(tmp_path), "--questions", str(questions))
    assert result.returncode == 1
    assert "Gold pages missing" in result.stdout and "no-such-page" in result.stdout


def test_eval_on_empty_index_misses(tmp_path: Path) -> None:
    (tmp_path / "q.yaml").write_text(yaml.safe_dump([{"q": "anything", "gold": []}]))
    result = run_eval("--dir", str(tmp_path), "--questions", str(tmp_path / "q.yaml"))
    assert result.returncode == 0
    assert "miss  anything  (top: -)" in result.stdout
