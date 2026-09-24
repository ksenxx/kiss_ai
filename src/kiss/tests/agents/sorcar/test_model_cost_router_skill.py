# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the bundled ``model-cost-router`` project skill.

The skill lives in ``.agents/skills/model-cost-router`` at the repo root.
These tests check that Sorcar discovers it, that its frontmatter satisfies
the Agent Skills specification, and that ``scripts/route.py`` behaves as
documented when run as a real subprocess against real catalog files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar.skills import discover_skills, load_skill_content, parse_frontmatter

REPO_ROOT = Path(__file__).resolve().parents[5]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "model-cost-router"
ROUTE = SKILL_DIR / "scripts" / "route.py"
TIERS = json.loads((SKILL_DIR / "assets" / "tiers.json").read_text(encoding="utf-8"))


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every user-level location at *tmp_path* so only repo skills are seen."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claudehome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    return tmp_path


def system_python() -> str:
    """Return a Python interpreter that cannot import ``kiss`` (exercises the fallback path)."""
    python = shutil.which("python3", path="/usr/bin:/bin")
    assert python is not None
    probe = subprocess.run(
        [python, "-c", "import kiss"],
        capture_output=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    if probe.returncode == 0:
        pytest.skip("system python can import kiss; cannot exercise the no-kiss fallback")
    return python


def run_route(*args: str, python: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``route.py`` with *args* and return the completed process.

    With *python* unset the venv interpreter runs with the developer's real
    environment, so ``kiss`` is importable and credential filtering is live.
    """
    env = os.environ.copy() if python is None else {"PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [python or sys.executable, str(ROUTE), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        check=False,
    )


def write_catalog(tmp_path: Path, entries: object) -> str:
    """Write *entries* as a MODEL_INFO.json under *tmp_path* and return its path."""
    path = tmp_path / "MODEL_INFO.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def test_skill_is_discovered_from_repo_root(isolated_homes: Path) -> None:
    skills = discover_skills(str(REPO_ROOT))
    skill = skills["model-cost-router"]
    assert skill.source == "agents-project"
    assert "cheapest model tier" in skill.description
    content = load_skill_content(skill)
    assert "<file>scripts/route.py</file>" in content
    assert "<file>assets/tiers.json</file>" in content
    assert "decide()" in content


def test_frontmatter_follows_agent_skills_spec() -> None:
    parsed = parse_frontmatter(SKILL_DIR / "SKILL.md")
    assert parsed is not None
    meta, body = parsed
    assert meta["name"] == SKILL_DIR.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    assert 0 < len(meta["description"]) <= 1024
    assert 0 < len(meta["compatibility"]) <= 500
    assert isinstance(meta["metadata"], dict)
    assert body.count("\n") < 500
    for reference in (
        "references/routing-table.md",
        "references/evidence.md",
        "references/other-clients.md",
    ):
        assert reference in body
        assert (SKILL_DIR / reference).is_file()


def test_tiers_json_models_exist_in_catalog() -> None:
    catalog = json.loads(
        (REPO_ROOT / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json").read_text()
    )
    for tier in ("small", "medium", "frontier"):
        assert TIERS[tier], tier
        for candidate in TIERS[tier]:
            assert candidate["model"] in catalog, candidate["model"]
            assert candidate["input"] <= candidate["output"]


def test_pick_uses_catalog_prices(tmp_path: Path) -> None:
    first = TIERS["small"][0]["model"]
    catalog = write_catalog(
        tmp_path, {first: {"input_price_per_1M": 9.0, "output_price_per_1M": 90.0}}
    )
    result = run_route(
        "pick",
        "--tier",
        "small",
        "--in",
        "1000000",
        "--out",
        "100000",
        "--catalog",
        catalog,
        python=system_python(),
    )
    assert result.returncode == 0, result.stderr
    chosen = json.loads(result.stdout)
    assert chosen["model"] == first
    assert chosen["input_per_1M"] == 9.0 and chosen["output_per_1M"] == 90.0
    assert chosen["estimated_usd"] == 18.0
    assert chosen["tier"] == "small" and chosen["runnable"] is None


def test_pick_with_kiss_filters_by_configured_credentials() -> None:
    result = run_route("pick", "--tier", "small")
    if result.returncode == 1:
        assert "no runnable model in tier 'small'" in result.stderr
        return
    chosen = json.loads(result.stdout)
    assert chosen["runnable"] is True
    assert chosen["model"] in {c["model"] for c in TIERS["small"]}


def test_pick_skips_every_candidate_when_no_credential_is_configured(tmp_path: Path) -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "KISS_HOME": str(tmp_path / "kiss"),
        "HOME": str(tmp_path / "home"),
    }
    for name in ("kiss", "home"):
        (tmp_path / name).mkdir()
    menu = subprocess.run(
        [sys.executable, str(ROUTE), "menu"], capture_output=True, text=True, env=env, check=False
    )
    assert menu.returncode == 0, menu.stderr
    assert all(c["runnable"] is False for tier in json.loads(menu.stdout).values() for c in tier)
    pick = subprocess.run(
        [sys.executable, str(ROUTE), "pick", "--tier", "medium"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert pick.returncode == 1
    assert "no runnable model in tier 'medium'" in pick.stderr


def test_pick_excludes_failed_models_and_fails_when_tier_is_empty(tmp_path: Path) -> None:
    models = [c["model"] for c in TIERS["frontier"]]
    catalog = write_catalog(tmp_path, {})
    result = run_route(
        "pick",
        "--tier",
        "frontier",
        "--exclude",
        models[0],
        "--catalog",
        catalog,
        python=system_python(),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model"] == models[1]

    result = run_route(
        "pick",
        "--tier",
        "frontier",
        "--exclude",
        *models,
        "--catalog",
        catalog,
        python=system_python(),
    )
    assert result.returncode == 1
    assert "no runnable model in tier 'frontier'" in result.stderr


def test_menu_ignores_non_dict_catalog_and_reports_unknown_availability(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path, [])
    result = run_route(
        "menu", "--in", "2000000", "--out", "0", "--catalog", catalog, python=system_python()
    )
    assert result.returncode == 0, result.stderr
    menu = json.loads(result.stdout)
    assert set(menu) == {"small", "medium", "frontier"}
    for tier in ("small", "medium", "frontier"):
        for candidate, expected in zip(menu[tier], TIERS[tier], strict=True):
            assert candidate["runnable"] is None
            assert candidate["model"] == expected["model"]
            assert candidate["estimated_usd"] == round(expected["input"] * 2, 4)


def test_missing_catalog_and_entry_without_prices_fall_back_to_tiers_json(tmp_path: Path) -> None:
    first = TIERS["medium"][0]
    result = run_route(
        "pick",
        "--tier",
        "medium",
        "--catalog",
        str(tmp_path / "absent.json"),
        python=system_python(),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model"] == first["model"]

    catalog = write_catalog(tmp_path, {first["model"]: {"context_length": 1}})
    result = run_route("pick", "--tier", "medium", "--catalog", catalog, python=system_python())
    assert result.returncode == 0, result.stderr
    chosen = json.loads(result.stdout)
    assert chosen["input_per_1M"] == first["input"] and chosen["output_per_1M"] == first["output"]


def test_estimate_from_catalog_tiers_and_unknown(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path, {"custom/model": {"input_price_per_1M": 1.0, "output_price_per_1M": 2.0}}
    )
    result = run_route(
        "estimate",
        "--model",
        "custom/model",
        "--in",
        "1000000",
        "--out",
        "1000000",
        "--catalog",
        catalog,
    )
    assert result.returncode == 0 and json.loads(result.stdout) == {
        "model": "custom/model",
        "estimated_usd": 3.0,
    }

    tier_model = TIERS["frontier"][0]
    result = run_route(
        "estimate",
        "--model",
        tier_model["model"],
        "--in",
        "1000000",
        "--out",
        "0",
        "--catalog",
        catalog,
    )
    assert (
        result.returncode == 0 and json.loads(result.stdout)["estimated_usd"] == tier_model["input"]
    )

    result = run_route("estimate", "--model", "does/not-exist", "--catalog", catalog)
    assert result.returncode == 1 and "unknown model 'does/not-exist'" in result.stderr


def test_log_creates_header_then_appends_rows(tmp_path: Path) -> None:
    ledger = tmp_path / "nested" / "MODEL_DECISIONS.md"
    first = run_route(
        "log",
        "--unit",
        "grep | report",
        "--tier",
        "small",
        "--model",
        "m1",
        "--reason",
        "lookup",
        "--file",
        str(ledger),
    )
    assert first.returncode == 0 and first.stdout.strip() == f"logged to {ledger}"
    second = run_route(
        "log",
        "--unit",
        "fix",
        "--tier",
        "medium",
        "--model",
        "m2",
        "--reason",
        "known\n  cause",
        "--outcome",
        "passed",
        "--file",
        str(ledger),
    )
    assert second.returncode == 0
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# Model routing decisions"
    assert lines[2] == "| time (UTC) | unit | tier | model | reason | outcome |"
    assert lines[4].endswith("| grep / report | small | m1 | lookup | pending |")
    assert lines[5].endswith("| fix | medium | m2 | known cause | passed |")
    assert len(lines) == 6


def test_log_requires_a_known_tier(tmp_path: Path) -> None:
    result = run_route(
        "log",
        "--unit",
        "u",
        "--tier",
        "huge",
        "--model",
        "m",
        "--reason",
        "r",
        "--file",
        str(tmp_path / "l.md"),
    )
    assert result.returncode == 2
    assert "invalid choice: 'huge'" in result.stderr
