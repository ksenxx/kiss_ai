"""End-to-end README checks for Z.AI/Moonshot support and MiniMax removal.

The README is a user-facing integration surface: it documents which API keys
users should configure and which provider categories/models the bundled catalog
contains. These tests ensure that the README stays aligned with
``MODEL_INFO.json`` after replacing MiniMax support with Z.AI and Moonshot AI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kiss.core.models.model_info import get_model_provider

_REPO_ROOT = Path(__file__).resolve().parents[3]
_README = _REPO_ROOT / "README.md"
_MODEL_INFO = _REPO_ROOT / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"


def _readme_text() -> str:
    # UTF-8 explicitly: the README has non-cp1252 characters and Windows'
    # default text encoding is cp1252.
    return _README.read_text(encoding="utf-8")


def _model_info() -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = json.loads(_MODEL_INFO.read_text(encoding="utf-8"))
    return data


# Runtime routing label -> README category label; mirrors
# kiss.scripts.update_models._PROVIDER_LABEL_TO_README_CATEGORY.
_README_CATEGORY_LABELS = {
    "OpenAI": "OpenAI",
    "Anthropic": "Anthropic",
    "Gemini": "Gemini",
    "Together": "Together AI",
    "Z.AI": "Z.AI",
    "Moonshot": "Moonshot AI",
    "OpenRouter": "OpenRouter",
    "Claude Code CLI": "Claude Code CLI (`cc/*`)",
    "Codex CLI": "Codex CLI (`codex/*`)",
}


def _provider_category(model_name: str) -> str:
    """README category for ``model_name``, grouped by ROUTING provider.

    Uses the runtime routing table
    (:func:`kiss.core.models.model_info.get_model_provider`) — the same
    single source the README rewriter consults — so the open-weight
    ``openai/gpt-oss-*`` and ``google/gemma-*`` models count under
    Together AI (they are served with a ``TOGETHER_API_KEY``) rather than
    under the OpenAI/Gemini vendor families. A ``KeyError`` here means a
    catalog model that no registered provider routes.
    """
    return _README_CATEGORY_LABELS[get_model_provider(model_name)]


def _provider_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for model_name in _model_info():
        category = _provider_category(model_name)
        counts[category] = counts.get(category, 0) + 1
    return counts


def test_readme_has_no_minimax_references() -> None:
    """The user-facing README should not advertise removed MiniMax support."""
    readme = _readme_text()
    assert re.search(r"minimax", readme, flags=re.IGNORECASE) is None
    assert "| MiniMax |" not in readme
    assert "<strong>MiniMax" not in readme


def test_readme_documents_zai_and_moonshot_api_keys() -> None:
    """The setup snippet documents the new Z.AI and Moonshot env vars."""
    readme = _readme_text()
    assert "export ZAI_API_KEY=..." in readme
    assert "export MOONSHOT_API_KEY=..." in readme
    assert "export MINIMAX_API_KEY=..." not in readme


def test_readme_provider_table_matches_catalog_categories() -> None:
    """Provider-category totals in the README match MODEL_INFO.json."""
    readme = _readme_text()
    counts = _provider_counts()
    total = sum(counts.values())
    header = f"**{total} models** across **{len(counts)} provider categories**"
    assert header in readme
    for provider, count in counts.items():
        assert f"| {provider} | {count} |" in readme
    assert "| Z.AI | 8 |" in readme
    assert "| Moonshot AI | 10 |" in readme


def test_readme_capability_totals_match_catalog() -> None:
    """Capability totals in the README match MODEL_INFO.json."""
    readme = _readme_text()
    model_info = _model_info()
    generation_count = sum(1 for entry in model_info.values() if entry.get("gen"))
    function_calling_count = sum(1 for entry in model_info.values() if entry.get("fc"))
    embedding_count = sum(1 for entry in model_info.values() if entry.get("emb"))
    decisions_count = sum(1 for entry in model_info.values() if entry.get("dec"))
    assert f"- **{generation_count}** generation-capable models" in readme
    assert f"- **{function_calling_count}** function-calling-capable models" in readme
    assert f"- **{embedding_count}** embedding models" in readme
    assert f"- **{decisions_count}** decision models" in readme


def test_readme_lists_zai_and_moonshot_models() -> None:
    """The full model list includes direct Z.AI and Moonshot AI sections."""
    readme = _readme_text()
    assert "<summary><strong>Z.AI (8)</strong></summary>" in readme
    assert "<summary><strong>Moonshot AI (10)</strong></summary>" in readme
    for model_name in (
        "glm-4.6",
        "glm-4.7",
        "kimi-k2.6",
        "kimi-k3-low",
        "kimi-k3-high",
        "kimi-k3-max",
        "moonshot-v1-32k",
    ):
        assert f"- `{model_name}`" in readme
