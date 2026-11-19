"""End-to-end README/MODELS.md checks for the retired providers and the catalog totals.

The README and MODELS.md are user-facing integration surfaces: the README
documents which API keys users should configure and the catalog totals, and
MODELS.md the provider categories and models the bundled catalog contains.
These tests ensure that both stay aligned with ``MODEL_INFO.json`` now that
MiniMax and Together AI are gone and Z.AI's and Moonshot AI's models are
reachable through OpenRouter only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from kiss.core.models.model_info import get_model_provider

_REPO_ROOT = Path(__file__).resolve().parents[3]
_README = _REPO_ROOT / "README.md"
_MODELS = _REPO_ROOT / "MODELS.md"
_MODEL_INFO = _REPO_ROOT / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"


def _readme_text() -> str:
    # UTF-8 explicitly: the README has non-cp1252 characters and Windows'
    # default text encoding is cp1252.
    return _README.read_text(encoding="utf-8")


def _models_text() -> str:
    return _MODELS.read_text(encoding="utf-8")


def _model_info() -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = json.loads(_MODEL_INFO.read_text(encoding="utf-8"))
    return data


# Runtime routing label -> README category label; mirrors
# kiss.scripts.update_models._PROVIDER_LABEL_TO_README_CATEGORY.
_README_CATEGORY_LABELS = {
    "OpenAI": "OpenAI",
    "Anthropic": "Anthropic",
    "Bedrock": "Amazon Bedrock (`bedrock/*`)",
    "Gemini": "Gemini",
    "OpenRouter": "OpenRouter",
    "Claude Code CLI": "Claude Code CLI (`cc/*`)",
    "Codex CLI": "Codex CLI (`codex/*`)",
}


def _provider_category(model_name: str) -> str:
    """README category for ``model_name``, grouped by ROUTING provider.

    Uses the runtime routing table
    (:func:`kiss.core.models.model_info.get_model_provider`) — the same
    single source the README rewriter consults. A ``KeyError`` here means a
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
    """The user-facing README and MODELS.md should not advertise removed MiniMax support."""
    for text in (_readme_text(), _models_text()):
        assert re.search(r"minimax", text, flags=re.IGNORECASE) is None
        assert "| MiniMax |" not in text
        assert "<strong>MiniMax" not in text


def test_readme_documents_the_supported_api_keys_and_no_retired_one() -> None:
    """The setup snippet exports the four supported keys; no retired key is documented."""
    readme = _readme_text()
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        assert f"export {key}=..." in readme
    for text in (readme, _models_text()):
        for key in ("MINIMAX_API_KEY", "TOGETHER_API_KEY", "ZAI_API_KEY", "MOONSHOT_API_KEY"):
            assert key not in text


def test_readme_provider_table_matches_catalog_categories() -> None:
    """Provider-category totals in README.md and MODELS.md match MODEL_INFO.json."""
    models = _models_text()
    counts = _provider_counts()
    total = sum(counts.values())
    header = f"**{total} models** across **{len(counts)} provider categories**"
    assert header in _readme_text()
    assert header in models
    for provider, count in counts.items():
        assert f"| {provider} | {count} |" in models


def test_readme_capability_totals_match_catalog() -> None:
    """Capability totals in MODELS.md match MODEL_INFO.json."""
    readme = _models_text()
    model_info = _model_info()
    generation_count = sum(1 for entry in model_info.values() if entry.get("gen"))
    function_calling_count = sum(1 for entry in model_info.values() if entry.get("fc"))
    embedding_count = sum(1 for entry in model_info.values() if entry.get("emb"))
    decisions_count = sum(1 for entry in model_info.values() if entry.get("dec"))
    assert f"- **{generation_count}** generation-capable models" in readme
    assert f"- **{function_calling_count}** function-calling-capable models" in readme
    assert f"- **{embedding_count}** embedding models" in readme
    assert f"- **{decisions_count}** decision models" in readme


def test_models_md_lists_zai_and_moonshot_models_under_openrouter_only() -> None:
    """MODELS.md has no Together AI, Z.AI or Moonshot AI section; their models are OpenRouter's."""
    models = _models_text()
    for provider in ("Together AI", "Z.AI", "Moonshot AI"):
        assert f"| {provider} |" not in models
        assert f"<summary><strong>{provider} (" not in models
    for retired in ("glm-4.7", "kimi-k3", "moonshot-v1-32k", "moonshotai/Kimi-K3"):
        assert f"- `{retired}`" not in models
    for routed in ("openrouter/z-ai/glm-4.7", "openrouter/moonshotai/kimi-k3"):
        assert f"- `{routed}`" in models
