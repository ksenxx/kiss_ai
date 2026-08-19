# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cross-vendor regression audit for reasoning-effort alias generation.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.scripts.test_update_models_thinking_audit``; the non-core tests remain there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MOONSHOT_LEVELS = ("low", "high", "max")


GROK_EFFORT_LEVELS = ("low", "medium", "high")


GLM_5_2_LEVELS = ("high", "max")


ALL_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _catalog() -> dict[str, dict]:
    """Return the on-disk ``MODEL_INFO.json`` as a plain dict.

    Used by the positive lock-in tests to guarantee the currently
    correct catalog shape (K3 + gpt-5.x aliases present, no aliases for
    unsupported vendors) does not silently regress.
    """
    from kiss.core.models import model_info as mi

    return json.loads(  # type: ignore[no-any-return]
        Path(mi.__file__).parent.joinpath("MODEL_INFO.json").read_text()
    )


def _assert_alias_field_parity(data: dict[str, dict], base: str, alias: str) -> None:
    """Assert ``alias`` mirrors ``base`` in every non-thinking/alias field."""
    base_fields = {k: v for k, v in data[base].items() if k not in ("thinking", "alias_of")}
    alias_fields = {k: v for k, v in data[alias].items() if k not in ("thinking", "alias_of")}
    assert alias_fields == base_fields, (
        f"{alias} must match {base} byte-for-byte outside thinking/alias_of; "
        f"got {alias_fields!r} vs {base_fields!r}"
    )


class TestCatalogAliasesFrozen:
    """Currently-shipping alias sets must not silently disappear."""

    def test_kimi_k3_family_has_low_high_max_aliases(self) -> None:
        """The K3 fix guarantees three routes × three-level ladder = nine aliases."""
        data = _catalog()
        bases = ("kimi-k3", "moonshotai/Kimi-K3", "openrouter/moonshotai/kimi-k3")
        for base in bases:
            assert base in data, f"Base {base} missing"
            assert data[base].get("thinking") == "high", (
                f"Base {base} must be capped at high; got {data[base].get('thinking')!r}"
            )
            assert "alias_of" not in data[base]
            for level in MOONSHOT_LEVELS:
                alias = f"{base}-{level}"
                assert alias in data, f"Missing alias {alias}"
                assert data[alias].get("thinking") == level, (
                    f"{alias} must have thinking={level!r}"
                )
                assert data[alias].get("alias_of") == base, (
                    f"{alias} must carry alias_of={base!r}"
                )
            # Guard: never emit off-scale aliases for Moonshot.
            for off_scale in ("medium", "xhigh"):
                assert f"{base}-{off_scale}" not in data, (
                    f"{base}-{off_scale} must not exist on the Moonshot scale"
                )

    def test_gpt_5_family_has_openai_ladder_aliases(self) -> None:
        """The historical gpt-5.5+ xhigh split still produces the full ladder."""
        data = _catalog()
        # A representative subset covering direct + OpenRouter routes.
        for base in (
            "gpt-5.5",
            "gpt-5.6-sol",
            "openrouter/openai/gpt-5.5",
            "openrouter/openai/gpt-5.6-sol",
            "openrouter/~openai/gpt-latest",
        ):
            assert base in data, f"Base {base} missing"
            assert data[base].get("thinking") == "high", base
            for level in ("low", "medium", "high", "xhigh"):
                alias = f"{base}-{level}"
                assert alias in data, f"Missing OpenAI-ladder alias {alias}"
                assert data[alias].get("thinking") == level
                assert data[alias].get("alias_of") == base

    def test_no_stray_max_aliases_on_openai_models(self) -> None:
        """The Moonshot-only ``-max`` suffix must never appear on OpenAI keys."""
        data = _catalog()
        for name in data:
            if not name.endswith("-max"):
                continue
            base = name.removesuffix("-max")
            # An OpenAI base would never carry a -max alias because the
            # OpenAI ladder tops out at xhigh.
            if base in data and base.startswith(("gpt-", "o1", "o3", "o4")):
                pytest.fail(
                    f"OpenAI-family base {base!r} has a -max alias; "
                    "Moonshot scale mistakenly applied?"
                )

    def test_no_stray_xhigh_aliases_on_moonshot_models(self) -> None:
        """The OpenAI-only ``-xhigh`` suffix must never appear on Moonshot keys."""
        data = _catalog()
        for name in data:
            if not name.endswith("-xhigh"):
                continue
            base = name.removesuffix("-xhigh")
            if base in data:
                if base.startswith("kimi-") or base.startswith("moonshotai/") or (
                    base.startswith("openrouter/moonshotai/")
                ):
                    pytest.fail(
                        f"Moonshot base {base!r} has an -xhigh alias; "
                        "OpenAI scale mistakenly applied?"
                    )

    def test_no_medium_aliases_on_moonshot_or_grok_3_mini(self) -> None:
        """Moonshot scale has no ``medium``.  grok-3-mini (now on the
        2-level scale) also has no ``medium`` — its aliases are not in
        the catalog yet (they materialize on the next re-probe) so this
        test is the guard against a mis-fix that reintroduces it."""
        data = _catalog()
        for name in data:
            if not name.endswith("-medium"):
                continue
            base = name.removesuffix("-medium")
            if base.startswith("kimi-") or base.startswith("moonshotai/") or (
                base.startswith("openrouter/moonshotai/")
            ):
                pytest.fail(
                    f"Moonshot base {base!r} has a bogus -medium alias"
                )
            base_last = base.rsplit("/", 1)[-1].lower()
            if base_last.startswith("grok-3-mini"):
                pytest.fail(
                    f"grok-3-mini base {base!r} has a bogus -medium alias; "
                    "the model only accepts low/high"
                )

    def test_generated_aliases_carry_alias_of_marker(self) -> None:
        """Every ``-{level}`` sibling on every scale must carry ``alias_of``.

        The marker is what lets ``update_models.py`` and the runtime
        alias resolver distinguish synthetic aliases from real upstream
        models whose names happen to end in ``-low`` / ``-medium`` /
        ``-high`` / ``-xhigh`` / ``-max``.
        """
        data = _catalog()
        for name, entry in data.items():
            if entry.get("alias_of"):
                assert name.endswith(tuple(f"-{lvl}" for lvl in ALL_LEVELS)), (
                    f"alias_of set but name {name!r} does not end in a level suffix"
                )
                base = entry["alias_of"]
                assert base in data, (
                    f"Dangling alias {name!r} points at missing base {base!r}"
                )
                assert base == name.rsplit("-", 1)[0], (
                    f"alias_of on {name!r} inconsistent with its suffix"
                )


class TestAuditGapTripwires:
    """Positive lock-ins for the four audit gaps, now fixed.

    Each docstring still records the gap letter (A/B/C/D) from the audit
    report so a bisector can jump straight to the corresponding
    recommended-fix section.
    """

    def test_gap_d_openrouter_gpt_oss_has_effort_aliases(self) -> None:
        """Post-fix: OpenRouter gpt-oss rows carry the 3-level ladder."""
        data = _catalog()
        for base in (
            "openrouter/openai/gpt-oss-120b",
            "openrouter/openai/gpt-oss-20b",
            "openrouter/openai/gpt-oss-safeguard-20b",
        ):
            assert data.get(base, {}).get("thinking") == "high", (
                f"{base} must have thinking='high' after materialization"
            )
            for level in ("low", "medium", "high"):
                alias = f"{base}-{level}"
                assert alias in data, f"Missing gpt-oss alias {alias}"
                assert data[alias].get("thinking") == level
                assert data[alias].get("alias_of") == base
            # Never emit -xhigh for gpt-oss.
            assert f"{base}-xhigh" not in data, (
                f"{base}-xhigh must not exist; gpt-oss ladder tops at 'high'"
            )


class TestNewFamilyCatalogAliasesFrozen:
    """Shipped alias sets for Grok / GLM-5.2 / gpt-oss must not regress."""

    def test_grok_effort_family_has_low_medium_high_aliases(self) -> None:
        """Gap A lock-in: grok-4.5 / grok-4.3 ship the 3-level Grok ladder."""
        data = _catalog()
        for base in ("openrouter/x-ai/grok-4.5", "openrouter/x-ai/grok-4.3"):
            assert base in data, f"Base {base} missing"
            assert data[base].get("thinking") == "high", (
                f"Base {base} must store thinking='high'; "
                f"got {data[base].get('thinking')!r}"
            )
            assert "alias_of" not in data[base]
            for level in GROK_EFFORT_LEVELS:
                alias = f"{base}-{level}"
                assert alias in data, f"Missing Grok alias {alias}"
                assert data[alias].get("thinking") == level
                assert data[alias].get("alias_of") == base
                _assert_alias_field_parity(data, base, alias)
            for off_scale in ("xhigh", "max"):
                assert f"{base}-{off_scale}" not in data, (
                    f"{base}-{off_scale} must not exist; the Grok effort "
                    "ladder tops at 'high'"
                )

    def test_glm_5_2_family_has_high_max_aliases(self) -> None:
        """Gap B lock-in: GLM-5.2 ships the 2-level (high, max) ladder."""
        data = _catalog()
        for base in ("zai-org/GLM-5.2", "openrouter/z-ai/glm-5.2"):
            assert base in data, f"Base {base} missing"
            assert data[base].get("thinking") == "high", (
                f"Base {base} must be capped at high; "
                f"got {data[base].get('thinking')!r}"
            )
            assert "alias_of" not in data[base]
            for level in GLM_5_2_LEVELS:
                alias = f"{base}-{level}"
                assert alias in data, f"Missing GLM-5.2 alias {alias}"
                assert data[alias].get("thinking") == level
                assert data[alias].get("alias_of") == base
                _assert_alias_field_parity(data, base, alias)
            for off_scale in ("low", "medium", "xhigh"):
                assert f"{base}-{off_scale}" not in data, (
                    f"{base}-{off_scale} must not exist; GLM-5.2's ladder "
                    "is exactly (high, max)"
                )

    def test_together_gpt_oss_has_low_medium_high_aliases(self) -> None:
        """Gap C lock-in: Together-route gpt-oss ships the 3-level ladder."""
        data = _catalog()
        for base in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
            assert base in data, f"Base {base} missing"
            assert data[base].get("thinking") == "high", (
                f"Base {base} must store thinking='high'; "
                f"got {data[base].get('thinking')!r}"
            )
            assert "alias_of" not in data[base]
            for level in ("low", "medium", "high"):
                alias = f"{base}-{level}"
                assert alias in data, f"Missing gpt-oss alias {alias}"
                assert data[alias].get("thinking") == level
                assert data[alias].get("alias_of") == base
                _assert_alias_field_parity(data, base, alias)
            for off_scale in ("xhigh", "max"):
                assert f"{base}-{off_scale}" not in data, (
                    f"{base}-{off_scale} must not exist; gpt-oss rejects "
                    "xhigh and max"
                )
