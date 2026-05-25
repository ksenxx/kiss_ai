"""Integration tests for the VS Code model-picker default value.

Bug: on a fresh install the picker rendered blank because
``getDefaultModel()`` in ``DependencyInstaller.ts`` returned ``""`` whenever
``uv`` / the embedded KISS project ``.venv`` was not yet available.  The
extension constructor then assigned that empty string to ``_selectedModel``
and the initial HTML rendered ``<span id="model-name"></span>``.  The
backend's ``models`` event eventually arrives and corrects the picker,
but only after the daemon is up — which can be minutes during first-time
dependency installation, leaving the picker visibly empty for the user.

The fix introduces a pure-TypeScript fallback that mirrors Python's
``kiss.core.models.model_info.get_default_model`` priority order and is
invoked whenever the Python call fails.  These tests pin the contract:

* The TS-side ``getFallbackDefaultModel`` exists, is exported, and is
  reachable from every failure branch of ``getDefaultModel``.
* The TS fallback's model strings and priority match the Python source
  one-for-one, so the picker shows the same value the backend would
  eventually broadcast.
* ``SorcarSidebarView``'s constructor still consults the fallback via
  ``getDefaultModel()``, so the initial HTML embeds a non-empty model
  name even before the daemon is reachable.
"""

import re
import unittest
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[3]
VSCODE_SRC = REPO_SRC / "agents" / "vscode" / "src"
INSTALLER_SOURCE = (VSCODE_SRC / "DependencyInstaller.ts").read_text()
SIDEBAR_SOURCE = (VSCODE_SRC / "SorcarSidebarView.ts").read_text()
MODEL_INFO_SOURCE = (REPO_SRC / "core" / "models" / "model_info.py").read_text()


def _python_default_model_priority() -> list[tuple[str, str]]:
    """Parse the Python ``get_default_model`` body into (trigger, model) pairs.

    Returns the priority list in declared order.  ``trigger`` is the
    canonical key/CLI name (``ANTHROPIC_API_KEY``, ``claude``,
    ``codex``, ``""`` for the final ``"No model"`` sentinel).
    """
    body_match = re.search(
        r"^def get_default_model\(\)[^\n]*:\n(?P<body>.*?)\n(?=^\S|\Z)",
        MODEL_INFO_SOURCE,
        re.DOTALL | re.MULTILINE,
    )
    assert body_match is not None, (
        "could not locate get_default_model() in model_info.py"
    )
    body = body_match.group("body")

    pairs: list[tuple[str, str]] = []
    for trigger, model in re.findall(
        r"keys\.(\w+):\s*\n\s*return\s+\"([^\"]+)\"",
        body,
    ):
        pairs.append((trigger, model))
    for cli, model in re.findall(
        r"shutil\.which\(\"(\w+)\"\)[^\n]*\n\s*return\s+\"([^\"]+)\"",
        body,
    ):
        pairs.append((cli, model))
    for model in re.findall(
        r"find_codex_executable\(\)[^\n]*\n\s*return\s+\"([^\"]+)\"",
        body,
    ):
        pairs.append(("codex", model))
    final_match = re.search(r"return\s+\"(No model)\"\s*\n", body)
    assert final_match is not None, "Python fallback sentinel changed"
    pairs.append(("", final_match.group(1)))
    return pairs


def _ts_fallback_priority() -> list[tuple[str, str]]:
    """Parse ``getFallbackDefaultModel`` in TS into (trigger, model) pairs."""
    fn_match = re.search(
        r"export function getFallbackDefaultModel\(\):\s*string\s*\{"
        r"(?P<body>.*?)\n\}",
        INSTALLER_SOURCE,
        re.DOTALL,
    )
    assert fn_match is not None, (
        "getFallbackDefaultModel() is missing from DependencyInstaller.ts — "
        "the model picker will render blank on a fresh install"
    )
    body = fn_match.group("body")

    pairs: list[tuple[str, str]] = []
    for trigger, model in re.findall(
        r"env\.(\w+)\)\s*return\s+'([^']+)'",
        body,
    ):
        pairs.append((trigger, model))
    for cli, model in re.findall(
        r"execFileSync\(whichCmd,\s*\['(\w+)'\][^\n]*\n[^\n]*return\s+'([^']+)'",
        body,
    ):
        pairs.append((cli, model))
    final_match = re.search(r"return\s+'(No model)'\s*;\s*$", body)
    assert final_match is not None, (
        "TS fallback must terminate in 'No model' so the picker is never blank"
    )
    pairs.append(("", final_match.group(1)))
    return pairs


class TestFallbackModelExists(unittest.TestCase):
    """The TS fallback function must exist and be exported."""

    def test_fallback_function_is_exported(self) -> None:
        """``getFallbackDefaultModel`` must be exported from the installer
        so it is callable from ``SorcarSidebarView`` (directly or via
        ``getDefaultModel``)."""
        assert re.search(
            r"export function getFallbackDefaultModel\(\)",
            INSTALLER_SOURCE,
        ), "getFallbackDefaultModel must be exported"

    def test_fallback_returns_no_model_sentinel(self) -> None:
        """The fallback's terminal return must be the same ``"No model"``
        sentinel Python uses, so the picker text matches what the daemon
        will eventually broadcast."""
        pairs = _ts_fallback_priority()
        assert pairs[-1] == ("", "No model"), (
            f"TS fallback must end in 'No model', got {pairs[-1]}"
        )


class TestFallbackMatchesPython(unittest.TestCase):
    """The TS fallback must match Python's ``get_default_model`` exactly.

    Drift here means the picker shows a different value than the daemon
    would, defeating the entire point of the fallback (the daemon will
    later overwrite the picker, causing a visible flicker / surprise).
    """

    def test_priority_order_and_models_identical(self) -> None:
        """Walk both priority lists side-by-side; they must agree."""
        py = _python_default_model_priority()
        ts = _ts_fallback_priority()
        assert py == ts, (
            "TS fallback priority drifted from Python's get_default_model().\n"
            f"  python: {py}\n  typescript: {ts}\n"
            "Update getFallbackDefaultModel() in DependencyInstaller.ts so "
            "the picker matches the daemon's eventual answer."
        )


class TestGetDefaultModelUsesFallback(unittest.TestCase):
    """``getDefaultModel`` must consult the fallback on every failure path."""

    def _get_default_model_body(self) -> str:
        match = re.search(
            r"export function getDefaultModel\(\):\s*string\s*\{(?P<body>.*?)\n\}",
            INSTALLER_SOURCE,
            re.DOTALL,
        )
        assert match is not None, "getDefaultModel() not found"
        return match.group("body")

    def test_uv_or_project_missing_uses_fallback(self) -> None:
        """When ``uv`` or the bundled KISS project cannot be located, the
        function must call the fallback rather than return ``""``."""
        body = self._get_default_model_body()
        guard = re.search(
            r"if\s*\(!uvPath\s*\|\|\s*!kissProject\)\s*return\s+([^;]+);",
            body,
        )
        assert guard is not None, "Missing uv/project early-return guard"
        assert "getFallbackDefaultModel" in guard.group(1), (
            "Early-return on missing uv/project must call "
            "getFallbackDefaultModel() — returning '' makes the picker blank"
        )

    def test_python_exception_uses_fallback(self) -> None:
        """The ``catch`` branch (uv run failed: timeout / venv missing / "
        "import error) must call the fallback."""
        body = self._get_default_model_body()
        catch_match = re.search(
            r"catch\s*\{(?P<catch>[^}]*)\}",
            body,
            re.DOTALL,
        )
        assert catch_match is not None, "getDefaultModel() must have a catch"
        assert "getFallbackDefaultModel" in catch_match.group("catch"), (
            "catch branch must call getFallbackDefaultModel() so first-time "
            "setup (no .venv yet) does not yield a blank picker"
        )

    def test_empty_python_output_uses_fallback(self) -> None:
        """Even if uv runs successfully but prints nothing (corrupt venv,
        partial install), the picker must not be blank."""
        body = self._get_default_model_body()
        assert re.search(
            r"return\s+out\s*\|\|\s*getFallbackDefaultModel\(\)",
            body,
        ), (
            "Successful uv invocation that prints '' must still fall back, "
            "otherwise a corrupt .venv yields a blank picker"
        )


class TestSidebarSeedsModelFromInstaller(unittest.TestCase):
    """The sidebar constructor must seed ``_selectedModel`` via the
    installer so the very first webview HTML contains a non-empty model."""

    def test_constructor_calls_get_default_model(self) -> None:
        ctor_match = re.search(
            r"constructor\(extensionUri[^)]*\)\s*\{(?P<body>.*?)\n  \}",
            SIDEBAR_SOURCE,
            re.DOTALL,
        )
        assert ctor_match is not None, "SorcarSidebarView constructor not found"
        body = ctor_match.group("body")
        assert "getDefaultModel()" in body, (
            "Constructor must call getDefaultModel() so the first HTML render "
            "contains a non-empty <span id=\"model-name\">"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
