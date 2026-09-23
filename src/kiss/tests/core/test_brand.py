# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for product branding (``kiss.core.brand``).

The brand strings live in ``media/brand.json``; the prompt files carry
an ``{{IDENTITY}}`` placeholder; the chat pages of the remote web app and
the share page are rendered from the same brand.  These tests exercise
the real files and renderers: no mocks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from kiss.agents.third_party_agents import ask_sea
from kiss.core import brand as brand_module
from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE
from kiss.core.brand import BRAND, BRAND_FILE, DEFAULT_BRAND, PRODUCT_NAME, load_brand, render_brand
from kiss.server import web_server

_PLACEHOLDER = re.compile(r"\{\{(PRODUCT_NAME|SHORT_NAME|TAGLINE|IDENTITY|BRAND_JSON|BRAND_STYLE_HREF)\}\}")


def test_brand_module_reads_the_media_brand_file() -> None:
    """``BRAND`` is exactly what ``media/brand.json`` says; the fallback is KISS Sorcar.

    The checkout may carry a custom brand (a re-branded distribution), so
    the test pins the file-to-constant plumbing, not the stock strings.
    """
    assert BRAND_FILE.is_file(), BRAND_FILE
    assert BRAND_FILE.parent.name == "media"
    assert BRAND == load_brand(BRAND_FILE)
    assert PRODUCT_NAME == BRAND["product_name"]
    assert set(json.loads(BRAND_FILE.read_text(encoding="utf-8"))) <= set(DEFAULT_BRAND)
    assert DEFAULT_BRAND["product_name"] == "KISS Sorcar"


def test_load_brand_custom_partial_and_broken_files(tmp_path: Path) -> None:
    """A partial brand.json overrides only the keys it names; junk falls back."""
    custom = tmp_path / "brand.json"
    custom.write_text(
        json.dumps({
            "product_name": "Seamless Loop",
            "short_name": "s10s",
            "tagline": "",
            "identity": 42,
        }),
        encoding="utf-8",
    )
    loaded = load_brand(custom)
    assert loaded["product_name"] == "Seamless Loop"
    assert loaded["short_name"] == "s10s"
    assert loaded["tagline"] == DEFAULT_BRAND["tagline"]
    assert loaded["identity"] == DEFAULT_BRAND["identity"]
    assert loaded["extension_description"] == DEFAULT_BRAND["extension_description"]

    assert load_brand(tmp_path / "missing.json") == DEFAULT_BRAND
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert load_brand(tmp_path / "broken.json") == DEFAULT_BRAND
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert load_brand(tmp_path / "list.json") == DEFAULT_BRAND


def test_render_brand_fills_known_tokens_only() -> None:
    """Known tokens are filled; foreign ``{{...}}`` tokens survive untouched."""
    text = "{{IDENTITY}} {{PRODUCT_NAME}}/{{SHORT_NAME}} {{TAGLINE}} {{VERSION_SUFFIX}}"
    rendered = render_brand(text)
    assert rendered == (
        f"{BRAND['identity']} {BRAND['product_name']}/{BRAND['short_name']} "
        f"{BRAND['tagline']} {{{{VERSION_SUFFIX}}}}"
    )
    custom = dict(DEFAULT_BRAND, product_name="Seamless Loop", identity="You are Seamless Loop.")
    assert render_brand("{{IDENTITY}} {{PRODUCT_NAME}}", custom) == (
        "You are Seamless Loop. Seamless Loop"
    )


def test_prompt_files_carry_placeholder_and_prompts_are_rendered() -> None:
    """SYSTEM.md/SYSTEM_LITE.md hold ``{{IDENTITY}}``; the loaded prompts hold the sentence."""
    pkg = Path(brand_module.__file__).resolve().parents[1]
    for name in ("SYSTEM.md", "SYSTEM_LITE.md"):
        raw = (pkg / name).read_text(encoding="utf-8")
        assert "{{IDENTITY}}" in raw, name
        assert "You are KISS Sorcar" not in raw, name
    for prompt in (SYSTEM_PROMPT, SYSTEM_PROMPT_LITE, ask_sea.system_prompt()):
        assert prompt.startswith("<identity>\n\n" + BRAND["identity"])
        assert not _PLACEHOLDER.search(prompt)


def test_remote_webapp_page_is_branded() -> None:
    """The remote chat page carries the name, tagline, skin link and brand JSON."""
    page = web_server._build_html()
    assert f"<title>{PRODUCT_NAME}</title>" in page
    assert f"{PRODUCT_NAME} Server is starting ..." in page
    assert f"<h2>Welcome to {PRODUCT_NAME}</h2>" in page
    assert f"<p>{BRAND['tagline']}</p>" in page
    assert re.search(r'<link href="/media/brand\.css\?v=[0-9a-f]+" rel="stylesheet">', page)
    brand_json = re.search(r"window\.__BRAND__ = (\{.*?\});</script>", page)
    assert brand_json is not None
    assert json.loads(brand_json.group(1)) == {
        "productName": PRODUCT_NAME,
        "shortName": BRAND["short_name"],
    }
    assert not _PLACEHOLDER.search(page)
    # brand.css is part of the offline app shell the service worker precaches.
    assert any(url.startswith("/media/brand.css?v=") for url in web_server._app_shell_urls())


def test_share_page_inlines_brand_css_and_default_title() -> None:
    """The exported share page inlines brand.css and uses the branded default title."""
    brand_css = (web_server.MEDIA_DIR / "brand.css").read_text(encoding="utf-8")
    page = web_server._build_share_page("", "<div>x</div>")
    assert f"<title>{PRODUCT_NAME} chat</title>" in page
    assert brand_css.strip() in page


def test_custom_brand_json_rebrands_prompt_and_page_in_a_fresh_process(tmp_path: Path) -> None:
    """Swapping brand.json (the customization patch) re-brands a fresh interpreter.

    Runs in a subprocess against a copy of the kiss package so the
    installed checkout's brand.json is never touched.
    """
    src_root = Path(brand_module.__file__).resolve().parents[2]
    copy = tmp_path / "src"
    subprocess.run(
        ["cp", "-r", "--", str(src_root), str(copy)],
        check=True,
    )
    brand_file = copy / "kiss" / "agents" / "vscode" / "media" / "brand.json"
    brand_file.write_text(
        json.dumps({
            "product_name": "Seamless Loop",
            "short_name": "s10s",
            "tagline": "SeamlessLabs' assistant.",
            "identity": "You are Seamless Loop (s10s), the AI assistant of SeamlessLabs.",
        }),
        encoding="utf-8",
    )
    probe = (
        "from kiss.core.base import SYSTEM_PROMPT, SYSTEM_PROMPT_LITE\n"
        "from kiss.core.brand import PRODUCT_NAME, SHORT_NAME\n"
        "from kiss.server import web_server, tls_certs\n"
        "print(PRODUCT_NAME); print(SHORT_NAME)\n"
        "print(SYSTEM_PROMPT.splitlines()[2][:63])\n"
        "print(SYSTEM_PROMPT_LITE.splitlines()[2][:63])\n"
        "page = web_server._build_html()\n"
        "print('<title>Seamless Loop</title>' in page)\n"
        "print('Welcome to Seamless Loop</h2>' in page and 'SeamlessLabs&#x27; assistant.' in page)\n"
        "print(tls_certs._CA_COMMON_NAME_PREFIX)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=copy,
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(copy), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.stdout.splitlines() == [
        "Seamless Loop",
        "s10s",
        "You are Seamless Loop (s10s), the AI assistant of SeamlessLabs.",
        "You are Seamless Loop (s10s), the AI assistant of SeamlessLabs.",
        "True",
        "True",
        "Seamless Loop Local CA",
    ]
