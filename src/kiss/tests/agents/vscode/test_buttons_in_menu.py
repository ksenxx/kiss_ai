"""Tests that Frequent tasks, History, and Settings buttons are in the menu dropdown."""

import re
from pathlib import Path

VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"
SORCAR_TAB_TS = VSCODE_DIR / "src" / "SorcarTab.ts"
WEB_SERVER_PY = VSCODE_DIR / "web_server.py"
MAIN_JS = VSCODE_DIR / "media" / "main.js"
MAIN_CSS = VSCODE_DIR / "media" / "main.css"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ── HTML structure tests ────────────────────────────────


class TestButtonsRemovedFromTabBar:
    """The 3 buttons must NOT appear inside the #tab-bar element."""

    def _extract_tab_bar(self, html: str) -> str:
        """Return the #tab-bar element content (up to its closing </div>)."""
        m = re.search(r'<div id="tab-bar">(.*?)</div>', html, re.DOTALL)
        assert m, "Could not find #tab-bar in HTML"
        return m.group(1)

    def test_no_config_btn_in_tab_bar_ts(self) -> None:
        tab_bar = self._extract_tab_bar(_read(SORCAR_TAB_TS))
        assert "config-btn" not in tab_bar

    def test_no_frequent_btn_in_tab_bar_ts(self) -> None:
        tab_bar = self._extract_tab_bar(_read(SORCAR_TAB_TS))
        assert "frequent-btn" not in tab_bar

    def test_no_history_btn_in_tab_bar_ts(self) -> None:
        tab_bar = self._extract_tab_bar(_read(SORCAR_TAB_TS))
        assert "history-btn" not in tab_bar

    def test_no_config_btn_in_tab_bar_ws(self) -> None:
        tab_bar = self._extract_tab_bar(_read(WEB_SERVER_PY))
        assert "config-btn" not in tab_bar

    def test_no_frequent_btn_in_tab_bar_ws(self) -> None:
        tab_bar = self._extract_tab_bar(_read(WEB_SERVER_PY))
        assert "frequent-btn" not in tab_bar

    def test_no_history_btn_in_tab_bar_ws(self) -> None:
        tab_bar = self._extract_tab_bar(_read(WEB_SERVER_PY))
        assert "history-btn" not in tab_bar


class TestButtonsInMenuDropdown:
    """The 3 buttons must appear inside the #menu-dropdown as .menu-item elements."""

    def _extract_menu_dropdown(self, html: str) -> str:
        """Return the #menu-dropdown content."""
        start = html.find('id="menu-dropdown"')
        assert start != -1, "Could not find #menu-dropdown in HTML"
        # Find the content between the opening tag and the next closing </div>
        # that matches the dropdown depth
        tag_start = html.rfind("<", 0, start)
        depth = 0
        i = tag_start
        while i < len(html):
            if html[i] == "<":
                if html[i : i + 2] == "</":
                    depth -= 1
                    if depth == 0:
                        end = html.find(">", i)
                        return html[tag_start : end + 1]
                elif html[i + 1 : i + 2] != "!":
                    # Check for self-closing tags
                    close_bracket = html.find(">", i)
                    if html[close_bracket - 1] == "/":
                        pass  # self-closing, no depth change
                    else:
                        depth += 1
            i += 1
        return html[tag_start:]

    def test_config_btn_in_menu_ts(self) -> None:
        content = _read(SORCAR_TAB_TS)
        dropdown = self._extract_menu_dropdown(content)
        assert 'id="config-btn"' in dropdown
        assert "menu-item" in dropdown.split('id="config-btn"')[1][:50]

    def test_frequent_btn_in_menu_ts(self) -> None:
        content = _read(SORCAR_TAB_TS)
        dropdown = self._extract_menu_dropdown(content)
        assert 'id="frequent-btn"' in dropdown
        assert "menu-item" in dropdown.split('id="frequent-btn"')[1][:50]

    def test_history_btn_in_menu_ts(self) -> None:
        content = _read(SORCAR_TAB_TS)
        dropdown = self._extract_menu_dropdown(content)
        assert 'id="history-btn"' in dropdown
        assert "menu-item" in dropdown.split('id="history-btn"')[1][:50]

    def test_config_btn_in_menu_ws(self) -> None:
        content = _read(WEB_SERVER_PY)
        dropdown = self._extract_menu_dropdown(content)
        assert 'id="config-btn"' in dropdown

    def test_frequent_btn_in_menu_ws(self) -> None:
        content = _read(WEB_SERVER_PY)
        dropdown = self._extract_menu_dropdown(content)
        assert 'id="frequent-btn"' in dropdown

    def test_history_btn_in_menu_ws(self) -> None:
        content = _read(WEB_SERVER_PY)
        dropdown = self._extract_menu_dropdown(content)
        assert 'id="history-btn"' in dropdown

    def test_menu_divider_before_sidebar_buttons_ts(self) -> None:
        content = _read(SORCAR_TAB_TS)
        dropdown = self._extract_menu_dropdown(content)
        divider_pos = dropdown.find("menu-divider")
        frequent_pos = dropdown.find('id="frequent-btn"')
        assert divider_pos != -1, "No menu-divider found"
        assert frequent_pos != -1
        assert divider_pos < frequent_pos, "Divider should come before frequent-btn"

    def test_menu_divider_before_sidebar_buttons_ws(self) -> None:
        content = _read(WEB_SERVER_PY)
        dropdown = self._extract_menu_dropdown(content)
        divider_pos = dropdown.find("menu-divider")
        frequent_pos = dropdown.find('id="frequent-btn"')
        assert divider_pos != -1, "No menu-divider found"
        assert frequent_pos != -1
        assert divider_pos < frequent_pos


class TestButtonLabels:
    """Verify the button text labels in the menu dropdown."""

    def test_settings_label_ts(self) -> None:
        content = _read(SORCAR_TAB_TS)
        # config-btn should have "Settings" text
        idx = content.find('id="config-btn"')
        assert idx != -1
        snippet = content[idx : idx + 1000]
        assert "Settings" in snippet

    def test_frequent_tasks_label_ts(self) -> None:
        content = _read(SORCAR_TAB_TS)
        idx = content.find('id="frequent-btn"')
        assert idx != -1
        snippet = content[idx : idx + 300]
        assert "Frequent tasks" in snippet

    def test_history_label_ts(self) -> None:
        content = _read(SORCAR_TAB_TS)
        idx = content.find('id="history-btn"')
        assert idx != -1
        snippet = content[idx : idx + 300]
        assert "History" in snippet


# ── JavaScript tests ────────────────────────────────────


class TestMainJsUpdated:
    """Verify main.js changes for the button relocation."""

    def test_add_btn_uses_append(self) -> None:
        """The + button must use appendChild (not insertBefore config-btn)."""
        js = _read(MAIN_JS)
        assert "tabBar.appendChild(addBtn)" in js
        assert "insertBefore(addBtn" not in js

    def test_history_btn_uses_active_class(self) -> None:
        js = _read(MAIN_JS)
        assert "historyBtn.classList.add('active')" in js
        assert "historyBtn.classList.remove('active')" in js
        assert "historyBtn.classList.add('open')" not in js
        assert "historyBtn.classList.remove('open')" not in js

    def test_config_btn_uses_active_class(self) -> None:
        js = _read(MAIN_JS)
        assert "configBtn.classList.add('active')" in js
        assert "configBtn.classList.remove('active')" in js
        assert "configBtn.classList.add('open')" not in js
        assert "configBtn.classList.remove('open')" not in js

    def test_frequent_btn_uses_active_class(self) -> None:
        js = _read(MAIN_JS)
        assert "frequentBtn.classList.add('active')" in js
        assert "frequentBtn.classList.remove('active')" in js
        assert "frequentBtn.classList.add('open')" not in js
        assert "frequentBtn.classList.remove('open')" not in js


# ── CSS tests ────────────────────────────────────


class TestCssUpdated:
    """Verify CSS changes for the button relocation."""

    def test_no_tab_bar_history_btn_rule(self) -> None:
        css = _read(MAIN_CSS)
        assert "#tab-bar > #history-btn" not in css

    def test_no_tab_bar_config_btn_rule(self) -> None:
        css = _read(MAIN_CSS)
        assert "#tab-bar > #config-btn" not in css

    def test_no_tab_bar_frequent_btn_rule(self) -> None:
        css = _read(MAIN_CSS)
        assert "#tab-bar > #frequent-btn" not in css

    def test_menu_divider_css_exists(self) -> None:
        css = _read(MAIN_CSS)
        assert ".menu-divider" in css

    def test_menu_item_active_css_exists(self) -> None:
        """The .menu-item.active rule must exist for the active state."""
        css = _read(MAIN_CSS)
        assert ".menu-item.active" in css
