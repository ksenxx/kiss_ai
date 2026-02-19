"""Browser automation tool for LLM agents using Playwright.

Uses Playwright's native APIs (aria_snapshot, locators, keyboard/mouse)
with zero JavaScript injection to avoid bot detection.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_AUTO_DETECT = "auto"
KISS_PROFILE_DIR = str(Path.home() / ".kiss" / "browser_profile")

INTERACTIVE_ROLES = {
    "link", "button", "textbox", "searchbox", "combobox",
    "checkbox", "radio", "switch", "slider", "spinbutton",
    "tab", "menuitem", "menuitemcheckbox", "menuitemradio",
    "option", "treeitem",
}

_ROLE_LINE_RE = re.compile(r"^(\s*)-\s+([\w]+)\s*(.*)")

_SCROLL_DELTA = {"down": (0, 300), "up": (0, -300), "right": (300, 0), "left": (-300, 0)}


def _number_interactive_elements(snapshot: str) -> tuple[str, list[dict[str, str]]]:
    result_lines: list[str] = []
    elements: list[dict[str, str]] = []
    counter = 0
    for line in snapshot.splitlines():
        m = _ROLE_LINE_RE.match(line)
        if not m:
            result_lines.append(line)
            continue
        indent, role, rest = m.group(1), m.group(2), m.group(3)
        if role not in INTERACTIVE_ROLES:
            result_lines.append(line)
            continue
        counter += 1
        name_match = re.match(r'"([^"]*)"', rest)
        elements.append({"role": role, "name": name_match.group(1) if name_match else ""})
        result_lines.append(f"{indent}- [{counter}] {role} {rest}".rstrip())
    return "\n".join(result_lines), elements


class WebUseTool:
    """Browser automation tool using Playwright with zero JS injection."""

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = False,
        viewport: tuple[int, int] = (1280, 900),
        user_data_dir: str | None = _AUTO_DETECT,
    ) -> None:
        self.browser_type = browser_type
        self.headless = headless
        self.viewport = viewport
        self.user_data_dir: str | None
        if user_data_dir == _AUTO_DETECT:
            self.user_data_dir = KISS_PROFILE_DIR
        else:
            self.user_data_dir = user_data_dir
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._elements: list[dict[str, str]] = []

    def _context_args(self) -> dict[str, Any]:
        return {
            "viewport": {"width": self.viewport[0], "height": self.viewport[1]},
            "locale": "en-US",
            "timezone_id": "America/Los_Angeles",
            "java_script_enabled": True,
            "has_touch": False,
            "is_mobile": False,
            "device_scale_factor": 2,
        }

    def _launch_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"headless": self.headless}
        if self.browser_type == "chromium":
            kwargs["args"] = [
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-infobars",
                "--no-first-run",
                "--no-default-browser-check",
            ]
            if not self.headless:
                kwargs["channel"] = "chrome"
        return kwargs

    def _ensure_browser(self) -> None:
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        launcher = getattr(self._playwright, self.browser_type)
        kwargs = self._launch_kwargs()

        if self.user_data_dir:
            Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)
            self._context = launcher.launch_persistent_context(
                self.user_data_dir, **kwargs, **self._context_args()
            )
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        else:
            self._browser = launcher.launch(**kwargs)
            self._context = self._browser.new_context(**self._context_args())
            self._page = self._context.new_page()

    def _get_ax_tree(self, max_chars: int = 50000) -> str:
        self._ensure_browser()
        header = f"Page: {self._page.title()}\nURL: {self._page.url}\n\n"
        snapshot = self._page.locator("body").aria_snapshot()
        if not snapshot:
            self._elements = []
            return header + "(empty page)"
        numbered, self._elements = _number_interactive_elements(snapshot)
        if len(numbered) > max_chars:
            numbered = numbered[:max_chars] + "\n... [truncated]"
        return header + numbered

    def _wait_for_stable(self) -> None:
        try:
            self._page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        try:
            self._page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass

    def _check_for_new_tab(self) -> None:
        if self._context is None:
            return
        pages = self._context.pages
        if len(pages) > 1 and pages[-1] != self._page:
            self._page = pages[-1]

    def _resolve_locator(self, element_id: int) -> Any:
        if element_id < 1 or element_id > len(self._elements):
            snapshot = self._page.locator("body").aria_snapshot()
            if snapshot:
                _, self._elements = _number_interactive_elements(snapshot)
            if element_id < 1 or element_id > len(self._elements):
                raise ValueError(f"Element with ID {element_id} not found.")
        role = self._elements[element_id - 1]["role"]
        name = self._elements[element_id - 1]["name"]
        if name:
            locator = self._page.get_by_role(role, name=name, exact=True)
        else:
            locator = self._page.get_by_role(role)
        n = locator.count()
        if n == 0:
            raise ValueError(f"Element with ID {element_id} not found on page.")
        if n == 1:
            return locator
        for i in range(n):
            try:
                if locator.nth(i).is_visible():
                    return locator.nth(i)
            except Exception:
                continue
        return locator.first

    def go_to_url(self, url: str) -> str:
        """Navigate the browser to a URL and return the page accessibility tree.
        Use when you need to open a new page or switch pages. Special values: "tab:list"
        returns a list of open tabs; "tab:N" switches to tab N (0-based).

        Args:
            url: Full URL to open, or "tab:list" for tab list, or "tab:N" to switch to tab N.

        Returns:
            On success: page title, URL, and accessibility tree with [N] IDs. For "tab:list":
            list of open tabs with indices. On error: "Error navigating to <url>: <message>"."""
        self._ensure_browser()
        try:
            pages = self._context.pages
            if url == "tab:list":
                lines = [f"Open tabs ({len(pages)}):"]
                for i, page in enumerate(pages):
                    suffix = " (active)" if page == self._page else ""
                    lines.append(f"  [{i}] {page.title()} - {page.url}{suffix}")
                return "\n".join(lines)
            if url.startswith("tab:"):
                idx = int(url[4:])
                if 0 <= idx < len(pages):
                    self._page = pages[idx]
                    return self._get_ax_tree()
                return f"Error: Tab index {idx} out of range (0-{len(pages) - 1})."

            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self._wait_for_stable()
            return self._get_ax_tree()
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    def click(self, element_id: int, action: str = "click") -> str:
        """Click or hover on an interactive element by its [N] ID from the accessibility tree.
        Use after get_page_content or go_to_url to interact with links, buttons, tabs, etc.

        Args:
            element_id: Numeric ID shown in brackets [N] next to the element in the tree.
            action: "click" (default) to click the element, "hover" to only move focus.

        Returns:
            Updated accessibility tree (title, URL, numbered elements), or on error
            "Error clicking element <id>: <message>"."""
        self._ensure_browser()
        try:
            locator = self._resolve_locator(element_id)

            if action == "hover":
                locator.hover()
                self._page.wait_for_timeout(300)
                return self._get_ax_tree()

            pages_before = len(self._context.pages)
            locator.click()
            self._page.wait_for_timeout(500)
            self._wait_for_stable()
            if len(self._context.pages) > pages_before:
                self._check_for_new_tab()
                self._wait_for_stable()
            return self._get_ax_tree()
        except Exception as e:
            return f"Error clicking element {element_id}: {e}"

    def type_text(self, element_id: int, text: str, press_enter: bool = False) -> str:
        """Type text into a textbox, searchbox, or other editable element by its [N] ID.
        Clears existing content then types the given text. Use for forms, search boxes, etc.

        Args:
            element_id: Numeric ID from the accessibility tree (brackets [N]).
            text: String to type into the element.
            press_enter: If True, press Enter after typing (e.g. to submit a search).

        Returns:
            Updated accessibility tree, or "Error typing into element <id>: <message>" on error."""
        self._ensure_browser()
        try:
            locator = self._resolve_locator(element_id)
            select_all = "Meta+a" if sys.platform == "darwin" else "Control+a"
            locator.click()
            self._page.keyboard.press(select_all)
            self._page.keyboard.press("Backspace")
            self._page.keyboard.type(text, delay=50)
            if press_enter:
                self._page.keyboard.press("Enter")
                self._page.wait_for_timeout(500)
                self._wait_for_stable()
            return self._get_ax_tree()
        except Exception as e:
            return f"Error typing into element {element_id}: {e}"

    def press_key(self, key: str) -> str:
        """Press a single key or key combination. Use for navigation, closing dialogs, shortcuts.

        Args:
            key: Key name, e.g. "Enter", "Escape", "Tab", "ArrowDown", "PageDown", "Backspace",
                 or combination like "Control+a", "Shift+Tab".

        Returns:
            Updated accessibility tree, or "Error pressing key '<key>': <message>" on error."""
        self._ensure_browser()
        try:
            self._page.keyboard.press(key)
            self._page.wait_for_timeout(300)
            return self._get_ax_tree()
        except Exception as e:
            return f"Error pressing key '{key}': {e}"

    def scroll(self, direction: str = "down", amount: int = 3) -> str:
        """Scroll the current page to reveal more content. Use when needed elements are off-screen.

        Args:
            direction: "down", "up", "left", or "right".
            amount: Number of scroll steps (default 3).

        Returns:
            Updated accessibility tree after scrolling, or
            "Error scrolling <direction>: <message>" on error."""
        self._ensure_browser()
        try:
            dx, dy = _SCROLL_DELTA.get(direction, (0, 300))
            vw, vh = self.viewport[0] // 2, self.viewport[1] // 2
            self._page.mouse.move(vw, vh)
            for _ in range(amount):
                self._page.mouse.wheel(dx, dy)
                self._page.wait_for_timeout(100)
            self._page.wait_for_timeout(300)
            return self._get_ax_tree()
        except Exception as e:
            return f"Error scrolling {direction}: {e}"

    def screenshot(self, file_path: str = "screenshot.png") -> str:
        """Capture the visible viewport as an image. Use to verify layout, captchas, or
        visual state.

        Args:
            file_path: Path where the PNG will be saved (default "screenshot.png"). Parent
                directories are created if needed.

        Returns:
            "Screenshot saved to <resolved_path>", or
            "Error taking screenshot: <message>" on error."""
        self._ensure_browser()
        try:
            path = Path(file_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._page.screenshot(path=str(path), full_page=False)
            return f"Screenshot saved to {path}"
        except Exception as e:
            return f"Error taking screenshot: {e}"

    def get_page_content(self, text_only: bool = False) -> str:
        """Get the current page content. Use to decide what to click or type next.

        Args:
            text_only: If False (default), return accessibility tree with [N] IDs for interactive
                elements. If True, return plain text only (title, URL, body text).

        Returns:
            Accessibility tree or plain text as described above, or
            "Error getting page content: <message>" on error."""
        self._ensure_browser()
        try:
            if text_only:
                title = self._page.title()
                url = self._page.url
                body = self._page.inner_text("body")
                return f"Page: {title}\nURL: {url}\n\n{body}"
            return self._get_ax_tree()
        except Exception as e:
            return f"Error getting page content: {e}"

    def close(self) -> str:
        """Close the browser and release resources. Call when done with the session or before exit.

        Returns:
            "Browser closed." (always, even if nothing was open)."""
        try:
            if self.user_data_dir and self._context:
                self._context.close()
            elif self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._elements = []
        return "Browser closed."

    def get_tools(self) -> list[Callable[..., str]]:
        """Return callable web tools for registration with an agent.

        Returns:
            List of callables: go_to_url, click, type_text, press_key, scroll, screenshot,
            get_page_content. Does not include close."""
        return [
            self.go_to_url,
            self.click,
            self.type_text,
            self.press_key,
            self.scroll,
            self.screenshot,
            self.get_page_content,
        ]
