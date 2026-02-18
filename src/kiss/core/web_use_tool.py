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


def _number_interactive_elements(snapshot: str) -> tuple[str, list[dict[str, str]]]:
    lines = snapshot.splitlines()
    result_lines: list[str] = []
    elements: list[dict[str, str]] = []
    counter = 0

    for line in lines:
        m = _ROLE_LINE_RE.match(line)
        if m:
            indent, role, rest = m.group(1), m.group(2), m.group(3)
            if role in INTERACTIVE_ROLES:
                counter += 1
                name_match = re.match(r'"([^"]*)"', rest)
                name = name_match.group(1) if name_match else ""
                elements.append({"id": str(counter), "role": role, "name": name})
                result_lines.append(f"{indent}- [{counter}] {role} {rest}".rstrip())
                continue
        result_lines.append(line)

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

    def _chromium_args(self) -> list[str]:
        return [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
            "--disable-infobars",
            "--no-first-run",
            "--no-default-browser-check",
        ]

    def _ensure_browser(self) -> None:
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        launcher = getattr(self._playwright, self.browser_type)
        is_chromium = self.browser_type == "chromium"
        chromium_args = self._chromium_args() if is_chromium else []
        channel = "chrome" if is_chromium and not self.headless else None

        if self.user_data_dir:
            Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)
            launch_kwargs: dict[str, Any] = {
                "headless": self.headless,
                "args": chromium_args or None,
                **self._context_args(),
            }
            if channel:
                launch_kwargs["channel"] = channel
            self._context = launcher.launch_persistent_context(
                self.user_data_dir,
                **launch_kwargs,
            )
            pages = self._context.pages
            self._page = pages[0] if pages else self._context.new_page()
        else:
            launch_kwargs = {
                "headless": self.headless,
                "args": chromium_args or None,
            }
            if channel:
                launch_kwargs["channel"] = channel
            self._browser = launcher.launch(**launch_kwargs)
            self._context = self._browser.new_context(**self._context_args())
            self._page = self._context.new_page()

    def _get_ax_tree(self, max_chars: int = 50000) -> str:
        self._ensure_browser()
        title = self._page.title()
        url = self._page.url
        snapshot = self._page.locator("body").aria_snapshot()
        if not snapshot:
            self._elements = []
            return f"Page: {title}\nURL: {url}\n\n(empty page)"
        numbered, self._elements = _number_interactive_elements(snapshot)
        header = f"Page: {title}\nURL: {url}\n\n"
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
        elem = self._elements[element_id - 1]
        role, name = elem["role"], elem["name"]
        if name:
            locator = self._page.get_by_role(role, name=name, exact=True)
        else:
            locator = self._page.get_by_role(role)
        if locator.count() == 0:
            raise ValueError(f"Element with ID {element_id} not found on page.")
        if locator.count() == 1:
            return locator
        for i in range(locator.count()):
            nth = locator.nth(i)
            try:
                if nth.is_visible():
                    return nth
            except Exception:
                continue
        return locator.first

    def go_to_url(self, url: str) -> str:
        """Navigate to a URL. Use "tab:list" to list open tabs, "tab:N" to switch to tab N.

        Args:
            url: URL to navigate to, "tab:list" to list tabs, or "tab:N" to switch to tab N.
        """
        self._ensure_browser()
        try:
            if url == "tab:list":
                pages = self._context.pages
                lines = [f"Open tabs ({len(pages)}):"]
                for i, page in enumerate(pages):
                    marker = " (active)" if page == self._page else ""
                    lines.append(f"  [{i}] {page.title()} - {page.url}{marker}")
                return "\n".join(lines)

            if url.startswith("tab:"):
                idx = int(url[4:])
                pages = self._context.pages
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
        """Click or hover on an element by its numeric ID from the accessibility tree.

        Args:
            element_id: The numeric ID shown in brackets [N] in the accessibility tree.
            action: "click" (default) or "hover" to hover without clicking.
        """
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
        """Type text into an input or textarea element.

        Args:
            element_id: The numeric ID shown in brackets [N] in the accessibility tree.
            text: The text to type into the element.
            press_enter: Whether to press Enter after typing.
        """
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
        """Press a keyboard key or key combination.

        Args:
            key: Key name, e.g. "Enter", "Escape", "Tab", "ArrowDown", "PageDown",
                 "Backspace", or combo like "Control+a", "Shift+Tab".
        """
        self._ensure_browser()
        try:
            self._page.keyboard.press(key)
            self._page.wait_for_timeout(300)
            return self._get_ax_tree()
        except Exception as e:
            return f"Error pressing key '{key}': {e}"

    def scroll(self, direction: str = "down", amount: int = 3) -> str:
        """Scroll the page in a direction.

        Args:
            direction: "down", "up", "left", or "right".
            amount: Number of scroll clicks (default 3).
        """
        self._ensure_browser()
        try:
            delta_map = {
                "down": (0, 300),
                "up": (0, -300),
                "right": (300, 0),
                "left": (-300, 0),
            }
            dx, dy = delta_map.get(direction, (0, 300))
            vw = self.viewport[0] // 2
            vh = self.viewport[1] // 2
            self._page.mouse.move(vw, vh)
            for _ in range(amount):
                self._page.mouse.wheel(dx, dy)
                self._page.wait_for_timeout(100)
            self._page.wait_for_timeout(300)
            return self._get_ax_tree()
        except Exception as e:
            return f"Error scrolling {direction}: {e}"

    def screenshot(self, file_path: str = "screenshot.png") -> str:
        """Take a screenshot of the current page and save it to a file.

        Args:
            file_path: Path where the screenshot will be saved.
        """
        self._ensure_browser()
        try:
            path = Path(file_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._page.screenshot(path=str(path), full_page=False)
            return f"Screenshot saved to {path}"
        except Exception as e:
            return f"Error taking screenshot: {e}"

    def get_page_content(self, text_only: bool = False) -> str:
        """Get current page content: accessibility tree with interactive element IDs, or plain text.

        Args:
            text_only: If True, return full plain text instead of the accessibility tree.
        """
        self._ensure_browser()
        try:
            if text_only:
                title = self._page.title()
                url = self._page.url
                text: str = self._page.inner_text("body")
                return f"Page: {title}\nURL: {url}\n\n{text}"
            return self._get_ax_tree()
        except Exception as e:
            return f"Error getting page content: {e}"

    def close(self) -> str:
        """Close the browser and release all resources."""
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
        """Return all tool methods for passing to KISSAgent."""
        return [
            self.go_to_url,
            self.click,
            self.type_text,
            self.press_key,
            self.scroll,
            self.screenshot,
            self.get_page_content,
        ]
