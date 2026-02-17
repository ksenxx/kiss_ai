"""Browser automation tool for LLM agents using Playwright."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

DOM_EXTRACTION_JS = """
(() => {
    const INTERACTIVE_TAGS = new Set([
        'a', 'button', 'input', 'select', 'textarea', 'details', 'summary'
    ]);
    const INTERACTIVE_ROLES = new Set([
        'button', 'link', 'tab', 'menuitem', 'checkbox', 'radio',
        'switch', 'option', 'combobox', 'textbox', 'searchbox',
        'slider', 'spinbutton', 'treeitem'
    ]);
    const SKIP_TAGS = new Set([
        'script', 'style', 'noscript', 'svg', 'path', 'meta',
        'link', 'br', 'hr', 'wbr', 'template'
    ]);
    const STRUCTURAL_TAGS = new Set([
        'div', 'span', 'section', 'article', 'main', 'aside',
        'header', 'footer', 'nav', 'form', 'fieldset', 'ul', 'ol',
        'li', 'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th',
        'dl', 'dt', 'dd', 'figure', 'figcaption', 'dialog', 'body'
    ]);
    const MAX_TEXT_LEN = 80;

    let nextId = 1;
    window.__kiss_elements = {};

    function isVisible(el) {
        if (el.offsetWidth === 0 && el.offsetHeight === 0 &&
            el.getClientRects().length === 0) return false;
        const style = window.getComputedStyle(el);
        return style.display !== 'none' && style.visibility !== 'hidden';
    }

    function isInteractive(el) {
        const tag = el.tagName.toLowerCase();
        if (INTERACTIVE_TAGS.has(tag)) return true;
        const role = el.getAttribute('role');
        if (role && INTERACTIVE_ROLES.has(role)) return true;
        if (el.hasAttribute('onclick') || el.hasAttribute('contenteditable'))
            return true;
        const tabindex = el.getAttribute('tabindex');
        if (tabindex !== null && tabindex !== '-1') return true;
        return false;
    }

    function getDirectText(el) {
        let text = '';
        for (const child of el.childNodes) {
            if (child.nodeType === 3) {
                text += child.textContent;
            }
        }
        return text.trim();
    }

    function truncate(s) {
        s = s.replace(/\\s+/g, ' ').trim();
        if (s.length > MAX_TEXT_LEN) return s.substring(0, MAX_TEXT_LEN) + '...';
        return s;
    }

    function getAttrs(el) {
        const tag = el.tagName.toLowerCase();
        let attrs = '';
        if (tag === 'a') {
            const href = el.getAttribute('href');
            if (href) attrs += ' href="' + href + '"';
        }
        if (tag === 'input') {
            attrs += ' type="' + (el.getAttribute('type') || 'text') + '"';
            if (el.value) attrs += ' value="' + truncate(el.value) + '"';
            if (el.placeholder) attrs += ' placeholder="' + truncate(el.placeholder) + '"';
            if (el.name) attrs += ' name="' + el.name + '"';
        }
        if (tag === 'textarea') {
            if (el.placeholder) attrs += ' placeholder="' + truncate(el.placeholder) + '"';
            if (el.name) attrs += ' name="' + el.name + '"';
        }
        if (tag === 'select' && el.name) {
            attrs += ' name="' + el.name + '"';
        }
        if (tag === 'img') {
            const alt = el.getAttribute('alt');
            if (alt) attrs += ' alt="' + truncate(alt) + '"';
        }
        const role = el.getAttribute('role');
        if (role) attrs += ' role="' + role + '"';
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) attrs += ' aria-label="' + truncate(ariaLabel) + '"';
        return attrs;
    }

    function buildTree(el, depth) {
        if (!el || el.nodeType !== 1) return '';
        const tag = el.tagName.toLowerCase();
        if (SKIP_TAGS.has(tag)) return '';
        if (!isVisible(el)) return '';

        const indent = '  '.repeat(depth);
        const interactive = isInteractive(el);
        const attrs = getAttrs(el);
        const children = el.children;
        const directText = getDirectText(el);

        let childOutput = '';
        for (const child of children) {
            childOutput += buildTree(child, depth + 1);
        }

        const hasChildren = childOutput.length > 0;
        const hasText = directText.length > 0;

        function open(pfx) {
            return indent + pfx + '<' + tag + attrs + '>';
        }
        function close() { return '</' + tag + '>\\n'; }

        if (interactive) {
            const elemId = nextId++;
            window.__kiss_elements[elemId] = el;
            const pfx = '[' + elemId + '] ';
            const text = truncate(el.textContent || '');
            if (tag === 'select') {
                let opts = '';
                for (const opt of el.options) {
                    const sel = opt.selected ? ' (selected)' : '';
                    const t = truncate(opt.textContent || '');
                    opts += indent + '    <option>' + t + sel +
                        '</option>\\n';
                }
                return open(pfx) + '\\n' + opts +
                    indent + close();
            }
            if (tag === 'input' || tag === 'textarea') {
                return open(pfx) + '\\n';
            }
            if (hasChildren) {
                return open(pfx) + '\\n' + childOutput +
                    indent + close();
            }
            return open(pfx) + text + close();
        }

        if (!hasChildren && !hasText) return '';

        if (STRUCTURAL_TAGS.has(tag)) {
            if (hasChildren && hasText) {
                const t = truncate(directText);
                return open('') + '\\n' + indent + '  ' + t +
                    '\\n' + childOutput + indent + close();
            }
            if (hasChildren) {
                return childOutput;
            }
            return open('') + truncate(directText) + close();
        }

        if (hasChildren) {
            const text = hasText ? truncate(directText) : '';
            return open('') + text + '\\n' + childOutput +
                indent + close();
        }
        return open('') + truncate(directText) + close();
    }

    const body = document.body;
    if (!body) return 'Page has no body element.';
    return buildTree(body, 0);
})()
"""


CHROME_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = {runtime: {}, loadTimes: () => {}, csi: () => {}};
Object.defineProperty(navigator, 'permissions', {get: () => ({
    query: (p) => Promise.resolve({state: p.name === 'notifications'
        ? 'denied' : 'prompt'})
})});
"""


class WebUseTool:
    """Browser automation tool using Playwright for web interaction."""

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = False,
        viewport: tuple[int, int] = (1280, 900),
        user_data_dir: str | None = None,
    ) -> None:
        self.browser_type = browser_type
        self.headless = headless
        self.viewport = viewport
        self.user_data_dir = user_data_dir
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    def _context_args(self) -> dict[str, Any]:
        return {
            "viewport": {"width": self.viewport[0], "height": self.viewport[1]},
            "user_agent": CHROME_USER_AGENT,
            "locale": "en-US",
            "timezone_id": "America/Los_Angeles",
            "java_script_enabled": True,
            "has_touch": False,
            "is_mobile": False,
            "device_scale_factor": 2,
            "extra_http_headers": {
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"macOS"',
            },
        }

    def _chromium_args(self) -> list[str]:
        return [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
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
            self._context.add_init_script(STEALTH_JS)
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
            self._context.add_init_script(STEALTH_JS)
            self._page = self._context.new_page()

    def _get_dom_tree(self, max_chars: int = 50000) -> str:
        self._ensure_browser()
        title = self._page.title()
        url = self._page.url
        tree: str = self._page.evaluate(DOM_EXTRACTION_JS)
        header = f"Page: {title}\nURL: {url}\n\n"
        if len(tree) > max_chars:
            tree = tree[:max_chars] + "\n... [truncated]"
        return header + tree

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

    def go_to_url(self, url: str) -> str:  # noqa: N802
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
                    return self._get_dom_tree()
                return f"Error: Tab index {idx} out of range (0-{len(pages) - 1})."

            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self._wait_for_stable()
            return self._get_dom_tree()
        except Exception as e:
            return f"Error navigating to {url}: {e}"

    def click(self, element_id: int, action: str = "click") -> str:
        """Click or hover on an element by its numeric ID from the DOM tree.

        Args:
            element_id: The numeric ID shown in brackets [N] in the DOM tree.
            action: "click" (default) or "hover" to hover without clicking.
        """
        self._ensure_browser()
        try:
            if action == "hover":
                bbox = self._page.evaluate(
                    """(id) => {
                        const el = window.__kiss_elements[id];
                        if (!el) return null;
                        el.scrollIntoView({block: 'center'});
                        const rect = el.getBoundingClientRect();
                        return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
                    }""",
                    element_id,
                )
                if bbox is None:
                    return f"Error: Element with ID {element_id} not found."
                self._page.mouse.move(bbox["x"], bbox["y"])
                self._page.wait_for_timeout(300)
                return self._get_dom_tree()

            pages_before = len(self._context.pages)
            clicked = self._page.evaluate(
                """(id) => {
                    const el = window.__kiss_elements[id];
                    if (!el) return false;
                    el.scrollIntoView({block: 'center'});
                    el.click();
                    return true;
                }""",
                element_id,
            )
            if not clicked:
                return f"Error: Element with ID {element_id} not found."
            self._page.wait_for_timeout(500)
            self._wait_for_stable()
            if len(self._context.pages) > pages_before:
                self._check_for_new_tab()
                self._wait_for_stable()
            return self._get_dom_tree()
        except Exception as e:
            return f"Error clicking element {element_id}: {e}"

    def type_text(self, element_id: int, text: str, press_enter: bool = False) -> str:
        """Type text into an input or textarea element.

        Args:
            element_id: The numeric ID shown in brackets [N] in the DOM tree.
            text: The text to type into the element.
            press_enter: Whether to press Enter after typing.
        """
        self._ensure_browser()
        try:
            found = self._page.evaluate(
                """(args) => {
                    const el = window.__kiss_elements[args.id];
                    if (!el) return false;
                    el.scrollIntoView({block: 'center'});
                    el.focus();
                    el.value = '';
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    return true;
                }""",
                {"id": element_id},
            )
            if not found:
                return f"Error: Element with ID {element_id} not found."
            self._page.evaluate(
                """(id) => {
                    const el = window.__kiss_elements[id];
                    if (el) el.focus();
                }""",
                element_id,
            )
            self._page.keyboard.type(text, delay=20)
            if press_enter:
                self._page.keyboard.press("Enter")
                self._page.wait_for_timeout(500)
                self._wait_for_stable()
            return self._get_dom_tree()
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
            return self._get_dom_tree()
        except Exception as e:
            return f"Error pressing key '{key}': {e}"

    def execute_js(self, code: str) -> str:
        """Execute JavaScript in the browser. Access elements via window.__kiss_elements[id].

        Args:
            code: JavaScript code to execute. Return a value to see it in the output.
        """
        self._ensure_browser()
        try:
            result = self._page.evaluate(code)
            self._page.wait_for_timeout(300)
            dom = self._get_dom_tree()
            if result is not None:
                return f"Result: {result}\n\n{dom}"
            return dom
        except Exception as e:
            return f"Error executing JS: {e}"

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
        """Get current page content: DOM tree with interactive element IDs, or plain text.

        Args:
            text_only: If True, return full plain text instead of the DOM tree.
        """
        self._ensure_browser()
        try:
            if text_only:
                title = self._page.title()
                url = self._page.url
                text: str = self._page.evaluate("document.body.innerText")
                return f"Page: {title}\nURL: {url}\n\n{text}"
            return self._get_dom_tree()
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
        return "Browser closed."

    def get_tools(self) -> list[Callable[..., str]]:
        """Return all tool methods for passing to KISSAgent."""
        return [
            self.go_to_url,
            self.click,
            self.type_text,
            self.press_key,
            self.execute_js,
            self.screenshot,
            self.get_page_content,
        ]
