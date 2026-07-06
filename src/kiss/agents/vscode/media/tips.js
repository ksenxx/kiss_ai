// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Fresh-install Tips window for the chat webview.
//
// Defines the ``<kiss-tips-panel>`` web component: a fixed-size modal
// panel (min(560px, 90vw) x min(520px, 82vh)) centered horizontally
// and vertically that renders one tip at a time as formatted HTML
// (markdown via window.marked, loaded by chat.html before this
// script) with Previous / Next / Close controls.  The tip body
// scrolls when content overflows, every fenced code block gets a
// Copy-to-clipboard button, and the markdown is styled with the
// Rosé Pine Moon palette.
//
// The tip list and the show-on-startup flag arrive via
// ``window.__TIPS__ = {tips: [...], show: bool}`` injected by the HTML
// builder (``SorcarTab.buildChatHtml`` in the extension,
// ``web_server._build_html`` on the remote webapp).  ``show`` is true
// exactly once — on the first chat render after a fresh installation.

/* global customElements */

(function () {
  'use strict';

  // Rosé Pine Moon palette: #232136 background, #2a273f surfaces,
  // #44415a borders, #e0def4 text, #908caa muted, #c4a7e7 headings,
  // #9ccfd8 links/hover, #f6c177 inline code, #ea9a97 accents,
  // #3e8fb0 buttons.
  const PANEL_CSS =
    '.tips-overlay {' +
    '  position: fixed;' +
    '  inset: 0;' +
    '  z-index: 10000;' +
    '  display: flex;' +
    '  justify-content: center;' +
    '  align-items: center;' +
    '  background: rgba(35, 33, 54, 0.6);' +
    '}' +
    '.tips-panel {' +
    '  display: flex;' +
    '  flex-direction: column;' +
    '  box-sizing: border-box;' +
    '  width: min(560px, 90vw);' +
    '  height: min(520px, 82vh);' +
    '  border: 1px solid #44415a;' +
    '  border-radius: 8px;' +
    '  background: #232136;' +
    '  color: #e0def4;' +
    '  font-family: var(--vscode-font-family, sans-serif);' +
    '  font-size: calc(var(--vscode-font-size, 13px) * 1.25);' +
    '  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);' +
    '}' +
    '.tips-header {' +
    '  display: flex;' +
    '  align-items: center;' +
    '  gap: 8px;' +
    '  padding: 10px 12px;' +
    '  border-bottom: 1px solid #44415a;' +
    '}' +
    '.tips-title { font-weight: 600; flex: 1; color: #c4a7e7; }' +
    '.tips-counter {' +
    '  color: #908caa;' +
    '  font-size: 0.9em;' +
    '}' +
    '.tips-close {' +
    '  border: none;' +
    '  background: transparent;' +
    '  color: #908caa;' +
    '  font-size: 20px;' +
    '  line-height: 1;' +
    '  cursor: pointer;' +
    '  padding: 2px 6px;' +
    '}' +
    '.tips-close:hover { color: #e0def4; }' +
    '.tips-body {' +
    '  flex: 1 1 auto;' +
    '  min-height: 0;' +
    '  padding: 14px 16px;' +
    '  overflow: auto;' +
    '}' +
    '.tips-body h1, .tips-body h2, .tips-body h3, .tips-body h4 {' +
    '  color: #c4a7e7;' +
    '}' +
    '.tips-body a { color: #9ccfd8; }' +
    '.tips-body strong { color: #ea9a97; }' +
    '.tips-body blockquote {' +
    '  margin: 8px 0;' +
    '  padding-left: 10px;' +
    '  border-left: 3px solid #ea9a97;' +
    '  color: #908caa;' +
    '}' +
    '.tips-body pre {' +
    '  background: #2a273f;' +
    '  border: 1px solid #44415a;' +
    '  border-radius: 6px;' +
    '  padding: 10px 12px;' +
    '  margin: 8px 0;' +
    '  overflow-x: auto;' +
    '  white-space: pre-wrap;' +
    '}' +
    '.tips-body code {' +
    '  background: #2a273f;' +
    '  color: #f6c177;' +
    '  padding: 1px 4px;' +
    '  border-radius: 3px;' +
    '}' +
    '.tips-body pre code { background: transparent; padding: 0; }' +
    '.tips-code { position: relative; }' +
    '.tips-copy {' +
    '  position: absolute;' +
    '  top: 6px;' +
    '  right: 6px;' +
    '  border: none;' +
    '  border-radius: 3px;' +
    '  padding: 2px 8px;' +
    '  font-size: 0.85em;' +
    '  cursor: pointer;' +
    '  background: #3e8fb0;' +
    '  color: #232136;' +
    '}' +
    '.tips-copy:hover { background: #9ccfd8; }' +
    '.tips-footer {' +
    '  display: flex;' +
    '  justify-content: space-between;' +
    '  gap: 8px;' +
    '  padding: 10px 12px;' +
    '  border-top: 1px solid #44415a;' +
    '}' +
    '.tips-prev, .tips-next {' +
    '  border: none;' +
    '  border-radius: 3px;' +
    '  padding: 4px 12px;' +
    '  cursor: pointer;' +
    '  background: #3e8fb0;' +
    '  color: #232136;' +
    '}' +
    '.tips-prev:hover:enabled, .tips-next:hover:enabled {' +
    '  background: #9ccfd8;' +
    '}' +
    '.tips-prev:disabled, .tips-next:disabled {' +
    '  opacity: 0.4;' +
    '  cursor: default;' +
    '}';

  /**
   * Copy ``text`` via a hidden textarea + ``document.execCommand('copy')``.
   * Returns true when the copy command reports success.
   */
  function copyViaExecCommand(text) {
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.left = '-9999px';
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch (_err) {
      ok = false;
    }
    area.remove();
    return ok;
  }

  /**
   * Copy ``text`` to the clipboard, preferring the async clipboard API
   * and falling back to ``execCommand('copy')``.  Resolves to true on
   * success and false on failure.
   */
  function copyTextToClipboard(text) {
    const clip = navigator.clipboard;
    if (clip && typeof clip.writeText === 'function') {
      let written;
      try {
        written = clip.writeText(text);
      } catch (_err) {
        return Promise.resolve(copyViaExecCommand(text));
      }
      return written.then(
        () => true,
        () => copyViaExecCommand(text),
      );
    }
    return Promise.resolve(copyViaExecCommand(text));
  }

  /**
   * Flash "Copied!" / "Failed" feedback on a copy button, then restore
   * the "Copy" label after 1.5 seconds.
   */
  function flashCopyResult(btn, ok) {
    btn.textContent = ok ? 'Copied!' : 'Failed';
    setTimeout(() => {
      btn.textContent = 'Copy';
    }, 1500);
  }

  /**
   * Click handler for copy buttons: copy the text of the code block
   * sharing the button's ``.tips-code`` wrapper and flash feedback.
   */
  function onCopyClick(event) {
    const btn = event.currentTarget;
    const code = btn.parentElement.querySelector('pre code');
    const text = code ? code.textContent : '';
    copyTextToClipboard(text).then(ok => {
      flashCopyResult(btn, ok);
    });
  }

  /**
   * Centered modal panel showing one markdown tip at a time.
   *
   * Set ``el.tips`` to a list of markdown strings; the panel renders
   * the first tip and enables Previous / Next / Close with the usual
   * semantics (Previous disabled on the first tip, Next disabled on
   * the last, Close removes the panel from the DOM).
   */
  class KissTipsPanel extends HTMLElement {
    constructor() {
      super();
      this._tips = [];
      this._index = 0;
      const root = this.attachShadow({mode: 'open'});
      const style = document.createElement('style');
      style.textContent = PANEL_CSS;
      root.appendChild(style);

      const overlay = document.createElement('div');
      overlay.className = 'tips-overlay';
      const panel = document.createElement('div');
      panel.className = 'tips-panel';
      panel.setAttribute('role', 'dialog');
      panel.setAttribute('aria-modal', 'true');
      panel.setAttribute('aria-label', 'Tips');

      const header = document.createElement('div');
      header.className = 'tips-header';
      const title = document.createElement('span');
      title.className = 'tips-title';
      title.textContent = 'Tips';
      this._counter = document.createElement('span');
      this._counter.className = 'tips-counter';
      this._close = document.createElement('button');
      this._close.className = 'tips-close';
      this._close.type = 'button';
      this._close.setAttribute('aria-label', 'Close tips');
      this._close.textContent = '\u00d7';
      header.appendChild(title);
      header.appendChild(this._counter);
      header.appendChild(this._close);

      this._body = document.createElement('div');
      this._body.className = 'tips-body';

      const footer = document.createElement('div');
      footer.className = 'tips-footer';
      this._prev = document.createElement('button');
      this._prev.className = 'tips-prev';
      this._prev.type = 'button';
      this._prev.textContent = 'Previous';
      this._next = document.createElement('button');
      this._next.className = 'tips-next';
      this._next.type = 'button';
      this._next.textContent = 'Next';
      footer.appendChild(this._prev);
      footer.appendChild(this._next);

      panel.appendChild(header);
      panel.appendChild(this._body);
      panel.appendChild(footer);
      overlay.appendChild(panel);
      root.appendChild(overlay);

      const self = this;
      this._prev.addEventListener('click', () => {
        if (self._index > 0) {
          self._index -= 1;
          self._update();
        }
      });
      this._next.addEventListener('click', () => {
        if (self._index < self._tips.length - 1) {
          self._index += 1;
          self._update();
        }
      });
      this._close.addEventListener('click', () => {
        self.remove();
      });
    }

    /** Markdown tip strings shown by the panel, one at a time. */
    get tips() {
      return this._tips;
    }

    set tips(list) {
      this._tips = Array.isArray(list) ? list : [];
      this._index = 0;
      this._update();
    }

    /** Re-render the current tip, counter, and button states. */
    _update() {
      const total = this._tips.length;
      const text = total > 0 ? this._tips[this._index] : '';
      const md = window.marked;
      if (md && typeof md.parse === 'function') {
        this._body.innerHTML = md.parse(text);
      } else {
        const pre = document.createElement('pre');
        pre.textContent = text;
        this._body.replaceChildren(pre);
      }
      this._addCopyButtons();
      this._counter.textContent =
        total > 0 ? this._index + 1 + ' / ' + total : '';
      this._prev.disabled = this._index <= 0;
      this._next.disabled = this._index >= total - 1;
    }

    /**
     * Wrap every fenced code block (``pre > code``) in the rendered
     * tip with a ``.tips-code`` container and attach a Copy button.
     */
    _addCopyButtons() {
      for (const pre of this._body.querySelectorAll('pre')) {
        if (!pre.querySelector('code')) continue;
        const wrapper = document.createElement('div');
        wrapper.className = 'tips-code';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);
        const btn = document.createElement('button');
        btn.className = 'tips-copy';
        btn.type = 'button';
        btn.textContent = 'Copy';
        btn.setAttribute('aria-label', 'Copy code to clipboard');
        btn.addEventListener('click', onCopyClick);
        wrapper.appendChild(btn);
      }
    }
  }

  customElements.define('kiss-tips-panel', KissTipsPanel);

  /**
   * Create a ``<kiss-tips-panel>`` for ``tips`` and mount it on
   * document.body.  Exposed for reuse (e.g. a future "Show tips" menu
   * entry) and for tests.
   */
  function showTipsPanel(tips) {
    const el = document.createElement('kiss-tips-panel');
    el.tips = tips;
    document.body.appendChild(el);
    return el;
  }

  window.__kissShowTipsPanel = showTipsPanel;

  /** Tips from ``window.__TIPS__``, or ``[]`` when absent/malformed. */
  function configuredTips() {
    const cfg = window.__TIPS__;
    return cfg && Array.isArray(cfg.tips) ? cfg.tips : [];
  }

  /**
   * Wire the bulb button (``#tips-btn``) in the chat input toolbar:
   * clicking it shows the tips window.  Only one panel instance may
   * exist at a time — clicking while it is open is a no-op.  No-op
   * when the button is absent (e.g. non-chat pages reusing tips.js).
   */
  function wireTipsButton() {
    const btn = document.getElementById('tips-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      if (document.body.querySelector('kiss-tips-panel')) return;
      showTipsPanel(configuredTips());
    });
  }

  wireTipsButton();

  const cfg = window.__TIPS__;
  if (cfg && cfg.show && configuredTips().length > 0) {
    showTipsPanel(cfg.tips);
  }
})();
