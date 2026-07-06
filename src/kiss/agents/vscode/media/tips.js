// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Fresh-install Tips window for the chat webview.
//
// Defines the ``<kiss-tips-panel>`` web component: a centered modal
// panel that renders one tip at a time as formatted HTML (markdown via
// window.marked, loaded by chat.html before this script) with
// Previous / Next / Close controls.
//
// The tip list and the show-on-startup flag arrive via
// ``window.__TIPS__ = {tips: [...], show: bool}`` injected by the HTML
// builder (``SorcarTab.buildChatHtml`` in the extension,
// ``web_server._build_html`` on the remote webapp).  ``show`` is true
// exactly once — on the first chat render after a fresh installation.

/* global customElements */

(function () {
  'use strict';

  const PANEL_CSS =
    '.tips-overlay {' +
    '  position: fixed;' +
    '  inset: 0;' +
    '  z-index: 10000;' +
    '  display: flex;' +
    '  justify-content: center;' +
    '  align-items: center;' +
    '  background: rgba(0, 0, 0, 0.45);' +
    '}' +
    '.tips-panel {' +
    '  display: flex;' +
    '  flex-direction: column;' +
    '  max-width: min(560px, 90vw);' +
    '  max-height: 80vh;' +
    '  min-width: min(360px, 90vw);' +
    '  border: 2px solid var(--vscode-focusBorder, #007fd4);' +
    '  border-radius: 6px;' +
    '  background: var(--vscode-editorWidget-background, #252526);' +
    '  color: var(--vscode-editor-foreground, #ccc);' +
    '  font-family: var(--vscode-font-family, sans-serif);' +
    '  font-size: var(--vscode-font-size, 13px);' +
    '  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);' +
    '}' +
    '.tips-header {' +
    '  display: flex;' +
    '  align-items: center;' +
    '  gap: 8px;' +
    '  padding: 10px 12px;' +
    '  border-bottom: 1px solid var(--vscode-panel-border, #80808059);' +
    '}' +
    '.tips-title { font-weight: 600; flex: 1; }' +
    '.tips-counter {' +
    '  color: var(--vscode-descriptionForeground, #8b8b8b);' +
    '  font-size: 0.9em;' +
    '}' +
    '.tips-close {' +
    '  border: none;' +
    '  background: transparent;' +
    '  color: inherit;' +
    '  font-size: 16px;' +
    '  line-height: 1;' +
    '  cursor: pointer;' +
    '  padding: 2px 6px;' +
    '}' +
    '.tips-body { padding: 12px; overflow-y: auto; }' +
    '.tips-body pre { white-space: pre-wrap; margin: 0; }' +
    '.tips-body code {' +
    '  background: var(--vscode-input-background, #3c3c3c);' +
    '  padding: 1px 4px;' +
    '  border-radius: 3px;' +
    '}' +
    '.tips-footer {' +
    '  display: flex;' +
    '  justify-content: space-between;' +
    '  gap: 8px;' +
    '  padding: 10px 12px;' +
    '  border-top: 1px solid var(--vscode-panel-border, #80808059);' +
    '}' +
    '.tips-prev, .tips-next {' +
    '  border: none;' +
    '  border-radius: 3px;' +
    '  padding: 4px 12px;' +
    '  cursor: pointer;' +
    '  background: var(--vscode-button-background, #0e639c);' +
    '  color: var(--vscode-button-foreground, #fff);' +
    '}' +
    '.tips-prev:disabled, .tips-next:disabled {' +
    '  opacity: 0.4;' +
    '  cursor: default;' +
    '}';

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
      this._counter.textContent =
        total > 0 ? this._index + 1 + ' / ' + total : '';
      this._prev.disabled = this._index <= 0;
      this._next.disabled = this._index >= total - 1;
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
