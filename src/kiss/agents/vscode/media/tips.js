// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

/* global customElements */

(function () {
  'use strict';

  // Every colour, radius, shadow and duration is a design token that
  // main.css defines on :root (remote-codex.css re-derives the
  // theme-dependent ones on body.remote-chat, so the light theme of the
  // remote page flows through as well).  Custom properties inherit into
  // the shadow root, so this one stylesheet follows the active VS Code
  // theme in the sidebar and the editor tab, and both remote themes.
  // The fallbacks are VS Code Dark Modern.
  const PANEL_CSS =
    '.tips-overlay {' +
    '  position: fixed;' +
    '  inset: 0;' +
    '  z-index: var(--z-dialog, 300);' +
    '  display: flex;' +
    '  justify-content: center;' +
    '  align-items: center;' +
    '  background: var(--scrim, rgb(0 0 0 / 50%));' +
    '  animation: tips-fade var(--dur-slow, 0.2s) var(--ease, ease-out) both;' +
    '}' +
    '@keyframes tips-fade { from { opacity: 0; } }' +
    '@keyframes tips-rise { from { opacity: 0; transform: translateY(8px); } }' +
    '.tips-panel {' +
    '  display: flex;' +
    '  flex-direction: column;' +
    '  box-sizing: border-box;' +
    '  width: min(560px, 90vw);' +
    '  height: min(520px, 82vh);' +
    '  overflow: hidden;' +
    '  border: 1px solid var(--settings-border, var(--border, #2b2b2b));' +
    '  border-radius: var(--radius-xl, 10px);' +
    '  background: var(--bg2, #181818);' +
    '  color: var(--fg, #ccc);' +
    '  font-family: var(--vscode-font-family, sans-serif);' +
    '  font-size: var(--vscode-font-size, 13px);' +
    '  line-height: 1.5;' +
    '  box-shadow: var(--shadow-lg, 0 8px 24px rgb(0 0 0 / 36%));' +
    '  animation: tips-rise var(--dur-slow, 0.2s) var(--ease, ease-out) both;' +
    '}' +
    '.tips-header {' +
    '  display: flex;' +
    '  align-items: center;' +
    '  gap: var(--space-2, 8px);' +
    '  padding: var(--space-3, 12px) var(--space-3, 12px)' +
    '    var(--space-3, 12px) var(--space-4, 16px);' +
    '}' +
    '.tips-icon {' +
    '  display: inline-flex;' +
    '  flex: none;' +
    '  align-items: center;' +
    '  justify-content: center;' +
    '  width: 26px;' +
    '  height: 26px;' +
    '  border-radius: var(--radius-sm, 4px);' +
    '  color: var(--accent, #4daafc);' +
    '  background: var(--accent-tint, rgb(77 170 252 / 8%));' +
    '}' +
    '.tips-icon svg { width: 16px; height: 16px; }' +
    '.tips-title {' +
    '  flex: 1;' +
    '  font-weight: 600;' +
    '  font-size: var(--fs-lg, 1.1em);' +
    '  letter-spacing: 0.01em;' +
    '}' +
    '.tips-counter {' +
    '  padding: 1px var(--space-2, 8px);' +
    '  border: 1px solid var(--panel-line, rgb(255 255 255 / 10%));' +
    '  border-radius: var(--radius-pill, 999px);' +
    '  background: var(--panel-tint, rgb(255 255 255 / 4%));' +
    '  color: var(--dim, #9d9d9d);' +
    '  font-size: var(--fs-sm, 0.85em);' +
    '  font-variant-numeric: tabular-nums;' +
    '}' +
    '.tips-close {' +
    '  display: inline-flex;' +
    '  align-items: center;' +
    '  justify-content: center;' +
    '  width: 30px;' +
    '  height: 30px;' +
    '  padding: 0;' +
    '  border: none;' +
    '  border-radius: var(--radius-sm, 4px);' +
    '  background: transparent;' +
    '  color: color-mix(in srgb, var(--fg, #ccc) 55%, transparent);' +
    '  font-size: 26px;' +
    '  line-height: 1;' +
    '  cursor: pointer;' +
    '}' +
    '.tips-close:hover {' +
    '  color: var(--fg, #ccc);' +
    '  background: color-mix(in srgb, var(--fg, #ccc) 8%, transparent);' +
    '}' +
    // A hairline under the header doubles as the reading progress:
    // the accent bar grows with the tip index.
    '.tips-progress {' +
    '  flex: none;' +
    '  height: 2px;' +
    '  background: var(--panel-line, rgb(255 255 255 / 10%));' +
    '}' +
    '.tips-progress-bar {' +
    '  height: 100%;' +
    '  background: var(--accent, #4daafc);' +
    '  transition: width var(--dur-slow, 0.2s) var(--ease, ease-out);' +
    '}' +
    '.tips-body {' +
    '  flex: 1 1 auto;' +
    '  min-height: 0;' +
    '  padding: var(--space-4, 16px) var(--space-5, 20px);' +
    '  overflow: auto;' +
    '  overflow-wrap: anywhere;' +
    '  scrollbar-width: thin;' +
    '}' +
    '.tips-body > :first-child { margin-top: 0; }' +
    '.tips-body > :last-child { margin-bottom: 0; }' +
    '.tips-body h1, .tips-body h2, .tips-body h3, .tips-body h4 {' +
    '  margin: 0 0 var(--space-2, 8px);' +
    '  color: var(--fg, #ccc);' +
    '  font-weight: 600;' +
    '  line-height: 1.3;' +
    '}' +
    '.tips-body h1, .tips-body h2 { font-size: var(--fs-xl, 1.25em); }' +
    '.tips-body h3, .tips-body h4 { font-size: var(--fs-lg, 1.1em); }' +
    '.tips-body p, .tips-body ul, .tips-body ol {' +
    '  margin: 0 0 var(--space-2-5, 10px);' +
    '}' +
    '.tips-body ul, .tips-body ol { padding-left: var(--space-5, 20px); }' +
    '.tips-body li { margin: var(--space-0-5, 2px) 0; }' +
    '.tips-body a { color: var(--accent, #4daafc); text-decoration: none; }' +
    '.tips-body a:hover { text-decoration: underline; }' +
    '.tips-body strong { color: var(--fg, #ccc); font-weight: 600; }' +
    '.tips-body blockquote {' +
    '  margin: var(--space-2, 8px) 0;' +
    '  padding: var(--space-1, 4px) 0 var(--space-1, 4px) var(--space-3, 12px);' +
    '  border-left: 3px solid var(--accent-line, var(--accent, #4daafc));' +
    '  color: var(--dim, #9d9d9d);' +
    '}' +
    '.tips-body pre {' +
    '  margin: var(--space-2, 8px) 0;' +
    '  padding: var(--space-2-5, 10px) var(--space-3, 12px);' +
    '  border: 1px solid var(--panel-line, rgb(255 255 255 / 10%));' +
    '  border-radius: var(--radius-md, 6px);' +
    '  background: var(--vscode-textCodeBlock-background, #2b2b2b);' +
    '  overflow-x: auto;' +
    '  white-space: pre-wrap;' +
    '  font-family: var(--vscode-editor-font-family, monospace);' +
    '  font-size: var(--fs-sm, 0.85em);' +
    '  line-height: 1.45;' +
    '}' +
    '.tips-body code {' +
    '  padding: 1px var(--space-1, 4px);' +
    '  border-radius: var(--radius-sm, 4px);' +
    '  background: var(--vscode-textCodeBlock-background, #2b2b2b);' +
    '  color: var(--vscode-textPreformat-foreground, var(--fg, #ccc));' +
    '  font-family: var(--vscode-editor-font-family, monospace);' +
    '  font-size: 0.92em;' +
    '}' +
    '.tips-body pre code {' +
    '  padding: 0;' +
    '  background: transparent;' +
    '  color: inherit;' +
    '  font-size: inherit;' +
    '}' +
    '.tips-body table { margin: var(--space-2, 8px) 0; border-collapse: collapse; }' +
    '.tips-body th, .tips-body td {' +
    '  padding: var(--space-1, 4px) var(--space-2, 8px);' +
    '  border: 1px solid var(--panel-line, rgb(255 255 255 / 10%));' +
    '  text-align: left;' +
    '}' +
    '.tips-body th { background: var(--panel-tint, rgb(255 255 255 / 4%)); font-weight: 600; }' +
    '.tips-body hr {' +
    '  margin: var(--space-3, 12px) 0;' +
    '  border: none;' +
    '  border-top: 1px solid var(--panel-line, rgb(255 255 255 / 10%));' +
    '}' +
    '.tips-code { position: relative; }' +
    '.tips-copy {' +
    '  position: absolute;' +
    '  top: var(--space-1-5, 6px);' +
    '  right: var(--space-1-5, 6px);' +
    '  padding: 1px var(--space-2, 8px);' +
    '  border: 1px solid var(--border, #2b2b2b);' +
    '  border-radius: var(--radius-sm, 4px);' +
    '  background: var(--panel-tint-solid, #232323);' +
    '  color: var(--dim, #9d9d9d);' +
    '  font: inherit;' +
    '  font-size: var(--fs-xs, 0.7em);' +
    '  cursor: pointer;' +
    '}' +
    '.tips-copy:hover {' +
    '  color: var(--fg, #ccc);' +
    '  border-color: color-mix(in srgb, var(--fg, #ccc) 24%, transparent);' +
    '  background: color-mix(in srgb, var(--fg, #ccc) 10%, var(--bg, #1f1f1f));' +
    '}' +
    '.tips-footer {' +
    '  display: flex;' +
    '  flex-wrap: wrap;' +
    '  align-items: center;' +
    '  gap: var(--space-2, 8px);' +
    '  padding: var(--space-3, 12px) var(--space-4, 16px);' +
    '  border-top: 1px solid var(--panel-line, rgb(255 255 255 / 10%));' +
    '  background: var(--panel-tint, rgb(255 255 255 / 4%));' +
    '}' +
    '.tips-prev, .tips-next {' +
    '  display: inline-flex;' +
    '  align-items: center;' +
    '  gap: var(--space-1, 4px);' +
    '  min-height: 26px;' +
    '  padding: var(--space-1, 4px) var(--space-3, 12px);' +
    '  border: 1px solid transparent;' +
    '  border-radius: var(--radius-sm, 4px);' +
    '  font: inherit;' +
    '  font-size: var(--fs-md, 0.9em);' +
    '  cursor: pointer;' +
    '}' +
    '.tips-prev {' +
    '  border-color: var(--border, #2b2b2b);' +
    '  background: var(--vscode-button-secondaryBackground, transparent);' +
    '  color: var(--vscode-button-secondaryForeground, var(--fg, #ccc));' +
    '}' +
    '.tips-prev:hover:enabled {' +
    '  background: var(--vscode-button-secondaryHoverBackground,' +
    '    color-mix(in srgb, var(--fg, #ccc) 10%, transparent));' +
    '}' +
    '.tips-next {' +
    '  margin-left: auto;' +
    '  background: var(--vscode-button-background, #0078d4);' +
    '  color: var(--vscode-button-foreground, #fff);' +
    '}' +
    '.tips-next:hover:enabled {' +
    '  background: var(--vscode-button-hoverBackground, #026ec1);' +
    '}' +
    '.tips-prev:disabled, .tips-next:disabled {' +
    '  opacity: 0.4;' +
    '  cursor: default;' +
    '}' +
    '.tips-optout {' +
    '  display: flex;' +
    '  flex: 1;' +
    '  align-items: center;' +
    '  justify-content: center;' +
    '  gap: var(--space-1-5, 6px);' +
    '  color: var(--dim, #9d9d9d);' +
    '  font-size: var(--fs-sm, 0.85em);' +
    '  cursor: pointer;' +
    '  user-select: none;' +
    '}' +
    '.tips-optout input { margin: 0; accent-color: var(--accent, #4daafc); }' +
    // Phones: the paging buttons keep the first row, the checkbox
    // moves to a row of its own instead of squeezing the labels.
    '@media (width <= 480px) {' +
    '  .tips-optout { flex: 1 1 100%; order: 1; justify-content: flex-start; }' +
    '}' +
    '.tips-close:focus-visible, .tips-prev:focus-visible,' +
    '.tips-next:focus-visible, .tips-copy:focus-visible,' +
    '.tips-optout input:focus-visible {' +
    '  outline: 1px solid var(--vscode-focusBorder, var(--accent, #4daafc));' +
    '  outline-offset: 1px;' +
    '}';

  // Lightbulb, 16x16 stroke icon (no external asset: the shadow root
  // cannot see the webview's media URIs).
  const TIP_ICON_SVG =
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor"' +
    ' stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"' +
    ' aria-hidden="true">' +
    '<path d="M6 13h4M6.5 15h3"/>' +
    '<path d="M8 1.5a4.5 4.5 0 0 0-2.6 8.2c.4.3.6.7.6 1.1V11h4v-.2' +
    'c0-.4.2-.8.6-1.1A4.5 4.5 0 0 0 8 1.5z"/>' +
    '</svg>';

  // The webview remembers the user's "don't show tips automatically"
  // choice under this localStorage key so the auto-open on cfg.show is
  // skipped even before the host has persisted the choice.
  const OPT_OUT_KEY = 'kissTipsOptOut';

  // The product version whose tips this browser has already auto-opened.
  // The remote server serves many devices and cannot tell which of them
  // saw the tips, so it sends `show: true` with the running `version`
  // and each browser remembers the last version it showed here: the
  // tips come back once per device after every update.
  const SEEN_VERSION_KEY = 'kissTipsSeenVersion';

  function readOptOut() {
    try {
      return window.localStorage.getItem(OPT_OUT_KEY) === '1';
    } catch {
      return false;
    }
  }

  function readSeenVersion() {
    try {
      return window.localStorage.getItem(SEEN_VERSION_KEY);
    } catch {
      return null;
    }
  }

  function writeSeenVersion(version) {
    try {
      window.localStorage.setItem(SEEN_VERSION_KEY, version);
    } catch {
      // No storage: the tips simply reopen on the next page load.
    }
  }

  /**
   * Persist the "don't show tips automatically" choice.
   *
   * Stores it in localStorage for the webview itself and dispatches the
   * `kiss-tips-opt-out` CustomEvent on `window` with
   * `detail = {type: 'tipsOptOut', optOut: <boolean>}`.  main.js relays
   * that detail to the extension host / remote server as a
   * `tipsOptOut` message so the choice survives extension updates
   * (the host then stops claiming the popup for new versions).
   */
  function writeOptOut(optOut) {
    try {
      if (optOut) window.localStorage.setItem(OPT_OUT_KEY, '1');
      else window.localStorage.removeItem(OPT_OUT_KEY);
    } catch {
      // Storage may be unavailable (private mode); the host copy is
      // the durable one.
    }
    window.dispatchEvent(
      new CustomEvent('kiss-tips-opt-out', {
        detail: {type: 'tipsOptOut', optOut: !!optOut},
      }),
    );
  }

  function copyViaExecCommand(text) {
    // Looked up at call time: panelCopy.js loads after tips.js.
    const pc = window.PanelCopy;
    return !!(pc && pc.fallbackCopyText && pc.fallbackCopyText(text));
  }

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

  // tipsflash0903-coverage:start
  // The revert timer lives on the button and is restarted on every
  // copy: a bare setTimeout let a rapid second click's "Copied!" be
  // wiped early by the first click's stale timer.
  function flashCopyResult(btn, ok) {
    btn.textContent = ok ? 'Copied!' : 'Failed';
    if (btn._kissFlashTimer) clearTimeout(btn._kissFlashTimer);
    btn._kissFlashTimer = setTimeout(() => {
      btn._kissFlashTimer = null;
      btn.textContent = 'Copy';
    }, 1500);
  }
  // tipsflash0903-coverage:end

  function onCopyClick(event) {
    const btn = event.currentTarget;
    const code = btn.parentElement.querySelector('pre code');
    const text = code ? code.textContent : '';
    // tipsflash0903-coverage:start
    // Async clipboard writes can settle out of order: without an
    // ownership check, a rapid first click failing AFTER a second
    // click succeeded replaced the latest "Copied!" with "Failed" and
    // restarted the revert timer from the stale operation.  Each click
    // takes a new per-button generation; only the completion that
    // still owns the latest generation may update the label.
    const gen = (btn._kissCopyGen || 0) + 1;
    btn._kissCopyGen = gen;
    copyTextToClipboard(text).then(ok => {
      if (btn._kissCopyGen !== gen) return;
      flashCopyResult(btn, ok);
    });
    // tipsflash0903-coverage:end
  }

  class KissTipsPanel extends HTMLElement {
    constructor() {
      super();
      this._tips = [];
      this._tipsAssigned = false;
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
      const icon = document.createElement('span');
      icon.className = 'tips-icon';
      icon.innerHTML = TIP_ICON_SVG;
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
      header.appendChild(icon);
      header.appendChild(title);
      header.appendChild(this._counter);
      header.appendChild(this._close);

      const progress = document.createElement('div');
      progress.className = 'tips-progress';
      this._progressBar = document.createElement('div');
      this._progressBar.className = 'tips-progress-bar';
      progress.appendChild(this._progressBar);

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
      const optOutLabel = document.createElement('label');
      optOutLabel.className = 'tips-optout';
      this._optOut = document.createElement('input');
      this._optOut.type = 'checkbox';
      this._optOut.className = 'tips-optout-input';
      this._optOut.checked = readOptOut();
      optOutLabel.appendChild(this._optOut);
      optOutLabel.appendChild(
        document.createTextNode("Don't show tips automatically"),
      );
      footer.appendChild(this._prev);
      footer.appendChild(optOutLabel);
      footer.appendChild(this._next);

      panel.appendChild(header);
      panel.appendChild(progress);
      panel.appendChild(this._body);
      panel.appendChild(footer);
      overlay.appendChild(panel);
      root.appendChild(overlay);
      this._overlay = overlay;
      this._panel = panel;
      this._opener = null;

      const self = this;
      this._prev.addEventListener('click', () => self._step(-1));
      this._next.addEventListener('click', () => self._step(1));
      this._close.addEventListener('click', () => {
        self.remove();
      });
      this._optOut.addEventListener('change', () => {
        writeOptOut(self._optOut.checked);
      });
      // A click on the dimmed backdrop (not inside the panel) closes.
      overlay.addEventListener('click', event => {
        if (event.target === overlay) self.remove();
      });
      this._onKeyDown = function (event) {
        self._handleKeyDown(event);
      };
    }

    /** Move `delta` tips (-1 / +1), clamped to the list. */
    _step(delta) {
      const target = this._index + delta;
      if (target < 0 || target >= this._tips.length) return;
      this._index = target;
      this._update();
    }

    /**
     * Escape closes the dialog; Left/Right arrows page through the
     * tips; Tab and Shift+Tab cycle inside it so focus never lands on
     * the page behind the modal overlay.
     */
    _handleKeyDown(event) {
      if (event.key === 'Escape') {
        event.preventDefault();
        this.remove();
        return;
      }
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        this._step(event.key === 'ArrowLeft' ? -1 : 1);
        return;
      }
      if (event.key !== 'Tab') return;
      const items = Array.from(
        this._panel.querySelectorAll('button, input'),
      ).filter(el => !el.disabled);
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      const current = this.shadowRoot.activeElement;
      if (event.shiftKey && (current === first || current === null)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && current === last) {
        event.preventDefault();
        first.focus();
      }
    }

    get tips() {
      return this._tips;
    }

    set tips(list) {
      this._tips = Array.isArray(list) ? list : [];
      this._tipsAssigned = true;
      this._index = 0;
      this._update();
    }

    connectedCallback() {
      if (!this._tipsAssigned) {
        this.remove();
        return;
      }
      // Remember who opened the dialog so closing can hand focus back,
      // then move focus into the dialog (the close button).
      this._opener = document.activeElement;
      document.addEventListener('keydown', this._onKeyDown, true);
      this._close.focus();
    }

    disconnectedCallback() {
      document.removeEventListener('keydown', this._onKeyDown, true);
      const opener = this._opener;
      this._opener = null;
      if (
        opener &&
        opener !== document.body &&
        typeof opener.focus === 'function' &&
        document.contains(opener)
      ) {
        opener.focus();
      }
    }

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
      this._progressBar.style.width =
        total > 0 ? ((this._index + 1) / total) * 100 + '%' : '0%';
      this._prev.disabled = this._index <= 0;
      this._next.disabled = this._index >= total - 1;
    }

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

  function showTipsPanel(tips) {
    const el = document.createElement('kiss-tips-panel');
    el.tips = tips;
    document.body.appendChild(el);
    return el;
  }

  window.__kissShowTipsPanel = showTipsPanel;

  function configuredTips() {
    const cfg = window.__TIPS__;
    return cfg && Array.isArray(cfg.tips) ? cfg.tips : [];
  }

  function wireTipsButton() {
    const btn = document.getElementById('tips-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      if (document.body.querySelector('kiss-tips-panel')) return;
      showTipsPanel(configuredTips());
    });
  }

  wireTipsButton();

  /**
   * Whether to open the tips now without a click.  `cfg.show` is the
   * host's verdict (the VS Code extension claims the popup once per
   * version in $KISS_HOME; the remote server says yes unless the user
   * opted out), the opt-out checkbox overrides it, and `cfg.version`,
   * when the host sends one, limits the auto-open to once per version
   * per browser.
   */
  function shouldAutoShow(cfg) {
    if (!cfg || !cfg.show || readOptOut()) return false;
    if (configuredTips().length === 0) return false;
    if (document.body.querySelector('kiss-tips-panel')) return false;
    return !cfg.version || readSeenVersion() !== cfg.version;
  }

  const cfg = window.__TIPS__;
  if (shouldAutoShow(cfg)) {
    showTipsPanel(cfg.tips);
    if (cfg.version) writeSeenVersion(cfg.version);
  }
})();
