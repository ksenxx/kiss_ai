// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The terminal tab: a shell running on the daemon's machine, shown on
// this surface with xterm.js (media/xterm.js + media/xterm-addon-fit.js).
//
// The daemon (kiss/server/terminal_tab.py) owns the shell and its
// pseudo-terminal.  This module builds the view for one such shell:
// an xterm.js terminal that sends what the user types as terminalInput,
// its size as terminalResize, and paints the terminalData bytes the
// daemon streams back.
//
//   const view = TerminalTabView.create(tabId, {
//     send: msg => api[msg.type](msg),   // terminalAttach / terminalInput /
//                                        // terminalResize
//   });
//   view.el                 -> element to mount in the content area
//   view.data(base64)       -> paint a terminalData event
//   view.setVisible(bool)   -> attach (first show) / fit + focus
//   view.resubscribe()      -> re-attach after a reconnect
//   view.dispose()          -> stop observers, free the xterm instance
//
// A surface attaches lazily, the first time it shows the tab: the
// daemon then replays the scrollback it kept and streams from there.
// Re-attaching (reconnect, re-announcement) resets the terminal before
// the replay so nothing is painted twice.

(function (global) {
  'use strict';

  const FIT_DEBOUNCE_MS = 50;
  const ANSI_NAMES = [
    'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white',
    'brightBlack', 'brightRed', 'brightGreen', 'brightYellow',
    'brightBlue', 'brightMagenta', 'brightCyan', 'brightWhite',
  ];

  function cssVar(style, names, fallback) {
    for (const name of names) {
      const v = style.getPropertyValue(name).trim();
      if (v) return v;
    }
    return fallback;
  }

  // The surface's palette: VS Code exposes its terminal.* theme colours
  // as --vscode-terminal-* variables and the remote webapp defines the
  // same names, so one routine serves both.
  function themeFromCss() {
    const style = getComputedStyle(document.body);
    const fg = cssVar(style, ['--vscode-terminal-foreground', '--vscode-editor-foreground'], '#cccccc');
    const bg = cssVar(style, ['--vscode-terminal-background', '--vscode-editor-background'], '#1f1f1f');
    const theme = {
      foreground: fg,
      background: bg,
      cursor: cssVar(style, ['--vscode-terminalCursor-foreground'], fg),
      cursorAccent: cssVar(style, ['--vscode-terminalCursor-background'], bg),
      selectionBackground: cssVar(
        style,
        ['--vscode-terminal-selectionBackground', '--vscode-editor-selectionBackground'],
        'rgba(128, 128, 128, 0.4)',
      ),
    };
    for (const name of ANSI_NAMES) {
      const key = '--vscode-terminal-ansi' + name[0].toUpperCase() + name.slice(1);
      const v = style.getPropertyValue(key).trim();
      if (v) theme[name] = v;
    }
    return theme;
  }

  function fontFromCss() {
    const style = getComputedStyle(document.body);
    return {
      fontFamily: cssVar(
        style,
        ['--vscode-terminal-font-family', '--vscode-editor-font-family'],
        "Menlo, Monaco, Consolas, 'Courier New', monospace",
      ),
      fontSize: parseInt(cssVar(style, ['--vscode-editor-font-size'], '13'), 10) || 13,
    };
  }

  function bytesFromBase64(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // Keys the browser (or the host page) must keep: copy/paste with
  // Ctrl+Shift+C/V as in VS Code's terminal, and the page's own
  // shortcuts that do not make sense inside a shell.
  function keepForBrowser(e) {
    if (e.type !== 'keydown') return false;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.shiftKey && (e.key === 'C' || e.key === 'c' || e.key === 'V' || e.key === 'v')) {
      return true;
    }
    return false;
  }

  function create(tabId, opts) {
    const send = opts.send;
    const el = document.createElement('div');
    el.className = 'terminal-view';
    el.setAttribute('role', 'region');
    el.setAttribute('aria-label', 'Terminal');
    const holder = document.createElement('div');
    holder.className = 'terminal-xterm';
    el.appendChild(holder);

    const font = fontFromCss();
    const term = new global.Terminal({
      cursorBlink: true,
      scrollback: 10000,
      fontFamily: font.fontFamily,
      fontSize: font.fontSize,
      theme: themeFromCss(),
    });
    const fit = new global.FitAddon.FitAddon();
    term.loadAddon(fit);
    term.attachCustomKeyEventHandler(e => !keepForBrowser(e));

    let opened = false;
    let visible = false;
    let attached = false;
    let disposed = false;
    let fitTimer = 0;
    let lastCols = 0;
    let lastRows = 0;

    term.onData(data => {
      if (attached) send({type: 'terminalInput', tab_id: tabId, data});
    });
    term.onBinary(data => {
      if (attached) send({type: 'terminalInput', tab_id: tabId, data, binary: true});
    });
    term.onResize(size => {
      if (!attached || (size.cols === lastCols && size.rows === lastRows)) return;
      lastCols = size.cols;
      lastRows = size.rows;
      send({type: 'terminalResize', tab_id: tabId, cols: size.cols, rows: size.rows});
    });

    function fitNow() {
      if (!opened || !visible || disposed) return;
      try {
        fit.fit();
      } catch (_e) {
        // The holder has no size yet (hidden mid-layout); the observer
        // fits again once it does.
      }
    }

    function scheduleFit() {
      clearTimeout(fitTimer);
      fitTimer = setTimeout(fitNow, FIT_DEBOUNCE_MS);
    }

    const resizeObserver =
      typeof ResizeObserver === 'function' ? new ResizeObserver(scheduleFit) : null;
    // Theme toggles flip a body class (remote webapp) or rewrite the
    // body's style variables (VS Code); either way re-read the palette.
    const themeObserver =
      typeof MutationObserver === 'function'
        ? new MutationObserver(() => {
            term.options.theme = themeFromCss();
          })
        : null;
    if (themeObserver) {
      themeObserver.observe(document.body, {attributes: true, attributeFilter: ['class', 'style']});
    }

    function attach() {
      attached = true;
      lastCols = term.cols;
      lastRows = term.rows;
      send({type: 'terminalAttach', tab_id: tabId, cols: term.cols, rows: term.rows});
    }

    const view = {
      el,
      term,
      data(b64) {
        if (disposed || !b64) return;
        term.write(bytesFromBase64(b64));
      },
      setVisible(on) {
        visible = !!on;
        if (!visible || disposed) return;
        if (!opened) {
          opened = true;
          term.open(holder);
          if (resizeObserver) resizeObserver.observe(holder);
        }
        fitNow();
        if (!attached) attach();
        term.focus();
      },
      focus() {
        if (opened && !disposed) term.focus();
      },
      resubscribe() {
        // The daemon forgot us (reconnect) or announced the tab again:
        // start over from its backlog so nothing is painted twice.
        if (!attached) return;
        term.reset();
        attach();
      },
      dispose() {
        disposed = true;
        clearTimeout(fitTimer);
        if (resizeObserver) resizeObserver.disconnect();
        if (themeObserver) themeObserver.disconnect();
        term.dispose();
      },
    };
    return view;
  }

  global.TerminalTabView = {create};
})(typeof window !== 'undefined' ? window : globalThis);
