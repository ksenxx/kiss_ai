// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The terminal tab: a shell on the machine running the daemon, shown
// in an xterm.js terminal (remote webapp only; a VS Code window has its
// own integrated terminal).  The daemon half is
// kiss/server/terminal_tab.py: it runs the shell on a pty and streams
// its output as terminalData events; keystrokes go back as
// terminalInput.  This module builds the view for one such tab:
//
//   const view = TerminalTabView.create(tabId, {
//     send: msg => api[msg.type](msg),   // terminalOpen / terminalInput /
//                                        // terminalResize / terminalClose
//     workDir: '/path/the/shell/starts/in',   // '' = the daemon's work dir
//   });
//   view.el                 -> element to mount in the content area
//   view.setVisible(bool)   -> shown: load xterm, open the shell, fit, focus
//   view.data(text)         -> write a terminalData event
//   view.opened(ev)         -> a terminalOpened reply (new or re-attached)
//   view.exit(code)         -> the shell ended (terminalExit)
//   view.error(text)        -> the shell could not start (terminalError)
//   view.reconnect()        -> the WebSocket came back: re-attach the shell
//   view.retheme()          -> the page theme changed
//   view.dispose()          -> drop the terminal (the daemon is told
//                              separately, through terminalClose)
//
// xterm.js (6.x) and its fit addon come from jsDelivr on first use.
// Their UMD bundles register as an AMD module whenever a global
// `define` exists, which it does once Monaco's loader has run, so the
// bundles are evaluated with `define` shadowed and attach `Terminal`
// and `FitAddon` to the window as on a plain page.

(function (global) {
  'use strict';

  const XTERM_BASE = 'https://cdn.jsdelivr.net/npm/@xterm/xterm@6.0.0';
  const FIT_BASE = 'https://cdn.jsdelivr.net/npm/@xterm/addon-fit@0.11.0';
  const LOAD_TIMEOUT_MS = 15000;
  const RESIZE_DEBOUNCE_MS = 100;
  const DIM = '\x1b[2m';
  const RED = '\x1b[31m';
  const RESET = '\x1b[0m';

  let xtermPromise = null;

  function fetchText(url) {
    const ctl = new global.AbortController();
    const timer = setTimeout(() => ctl.abort(), LOAD_TIMEOUT_MS);
    return fetch(url, {signal: ctl.signal})
      .then(r => {
        if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
        return r.text();
      })
      .finally(() => clearTimeout(timer));
  }

  function evalUmd(code) {
    // Neither CommonJS nor AMD in scope: the bundle takes its
    // browser-global branch.
    new Function('define', 'module', 'exports', code)(
      undefined,
      undefined,
      undefined,
    );
  }

  function ensureXterm() {
    if (xtermPromise) return xtermPromise;
    if (global.Terminal && global.FitAddon) {
      xtermPromise = Promise.resolve();
      return xtermPromise;
    }
    xtermPromise = Promise.all([
      fetchText(XTERM_BASE + '/css/xterm.css'),
      fetchText(XTERM_BASE + '/lib/xterm.js'),
      fetchText(FIT_BASE + '/lib/addon-fit.js'),
    ])
      .then(parts => {
        const style = document.createElement('style');
        style.id = 'xterm-css';
        style.textContent = parts[0];
        document.head.appendChild(style);
        evalUmd(parts[1]);
        evalUmd(parts[2]);
        if (!global.Terminal || !global.FitAddon) {
          throw new Error('xterm.js did not initialise');
        }
      })
      .catch(err => {
        // A failed download must not poison every later open.
        xtermPromise = null;
        throw err;
      });
    return xtermPromise;
  }

  function cssVar(name, fallback) {
    const v = global
      .getComputedStyle(document.body)
      .getPropertyValue(name)
      .trim();
    return v || fallback;
  }

  // The page palette (VS Code Dark/Light Modern, injected by
  // web_server.py) as an xterm theme.  Only names the palette defines
  // are read (test_remote_vscode_theme keeps that list honest); the
  // terminal shares the editor's background, as VS Code's panel does.
  function themeFromPage() {
    const bg = cssVar('--vscode-editor-background', '#1f1f1f');
    const fg =
      cssVar('--vscode-terminal-foreground', '') ||
      cssVar('--vscode-editor-foreground', '#cccccc');
    const ansi = name => cssVar('--vscode-terminal-ansi' + name, '');
    const theme = {
      background: bg,
      foreground: fg,
      cursor: fg,
      cursorAccent: bg,
      selectionBackground: cssVar(
        '--vscode-editor-selectionBackground',
        '#264f78',
      ),
    };
    const names = [
      'Black',
      'Red',
      'Green',
      'Yellow',
      'Blue',
      'Magenta',
      'Cyan',
      'White',
      'BrightBlack',
      'BrightRed',
      'BrightGreen',
      'BrightYellow',
      'BrightBlue',
      'BrightMagenta',
      'BrightCyan',
      'BrightWhite',
    ];
    names.forEach(n => {
      const c = ansi(n);
      if (c) theme[n.charAt(0).toLowerCase() + n.slice(1)] = c;
    });
    return theme;
  }

  function create(tabId, opts) {
    const send = opts.send;
    const el = document.createElement('div');
    el.className = 'terminal-view';
    const surface = document.createElement('div');
    surface.className = 'terminal-surface';
    el.appendChild(surface);

    const view = {el: el, tabId: tabId, disposed: false, exited: false};
    let term = null;
    let fit = null;
    let observer = null;
    let resizeTimer = null;
    let visible = false;
    let starting = false;
    // Output that arrived before xterm was ready (a re-attached shell
    // may print before the terminal exists on this page).
    let pending = [];
    // Set once a shell was opened for this tab, so a later
    // terminalOpened with attached:false means the old shell is gone.
    let hadShell = false;
    // Set once terminalOpen was sent, so a reconnect re-sends it even
    // when the socket dropped before the first terminalOpened came.
    let openRequested = false;
    // Set while an Enter on an ended shell asks for a new one, so the
    // fresh shell's terminalOpened is not reported as a lost session.
    let restarting = false;
    let lastSize = {cols: 0, rows: 0};

    function note(text, colour) {
      const line = '\r\n' + (colour || DIM) + text + RESET + '\r\n';
      if (term) term.write(line);
      else pending.push(line);
    }

    function openShell() {
      lastSize = {cols: term ? term.cols : 80, rows: term ? term.rows : 24};
      openRequested = true;
      send({
        type: 'terminalOpen',
        tab_id: tabId,
        cols: lastSize.cols,
        rows: lastSize.rows,
        workDir: opts.workDir || '',
      });
    }

    function restartShell() {
      view.exited = false;
      restarting = true;
      if (term) term.write('\r\n');
      openShell();
    }

    // Runs before xterm's own key handling (which stops propagation).
    function keyHandler(ev) {
      if (ev.type !== 'keydown') return true;
      // Enter on an ended shell starts a new one in the same tab.
      if (ev.key === 'Enter' && view.exited && hadShell) {
        restartShell();
        return false;
      }
      // Keep the browser's copy/paste chords: Ctrl/Cmd+Shift+C copies
      // the selection, Ctrl/Cmd+Shift+V pastes through the textarea's
      // native paste event (which xterm handles).
      if (!ev.shiftKey || !(ev.ctrlKey || ev.metaKey)) return true;
      if (ev.key === 'C' || ev.key === 'c') {
        const sel = term.getSelection();
        if (sel && navigator.clipboard) navigator.clipboard.writeText(sel);
        return false;
      }
      return !(ev.key === 'V' || ev.key === 'v');
    }

    function start() {
      if (term || starting || view.disposed) return;
      starting = true;
      ensureXterm().then(
        () => {
          starting = false;
          if (view.disposed || term) return;
          term = new global.Terminal({
            cursorBlink: true,
            fontSize: 13,
            fontFamily: cssVar(
              '--vscode-editor-font-family',
              "'Fira Code', Menlo, Consolas, monospace",
            ),
            scrollback: 5000,
            theme: themeFromPage(),
          });
          fit = new global.FitAddon.FitAddon();
          term.loadAddon(fit);
          surface.replaceChildren(); // a load-error note from an earlier try
          term.open(surface);
          term.attachCustomKeyEventHandler(keyHandler);
          fit.fit();
          term.onData(data => {
            if (!view.exited)
              send({type: 'terminalInput', tab_id: tabId, data: data});
          });
          term.onResize(size => {
            if (size.cols === lastSize.cols && size.rows === lastSize.rows)
              return;
            lastSize = {cols: size.cols, rows: size.rows};
            send({
              type: 'terminalResize',
              tab_id: tabId,
              cols: size.cols,
              rows: size.rows,
            });
          });
          // Refit the terminal to its surface; the onResize handler
          // above forwards a changed size to the pty.  fit() is a no-op
          // until xterm has measured a cell size, and Terminal.resize()
          // ignores an unchanged grid, so nothing is sent unless the
          // grid really changed.
          observer = new ResizeObserver(() => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
              if (visible && !view.disposed) fit.fit();
            }, RESIZE_DEBOUNCE_MS);
          });
          observer.observe(el);
          pending.forEach(chunk => term.write(chunk));
          pending = [];
          openShell();
          if (visible) term.focus();
        },
        err => {
          starting = false;
          // No terminal to write into: say so in the tab itself.  A
          // later show of the tab (switch away and back) retries.
          const msg = document.createElement('div');
          msg.className = 'terminal-load-error';
          msg.textContent =
            'The terminal could not load xterm.js from jsDelivr (' +
            (err && err.message ? err.message : err) +
            ').  Check the connection, then switch to another tab and back to retry.';
          surface.replaceChildren(msg);
        },
      );
    }

    view.setVisible = function (show) {
      visible = !!show;
      if (!visible) return;
      if (!term) {
        start();
        return;
      }
      // The tab is on screen already (the caller set its display), so
      // keystrokes may follow at once; the fit waits for the layout.
      term.focus();
      requestAnimationFrame(() => {
        if (view.disposed || !visible) return;
        fit.fit();
      });
    };

    view.data = function (text) {
      if (term) term.write(text);
      else pending.push(text);
    };

    view.opened = function (ev) {
      if (hadShell && !ev.attached && !restarting) {
        note(
          'The shell ended while this page was disconnected; a new one started.',
        );
      }
      hadShell = true;
      restarting = false;
      view.exited = false;
    };

    view.exit = function (code) {
      view.exited = true;
      note(
        'The shell exited' +
          (code ? ' with code ' + code : '') +
          '.  Close the tab, or press Enter to start a new one.',
      );
    };

    view.error = function (text) {
      view.exited = true;
      note(text, RED);
    };

    view.reconnect = function () {
      if (view.disposed || view.exited || !openRequested) return;
      openShell();
    };

    view.retheme = function () {
      if (term) term.options.theme = themeFromPage();
    };

    view.dispose = function () {
      if (view.disposed) return;
      view.disposed = true;
      clearTimeout(resizeTimer);
      if (observer) observer.disconnect();
      if (term) term.dispose();
      term = null;
      fit = null;
    };

    return view;
  }

  global.TerminalTabView = {create};
})(typeof window !== 'undefined' ? window : globalThis);
