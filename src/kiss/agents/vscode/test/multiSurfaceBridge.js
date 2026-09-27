// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// Test helper: host several REAL chat webviews (media/main.js under
// jsdom), each connected to a running kiss daemon over its Unix
// socket, and expose them to a driving process over stdin/stdout.
//
// Every webview is one "surface" — a VS Code sidebar view, a VS Code
// editor-tab panel, or a browser tab of the remote web app — and each
// gets its OWN daemon connection, exactly like production (the
// extension host forwards webview commands verbatim over the UDS; the
// remote page's acquireVsCodeApi shim posts them over its WebSocket).
// Everything a webview posts is written to its socket as one JSON
// line; every event line the daemon writes is dispatched to the
// webview as a `message` event.
//
// Protocol (one JSON object per line):
//   in : {op:'open', name, bodyAttrs, state}   -> out {op:'opened', name}
//   in : {op:'close', name}                    -> out {op:'closed', name}
//   in : {op:'tabs', name}                     -> out {op:'tabs', name, tabs, errors}
//   in : {op:'submit', name, text}             -> out {op:'submitted', name}
//   in : {op:'closeTab', name, tabId}          -> out {op:'tabClosed', name}
//   in : {op:'events', name}                   -> out {op:'events', name, events}
//   in : {op:'quit'}
//
// `tabs` describes the rendered tab bar (#tab-list, the user-visible
// truth) joined with the webview's own tab records (parent, task id,
// running/done flags) taken from the state it persists via setState.

const fs = require('fs');
const net = require('net');
const path = require('path');
const readline = require('readline');
const {JSDOM, VirtualConsole} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');
const SOCK_PATH = process.argv[2];
if (!SOCK_PATH) {
  process.stderr.write('usage: node multiSurfaceBridge.js <uds-path>\n');
  process.exit(2);
}

const surfaces = new Map();
const DETAILED_EVENTS = new Set([
  'new_tab',
  'openSubagentTab',
  'subagentDone',
  'closeSubagentTab',
  'error',
]);

function out(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n');
}

function makeWebview(bodyAttrs, initialState, onPost, errors) {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace('{{BODY_CLASS_ATTR}}', bodyAttrs || '');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', e => {
    errors.push(String((e && e.stack) || e));
  });
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
    virtualConsole,
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  let state = initialState || null;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: onPost,
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, getState: () => state};
}

function openSurface(cmd) {
  const errors = [];
  const events = [];
  const socket = net.createConnection(SOCK_PATH);
  const pending = [];
  let connected = false;
  socket.on('connect', () => {
    connected = true;
    pending.splice(0).forEach(line => socket.write(line));
    out({op: 'opened', name: cmd.name});
  });
  function post(msg) {
    const line = JSON.stringify(msg) + '\n';
    if (connected) socket.write(line);
    else pending.push(line);
  }
  const view = makeWebview(cmd.bodyAttrs, cmd.state, post, errors);
  const rl = readline.createInterface({input: socket});
  rl.on('line', line => {
    if (!line.trim()) return;
    let ev;
    try {
      ev = JSON.parse(line);
    } catch (_e) {
      return;
    }
    // Tab-lifecycle events are kept whole (the driver asserts on their
    // fields); everything else only by type.
    events.push(DETAILED_EVENTS.has(ev.type) ? ev : ev.type);
    try {
      view.win.dispatchEvent(
        new view.win.MessageEvent('message', {data: ev}),
      );
    } catch (e) {
      errors.push('dispatch ' + ev.type + ': ' + String((e && e.stack) || e));
    }
  });
  socket.on('error', e => errors.push('socket: ' + String(e)));
  surfaces.set(cmd.name, {view, socket, errors, events});
}

// The rendered tab bar is the user-visible truth: one entry per
// `#tab-list [data-tab-id]` element.  A sub-agent tab's id is
// `${parentTabId}__sub_${taskId}` (main.js subagentTabIdFor), so the
// driver derives parent and task from the id.
function describeTabs(surface) {
  const {win} = surface.view;
  const els = Array.from(
    win.document.querySelectorAll('#tab-list [data-tab-id]'),
  );
  return els.map(el => {
    const indicator = el.querySelector('.subagent-indicator');
    return {
      id: el.dataset.tabId,
      title: (el.querySelector('.chat-tab-label') || {}).textContent || '',
      isSubagentTab: el.classList.contains('subagent-tab'),
      subagentDone: !!(indicator && indicator.classList.contains('done')),
      running: !!el.querySelector('.status-spinner'),
    };
  });
}

function handle(cmd) {
  if (cmd.op === 'quit') {
    surfaces.forEach(s => s.socket.destroy());
    process.exit(0);
  }
  if (cmd.op === 'open') {
    openSurface(cmd);
    return;
  }
  const surface = surfaces.get(cmd.name);
  if (!surface) {
    out({op: 'error', name: cmd.name, error: 'no such surface'});
    return;
  }
  const doc = surface.view.win.document;
  switch (cmd.op) {
    case 'close':
      surface.socket.destroy();
      surface.view.win.close();
      surfaces.delete(cmd.name);
      out({op: 'closed', name: cmd.name});
      break;
    case 'tabs':
      out({
        op: 'tabs',
        name: cmd.name,
        tabs: describeTabs(surface),
        errors: surface.errors,
      });
      break;
    case 'submit':
      doc.getElementById('task-input').value = cmd.text;
      doc.getElementById('send-btn').click();
      out({op: 'submitted', name: cmd.name});
      break;
    case 'closeTab': {
      const el = doc.querySelector(
        `#tab-list [data-tab-id="${cmd.tabId}"] .chat-tab-close`,
      );
      if (el) el.click();
      out({op: 'tabClosed', name: cmd.name, found: !!el});
      break;
    }
    case 'events':
      out({op: 'events', name: cmd.name, events: surface.events});
      break;
    default:
      out({op: 'error', name: cmd.name, error: 'unknown op ' + cmd.op});
  }
}

readline.createInterface({input: process.stdin}).on('line', line => {
  if (!line.trim()) return;
  handle(JSON.parse(line));
});
