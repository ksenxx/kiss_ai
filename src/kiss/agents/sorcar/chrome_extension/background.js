// KISS Sorcar Bridge — service worker.
//
// Connects to the local KISS daemon's chrome-bridge WebSocket server
// (ws://127.0.0.1:<port>; the ephemeral port and per-start token come from
// bridge-config.json, which KISS writes next to this file every time its
// server starts and which is re-read on every connection attempt and on
// every keepalive of a live connection) and executes the commands KISS
// sends: list/open/activate/close tabs, navigate, evaluate JavaScript in a
// page, screenshot.
//
// Handshake is mutual: the server sends a one-time nonce, this worker
// answers with sha256(token + ":" + nonce), and accepts commands only
// after the server proves itself with sha256(nonce + ":" + token). The
// raw token never crosses the wire, so a rogue local process occupying
// the port can neither learn it nor drive this browser.
//
// MV3 service workers are killed when idle; a ping every 25 s counts as
// activity and keeps this worker alive while the socket is open, and a
// chrome.alarms tick, armed from the first connection attempt on,
// starts the next worker after Chrome killed this one anyway.

const RECONNECT_ALARM = 'kiss-bridge-reconnect';
const PING_INTERVAL_MS = 25000;
const QUICK_RETRY_MS = 3000;
const HANDSHAKE_DEADLINE_MS = 10000; // KISS waits as long for the hello
const COMMAND_DEADLINE_MS = 30000; // when KISS sends no timeout_ms (an older KISS)
const CAPTURE_GAP_MS = 1000; // Chrome refuses a third captureVisibleTab within one second
const MAX_FRAME_BYTES = 64 * 1024 * 1024; // KISS takes no larger frame (MAX_MESSAGE_BYTES in chrome_bridge.py)
// Commands that move focus or tabs around must not interleave (two
// concurrent screenshots would both capture whichever tab lost the race).
const FOCUS_MUTATING = new Set(['open_tab', 'navigate', 'activate_tab', 'close_tab', 'screenshot']);

let ws = null;
let connecting = false;
let pingTimer = null;
let retryTimer = null;
let uidSpace = null; // names the live connection's uids (see the interaction layer)
let lastCaptureAt = 0; // when the latest screenshot was taken (Date.now())
let focusQueue = Promise.resolve();
const tabChains = new Map(); // tabId -> promise chain (one debugger attach per tab at a time)

// A JSON file of the extension's folder, as it is on disk now.
async function readJson(name) {
  const resp = await fetch(chrome.runtime.getURL(name), { cache: 'no-store' });
  return resp.json();
}

// Rejects while the file is missing (KISS has not started its bridge yet).
function getConfig() {
  return readJson('bridge-config.json');
}

// Chrome runs the copy of this script it read when it loaded the
// extension: a newer one that KISS installed since runs only once the
// extension is reloaded (or Chrome starts again). The manifest KISS
// installs names its build in "version" and is written after the worker
// it points at, so a version on disk other than the loaded one is another
// install, complete.
async function outdated() {
  return (await readJson('manifest.json')).version !== chrome.runtime.getManifest().version;
}

// KISS takes only the worker of the extension it installed, and closed
// this one's socket with 4004: Chrome loaded this worker from another
// install. Reloading the extension loads the installed one, whose worker
// connects; no command is cut off, the refused socket never carried one.
// With the loaded version on disk there is nothing to load (KISS could not
// install its files), and a manifest being replaced right now cannot be
// read: the next attempt asks again.
//
// With Developer mode off Chrome switches a reloaded unpacked extension
// off until the mode is on again. No worker can tell beforehand, and the
// user's own reload ends the same way: KISS, having refused this worker,
// tells the user what to switch on.
async function refused() {
  let reload = false;
  try {
    reload = await outdated();
  } catch (e) {
    // the next attempt reads the manifest again
  }
  if (reload) {
    chrome.runtime.reload();
  } else {
    scheduleReconnect();
  }
}

async function sha256hex(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function sendOn(sock, obj) {
  // Replies go to the socket that carried the command, never to a newer
  // replacement connection: a stale answer must not satisfy an unrelated
  // command after a server restart reuses ids.
  if (!sock || sock.readyState !== WebSocket.OPEN) return;
  let text = JSON.stringify(obj);
  // KISS closes the connection, with every other command on it, over a
  // frame larger than it takes: a result that large is answered as an
  // error. A text is at most three bytes of UTF-8 for each unit of its
  // length, so only a long one is measured.
  if (text.length * 3 > MAX_FRAME_BYTES && new Blob([text]).size > MAX_FRAME_BYTES) {
    text = JSON.stringify({ id: obj.id, ok: false, error: `the result is larger than the ${MAX_FRAME_BYTES >> 20} MB a reply may be: return less of it` });
  }
  sock.send(text);
}

function scheduleReconnect() {
  clearTimeout(retryTimer);
  retryTimer = setTimeout(connect, QUICK_RETRY_MS); // while the worker lives
}

// Gives up on the live socket and reconnects on the usual schedule, without
// waiting for the close event: a peer that answers no close frame delays
// it, and the socket is obsolete from here on, so onclose ignores it.
function giveUp(sock) {
  if (sock !== ws) return; // closed or replaced meanwhile: not this caller's to close
  ws = null;
  clearInterval(pingTimer);
  sock.close();
  scheduleReconnect();
}

// The keepalive of an authenticated socket: the ping, and a look at
// bridge-config.json. KISS's daemon publishes its bridge over a live one
// (a CLI run's), whose process keeps this socket open: only the file's
// token, new with every bridge start, tells that the browser is another
// bridge's now. No readable file: the bridge this socket has is the only
// one.
async function keepAlive(sock, cfg) {
  sendOn(sock, { type: 'ping' });
  try {
    if ((await getConfig()).token !== cfg.token) giveUp(sock);
  } catch (e) {
    // the connection stays
  }
}

// Handles one frame of the connection *link* (made by connect). The
// handshake: the server's challenge is answered with a proof of the
// token, and no command runs before the server has proved the token in
// turn (hello_ok); a rogue listener on the port knows no token.
async function onFrame(link, data) {
  let msg = null;
  try {
    msg = JSON.parse(data);
  } catch (e) {
    // not JSON: no frame of the server's
  }
  if (!msg) return;
  if (msg.type === 'challenge') {
    link.nonce = msg.nonce;
    // The version tells KISS which install Chrome runs: it takes only its own.
    const proof = await sha256hex(`${link.token}:${msg.nonce}`);
    sendOn(link.sock, { type: 'hello', proof, version: chrome.runtime.getManifest().version });
  } else if (msg.type === 'hello_ok') {
    const expected = await sha256hex(`${link.nonce}:${link.token}`);
    link.nonce = null; // a proof answers one challenge
    // On a socket given up meanwhile the verdict changes nothing:
    // handleCommand runs no command on a socket that is not open.
    if (msg.proof === expected) link.authed = true;
    else link.sock.close(4003, 'bad server proof');
  } else if (link.authed && typeof msg.id === 'number' && msg.cmd) {
    handleCommand(link.sock, msg);
  }
}

async function connect() {
  if (connecting) return; // single-flight: lifecycle events can fire together
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  connecting = true;
  // Starts the next worker after Chrome killed this one, on a live socket
  // as well as between attempts: every attempt arms it, and only a
  // replacement by another profile (close code 4002) disarms it.
  chrome.alarms.create(RECONNECT_ALARM, { periodInMinutes: 0.5 });
  try {
    // allSettled: whatever became of the cleanup, the worker connects.
    await Promise.allSettled([orphansDropped]);
    const cfg = await getConfig();
    const sock = new WebSocket(`ws://127.0.0.1:${cfg.port}`);
    ws = sock;
    // The connection's own from its first moment: the name of its uids
    // (see the interaction layer) and its keepalive, whose first tick
    // only an authenticated socket lives to see (the handshake deadline
    // below is the shorter of the two).
    uidSpace = crypto.randomUUID();
    clearInterval(pingTimer);
    pingTimer = setInterval(() => keepAlive(sock, cfg), PING_INTERVAL_MS);
    const link = { sock, token: cfg.token, nonce: null, authed: false };
    // A peer that accepts the connection and never completes the handshake
    // (a stalled bridge, an unrelated listener on a stale port) would hold
    // the one connection for good: connect() leaves an open socket alone.
    // The deadline is well inside the 30 s an idle worker lives, so the
    // retry is armed before Chrome kills this one.
    setTimeout(() => {
      if (!link.authed) giveUp(sock);
    }, HANDSHAKE_DEADLINE_MS);
    // One frame at a time, in the order they came: a command right behind
    // the server's proof (KISS may send one) waits for the check of that
    // proof, which is async, and a command ahead of it is dropped.
    // finally: a frame is handled whatever became of the one before it.
    let inbox = Promise.resolve();
    sock.onmessage = (event) => {
      inbox = inbox.finally(() => onFrame(link, event.data));
    };
    sock.onclose = (event) => {
      if (sock !== ws) return; // an obsolete socket must not clear the live one
      ws = null;
      clearInterval(pingTimer);
      if (event.code === 4002) {
        // Replaced by another Chrome profile: stay down until the user
        // wakes us (toolbar click) — and disarm the reconnect alarm, or
        // two profiles replace each other forever.
        clearTimeout(retryTimer);
        chrome.alarms.clear(RECONNECT_ALARM);
        return;
      }
      if (event.code === 4004) refused(); // KISS installed another build than this one
      else scheduleReconnect();
    };
    sock.onerror = () => {
      // onclose follows and handles reconnect.
    };
  } catch (e) {
    scheduleReconnect();
  } finally {
    connecting = false;
  }
}

// ---------------------------------------------------------------- commands

function tabInfo(t) {
  return { id: t.id, windowId: t.windowId, title: t.title, url: t.url, active: t.active, status: t.status };
}

// A command's deadline (handleCommand's limit) travels into every stage
// of an evaluate: once it has passed, the command was answered "exceeded"
// and its queue slot released, so no stage that has not started its side
// effects yet may start them (a renderer that was busy for the whole
// deadline must not run the injection once it recovers, nor the CSP
// fallback's debugger attach).
function pastDeadline(limit) {
  // `expired` is set by the timer that answered "exceeded": a timer may
  // fire a fraction of a millisecond before Date.now() reaches `at`, and
  // the answered command must never pass this check again.
  return limit.expired === true || Date.now() >= limit.at;
}

function deadlineError(limit) {
  return new Error(`command exceeded its ${limit.ms / 1000}s deadline`);
}

// The tokens of a script's directive prologue, each matched where
// lastIndex points (sticky) so the scan below never backtracks: the
// whitespace and comments between tokens, a string literal (a
// backslash escapes the next character, or a CRLF), and the operators
// that continue an expression after a string literal, before which
// automatic semicolon insertion does not end the statement (a ++ or
// -- after a line break is a restricted production: it does; a dot
// before a digit starts a number; in / instanceof followed by an
// identifier character, an escape or a non-ASCII letter are names).
const TRIVIA = /\s+|\/\/[^\n\r\u2028\u2029]*|\/\*[\s\S]*?\*\//y;
const STRING_LITERAL = /'(?:[^'\\\n\r]|\\(?:\r\n|[\s\S]))*'|"(?:[^"\\\n\r]|\\(?:\r\n|[\s\S]))*"/y;
const CONTINUES_EXPRESSION =
  /[[(`,?:=<>*\/%&|^]|\.(?!\d)|\+(?!\+)|-(?!-)|!=|(?:in|instanceof)(?![\w$\\]|[^\x00-\x7f\s])/y;
const LINE_BREAK = /[\n\r\u2028\u2029]/;

// Index of the first character after the whitespace and comments at pos.
function skipTrivia(script, pos) {
  TRIVIA.lastIndex = pos;
  while (TRIVIA.exec(script)) pos = TRIVIA.lastIndex;
  return pos;
}

// Whether the script's directive prologue (the string literal
// statements it opens with, ECMAScript 11.2.1) holds a 'use strict'
// directive. A string literal is such a statement when a semicolon,
// the end of the script, or a line break before a token that cannot
// continue the expression ends it; 'use strict'.length and
// 'use strict'\n+ x are expressions, not directives. The prologue
// ends at the first statement that is not a lone string literal.
function hasUseStrictDirective(script) {
  let pos = 0;
  for (;;) {
    pos = skipTrivia(script, pos);
    STRING_LITERAL.lastIndex = pos;
    const literal = STRING_LITERAL.exec(script);
    if (!literal) return false;
    const literalEnd = STRING_LITERAL.lastIndex;
    pos = skipTrivia(script, literalEnd);
    if (script[pos] === ';') {
      pos += 1;
    } else if (pos < script.length) {
      CONTINUES_EXPRESSION.lastIndex = pos;
      if (!LINE_BREAK.test(script.slice(literalEnd, pos)) || CONTINUES_EXPRESSION.test(script)) {
        return false;
      }
    }
    if (literal[0].slice(1, -1) === 'use strict') return true;
  }
}

// Injected by chrome.scripting.executeScript into the page's MAIN world.
// The envelope distinguishes "script threw" from "script returned
// undefined" — Chrome resolves an uncaught injected throw as a bare
// {result: null}, which is indistinguishable from a null return. The
// probe runs FIRST: cspBlocked can only be set before the user's code
// has executed, so a matching error thrown by the script itself never
// triggers a fallback that would replay its side effects. deadlineAt is
// the command's deadline (Date.now() based, the same clock in the page):
// an injection that waited on a busy renderer past it refuses the work,
// since KISS was already told the command failed. The function runs in
// the page's script scope, where a page's (or an earlier script's)
// top-level `let Date` shadows the global, so the clock is read off the
// global object itself: `this` of this sloppy plain call.
//
// A page left under a script that still waits (a link, a form, the
// history) is kept in Chrome's back/forward cache, frozen: no timer of
// the script fires again and executeScript never answers, so the command
// would wait out its whole deadline. The page's own pagehide ends the
// wait at once. (A page whose earlier listener stops that event keeps
// the wait, which then ends at the deadline.) The deadline ends the wait
// in the page as well, where KISS was answered "exceeded" by then: a
// script that never settles must leave the page it stays in nothing to
// carry, no listener and no pending wait.
async function runCode(code, deadlineAt) {
  if (this.Date.now() >= deadlineAt) {
    return { ok: false, error: 'command exceeded its deadline before the page ran it' };
  }
  try {
    // eslint-disable-next-line no-eval
    (0, eval)(''); // CSP probe: throws EvalError before any user code runs
  } catch (e) {
    return { ok: false, cspBlocked: true, error: String((e && e.message) || e) };
  }
  const gone = {};
  const late = {};
  let end;
  const ended = new this.Promise((resolve) => { end = resolve; });
  const leave = () => end(gone);
  this.addEventListener('pagehide', leave);
  const timer = this.setTimeout(() => end(late), deadlineAt - this.Date.now());
  try {
    // eslint-disable-next-line no-eval
    const value = await this.Promise.race([(0, eval)(code), ended]); // await: async scripts work
    if (value === gone) return { ok: false, error: 'the page went away (it navigated, reloaded or closed) before the script finished' };
    if (value === late) return { ok: false, expired: true };
    return { ok: true, value };
  } catch (e) {
    return { ok: false, error: String((e && (e.stack || e.message)) || e) };
  } finally {
    this.removeEventListener('pagehide', leave);
    this.clearTimeout(timer);
  }
}

async function evalViaScripting(tabId, script, limit) {
  const [injection] = await chrome.scripting.executeScript({
    target: { tabId },
    world: 'MAIN',
    func: runCode,
    args: [script, limit.at],
  });
  const r = injection && injection.result;
  if (!r) throw new Error('the page returned nothing (it may have navigated away)');
  if (r.expired) throw deadlineError(limit); // the page's word for what the deadline answered
  if (!r.ok) {
    const err = new Error(r.error);
    if (r.cspBlocked) err.cspBlocked = true;
    throw err;
  }
  return r.value === undefined ? null : r.value;
}

// Pages with a strict CSP (Gmail, most SSO apps) reject eval even for
// extension-injected code in the MAIN world; chrome.debugger is privileged
// and bypasses page CSP. Attaching shows Chrome's "debugging this browser"
// banner for the duration.
//
// chrome.debugger knows one attachment per tab, with no handle to name it
// by: a detach is for whatever is attached right now. debuggerOwners
// records which evalViaDebugger call owns the tab's attachment, so a
// cleanup (the finally below, or the deadline's detachDebugger) only
// detaches its own and never the attachment of the next command on the
// tab, which may have attached while a timed-out command's debugger
// call was still failing.
const debuggerOwners = new Map(); // tabId -> the owner token of the live attachment

function detachDebugger(tabId, owner) {
  if (debuggerOwners.get(tabId) !== owner) return Promise.resolve();
  debuggerOwners.delete(tabId);
  return chrome.debugger.detach({ tabId }).catch(() => {});
}

// Chrome keeps the attachments of a worker it ended (mid-script, say),
// with their "debugging this browser" banner, and refuses the next
// worker a second attachment on the same tab. A starting worker drops
// what it finds before it takes its first command (connect waits for
// orphansDropped), when no command of its own can be attached yet. An
// attachment that is not this extension's (an open DevTools) refuses the
// detach and stays.
async function dropOrphanedDebuggers() {
  for (const target of await chrome.debugger.getTargets()) {
    if (target.attached && target.tabId !== undefined) {
      await chrome.debugger.detach({ tabId: target.tabId }).catch(() => {});
    }
  }
}
const orphansDropped = dropOrphanedDebuggers();

async function evalViaDebugger(tabId, script, limit) {
  const target = { tabId };
  await chrome.debugger.attach(target, '1.3');
  const owner = {};
  debuggerOwners.set(tabId, owner);
  limit.debuggerOwner = owner; // lets the deadline detach this attachment, and only this one
  try {
    // The attach itself may have waited past the deadline: this attach
    // is detached by the finally below, not by the deadline's one-shot
    // detach, which ran before it existed.
    if (pastDeadline(limit)) throw deadlineError(limit);
    // The same gate runCode applies, in the page: the evaluation can
    // queue behind a busy renderer past the deadline, and the deadline's
    // detach does not cancel it. A statement before the script leaves
    // the script's completion value (what the agent gets back) as is,
    // but ends its directive prologue: a script that opens with
    // 'use strict' gets the directive restated ahead of the gate, so
    // it runs as strict as it did unguarded. The gate shares the
    // script's scope, so it names no identifier the script could
    // shadow: a top-level `let Date = 1` hoists over the whole script
    // and a bare `Date` here would be read in its temporal dead zone.
    // `this` of a classic script is the global object, strict or not.
    const gate = `if (this.Date.now() >= ${limit.at}) throw new this.Error('command exceeded its deadline before the page ran it');\n`;
    const strict = hasUseStrictDirective(script) ? "'use strict';\n" : '';
    const resp = await chrome.debugger.sendCommand(target, 'Runtime.evaluate', {
      expression: strict + gate + script,
      returnByValue: true,
      awaitPromise: true,
    });
    if (resp.exceptionDetails) {
      const ex = resp.exceptionDetails.exception;
      throw new Error((ex && (ex.description || ex.value)) || resp.exceptionDetails.text);
    }
    return resp.result && resp.result.value !== undefined ? resp.result.value : null;
  } finally {
    await detachDebugger(tabId, owner);
  }
}

// A tab Chrome unloaded to save memory, one not loaded yet since the
// session was restored, and one whose page crashed hold no document: a
// script there reads an empty page (or is refused as if the extension
// lacked the permission) and a capture shows a blank one. Such a tab is
// loaded again where it is, as it is for a user who turns to it, before
// a command reads or shows its page; past the command's deadline nothing
// is started. A tab that cannot be read is the command's own to report.
//
// One load serves every command that needs the page. While it is under
// way the tab says "loading", like any page a user is waiting for, and
// still holds no document: a command that comes then (a screenshot beside
// the evaluate that started the load) waits for that same load.
const restores = new Map(); // tabId -> the load that brings the tab's page back

async function restore(tabId) {
  await chrome.tabs.reload(tabId);
  await waitForLoad(tabId, 15000);
}

async function loaded(tabId, limit) {
  let tab;
  try {
    tab = await chrome.tabs.get(tabId);
  } catch (e) {
    return;
  }
  if (!restores.has(tabId)) {
    if (tab.status !== 'unloaded' || pastDeadline(limit)) return;
    const load = restore(tabId);
    restores.set(tabId, load);
    const forget = () => restores.delete(tabId);
    load.then(forget, forget);
  }
  await restores.get(tabId);
}

async function evaluate(args, limit) {
  const { tabId, script } = args;
  await loaded(tabId, limit);
  return evalViaScripting(tabId, script, limit).catch((e1) => {
    // Only a CSP refusal (probe failed, the script never ran) may fall
    // back to the debugger; a script's own error must not replay its
    // side effects in another world, and a refusal learned after the
    // deadline starts no debugger at all.
    if (!e1.cspBlocked) throw e1;
    if (pastDeadline(limit)) throw deadlineError(limit);
    return evalViaDebugger(tabId, script, limit).catch((e2) => {
      throw new Error(`evaluate failed (page world blocked by CSP: ${e1.message}; debugger: ${e2.message})`);
    });
  });
}

// -------------------------------------------------- interaction layer
// The agent-facing commands (snapshot/click/type_text) run this page code
// through evaluate(), so they inherit its MAIN-world + CSP-debugger cascade
// and its per-tab serialization. Uids are stored in data-kiss-uid
// attributes, which click/type resolve, and owned through the document's
// registry (document.kissUids, a WeakMap from element to uid), which they
// check: cloneNode (and an innerHTML round trip) copies attributes, not
// registry entries, so a copy of an element, or a replacement that took
// its place, never answers to its uid. A uid is stable across snapshots
// and never reused, for another element or in another document: a fresh
// one comes from the block of numbers that is the snapshot command's own
// (KISS numbers its commands 1, 2, 3, ... for as long as its bridge
// runs), and a registry is one connection's (uidSpace): the next
// connection may be another bridge's, which numbers its commands from 1
// again, so the uids of an earlier connection are void. Every task
// shares the one Chrome, so a handle from an earlier snapshot, another
// agent's or one taken before the page changed or the tab navigated,
// still names its element or fails as stale, never a control that took
// its place. The embedded sources use
// only plain string concatenation (no backticks, no ${) so they can live
// inside this file's template literals safely; the one interpolation is
// KISS_TARGET_JS, such a source itself, into the two that share it.
const KISS_SNAPSHOT_JS = `(function (id, space) {
  var MAX = 200;
  var ROLES = { button: 1, link: 1, menuitem: 1, menuitemcheckbox: 1, menuitemradio: 1, tab: 1, option: 1, checkbox: 1, switch: 1, radio: 1, combobox: 1, listbox: 1, textbox: 1, searchbox: 1, slider: 1, spinbutton: 1, treeitem: 1 };
  // What a person can press or edit whatever role and tabindex it carries
  // (an editing host is contenteditable in one of its three states that
  // make it one: "", "true" and "plaintext-only"), and what a role can
  // make interactive.
  var NATIVE = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[onclick],[contenteditable=""],[contenteditable=true i],[contenteditable=plaintext-only i]';
  var SEL = NATIVE + ',[role]';
  function walk(root, visit) {
    var nodes = root.querySelectorAll('*');
    for (var i = 0; i < nodes.length; i++) {
      visit(nodes[i]);
      if (nodes[i].shadowRoot) walk(nodes[i].shadowRoot, visit);
    }
  }
  // Candidates: semantic interactives (SEL), what the keyboard reaches (a
  // tabindex that is not negative: -1 marks a focus target of scripts, and
  // takes nothing away from a control that is one by SEL, like the buttons
  // of a toolbar that share one tab stop) plus anything styled
  // cursor:pointer — React apps routinely make bare divs clickable with no
  // role/tabindex at all (cursor inherits, so a clickable container makes
  // every descendant a candidate too; the innermost-wins pass below
  // collapses those).
  var cands = [];
  walk(document, function (el) {
    var cs = getComputedStyle(el);
    // Hidden elements are no candidates at all: a hidden text-bearing
    // descendant (a collapsed label inheriting the cursor) would
    // otherwise suppress its visible clickable ancestor below.
    if (cs.visibility === 'hidden' || cs.display === 'none' || !el.getClientRects().length) return;
    if (el.matches(SEL) || parseInt(el.getAttribute('tabindex'), 10) >= 0) {
      // A layout role rules out a bare element only: a button or a field
      // is pressed and edited all the same.
      var role = el.getAttribute('role');
      if (role && !ROLES[role] && !el.matches(NATIVE)) return;
      cands.push(el);
    } else if (cs.cursor === 'pointer') {
      cands.push(el);
    }
  });
  function textOf(el) {
    var tag = el.tagName.toLowerCase();
    var text = (el.innerText || '').replace(/\\s+/g, ' ').trim();
    if (tag === 'select') {
      // innerText lists EVERY option; what matters is the selected one.
      text = el.selectedOptions && el.selectedOptions.length ? el.selectedOptions[0].textContent.trim() : '';
    } else if (!text && (tag === 'input' || tag === 'textarea')) {
      // checkbox/radio have a meaningless value ('on'): label or nothing.
      // What a password field holds is no text of the listing: the page
      // itself shows dots for it.
      if (el.type === 'checkbox' || el.type === 'radio') text = el.getAttribute('aria-label') || el.name || '';
      else text = el.placeholder || (el.type !== 'password' && el.value) || el.name || '';
    }
    if (!text) text = el.getAttribute('aria-label') || el.getAttribute('alt') || el.getAttribute('title') || '';
    return text;
  }
  var texts = new Map();
  cands.forEach(function (el) { texts.set(el, textOf(el)); });
  // Innermost-with-text wins: a click on the inner element bubbles up
  // through every clickable ancestor (firing its handlers), but the reverse
  // is not true — so a candidate whose text names the action suppresses its
  // clickable ancestors. Textless candidates (icon glyphs) suppress nothing:
  // a labeled container beats its own <path>/<i> leaves.
  var suppressed = new Set();
  cands.forEach(function (el) {
    if (!texts.get(el)) return;
    for (var p = el.parentElement; p; p = p.parentElement) {
      if (suppressed.has(p)) break;
      suppressed.add(p);
    }
  });
  var candSet = new Set(cands);
  var ICON_TAGS = { path: 1, svg: 1, i: 1, img: 1, span: 1 };
  var els = cands.filter(function (el) {
    if (suppressed.has(el)) return false;
    // A textless icon-ish leaf under a surviving clickable is noise: the
    // ancestor is the same click target with a better (or equal) label.
    if (!texts.get(el) && ICON_TAGS[el.tagName.toLowerCase()]) {
      for (var p = el.parentElement; p; p = p.parentElement) {
        if (candSet.has(p) && !suppressed.has(p)) return false;
      }
    }
    return true;
  });
  var lines = [];
  var shown = Math.min(els.length, MAX);
  // Only the listed elements keep a uid: a handle to an element that is
  // gone from the snapshot fails instead of resolving to its successor.
  var listed = new Set(els.slice(0, shown));
  walk(document, function (el) { if (!listed.has(el)) el.removeAttribute('data-kiss-uid'); });
  var next = (id - 1) * MAX + 1; // this command's block of MAX uids
  // Another connection's registry is dropped, and its uids with it.
  var uids = document.kissUids;
  if (!uids || uids.space !== space) uids = document.kissUids = { space: space, of: new WeakMap() };
  for (var i = 0; i < shown; i++) {
    var el = els[i];
    // The uid the element owns is the registry's, not the attribute: a
    // cloned node carries the attribute along (the copy may well come
    // first, or have replaced the original), never the registry entry.
    var uid = uids.of.get(el);
    if (!uid) {
      uid = next++;
      uids.of.set(el, uid);
    }
    el.setAttribute('data-kiss-uid', String(uid));
    var tag = el.tagName.toLowerCase();
    var role = el.getAttribute('role');
    var text = texts.get(el) || '';
    if (text.length > 80) text = text.slice(0, 77) + '...';
    var line = '[' + uid + '] <' + tag + (role ? ' role=' + role : '') + (tag === 'input' && el.type ? ' type=' + el.type : '') + '>';
    if (text) line += ' "' + text + '"';
    // The href of a drawing's link (SVG) is an object that holds the
    // address in baseVal, as the page wrote it.
    var href = tag === 'a' && el.href ? String(el.href.baseVal ?? el.href) : '';
    if (href) line += ' -> ' + href.slice(0, 80);
    // :disabled is the state a control is in: its fieldset disables it too.
    if (el.disabled || el.matches(':disabled')) line += ' [disabled]';
    if (el.checked || el.getAttribute('aria-checked') === 'true') line += ' [checked]';
    if (el.getAttribute('aria-selected') === 'true') line += ' [selected]';
    lines.push(line);
  }
  var head = document.title + ' — ' + location.href + ' — ' + els.length + ' interactive elements';
  if (els.length > shown) head += ' (showing first ' + shown + ')';
  return head + '\\n' + lines.join('\\n');
})`;

// click/type resolve uids with a deep find: uids may live inside open
// shadow roots, where document.querySelector cannot reach. Of the
// elements carrying the attribute only the one owning the uid (in the
// document's registry, which cloneNode does not copy into) is the element
// the snapshot listed; without a registry of this connection (no snapshot
// ran in the document, or an earlier connection's did) nothing owns a
// uid. The element found is scrolled into view. One source, interpolated
// into both interaction sources so each stays self-contained for injection.
const KISS_TARGET_JS = `function kissTarget(uid, space) {
    function kissFind(u) {
      var uids = document.kissUids;
      if (!uids || uids.space !== space) return null;
      var sel = '[data-kiss-uid="' + u + '"]';
      function owner(scope) {
        var hits = scope.querySelectorAll(sel);
        for (var i = 0; i < hits.length; i++) if (uids.of.get(hits[i]) === u) return hits[i];
        return null;
      }
      var direct = owner(document);
      if (direct) return direct;
      var queue = [document];
      while (queue.length) {
        var nodes = queue.shift().querySelectorAll('*');
        for (var i = 0; i < nodes.length; i++) {
          if (nodes[i].shadowRoot) {
            var hit = owner(nodes[i].shadowRoot);
            if (hit) return hit;
            queue.push(nodes[i].shadowRoot);
          }
        }
      }
      return null;
    }
    var el = kissFind(uid);
    if (!el) throw new Error('element [' + uid + '] no longer exists — the page changed; take a fresh snapshot');
    // Scroll only when the element is actually out of view: an unconditional
    // scrollIntoView scrolls the page, and open menus/poppers close on scroll.
    var vr = el.getBoundingClientRect();
    if (vr.top < 0 || vr.bottom > innerHeight || vr.left < 0 || vr.right > innerWidth) {
      el.scrollIntoView({ block: 'center', inline: 'center' });
    }
    return el;
  }`;

const KISS_CLICK_JS = `(function (uid, space) {
  ${KISS_TARGET_JS}
  var el = kissTarget(uid, space);
  // A disabled control takes no click, a user's or a script's: the page
  // would hear the pointer events below and no click.
  if (el.matches(':disabled')) throw new Error('element [' + uid + '] is disabled: nothing was clicked');
  if (el.focus) el.focus();
  // composed, as a user's pointer events are: a listener outside the
  // element's shadow root (on its host, on the document) hears them.
  var opts = { bubbles: true, cancelable: true, composed: true, view: window };
  el.dispatchEvent(new PointerEvent('pointerdown', opts));
  el.dispatchEvent(new MouseEvent('mousedown', opts));
  el.dispatchEvent(new PointerEvent('pointerup', opts));
  el.dispatchEvent(new MouseEvent('mouseup', opts));
  // SVG elements (an <svg role=button>) have no click(): a dispatched
  // click reaches their handlers all the same.
  if (typeof el.click === 'function') el.click();
  else el.dispatchEvent(new MouseEvent('click', opts));
  // String(): the value of a <progress>, a <meter> or an <li> is a number.
  // What a password field holds describes nothing: the page shows dots.
  var desc = String(el.innerText || el.getAttribute('aria-label') || (el.type !== 'password' && el.value) || el.tagName).replace(/\\s+/g, ' ').trim().slice(0, 60);
  var said = 'clicked [' + uid + '] <' + el.tagName.toLowerCase() + '> "' + desc + '"';
  // A link into another tab (_blank, or a name no tab or frame has yet)
  // opens one for a user's click and none for a scripted one (Chrome's
  // pop-up blocker): the reply says so and gives the link's address for
  // open_tab, resolved as the page resolves it (a drawing's link holds
  // its own as written, in href.baseVal).
  var link = el.closest('a:any-link[target]');
  var target = link ? link.getAttribute('target') : '';
  if (!/^(_self|_parent|_top|)$/i.test(target)) {
    var to = document.createElement('a');
    to.href = link.href.baseVal ?? link.href;
    said += ' — the link targets ' + target + ', and Chrome blocks a new tab for a scripted click: if none opened, open_tab ' + to.href;
  }
  return said;
})`;

// Typing replaces what a text field holds and tells the page as a user's
// typing does: an input event that leaves the field's shadow root
// (composed), and for an input or a text area the change event, which
// does not. An input or a text area is typed into through its value,
// also inside an editor (where it is editable content too); the value of
// a checkbox, a radio button, a button or a file input is not what a user
// types, so these are refused like everything else that is no text field,
// before anything is touched, and so is a field a user cannot type into:
// a disabled one (by itself or by its fieldset) and a read-only one. A
// field whose type drops the text (letters in a number field) is refused
// too, holding what it held; one that keeps it in its own form (a
// one-line field without the line breaks) has taken it. The inputs
// whose value a user picks rather than types (PICKED) turn what is no
// value of theirs into a default of their own, a slider into its middle
// and a colour well into black: they have taken a text only when they
// hold it as given, less the spaces around it and the case of its
// letters (Chrome writes a colour in lower case).
const KISS_TYPE_JS = `(function (uid, text, space) {
  ${KISS_TARGET_JS}
  var NO_TEXT = { checkbox: 1, radio: 1, button: 1, submit: 1, reset: 1, image: 1, file: 1 };
  var PICKED = { color: 1, range: 1, date: 1, time: 1, 'datetime-local': 1, month: 1, week: 1 };
  var el = kissTarget(uid, space);
  var tag = el.tagName.toLowerCase();
  var input = el.tagName === 'INPUT';
  var kind = '<' + tag + (input ? ' type=' + el.type : '') + '>';
  var field = input ? !NO_TEXT[el.type] : el.tagName === 'TEXTAREA';
  if (!field && (input || !el.isContentEditable)) {
    throw new Error('element [' + uid + '] is a ' + kind + ', not a text field');
  }
  // The readonly of a slider or a colour well is no lock: the attribute
  // does not apply to them, and a user moves them all the same.
  var readOnly = el.readOnly && el.type !== 'range' && el.type !== 'color';
  var locked = el.matches(':disabled') ? 'disabled' : readOnly ? 'read-only' : '';
  if (locked) throw new Error('element [' + uid + '] ' + kind + ' is ' + locked + ' and was left unchanged');
  el.focus();
  if (field) {
    var proto = input ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
    var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    var held = el.value;
    var picked = PICKED[el.type];
    var given = picked ? text.trim() : text;
    setter.call(el, given);
    if (picked ? el.value.toLowerCase() !== given.toLowerCase() : text && !el.value) {
      setter.call(el, held);
      throw new Error('element [' + uid + '] ' + kind + ' did not take the text and was left unchanged');
    }
    el.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  } else {
    // An editor, replaced as a person does it, select all and type (or
    // delete): an editing command fires the input events an editor keeps
    // its model by, for an empty text too (inserting an empty text
    // inserts nothing and tells the page nothing); emptying the DOM
    // under it fires none.
    var all = document.createRange();
    all.selectNodeContents(el);
    var selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(all);
    document.execCommand(text ? 'insertText' : 'delete', false, text);
  }
  return 'typed ' + text.length + ' chars into [' + uid + '] <' + tag + '>';
})`;

// Runs one of the interaction functions in the page through evaluate()'s
// CSP cascade. Args are interpolated, never the page's data: uid is checked
// to be an integer and text is JSON-encoded, so a page cannot smuggle code
// into these strings.
function runInteraction(args, src, extra, limit) {
  const uid = args.uid;
  if (!Number.isInteger(uid)) throw new Error(`command needs an integer 'uid' from a snapshot`);
  const script = `(${src})(${uid}${extra || ''}, ${JSON.stringify(uidSpace)})`;
  return evaluate({ tabId: args.tabId, script }, limit);
}

function waitForLoad(tabId, timeoutMs) {
  // Poll instead of listening for tabs.onUpdated: a fast (cached) navigation
  // can complete before a listener attached after tabs.update would see it.
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve) => {
    async function poll() {
      try {
        const tab = await chrome.tabs.get(tabId);
        if (tab.status === 'complete') return resolve();
      } catch (e) {
        return resolve(); // tab closed mid-navigation
      }
      if (Date.now() >= deadline) return resolve();
      setTimeout(poll, 200);
    }
    poll();
  });
}

// chrome.tabs opens what is no URL on its own ("example.com", "/inbox")
// as a file of this extension, a page that does not exist: refused.
function completeUrl(url) {
  if (!URL.canParse(url)) throw new Error(`'${url}' is not a complete URL: write it in full, as in https://example.com`);
  return url;
}

const COMMANDS = {
  async list_tabs() {
    const tabs = await chrome.tabs.query({});
    return tabs.filter((t) => t.id !== undefined).map(tabInfo);
  },

  async open_tab(args, limit) {
    const url = completeUrl(args.url || 'about:blank');
    const active = args.active !== false;
    let tab;
    try {
      tab = await chrome.tabs.create({ url, active });
    } catch (e) {
      // Chrome can run with no window (on a Mac, once the last one is
      // closed) and then has no place for a tab: it gets a window of its
      // own. Any other refusal is Chrome's answer.
      const windows = await chrome.windows.getAll({ windowTypes: ['normal'] });
      if (windows.length || pastDeadline(limit)) throw e;
      tab = (await chrome.windows.create({ url, focused: active })).tabs[0];
    }
    if (args.url) await waitForLoad(tab.id, 15000);
    return tabInfo(await chrome.tabs.get(tab.id));
  },

  async navigate(args) {
    await chrome.tabs.update(args.tabId, { url: completeUrl(args.url) });
    await waitForLoad(args.tabId, 15000);
    return tabInfo(await chrome.tabs.get(args.tabId));
  },

  // The deadline is rechecked before every focus mutation that follows an
  // await: a command answered "exceeded" has released the focus queue, and
  // resuming into the next command's tab would steal its focus.
  async activate_tab(args, limit) {
    await loaded(args.tabId, limit);
    if (pastDeadline(limit)) throw deadlineError(limit);
    const tab = await chrome.tabs.update(args.tabId, { active: true });
    if (pastDeadline(limit)) throw deadlineError(limit);
    await chrome.windows.update(tab.windowId, { focused: true });
    return tabInfo(tab);
  },

  async close_tab(args) {
    await chrome.tabs.remove(args.tabId);
    return { closed: args.tabId };
  },

  evaluate,

  async snapshot(args, limit, id) {
    const script = `(${KISS_SNAPSHOT_JS})(${id}, ${JSON.stringify(uidSpace)})`;
    return evaluate({ tabId: args.tabId, script }, limit);
  },

  click(args, limit) {
    return runInteraction(args, KISS_CLICK_JS, '', limit);
  },

  type_text(args, limit) {
    // An empty text is a value: it clears the field.
    if (typeof args.text !== 'string') throw new Error("type_text needs a 'text' string");
    return runInteraction(args, KISS_TYPE_JS, `, ${JSON.stringify(args.text)}`, limit);
  },

  async screenshot(args, limit) {
    await loaded(args.tabId, limit);
    const tab = await chrome.tabs.get(args.tabId);
    if (pastDeadline(limit)) throw deadlineError(limit);
    await chrome.windows.update(tab.windowId, { focused: true });
    if (pastDeadline(limit)) throw deadlineError(limit);
    await chrome.tabs.update(args.tabId, { active: true });
    if (pastDeadline(limit)) throw deadlineError(limit);
    // Let the tab paint, and keep the captures inside Chrome's quota: a
    // third one within a second fails, whichever task asked for it.
    const pause = Math.max(250, lastCaptureAt + CAPTURE_GAP_MS - Date.now());
    await new Promise((r) => setTimeout(r, pause));
    // Answered "exceeded" meanwhile: the window's active tab may be the
    // next command's by now, and a capture spends Chrome's capture quota.
    if (pastDeadline(limit)) throw deadlineError(limit);
    lastCaptureAt = Date.now();
    return chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' }); // data URL
  },
};

// A command gets a hard deadline: a never-settling page promise (or a wedged
// debugger attach) must fail the command and free the queue, not pin the
// debugger to the tab forever. The deadline is the caller's: KISS sends how
// long it still waits for the reply (timeout_ms, a little less than its own
// timeout), counted from the command's arrival, so the "exceeded" reply
// reaches KISS while it listens instead of KISS timing out and dropping the
// connection with every other command in flight.
function withDeadline(promise, onTimeout, limit) {
  let timer = null;
  return Promise.race([
    // Cancel the timer when the work settles: a completed command's
    // leftover timeout must not fire onTimeout into a LATER command's
    // session (e.g. detach its debugger mid-evaluate).
    promise.finally(() => clearTimeout(timer)),
    new Promise((_, reject) => {
      timer = setTimeout(() => {
        limit.expired = true;
        if (onTimeout) onTimeout();
        reject(deadlineError(limit));
      }, Math.max(limit.at - Date.now(), 0));
    }),
  ]);
}

// The deadline is applied INSIDE the queue slot: a wedged command fails
// at its deadline and releases the FIFO, instead of stalling every later
// command behind it while its own function never even starts.
function enqueue(key, fn, onTimeout, limit) {
  const prev = key === 'focus' ? focusQueue : tabChains.get(key) || Promise.resolve();
  const next = prev.then(() => withDeadline(fn(), onTimeout, limit));
  const kept = next.catch(() => {});
  if (key === 'focus') {
    focusQueue = kept;
  } else {
    tabChains.set(key, kept);
    kept.finally(() => {
      if (tabChains.get(key) === kept) tabChains.delete(key); // no per-tab leak
    });
  }
  return next;
}

async function handleCommand(sock, msg) {
  // Object.hasOwn: '__proto__' and friends must not resolve to Object members.
  if (!Object.hasOwn(COMMANDS, msg.cmd)) {
    sendOn(sock, { id: msg.id, ok: false, error: `unknown command '${msg.cmd}'` });
    return;
  }
  const fn = COMMANDS[msg.cmd];
  try {
    const args = msg.args || {};
    const budget = Number.isFinite(msg.timeout_ms) ? msg.timeout_ms : COMMAND_DEADLINE_MS;
    const limit = { at: Date.now() + budget, ms: budget };
    // A command whose socket closed while it waited in a queue (KISS timed
    // it out and dropped the connection), or whose deadline passed there,
    // must not run: its effect (a navigation, a click) would land after the
    // caller gave up on it, and nobody is listening for its reply. The
    // deadline check is pastDeadline's: a command the outer deadline below
    // already answered "exceeded" (its timer may fire a tick before
    // Date.now() reaches the deadline) must not run once its slot frees.
    const run = () => {
      if (sock.readyState !== WebSocket.OPEN) {
        throw new Error('connection closed before the command ran');
      }
      if (pastDeadline(limit)) {
        throw new Error(`command waited out its ${budget / 1000}s deadline in the queue`);
      }
      return fn(args, limit, msg.id); // the evaluate family rechecks the deadline at each stage
    };
    // Focus-mutating commands serialize globally; evaluate serializes per
    // tab (one debugger attach per tab at a time); read-only commands run
    // free but still under the deadline.
    let work;
    if (FOCUS_MUTATING.has(msg.cmd)) {
      work = enqueue('focus', run, null, limit);
    } else if (msg.cmd === 'evaluate' || msg.cmd === 'snapshot' || msg.cmd === 'click' || msg.cmd === 'type_text') {
      // At the deadline, a debugger this command attached (a wedged
      // Runtime.evaluate) is detached; one attached by another command
      // on the tab is not its to detach.
      work = enqueue(args.tabId || 0, run, () => {
        detachDebugger(args.tabId, limit.debuggerOwner);
      }, limit);
    } else {
      work = Promise.resolve().then(run);
    }
    // Raced here as well as inside the queue slot: a command still waiting
    // in a queue at its deadline is answered then, not once its slot starts.
    const result = await withDeadline(work, null, limit);
    sendOn(sock, { id: msg.id, ok: true, result });
  } catch (e) {
    sendOn(sock, { id: msg.id, ok: false, error: String((e && e.message) || e) });
  }
}

// ---------------------------------------------------------------- lifecycle

chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(connect);
chrome.action.onClicked.addListener(connect); // clicking the toolbar icon wakes the bridge
// A tick that finds the alarm gone was on its way when a replacement by
// another profile (close code 4002) disarmed it: this profile stays down.
chrome.alarms.onAlarm.addListener(async () => {
  if (await chrome.alarms.get(RECONNECT_ALARM)) connect();
});

connect();
