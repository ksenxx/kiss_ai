// KISS Sorcar Bridge — service worker.
//
// Connects to the local KISS daemon's chrome-bridge WebSocket server
// (ws://127.0.0.1:<port>; the ephemeral port and per-start token come from
// bridge-config.json, which KISS writes next to this file every time its
// server starts and which is re-read on every connection attempt) and
// executes the commands KISS sends: list/open/activate/close tabs, navigate,
// evaluate JavaScript in a page, screenshot.
//
// Handshake is mutual: the server sends a one-time nonce, this worker
// answers with sha256(token + ":" + nonce), and accepts commands only
// after the server proves itself with sha256(nonce + ":" + token). The
// raw token never crosses the wire, so a rogue local process occupying
// the port can neither learn it nor drive this browser.
//
// MV3 service workers are killed when idle; a ping every 25 s counts as
// activity and keeps this worker alive while the socket is open, and a
// chrome.alarms tick reconnects after Chrome killed it anyway.

const RECONNECT_ALARM = 'kiss-bridge-reconnect';
const PING_INTERVAL_MS = 25000;
const QUICK_RETRY_MS = 3000;
const COMMAND_DEADLINE_MS = 30000; // when KISS sends no timeout_ms (an older KISS)
// Commands that move focus or tabs around must not interleave (two
// concurrent screenshots would both capture whichever tab lost the race).
const FOCUS_MUTATING = new Set(['open_tab', 'navigate', 'activate_tab', 'close_tab', 'screenshot']);

let ws = null;
let connecting = false;
let pingTimer = null;
let retryTimer = null;
let focusQueue = Promise.resolve();
const tabChains = new Map(); // tabId -> promise chain (one debugger attach per tab at a time)

async function getConfig() {
  const resp = await fetch(chrome.runtime.getURL('bridge-config.json'), { cache: 'no-store' });
  if (!resp.ok) throw new Error('bridge-config.json missing (KISS has not started its bridge yet)');
  return resp.json();
}

async function sha256hex(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function sendOn(sock, obj) {
  // Replies go to the socket that carried the command, never to a newer
  // replacement connection: a stale answer must not satisfy an unrelated
  // command after a server restart reuses ids.
  if (sock && sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify(obj));
}

function scheduleReconnect() {
  clearTimeout(retryTimer);
  retryTimer = setTimeout(connect, QUICK_RETRY_MS); // while the worker lives
  chrome.alarms.create(RECONNECT_ALARM, { periodInMinutes: 0.5 }); // wakes it after Chrome kills it
}

async function connect() {
  if (connecting) return; // single-flight: lifecycle events can fire together
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  connecting = true;
  try {
    const cfg = await getConfig();
    const sock = new WebSocket(`ws://127.0.0.1:${cfg.port}`);
    ws = sock;
    let pendingNonce = null; // socket-scoped handshake state
    let authed = false; // commands are ignored until the server proves itself
    // Commands that arrive while the server's proof is still being checked
    // (the server sends hello_ok and may send a command right behind it;
    // the check is async): held here, run once the proof passes, dropped
    // with the socket when it fails.
    const preAuth = [];
    let proofPending = false;
    sock.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (e) {
        return;
      }
      if (msg.type === 'challenge') {
        pendingNonce = msg.nonce;
        sha256hex(`${cfg.token}:${msg.nonce}`).then((proof) => {
          sendOn(sock, { type: 'hello', proof });
        });
        return;
      }
      if (msg.type === 'hello_ok') {
        const nonce = pendingNonce;
        pendingNonce = null;
        proofPending = true;
        sha256hex(`${nonce}:${cfg.token}`).then((expected) => {
          proofPending = false;
          if (msg.proof !== expected) {
            sock.close(4003, 'bad server proof'); // rogue listener: knows no token
            return;
          }
          authed = true;
          chrome.alarms.clear(RECONNECT_ALARM); // live now: the socket keeps us alive
          clearInterval(pingTimer);
          pingTimer = setInterval(() => sendOn(sock, { type: 'ping' }), PING_INTERVAL_MS);
          for (const held of preAuth.splice(0)) handleCommand(sock, held);
        });
        return;
      }
      if (msg.type === 'ping') return;
      if (typeof msg.id !== 'number' || !msg.cmd) return;
      if (authed) handleCommand(sock, msg);
      else if (proofPending) preAuth.push(msg);
    };
    sock.onclose = (event) => {
      if (sock !== ws) return; // an obsolete socket must not clear the live one
      ws = null;
      clearInterval(pingTimer);
      if (event.code === 4002) {
        // Replaced by another Chrome profile: stay down until the user
        // wakes us (toolbar click) — and disarm any reconnect alarm an
        // earlier disconnect armed, or two profiles replace each other
        // forever.
        clearTimeout(retryTimer);
        chrome.alarms.clear(RECONNECT_ALARM);
        return;
      }
      scheduleReconnect();
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

// Injected by chrome.scripting.executeScript into the page's MAIN world.
// The envelope distinguishes "script threw" from "script returned
// undefined" — Chrome resolves an uncaught injected throw as a bare
// {result: null}, which is indistinguishable from a null return. The
// probe runs FIRST: cspBlocked can only be set before the user's code
// has executed, so a matching error thrown by the script itself never
// triggers a fallback that would replay its side effects.
async function runCode(code) {
  try {
    // eslint-disable-next-line no-eval
    (0, eval)(''); // CSP probe: throws EvalError before any user code runs
  } catch (e) {
    return { ok: false, cspBlocked: true, error: String((e && e.message) || e) };
  }
  try {
    // eslint-disable-next-line no-eval
    return { ok: true, value: await (0, eval)(code) }; // await: async scripts work
  } catch (e) {
    return { ok: false, error: String((e && (e.stack || e.message)) || e) };
  }
}

async function evalViaScripting(tabId, script) {
  const [injection] = await chrome.scripting.executeScript({
    target: { tabId },
    world: 'MAIN',
    func: runCode,
    args: [script],
  });
  const r = injection && injection.result;
  if (!r) throw new Error('the page returned nothing (it may have navigated away)');
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
async function evalViaDebugger(tabId, script) {
  const target = { tabId };
  await chrome.debugger.attach(target, '1.3');
  try {
    const resp = await chrome.debugger.sendCommand(target, 'Runtime.evaluate', {
      expression: script,
      returnByValue: true,
      awaitPromise: true,
    });
    if (resp.exceptionDetails) {
      const ex = resp.exceptionDetails.exception;
      throw new Error((ex && (ex.description || ex.value)) || resp.exceptionDetails.text || 'page exception');
    }
    return resp.result && resp.result.value !== undefined ? resp.result.value : null;
  } finally {
    await chrome.debugger.detach(target).catch(() => {});
  }
}

function evaluate(args) {
  const { tabId, script } = args;
  return evalViaScripting(tabId, script).catch((e1) => {
    // Only a CSP refusal (probe failed, the script never ran) may fall
    // back to the debugger; a script's own error must not replay its
    // side effects in another world.
    if (!e1.cspBlocked) throw e1;
    return evalViaDebugger(tabId, script).catch((e2) => {
      throw new Error(`evaluate failed (page world blocked by CSP: ${e1.message}; debugger: ${e2.message})`);
    });
  });
}

// -------------------------------------------------- interaction layer
// The agent-facing commands (snapshot/click/type_text) run this page code
// through evaluate(), so they inherit its MAIN-world + CSP-debugger cascade
// and its per-tab serialization. Uids are reassigned by every snapshot and
// stored in data-kiss-uid attributes; click/type resolve them there. The
// embedded sources use only plain string concatenation (no backticks, no
// ${) so they can live inside this file's template literals safely.
const KISS_SNAPSHOT_JS = `(function () {
  var MAX = 200;
  var ROLES = { button: 1, link: 1, menuitem: 1, menuitemcheckbox: 1, menuitemradio: 1, tab: 1, option: 1, checkbox: 1, switch: 1, radio: 1, combobox: 1, listbox: 1, textbox: 1, searchbox: 1, slider: 1, spinbutton: 1, treeitem: 1 };
  var SEL = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role],[onclick],[tabindex],[contenteditable=true]';
  function walk(root, visit) {
    var nodes = root.querySelectorAll('*');
    for (var i = 0; i < nodes.length; i++) {
      visit(nodes[i]);
      if (nodes[i].shadowRoot) walk(nodes[i].shadowRoot, visit);
    }
  }
  walk(document, function (el) { el.removeAttribute('data-kiss-uid'); });
  // Candidates: semantic interactives (SEL) plus anything styled
  // cursor:pointer — React apps routinely make bare divs clickable with no
  // role/tabindex at all (cursor inherits, so a clickable container makes
  // every descendant a candidate too; the innermost-wins pass below
  // collapses those).
  var cands = [];
  var styles = new Map();
  walk(document, function (el) {
    var cs = getComputedStyle(el);
    styles.set(el, cs);
    if (el.matches(SEL)) {
      var role = el.getAttribute('role');
      if (role && !ROLES[role]) return;
      var ti = el.getAttribute('tabindex');
      if (ti !== null && !role && parseInt(ti, 10) < 0) return;
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
      if (el.type === 'checkbox' || el.type === 'radio') text = el.getAttribute('aria-label') || el.name || '';
      else text = el.placeholder || el.value || el.name || '';
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
  var innermost = cands.filter(function (el) {
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
  var els = innermost.filter(function (el) {
    if (!el.getClientRects().length) return false;
    var cs = styles.get(el);
    return cs.visibility !== 'hidden' && cs.display !== 'none';
  });
  var lines = [];
  var shown = Math.min(els.length, MAX);
  for (var i = 0; i < shown; i++) {
    var el = els[i];
    var uid = i + 1;
    el.setAttribute('data-kiss-uid', String(uid));
    var tag = el.tagName.toLowerCase();
    var role = el.getAttribute('role');
    var text = texts.get(el) || '';
    if (text.length > 80) text = text.slice(0, 77) + '...';
    var line = '[' + uid + '] <' + tag + (role ? ' role=' + role : '') + (tag === 'input' && el.type ? ' type=' + el.type : '') + '>';
    if (text) line += ' "' + text + '"';
    if (tag === 'a' && el.href) line += ' -> ' + String(el.href).slice(0, 80);
    if (el.disabled) line += ' [disabled]';
    if (el.checked || el.getAttribute('aria-checked') === 'true') line += ' [checked]';
    if (el.getAttribute('aria-selected') === 'true') line += ' [selected]';
    lines.push(line);
  }
  var head = document.title + ' — ' + location.href + ' — ' + els.length + ' interactive elements';
  if (els.length > shown) head += ' (showing first ' + shown + ')';
  return head + '\\n' + lines.join('\\n');
})()`;

// click/type resolve uids with a deep find: uids may live inside open
// shadow roots, where document.querySelector cannot reach. The helper is
// inlined into each source so every constant stays self-contained for
// injection (and for the tests that extract these constants verbatim).
const KISS_CLICK_JS = `(function (uid) {
  function kissFind(u) {
    var sel = '[data-kiss-uid="' + u + '"]';
    var direct = document.querySelector(sel);
    if (direct) return direct;
    var queue = [document];
    while (queue.length) {
      var nodes = queue.shift().querySelectorAll('*');
      for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].shadowRoot) {
          var hit = nodes[i].shadowRoot.querySelector(sel);
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
  if (el.focus) el.focus();
  var opts = { bubbles: true, cancelable: true, view: window };
  el.dispatchEvent(new PointerEvent('pointerdown', opts));
  el.dispatchEvent(new MouseEvent('mousedown', opts));
  el.dispatchEvent(new PointerEvent('pointerup', opts));
  el.dispatchEvent(new MouseEvent('mouseup', opts));
  el.click();
  var desc = (el.innerText || el.getAttribute('aria-label') || el.value || el.tagName).replace(/\\s+/g, ' ').trim().slice(0, 60);
  return 'clicked [' + uid + '] <' + el.tagName.toLowerCase() + '> "' + desc + '"';
})`;

const KISS_TYPE_JS = `(function (uid, text) {
  function kissFind(u) {
    var sel = '[data-kiss-uid="' + u + '"]';
    var direct = document.querySelector(sel);
    if (direct) return direct;
    var queue = [document];
    while (queue.length) {
      var nodes = queue.shift().querySelectorAll('*');
      for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].shadowRoot) {
          var hit = nodes[i].shadowRoot.querySelector(sel);
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
  el.focus();
  if (el.isContentEditable) {
    el.textContent = '';
    document.execCommand('insertText', false, text);
  } else if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
    var proto = el.tagName === 'INPUT' ? HTMLInputElement.prototype : HTMLTextAreaElement.prototype;
    var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(el, text);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  } else {
    throw new Error('element [' + uid + '] is a <' + el.tagName.toLowerCase() + '>, not a text field');
  }
  return 'typed ' + text.length + ' chars into [' + uid + '] <' + el.tagName.toLowerCase() + '>';
})`;

// Runs one of the interaction functions in the page through evaluate()'s
// CSP cascade. Args are interpolated, never the page's data: uid is checked
// to be an integer and text is JSON-encoded, so a page cannot smuggle code
// into these strings.
function runInteraction(args, src, extra) {
  const uid = args.uid;
  if (!Number.isInteger(uid)) throw new Error(`command needs an integer 'uid' from a snapshot`);
  return evaluate({ tabId: args.tabId, script: `(${src})(${uid}${extra || ''})` });
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

const COMMANDS = {
  async list_tabs() {
    const tabs = await chrome.tabs.query({});
    return tabs.filter((t) => t.id !== undefined).map(tabInfo);
  },

  async open_tab(args) {
    const tab = await chrome.tabs.create({ url: args.url || 'about:blank', active: args.active !== false });
    if (args.url) await waitForLoad(tab.id, 15000);
    return tabInfo(await chrome.tabs.get(tab.id));
  },

  async navigate(args) {
    await chrome.tabs.update(args.tabId, { url: args.url });
    await waitForLoad(args.tabId, 15000);
    return tabInfo(await chrome.tabs.get(args.tabId));
  },

  async activate_tab(args) {
    const tab = await chrome.tabs.update(args.tabId, { active: true });
    await chrome.windows.update(tab.windowId, { focused: true });
    return tabInfo(tab);
  },

  async close_tab(args) {
    await chrome.tabs.remove(args.tabId);
    return { closed: args.tabId };
  },

  evaluate,

  async snapshot(args) {
    return evaluate({ tabId: args.tabId, script: KISS_SNAPSHOT_JS });
  },

  click(args) {
    return runInteraction(args, KISS_CLICK_JS);
  },

  type_text(args) {
    if (typeof args.text !== 'string' || !args.text) throw new Error("type_text needs a non-empty 'text'");
    return runInteraction(args, KISS_TYPE_JS, `, ${JSON.stringify(args.text)}`);
  },

  async screenshot(args) {
    const tab = await chrome.tabs.get(args.tabId);
    await chrome.windows.update(tab.windowId, { focused: true });
    await chrome.tabs.update(args.tabId, { active: true });
    await new Promise((r) => setTimeout(r, 250)); // let the tab paint
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
        if (onTimeout) onTimeout();
        reject(new Error(`command exceeded its ${limit.ms / 1000}s deadline`));
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
    // caller gave up on it, and nobody is listening for its reply.
    const run = () => {
      if (sock.readyState !== WebSocket.OPEN) {
        throw new Error('connection closed before the command ran');
      }
      if (Date.now() >= limit.at) {
        throw new Error(`command waited out its ${budget / 1000}s deadline in the queue`);
      }
      return fn(args);
    };
    // Focus-mutating commands serialize globally; evaluate serializes per
    // tab (one debugger attach per tab at a time); read-only commands run
    // free but still under the deadline.
    let work;
    if (FOCUS_MUTATING.has(msg.cmd)) {
      work = enqueue('focus', run, null, limit);
    } else if (msg.cmd === 'evaluate' || msg.cmd === 'snapshot' || msg.cmd === 'click' || msg.cmd === 'type_text') {
      work = enqueue(args.tabId || 0, run, () => {
        chrome.debugger.detach({ tabId: args.tabId }).catch(() => {});
      }, limit);
    } else {
      work = Promise.resolve().then(run);
    }
    // Raced here as well as inside the queue slot: a command still waiting
    // in a queue at its deadline is answered then, not once its slot starts.
    const result = await withDeadline(work, null, limit);
    sendOn(sock, { id: msg.id, ok: true, result: result === undefined ? null : result });
  } catch (e) {
    sendOn(sock, { id: msg.id, ok: false, error: String((e && e.message) || e) });
  }
}

// ---------------------------------------------------------------- lifecycle

chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onInstalled.addListener(connect);
chrome.action.onClicked.addListener(connect); // clicking the toolbar icon wakes the bridge
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === RECONNECT_ALARM) connect();
});

connect();
