// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Shows a transcript to two chat pages, whole to one and in pieces to
// the other, and fails unless both end up showing the same:
//
//   node test/lazy_replay_diff.js <cases.json>
//
// tests/webapp/test_lazy_chat_loading_jsdom.py writes the cases with
// the real server and runs this script.  A case is one task: the
// daemon's `task_events` frame for it without its events (`frame`),
// the events of its whole replay (`whole`), of the first piece of its
// lazy replay (`first`), whether it runs (`running`), and the daemon's
// answer to every `getTaskEvents` a page can post for it (`pieces`, by
// the handle of the panel that asks: the request without its task, tab
// and token).
//
// The page sent the first piece opens its unloaded panels one by one,
// as a reader would, and is answered each request; the page sent the
// whole transcript opens the same panels.  A panel the page unfolded
// by itself (a picture landed in it) has asked already and is only
// answered.  Before the first click and after each piece both pages
// must show the same status row and the same transcript: every
// element, class, attribute and text of #output, the "N events" of a
// folded Trajectory panel among them.
// Left out are the insides of the panels still to be loaded and the
// time a panel of a running task has run so far, which each page
// reads off its own clock.  With every panel open, both pages must
// also have asked the daemon for the same pictures.

'use strict';

const assert = require('assert');
const fs = require('fs');
const {
  makePage,
  send,
  click,
  readyTabId,
  statusMetrics,
} = require('./simplify2_harness');

// What a fan-out panel remembers beside what it shows: its number
// among the transcript's fan-outs and when its call ran decide which
// sub-agents are its own.
const FAN_OUT_FIELDS = [
  '_rpCallIndex',
  '_rpCallTs',
  '_rpDone',
  '_rpDoneTs',
  '_rpExpectedCount',
  '_rpOpenEnded',
];

/** A page shown *events* as the transcript of the task of *kase*. */
function show(kase, events) {
  const page = makePage();
  send(page.win, {type: 'daemonStatus', connected: true});
  page.tabId = readyTabId(page.posted);
  page.O = page.win.document.getElementById('output');
  send(page.win, {type: 'status', running: kase.running, tabId: page.tabId});
  send(page.win, Object.assign({}, kase.frame, {tabId: page.tabId, events}));
  return page;
}

function header(panel) {
  return panel.querySelector(':scope > .collapse-header');
}

/** The part of *panel* that holds the panels folded into it. */
function body(panel) {
  return panel.querySelector(':scope > .summary-sub, :scope > .trajectory-sub');
}

/** Where *el* is under *root*: the child positions that lead to it. */
function pathTo(el, root) {
  const path = [];
  for (let node = el; node !== root; node = node.parentElement)
    path.unshift(Array.from(node.parentElement.children).indexOf(node));
  return path;
}

/** The element at *path* under *root*, if there is one. */
function at(root, path) {
  return path.reduce((node, index) => node && node.children[index], root);
}

/**
 * One line for element *el* of *page*: its tag, classes and attributes
 * (the page's own tab id written TAB), and a fan-out panel's memory.
 * The marks of a panel still to be loaded (its class, the piece it
 * asks for) are not written: it is the same panel on both pages.
 */
function describe(el, page) {
  const waits = el.classList.contains('lazy-unloaded');
  const classes = Array.from(el.classList)
    .filter(name => name !== 'lazy-unloaded')
    .sort();
  const attributes = Array.from(el.attributes)
    .filter(a => a.name !== 'class' && !(waits && a.name === 'data-lazy-seg'))
    .map(a => a.name + '=' + JSON.stringify(a.value.split(page.tabId).join('TAB')))
    .sort();
  const line = [[el.tagName.toLowerCase(), ...classes].join('.'), ...attributes];
  if (el.classList.contains('tc-run-parallel'))
    line.push(...FAN_OUT_FIELDS.map(field => field + '=' + el[field]));
  return line.join(' ');
}

/** Whether *el* is the time a panel that still runs has run so far. */
function ticks(el) {
  return (
    el.classList.contains('panel-elapsed') &&
    el.parentElement.parentElement.classList.contains('panel-active')
  );
}

/**
 * Write what *node* of *page* shows to *lines*, one line per element
 * and per text, indented by *depth*.  The insides of the panels in
 * *away* are left out, and so is the note of a panel that has asked
 * for its piece (it waits on one page only).
 */
function write(node, page, away, depth, lines) {
  const indent = '  '.repeat(depth);
  if (node.nodeType === 3) {
    if (node.textContent.trim())
      lines.push(indent + JSON.stringify(node.textContent));
    return;
  }
  if (node.nodeType !== 1 || node.classList.contains('lazy-note')) return;
  lines.push(indent + describe(node, page));
  if (ticks(node)) return;
  const inside = away.includes(node) ? body(node) : null;
  for (const child of node.childNodes) {
    if (child === inside) lines.push(indent + '  (not loaded)');
    else write(child, page, away, depth + 1, lines);
  }
}

/** The transcript on *page*, without the insides of the panels *away*. */
function transcript(page, away) {
  const lines = [];
  write(page.O, page, away, 0, lines);
  return lines;
}

/** The first position at which line lists *a* and *b* differ, or -1. */
function firstDifference(a, b) {
  const length = Math.max(a.length, b.length);
  for (let i = 0; i < length; i++) if (a[i] !== b[i]) return i;
  return -1;
}

/**
 * Fail unless the pages of *kase* show the same status row and the
 * same transcript *when*, the insides of the panels the lazy page has
 * still to load left out of both.
 */
function assertSame(kase, whole, lazy, when) {
  assert.deepStrictEqual(
    statusMetrics(lazy.win),
    statusMetrics(whole.win),
    kase.name + ': the status row ' + when,
  );
  const waiting = Array.from(lazy.O.querySelectorAll('.lazy-unloaded'));
  const twins = waiting.map(panel => at(whole.O, pathTo(panel, lazy.O)));
  const want = transcript(whole, twins);
  const got = transcript(lazy, waiting);
  const where = firstDifference(want, got);
  if (where < 0) return;
  assert.fail(
    [
      kase.name + ': the transcripts differ ' + when + ', after',
      ...want.slice(Math.max(0, where - 8), where),
      'whole:     ' + want[where],
      'in pieces: ' + got[where],
    ].join('\n'),
  );
}

/** What names the piece *request* asks for among those of its task. */
function pieceKey(request) {
  const handle = {};
  for (const name of Object.keys(request).sort())
    if (!['type', 'taskId', 'tabId', 'token'].includes(name))
      handle[name] = request[name];
  return JSON.stringify(handle);
}

/** The pieces *page* asked the daemon for so far, answered or not. */
function requested(page) {
  return page.posted.filter(m => m.type === 'getTaskEvents');
}

/** The panel of *page* that waits for the piece *request* asks for. */
function waiting(page, request) {
  return page.O.querySelector(
    request.seg === undefined
      ? '.trajectory.lazy-unloaded'
      : '.lazy-unloaded[data-lazy-seg="' + request.seg + '"]',
  );
}

/**
 * Open every panel the lazy page of *kase* has still to load, answer
 * the request each posts, and open the same panels of the whole page.
 * Returns the pieces asked for.
 */
function openEveryPanel(kase, whole, lazy) {
  const asked = [];
  for (;;) {
    let panel;
    if (requested(lazy).length > asked.length) {
      // Unfolded by the page itself, on both pages: no click.  One
      // piece can unfold several panels, each asking for its own.
      panel = waiting(lazy, requested(lazy)[asked.length]);
      assert.ok(panel, kase.name + ': a panel waits for the piece asked for');
      assert.ok(
        !panel.classList.contains('collapsed'),
        kase.name + ': a panel that asked by itself is unfolded',
      );
    } else {
      panel = lazy.O.querySelector('.lazy-unloaded');
      if (!panel) return asked;
      click(lazy.win, header(panel));
      click(whole.win, header(at(whole.O, pathTo(panel, lazy.O))));
      assert.strictEqual(
        requested(lazy).length,
        asked.length + 1,
        kase.name + ': one request a click',
      );
    }
    const request = requested(lazy)[asked.length];
    const key = pieceKey(request);
    assert.strictEqual(request.taskId, kase.frame.task_id, kase.name + ': ' + key);
    assert.ok(!asked.includes(key), kase.name + ': ' + key + ' is asked once');
    assert.ok(kase.pieces[key], kase.name + ': no panel was handed ' + key);
    asked.push(key);
    send(lazy.win, {
      type: 'taskEventsPiece',
      taskId: request.taskId,
      seg: request.seg === undefined ? -1 : request.seg,
      token: request.token,
      events: kase.pieces[key],
    });
    assert.ok(
      !panel.classList.contains('lazy-unloaded'),
      kase.name + ': the answer to ' + key + ' fills its panel',
    );
    assertSame(kase, whole, lazy, 'after piece ' + key);
  }
}

/** The pictures *page* asked the daemon for: their numbers in the transcript. */
function pictures(page) {
  return page.posted
    .filter(m => m.type === 'getReplayImage')
    .map(m => m.index)
    .sort();
}

/** Show the task of *kase* both ways and compare at every step. */
function check(kase) {
  const whole = show(kase, kase.whole);
  const lazy = show(kase, kase.first);
  assertSame(kase, whole, lazy, 'as loaded');
  const asked = openEveryPanel(kase, whole, lazy);
  assert.deepStrictEqual(
    asked.slice().sort(),
    Object.keys(kase.pieces).sort(),
    kase.name + ': every handle the daemon gave is a panel to open',
  );
  assert.deepStrictEqual(
    pictures(lazy),
    pictures(whole),
    kase.name + ': the pictures asked for',
  );
  // A finished task folded again: the header counts what was loaded.
  const loaded = lazy.O.querySelector(':scope > .trajectory');
  if (loaded) {
    click(lazy.win, header(loaded));
    click(whole.win, header(whole.O.querySelector(':scope > .trajectory')));
    assertSame(kase, whole, lazy, 'folded again');
  }
  whole.win.close();
  lazy.win.close();
  console.log('  ok - ' + kase.name + ': ' + asked.length + ' pieces');
}

if (process.argv.length !== 3) {
  console.error('usage: node test/lazy_replay_diff.js <cases.json>');
  process.exit(2);
}
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
cases.forEach(check);
console.log('lazy_replay_diff: ' + cases.length + ' transcripts match');
