// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end integration tests for the fast-complete picker.
//
// The chat webview's input textbox now shows ALL fast-complete
// options as a dropdown picker — the same UI the ``@``-mention file
// picker uses — instead of only the single inline ghost-text
// suggestion.  The backend emits both events:
//
//   * ``ghost``       — the legacy single-suffix inline overlay
//                       (still consumed by ``acceptGhost`` on Tab).
//   * ``completions`` — a new list-of-candidates event that opens
//                       the dropdown picker in ``#autocomplete``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/fast_complete_picker.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

/**
 * Build a jsdom window running the production chat webview.  Returns
 * ``{win, posted}`` where ``posted`` is the array of every message the
 * webview has posted to the (recording) ``vscode`` host stub.
 */
function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function setInput(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.setSelectionRange(text.length, text.length);
  return inp;
}

/** Simulate the user typing *ch* at the end of the input. */
function typeChar(win, ch) {
  const inp = win.document.getElementById('task-input');
  inp.value += ch;
  inp.setSelectionRange(inp.value.length, inp.value.length);
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  return inp;
}

function picker(win) {
  return win.document.getElementById('autocomplete');
}

function visible(win) {
  return picker(win).style.display === 'block';
}

function items(win) {
  return Array.from(picker(win).querySelectorAll('.ac-item'));
}

function sections(win) {
  return Array.from(picker(win).querySelectorAll('.ac-section')).map(
    e => e.textContent,
  );
}

const COMPLETIONS = [
  {type: 'task', text: 'fix the parser bug now'},
  {type: 'task', text: 'fix the parser then commit'},
  {type: 'identifier', text: 'fix parse_arguments'},
];

// 1. A matching completions event opens the dropdown picker.
function testCompletionsRendersPicker() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  assert.strictEqual(
    visible(win),
    true,
    'completions event must open the picker',
  );
  const its = items(win);
  assert.strictEqual(its.length, COMPLETIONS.length);
  const text = picker(win).textContent;
  assert.ok(text.includes('fix the parser bug now'));
  assert.ok(text.includes('fix parse_arguments'));
}

// 2. The picker groups items by section and uses an icon column.
function testCompletionsHasSectionsAndIcons() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const labels = sections(win);
  assert.ok(
    labels.length >= 2,
    'completions picker must show grouped sections',
  );
  const iconCount = picker(win).querySelectorAll('.ac-icon').length;
  assert.strictEqual(
    iconCount,
    COMPLETIONS.length,
    'every item must carry a leading icon column (.ac-icon)',
  );
}

// 3. The picker shows a footer with the kbd hints.
function testCompletionsRendersFooter() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const footer = picker(win).querySelector('.ac-footer');
  assert.ok(footer, 'completions picker must render the .ac-footer kbd row');
  assert.ok(/Tab/.test(footer.textContent));
  assert.ok(/Esc/.test(footer.textContent));
}

// 4. A stale completions reply for an OLD query is dropped.
function testCompletionsStaleQueryDropped() {
  const {win} = makeWebview();
  setInput(win, 'fix the new bug');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix', // user has typed past this — reply is stale
  });
  assert.strictEqual(
    visible(win),
    false,
    'completions for a stale query must not open the picker',
  );
}

// 5. The picker stays hidden while an @-mention is being typed —
//    the file picker takes precedence in that mode.
function testCompletionsSuppressedDuringAtMention() {
  const {win} = makeWebview();
  setInput(win, 'open @util');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'open @util',
  });
  assert.strictEqual(
    visible(win),
    false,
    'completions must NOT clobber the @-mention file picker',
  );
}

// 6. The picker stays hidden when the cursor is not at the end.
function testCompletionsSuppressedWhenCursorNotAtEnd() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix the bug now');
  inp.setSelectionRange(3, 3); // cursor in the middle
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix the bug now',
  });
  assert.strictEqual(
    visible(win),
    false,
    'completions must not appear when the cursor is not at end',
  );
}

// 7. Clicking an item replaces the input with that completion +
//    a trailing space.
function testClickAcceptsCompletion() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const its = items(win);
  its[1].click();
  assert.strictEqual(
    inp.value,
    'fix the parser then commit ',
    'click must replace input with the chosen completion + space',
  );
  assert.strictEqual(
    visible(win),
    false,
    'picker must hide after click-accept',
  );
}

// 8. The first item is auto-selected (sel class) and ArrowDown
//    advances the selection.
function testArrowDownMovesSelection() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const its = items(win);
  assert.ok(its[0].classList.contains('sel'), 'first item is auto-selected');
  const ev = new win.KeyboardEvent('keydown', {
    key: 'ArrowDown',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  const its2 = items(win);
  assert.ok(its2[1].classList.contains('sel'));
  assert.ok(!its2[0].classList.contains('sel'));
}

// 9. Pressing Tab while the picker is open accepts the highlighted
//    completion (does NOT fall through to inline ghost accept).
function testTabAcceptsCompletion() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Tab',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  assert.strictEqual(
    inp.value,
    'fix the parser bug now ',
    'Tab inside picker accepts the selected completion',
  );
  assert.strictEqual(visible(win), false);
}

// 10. Pressing Enter inside the picker accepts the selected item.
function testEnterAcceptsCompletion() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Enter',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  assert.strictEqual(
    inp.value,
    'fix the parser bug now ',
    'Enter inside picker accepts the selected completion',
  );
}

// 11. Pressing Escape dismisses the picker.
function testEscapeDismissesPicker() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  assert.strictEqual(visible(win), true);
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Escape',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  assert.strictEqual(visible(win), false);
}

// 12. An empty completions list closes any previously visible picker.
function testEmptyCompletionsHidesPicker() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  assert.strictEqual(visible(win), true);
  send(win, {
    type: 'completions',
    completions: [],
    query: 'fix',
  });
  assert.strictEqual(
    visible(win),
    false,
    'an empty completions reply must hide the picker',
  );
}

// 13. The legacy ``ghost`` event must still update inline ghost text
//    (back-compat).
function testGhostStillWorks() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {type: 'ghost', suggestion: ' the bug', query: 'fix'});
  const overlay = win.document.getElementById('ghost-overlay').textContent;
  assert.ok(
    overlay.includes('the bug'),
    'ghost event must still feed the inline overlay',
  );
  // Tab still accepts the inline ghost when no picker is open.
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Tab',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  assert.strictEqual(inp.value, 'fix the bug ');
}

// 14. Typing a query in the input still posts a ``complete`` command
//    to the backend (the front-end protocol contract).
function testTypingRequestsCompletions() {
  const {win, posted} = makeWebview();
  const inp = win.document.getElementById('task-input');
  inp.value = 'f';
  inp.setSelectionRange(1, 1);
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  typeChar(win, 'i');
  typeChar(win, 'x');
  // The webview debounces by 300 ms — fast-forward.
  const sentinel = Date.now() + 600;
  while (Date.now() < sentinel) {
    // Spin until debounce timer fires.
  }
  // Drain microtasks/timers — the debounce uses ``setTimeout``.
  return new Promise(resolve => {
    win.setTimeout(() => {
      const cmd = posted.find(p => p && p.type === 'complete');
      assert.ok(cmd, 'typing must dispatch a ``complete`` command');
      assert.strictEqual(cmd.query, 'fix');
      resolve();
    }, 500);
  });
}

// 15. A late completions reply for the *current* query that the user
//    has since cleared must not pop the picker over an empty input.
function testCompletionsForEmptyInputDoesNotShow() {
  const {win} = makeWebview();
  setInput(win, '');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: '',
  });
  assert.strictEqual(
    visible(win),
    false,
    'completions for an empty query must not open the picker',
  );
}

// 16. Click handlers stop propagation so they don't blur the input
//    (the picker uses ``mousedown`` to avoid losing focus).
function testPickerSurvivesMouseDown() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const its = items(win);
  // mousedown must not close the picker before click fires.
  const md = new win.MouseEvent('mousedown', {bubbles: true, cancelable: true});
  its[0].dispatchEvent(md);
  assert.strictEqual(visible(win), true);
}

// 17. Per-connection scoping: a completions event with a foreign
//    ``connId`` (a sibling VS Code window) must not affect this
//    webview.  The frontend never sets ``window._connId``; the test
//    just confirms that ``connId`` on the event is tolerated.
function testCompletionsConnIdTolerated() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
    connId: 'sibling-window-uuid',
  });
  // The webview currently does NOT filter by connId for events
  // delivered to it (the backend already routes to the right
  // connection), so it must still render normally.
  assert.strictEqual(visible(win), true);
}

// 18. Back-compat: a completions event with NO ``query`` field
//    (older backend / recorded streams) must still render.
function testCompletionsBackCompatNoQuery() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {type: 'completions', completions: COMPLETIONS});
  assert.strictEqual(
    visible(win),
    true,
    'a prefix-less completions reply must still render',
  );
}

// 19. Robustness: a completions event with NO ``completions`` field
//    must close the picker without crashing (defensive default).
function testCompletionsMissingFieldHandled() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {type: 'completions', query: 'fix'});
  assert.strictEqual(
    visible(win),
    false,
    'missing completions field must default to empty list',
  );
}

// 20. Frequent / trick / identifier icons render distinctly.
function testCompletionsIconsPerType() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: [
      {type: 'task', text: 'fix history task'},
      {type: 'trick', text: 'fix the issue by writing'},
      {type: 'identifier', text: 'fix_my_variable'},
      {type: 'frequent', text: 'fix a frequent one'},
    ],
    query: 'fix',
  });
  const icons = picker(win).querySelectorAll('.ac-icon');
  assert.strictEqual(icons.length, 4);
  const html = picker(win).innerHTML;
  // Each glyph must be uniquely present.
  assert.ok(/polygon points="13 2 3 14/.test(html), 'task bolt icon present');
  assert.ok(/M12 2l1.5 5L19 8.5/.test(html), 'trick sparkle icon present');
  assert.ok(/M8 4H6a2 2 0 00-2 2v4/.test(html), 'identifier code icon present');
  assert.ok(/M12 2l3.09 6.26/.test(html), 'frequent star icon present');
  // Section labels.
  const labels = sections(win);
  assert.ok(labels.includes('History'));
  assert.ok(labels.includes('Frequent'));
  assert.ok(labels.includes('Suggestions'));
  assert.ok(labels.includes('From editor'));
}

// 21. ``acceptCompletion`` must NOT add a second trailing space when
//    the chosen completion text already ends in whitespace.
function testAcceptCompletionPreservesTrailingSpace() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: [{type: 'task', text: 'fix that ends in space '}],
    query: 'fix',
  });
  items(win)[0].click();
  assert.strictEqual(
    inp.value,
    'fix that ends in space ',
    'no double-space when item already ends in whitespace',
  );
}

// 22. The footer must lock all three kbd hints (navigate/accept/dismiss).
function testCompletionsFooterContents() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const footer = picker(win).querySelector('.ac-footer');
  assert.ok(/navigate/.test(footer.textContent), 'navigate hint present');
  assert.ok(/accept/.test(footer.textContent), 'accept hint present');
  assert.ok(/dismiss/.test(footer.textContent), 'dismiss hint present');
}

// 23. ``hlMatch`` highlights the user's prefix in every item.
function testCompletionsHighlightsPrefix() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const hl = picker(win).querySelectorAll('strong.ac-hl');
  assert.ok(hl.length >= 1, 'matched prefix must be wrapped in .ac-hl');
  assert.strictEqual(hl[0].textContent, 'fix');
}

// 25. Regression: an EMPTY ``completions`` reply that arrives
//     while an ``@``-mention file picker is already open must NOT
//     close the file picker.  The earlier code path called
//     ``hideAC()`` on the empty-data branch before checking
//     ``getAtCtx()``, which collapsed the visible file picker when
//     a delayed empty completions reply for the previously-typed
//     text raced in.
function testEmptyCompletionsDoesNotClobberAtMentionPicker() {
  const {win} = makeWebview();
  setInput(win, 'open @util');
  // Open the @-mention file picker with a non-empty 'files' reply.
  send(win, {
    type: 'files',
    files: [
      {type: 'file', text: 'util/index.ts'},
      {type: 'file', text: 'util/parse.ts'},
    ],
    prefix: 'util',
  });
  assert.strictEqual(
    visible(win),
    true,
    'precondition: file picker is open after `files` reply',
  );
  // Now a stale empty completions reply lands.
  send(win, {
    type: 'completions',
    completions: [],
    query: 'open @util',
  });
  assert.strictEqual(
    visible(win),
    true,
    'empty completions reply must NOT close the file picker',
  );
  // Items still there.
  assert.ok(
    items(win).length >= 2,
    'file picker items must be preserved',
  );
}

// 26. Tab inside the picker must call preventDefault so the chat
//    is not submitted (which Tab + Enter shares with the inline
//    ghost-accept path).
function testTabInPickerPreventsDefault() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Tab',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  assert.strictEqual(ev.defaultPrevented, true);
}

const tests = [
  testCompletionsRendersPicker,
  testCompletionsHasSectionsAndIcons,
  testCompletionsRendersFooter,
  testCompletionsStaleQueryDropped,
  testCompletionsSuppressedDuringAtMention,
  testCompletionsSuppressedWhenCursorNotAtEnd,
  testClickAcceptsCompletion,
  testArrowDownMovesSelection,
  testTabAcceptsCompletion,
  testEnterAcceptsCompletion,
  testEscapeDismissesPicker,
  testEmptyCompletionsHidesPicker,
  testGhostStillWorks,
  testTypingRequestsCompletions,
  testCompletionsForEmptyInputDoesNotShow,
  testPickerSurvivesMouseDown,
  testCompletionsConnIdTolerated,
  testCompletionsBackCompatNoQuery,
  testCompletionsMissingFieldHandled,
  testCompletionsIconsPerType,
  testAcceptCompletionPreservesTrailingSpace,
  testCompletionsFooterContents,
  testCompletionsHighlightsPrefix,
  testEmptyCompletionsDoesNotClobberAtMentionPicker,
  testTabInPickerPreventsDefault,
];

async function main() {
  let failed = 0;
  for (const t of tests) {
    try {
      const r = t();
      if (r && typeof r.then === 'function') await r;
      console.log('PASS', t.name);
    } catch (err) {
      failed += 1;
      console.error('FAIL', t.name);
      console.error(err && err.stack ? err.stack : err);
    }
  }
  if (failed) {
    console.error(failed + ' test(s) failed');
    process.exit(1);
  }
  console.log('All ' + tests.length + ' tests passed');
}

main();
