// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
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

function testCompletionsStaleQueryDropped() {
  const {win} = makeWebview();
  setInput(win, 'fix the new bug');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  assert.strictEqual(
    visible(win),
    false,
    'completions for a stale query must not open the picker',
  );
}

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

function testCompletionsSuppressedWhenCursorNotAtEnd() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix the bug now');
  inp.setSelectionRange(3, 3);
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

function testEnterDoesNotAcceptCompletion() {
  const {win, posted} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'ArrowDown',
      bubbles: true,
      cancelable: true,
    }),
  );
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
    }),
  );
  const submits = posted.filter(m => m.type === 'submit');
  assert.strictEqual(
    submits.length,
    1,
    'Enter with the picker open must submit the typed text',
  );
  assert.strictEqual(
    submits[0].prompt,
    'fix',
    'Enter must submit what the user typed — never the completion',
  );
  assert.strictEqual(
    visible(win),
    false,
    'the picker must be dismissed by Enter',
  );
  assert.strictEqual(
    inp.value,
    '',
    'submit clears the input; the completion text must not appear',
  );
}

function testEnterAcceptsFileMention() {
  const {win, posted} = makeWebview();
  const inp = setInput(win, 'open @util');
  send(win, {
    type: 'files',
    files: [
      {type: 'file', text: 'util/index.ts'},
      {type: 'file', text: 'util/parse.ts'},
    ],
    prefix: 'util',
  });
  assert.strictEqual(visible(win), true, 'file picker opens');
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
    }),
  );
  assert.ok(
    inp.value.indexOf('util/index.ts') !== -1,
    `Enter in the @-mention picker must insert the mention: ${inp.value}`,
  );
  assert.strictEqual(
    posted.filter(m => m.type === 'submit').length,
    0,
    'Enter in the @-mention picker must not submit the message',
  );
}

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

function testGhostStillWorks() {
  const {win} = makeWebview();
  const inp = setInput(win, 'fix');
  send(win, {type: 'ghost', suggestion: ' the bug', query: 'fix'});
  const overlay = win.document.getElementById('ghost-overlay').textContent;
  assert.ok(
    overlay.includes('the bug'),
    'ghost event must still feed the inline overlay',
  );
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Tab',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  assert.strictEqual(inp.value, 'fix the bug ');
}

function testTypingRequestsCompletions() {
  const {win, posted} = makeWebview();
  const inp = win.document.getElementById('task-input');
  inp.value = 'f';
  inp.setSelectionRange(1, 1);
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  typeChar(win, 'i');
  typeChar(win, 'x');
  const sentinel = Date.now() + 600;
  while (Date.now() < sentinel) {
  }
  return new Promise(resolve => {
    win.setTimeout(() => {
      const cmd = posted.find(p => p && p.type === 'complete');
      assert.ok(cmd, 'typing must dispatch a ``complete`` command');
      assert.strictEqual(cmd.query, 'fix');
      resolve();
    }, 500);
  });
}

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

function testPickerSurvivesMouseDown() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
  });
  const its = items(win);
  const md = new win.MouseEvent('mousedown', {bubbles: true, cancelable: true});
  its[0].dispatchEvent(md);
  assert.strictEqual(visible(win), true);
}

function testCompletionsConnIdTolerated() {
  const {win} = makeWebview();
  setInput(win, 'fix');
  send(win, {
    type: 'completions',
    completions: COMPLETIONS,
    query: 'fix',
    connId: 'sibling-window-uuid',
  });
  assert.strictEqual(visible(win), true);
}

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
  assert.ok(/polygon points="13 2 3 14/.test(html), 'task bolt icon present');
  assert.ok(/M12 2l1.5 5L19 8.5/.test(html), 'trick sparkle icon present');
  assert.ok(/M8 4H6a2 2 0 00-2 2v4/.test(html), 'identifier code icon present');
  assert.ok(/M12 2l3.09 6.26/.test(html), 'frequent star icon present');
  const labels = sections(win);
  assert.ok(labels.includes('History'));
  assert.ok(labels.includes('Frequent'));
  assert.ok(labels.includes('Suggestions'));
  assert.ok(labels.includes('From editor'));
}

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

function testEmptyCompletionsDoesNotClobberAtMentionPicker() {
  const {win} = makeWebview();
  setInput(win, 'open @util');
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
  assert.ok(
    items(win).length >= 2,
    'file picker items must be preserved',
  );
}

function testIdentifierAcceptPreservesExistingText() {
  const {win} = makeWebview();
  const inp = setInput(win, 'please fix the bug in parse_arg');
  send(win, {
    type: 'completions',
    completions: [{type: 'identifier', text: 'parse_arguments'}],
    query: 'please fix the bug in parse_arg',
  });
  const ev = new win.KeyboardEvent('keydown', {
    key: 'Tab',
    bubbles: true,
    cancelable: true,
  });
  inp.dispatchEvent(ev);
  assert.strictEqual(
    inp.value,
    'please fix the bug in parse_arguments ',
    'accepting an identifier completion must keep the existing text',
  );
}

function testTrickAcceptPreservesEarlierSentences() {
  const {win} = makeWebview();
  const inp = setInput(win, 'Fix the crash. Then rep');
  send(win, {
    type: 'completions',
    completions: [
      {
        type: 'trick',
        text: 'Then reproduce the issue by writing an end-to-end test',
      },
    ],
    query: 'Fix the crash. Then rep',
  });
  items(win)[0].click();
  assert.strictEqual(
    inp.value,
    'Fix the crash. Then reproduce the issue by writing an ' +
      'end-to-end test ',
    'accepting a trick completion must keep earlier sentences',
  );
}

function testIdentifierClickAcceptPreservesExistingText() {
  const {win} = makeWebview();
  const inp = setInput(win, 'rename self.old_na');
  send(win, {
    type: 'completions',
    completions: [{type: 'identifier', text: 'self.old_name'}],
    query: 'rename self.old_na',
  });
  items(win)[0].click();
  assert.strictEqual(
    inp.value,
    'rename self.old_name ',
    'click-accepting an identifier completion must keep the existing text',
  );
}

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
  testEnterDoesNotAcceptCompletion,
  testEnterAcceptsFileMention,
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
  testIdentifierAcceptPreservesExistingText,
  testTrickAcceptPreservesEarlierSentences,
  testIdentifierClickAcceptPreservesExistingText,
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
