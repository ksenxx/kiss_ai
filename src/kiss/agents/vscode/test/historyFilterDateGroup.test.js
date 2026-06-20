// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the "From" / "To" date filter widgets in the
// History sidebar's filter bar.
//
// Requirement driven by this test:
//
//   The "From" label, the From date textbox (``#hf-from``), and the
//   From calendar-picker button (``#hf-from-btn``) MUST NEVER be
//   split across multiple visual lines.  When the sidebar is narrow
//   enough that the filter bar has to wrap, the three pieces must
//   wrap together as a single atomic unit.  Likewise for the "To"
//   trio (label + ``#hf-to`` + ``#hf-to-btn``).
//
// The filter bar (``.history-filter-bar``) is a ``flex-wrap: wrap``
// flex container, so by default every direct child is an
// independent wrap unit and the label, input, and button can land
// on three different rows.  This test enforces a structural fix:
// each trio must share a single direct parent element whose CSS
// keeps its children on one line.
//
// Concretely the test asserts:
//
//   * ``label[for="hf-from"]``, ``#hf-from`` and ``#hf-from-btn``
//     have THE SAME direct parent element (i.e. they are wrapped
//     in a single grouping element).
//   * Same for the To trio.
//   * That grouping element is NOT the ``.history-filter-bar``
//     itself (which would defeat the purpose).
//   * The grouping element carries the class
//     ``.history-filter-date-group`` so the CSS can target it.
//   * ``media/main.css`` declares a ``.history-filter-date-group``
//     rule that prevents internal wrapping — either
//     ``flex-wrap: nowrap`` (when the group is a flex container)
//     or ``white-space: nowrap``.  Either guarantees the label,
//     input, and button stay glued on the same line.
//
// jsdom cannot run real layout, so this test enforces the
// invariant structurally (shared parent + class) AND through a
// CSS-source check, exactly the same pattern as
// ``historyTaskMeta.test.js`` uses for ``flex-basis: 100%``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyFilterDateGroup.test.js

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

  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: () => {},
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win};
}

function assertGroupedTrio(win, which) {
  const doc = win.document;
  const inputId = `hf-${which}`;
  const btnId = `hf-${which}-btn`;
  const label = doc.querySelector(`label[for="${inputId}"]`);
  const input = doc.getElementById(inputId);
  const button = doc.getElementById(btnId);

  assert.ok(label, `"${which}" label[for="${inputId}"] must exist`);
  assert.ok(input, `"${which}" #${inputId} input must exist`);
  assert.ok(button, `"${which}" #${btnId} button must exist`);

  // 1) The label, input, and button must share the SAME direct
  //    parent element — so the parent (not its three children)
  //    becomes the wrap unit inside .history-filter-bar.
  assert.strictEqual(
    label.parentElement,
    input.parentElement,
    `"${which}" label and input must share the same direct parent`,
  );
  assert.strictEqual(
    input.parentElement,
    button.parentElement,
    `"${which}" input and button must share the same direct parent`,
  );

  const group = label.parentElement;
  assert.ok(group, `"${which}" trio must have a direct parent element`);

  // 2) That parent must NOT be the filter bar itself.  If it were,
  //    each of the three would still be an independent flex item
  //    of a ``flex-wrap: wrap`` container and could wrap apart.
  assert.ok(
    !group.classList.contains('history-filter-bar'),
    `"${which}" trio must be wrapped in a grouping element ` +
      'distinct from .history-filter-bar (so the three pieces ' +
      'cannot wrap onto separate lines individually)',
  );

  // 3) The grouping element must carry the class
  //    ``.history-filter-date-group`` so CSS can target it.
  assert.ok(
    group.classList.contains('history-filter-date-group'),
    `"${which}" trio's wrapper must have class ` +
      `"history-filter-date-group" (got: "${group.className}")`,
  );

  // 4) The grouping element itself must live directly inside the
  //    history filter bar — otherwise it would not behave as a
  //    flex item of that bar.
  const bar = group.parentElement;
  assert.ok(
    bar && bar.classList.contains('history-filter-bar'),
    `"${which}" date group must be a direct child of ` +
      '.history-filter-bar',
  );

  // 5) Order inside the group must be label → input → button so
  //    the textbox sits between its label and its picker icon.
  const kids = Array.from(group.children);
  const idx = el => kids.indexOf(el);
  assert.ok(
    idx(label) < idx(input) && idx(input) < idx(button),
    `"${which}" group children must be ordered: label, input, ` +
      `button (got: [${kids.map(k => k.tagName + (k.id ? '#' + k.id : '')).join(', ')}])`,
  );

  // 6) No stray elements between the three — keep the group tight.
  assert.strictEqual(
    kids.length,
    3,
    `"${which}" date group must contain exactly 3 children ` +
      `(label, input, button); got ${kids.length}`,
  );
}

function testFromAndToGroupsAreAtomic() {
  const {win} = makeWebview();
  assertGroupedTrio(win, 'from');
  assertGroupedTrio(win, 'to');
  win.close();
  console.log(
    '  ok - From and To label/input/button trios share a single ' +
      '.history-filter-date-group parent inside .history-filter-bar',
  );
}

function testDateGroupCssPreventsInternalWrap() {
  // jsdom never applies external stylesheets to layout, so we
  // read main.css directly (same trick as historyTaskMeta.test.js
  // uses for ``flex-basis: 100%``) and assert that the
  // ``.history-filter-date-group`` rule guarantees the label,
  // input and button stay on one line.
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const re = /\.history-filter-date-group\s*\{([^}]*)\}/g;
  let m;
  let body = null;
  while ((m = re.exec(css)) !== null) {
    body = m[1];
    break;
  }
  assert.ok(
    body !== null,
    'main.css must declare a ".history-filter-date-group" rule so ' +
      'the From/To trios stay on a single line',
  );

  // Accept either of the two standard ways of preventing the
  // group's children from breaking across lines:
  //
  //   * ``flex-wrap: nowrap`` (when the group is a flex container)
  //   * ``white-space: nowrap``
  //
  // The group must also be laid out so it stays together as ONE
  // atomic flex item of ``.history-filter-bar``.  That means it
  // needs an inline-friendly display (``inline-flex`` or
  // ``inline-block``) — otherwise a block-level wrapper would
  // force a line break of its own.
  const hasFlexNowrap = /flex-wrap\s*:\s*nowrap/.test(body);
  const hasWhitespaceNowrap = /white-space\s*:\s*nowrap/.test(body);
  assert.ok(
    hasFlexNowrap || hasWhitespaceNowrap,
    '.history-filter-date-group must declare "flex-wrap: nowrap" ' +
      'or "white-space: nowrap" so its label/input/button never ' +
      `split across lines; got body: ${body.trim()}`,
  );

  const hasInlineDisplay =
    /display\s*:\s*inline-flex/.test(body) ||
    /display\s*:\s*inline-block/.test(body);
  assert.ok(
    hasInlineDisplay,
    '.history-filter-date-group must use an inline display ' +
      '("inline-flex" or "inline-block") so it behaves as a ' +
      `single flex item of .history-filter-bar; got body: ${body.trim()}`,
  );

  console.log(
    '  ok - .history-filter-date-group CSS keeps label/input/button ' +
      'on the same line',
  );
}

function main() {
  testFromAndToGroupsAreAtomic();
  testDateGroupCssPreventsInternalWrap();
  console.log('historyFilterDateGroup.test.js: all assertions passed.');
}

main();
