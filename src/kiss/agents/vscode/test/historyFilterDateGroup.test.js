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

  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

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

  assert.ok(
    !group.classList.contains('history-filter-bar'),
    `"${which}" trio must be wrapped in a grouping element ` +
      'distinct from .history-filter-bar (so the three pieces ' +
      'cannot wrap onto separate lines individually)',
  );

  assert.ok(
    group.classList.contains('history-filter-date-group'),
    `"${which}" trio's wrapper must have class ` +
      `"history-filter-date-group" (got: "${group.className}")`,
  );

  const rangeWrap = group.parentElement;
  assert.ok(
    rangeWrap && rangeWrap.classList.contains('history-filter-daterange'),
    `"${which}" date group must be a direct child of ` +
      '.history-filter-daterange',
  );
  const bar = rangeWrap.parentElement;
  assert.ok(
    bar && bar.classList.contains('history-filter-bar'),
    '.history-filter-daterange must be a direct child of ' +
      '.history-filter-bar',
  );

  const kids = Array.from(group.children);
  const idx = el => kids.indexOf(el);
  assert.ok(
    idx(label) < idx(input) && idx(input) < idx(button),
    `"${which}" group children must be ordered: label, input, ` +
      `button (got: [${kids.map(k => k.tagName + (k.id ? '#' + k.id : '')).join(', ')}])`,
  );

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
