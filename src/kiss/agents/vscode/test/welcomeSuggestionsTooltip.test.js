// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end regression test for welcome suggestion tooltips.  The
// welcome chips clamp long SAMPLE_TASKS text to three visible lines;
// hovering a chip must expose the full suggested text through the
// production custom tooltip used inside VS Code webviews.

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
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testWelcomeSuggestionHoverShowsFullSuggestedText() {
  const {win} = makeWebview();
  const fullText =
    'Please run the complete multi-step benchmark on PWD/data/input.json, ' +
    'collect all logs, explain every failure mode, and include the exact ' +
    'commands, file paths, and final remediation plan without truncating ' +
    'this suggested task text.';

  send(win, {type: 'welcome_suggestions', suggestions: [{text: fullText}]});

  const chip = win.document.querySelector('.suggestion-chip');
  assert.ok(chip, 'welcome suggestion chip should be rendered');
  assert.strictEqual(
    chip.dataset.tooltip,
    fullText,
    'chip must carry the full suggestion text for the custom tooltip',
  );

  chip.dispatchEvent(new win.MouseEvent('mouseover', {bubbles: true}));
  await wait(450);

  const tooltip = win.document.getElementById('custom-tooltip');
  assert.ok(tooltip, 'custom tooltip element should exist');
  assert.strictEqual(
    tooltip.textContent,
    fullText,
    'hover tooltip must show the entire suggested text',
  );
  assert.ok(
    tooltip.classList.contains('visible'),
    'hover tooltip should become visible',
  );
}

const tests = [testWelcomeSuggestionHoverShowsFullSuggestedText];

(async function run() {
  let failed = 0;
  for (const t of tests) {
    try {
      await t();
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
  console.log('All tests passed');
})();
