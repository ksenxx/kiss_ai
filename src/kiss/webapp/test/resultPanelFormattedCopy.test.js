// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end page tests: the Result panel's copy button copies the
// FORMATTED text of the rendered summary — never the summary's raw
// HTML markup.
//
// Reproduces the issue where createResultPanel() stored the raw HTML
// summary string in dataset.rawText, so pressing the panel's copy
// button put "<h2>Report</h2><ul><li>..." on the clipboard.
//
// media/main.js + media/panelCopy.js are served to the remote webapp
// by src/kiss/server/web_server.py.

'use strict';

const assert = require('assert');
const harness = require('./simplify2_harness');
const {send, settle, click, stubClipboard} = harness;

// The harness page plus a navigator.clipboard stub that records the
// last text written; getClipboard() reads it back.
function makePage() {
  let clipboard = null;
  const {win} = harness.makePage({
    beforeScripts: w => {
      clipboard = stubClipboard(w);
    },
  });
  return {win, getClipboard: () => clipboard.last()};
}

function clickCopy(win) {
  const rc = win.document.querySelector('#output > .rc');
  assert.ok(rc, 'expected a Result panel');
  const btn = rc.querySelector('.panel-copy-btn');
  assert.ok(btn, 'Result panel must have a copy button');
  click(win, btn);
}

async function testHtmlSummaryCopiesFormattedText() {
  const {win, getClipboard} = makePage();
  send(win, {
    type: 'result',
    success: true,
    is_continue: false,
    summary:
      '<h2>Report</h2>' +
      '<p>All <b>good</b>.</p>' +
      '<ul><li>alpha</li><li>beta</li></ul>' +
      '<ol start="3"><li>three</li><li>four</li></ol>' +
      '<ul><li><p>loose one</p><p>more</p></li><li></li><li><p>loose two</p></li></ul>' +
      '<pre><code>x = 1\ny = 2</code></pre>' +
      '<table><tr><th>k</th><th>v</th></tr>' +
      '<tr><td>a</td><td>b</td></tr></table>',
    total_tokens: 20,
    cost: '$0.02',
  });
  clickCopy(win);
  await settle();
  const text = getClipboard();
  assert.ok(text, 'copy button must write to the clipboard');
  assert.ok(
    !/<\/?(h2|p|ul|li|pre|code|table|b)\b/i.test(text),
    'BUG: clipboard must not contain raw HTML tags; got: ' + text,
  );
  assert.ok(text.includes('Report'), 'heading text must survive');
  assert.ok(text.includes('All good.'), 'inline markup must flatten');
  assert.ok(text.includes('- alpha'), 'ul items get "- " markers');
  assert.ok(text.includes('- beta'), 'every ul item gets a marker');
  assert.ok(text.includes('3. three'), 'ol honours its start number');
  assert.ok(text.includes('4. four'), 'ol numbering increments');
  assert.ok(
    /- loose one\n\nmore\n\n-\n- loose two/.test(text),
    'a loose item (paragraph-wrapped) keeps its marker on its first ' +
      'line, its later paragraphs stay paragraphs, an empty item keeps ' +
      'its line; got: ' +
      text,
  );
  assert.ok(
    text.includes('x = 1\ny = 2'),
    'pre content keeps its own line breaks',
  );
  assert.ok(text.includes('k | v'), 'table header cells join with " | "');
  assert.ok(text.includes('a | b'), 'table row cells join with " | "');
  assert.ok(
    /Report\n\nAll good\./.test(text),
    'paragraph blocks separate with a blank line; got: ' + text,
  );
  win.close();
}

async function testPlainTextResultStillCopiesItsText() {
  const {win, getClipboard} = makePage();
  send(win, {
    type: 'result',
    success: true,
    is_continue: false,
    text: 'plain output line 1\nline 2 <not-a-tag>',
    total_tokens: 5,
    cost: '$0.01',
  });
  clickCopy(win);
  await settle();
  const text = getClipboard();
  assert.ok(
    text.includes('plain output line 1\nline 2 <not-a-tag>'),
    'a plain-text result copies its text unchanged; got: ' + text,
  );
  win.close();
}

async function testFailedStatusLineIsCopiedTooAndChromeIsNot() {
  const {win, getClipboard} = makePage();
  send(win, {
    type: 'result',
    success: false,
    is_continue: false,
    summary: '<p>it broke</p>',
    total_tokens: 5,
    cost: '$0.01',
  });
  clickCopy(win);
  await settle();
  const text = getClipboard();
  assert.ok(
    text.includes('Status: FAILED'),
    'the status line is part of the copied text; got: ' + text,
  );
  assert.ok(text.includes('it broke'));
  assert.ok(
    !/<\/?p\b/.test(text),
    'BUG: clipboard must not contain raw HTML tags; got: ' + text,
  );
  win.close();
}

async function testEmptyAndImageOnlySummariesNeverCopyRawHtml() {
  // An HTML summary whose formatted text is empty (or reduces to an
  // image's alt text) must still never fall back to the raw markup.
  const cases = [
    {summary: '<p></p>', expect: ''},
    {summary: '<br>', expect: ''},
    {summary: '<img src="x" alt="diagram">', expect: 'diagram'},
  ];
  for (const c of cases) {
    const {win, getClipboard} = makePage();
    send(win, {
      type: 'result',
      success: true,
      is_continue: false,
      summary: c.summary,
      total_tokens: 1,
      cost: '$0.01',
    });
    clickCopy(win);
    await settle();
    const text = getClipboard();
    assert.ok(
      !/<[a-z][^>]*>/i.test(text),
      'BUG: raw HTML must never reach the clipboard for ' +
        JSON.stringify(c.summary) +
        '; got: ' +
        text,
    );
    assert.strictEqual(
      text,
      c.expect,
      'formatted text for ' + JSON.stringify(c.summary),
    );
    win.close();
  }
}

function testFormattedTextFromNodeBranches() {
  const {win} = makePage();
  const fmt = win.PanelCopy.formattedTextFromNode;
  const doc = win.document;
  const el = doc.createElement('div');
  el.innerHTML =
    '<p>a<br>b</p><hr>' +
    '<ul><li>top<ul><li>nested</li></ul></li></ul>' +
    '<script>bad()</script>' +
    '<div class="panel-copy-btn">chrome</div>' +
    '<ol start="x"><li>fallback-one</li></ol>' +
    '<ol reversed><li>second</li><li>first</li></ol>' +
    '<ol start="5" reversed><li>five</li><li>four</li></ol>' +
    '<ol><li value="4">four</li><li>five</li><li value="x">six</li></ol>';
  const text = fmt(el).replace(/^\n+|\n+$/g, '');
  assert.ok(text.includes('a\nb'), '<br> becomes a newline; got: ' + text);
  assert.ok(text.includes('---'), '<hr> becomes a rule line');
  assert.ok(text.includes('- top'), 'outer li gets a marker');
  assert.ok(text.includes('  - nested'), 'nested li indents two spaces');
  assert.ok(!text.includes('bad()'), 'script content is skipped');
  assert.ok(!text.includes('chrome'), 'panel chrome (copy button) skipped');
  assert.ok(
    text.includes('1. fallback-one'),
    'a non-numeric ol start falls back to 1; got: ' + text,
  );
  // The numbers the browser shows: a reversed list counts down from its
  // item count (or its start), an li value restarts the count.
  assert.ok(text.includes('2. second\n1. first'), 'reversed; got: ' + text);
  assert.ok(text.includes('5. five\n4. four'), 'reversed start; got: ' + text);
  assert.ok(
    text.includes('4. four\n5. five\n6. six'),
    'li value restarts the count; got: ' + text,
  );
  win.close();
}

function testNestedListAsAnItemsFirstChildStartsItsOwnLine() {
  // A nested list that is the first thing in its item (nothing, or
  // only whitespace, before it) must start on its own indented line:
  // the parent's marker and the child's marker are never joined on one
  // line as "-   - nested" / "1.   1. nested".  The item with text
  // before its child list keeps its shape.
  const {win} = makePage();
  const fmt = win.PanelCopy.formattedTextFromNode;
  const doc = win.document;
  const cases = [
    {
      html: '<ul><li><ul><li>nested</li></ul></li><li>next</li></ul>',
      expect: '-\n\n  - nested\n\n- next',
    },
    {
      html: '<ol><li><ol><li>nested</li></ol></li></ol>',
      expect: '1.\n\n  1. nested',
    },
    {
      html: '<ul><li> <ul><li>nested</li></ul></li></ul>',
      expect: '-\n\n  - nested',
    },
    {
      html: '<ul><li><ul><li>nested</li></ul><p>after</p></li></ul>',
      expect: '-\n\n  - nested\n\nafter',
    },
    {
      html: '<ul><li>top<ul><li>nested</li></ul></li></ul>',
      expect: '- top\n\n  - nested',
    },
  ];
  for (const c of cases) {
    const el = doc.createElement('div');
    el.innerHTML = c.html;
    const text = fmt(el).replace(/^\n+|\n+$/g, '');
    assert.strictEqual(
      text,
      c.expect,
      'BUG: nested list markers joined on one line for ' + c.html,
    );
  }
  win.close();
}

async function main() {
  await testHtmlSummaryCopiesFormattedText();
  console.log('  ok - HTML summary copies as formatted text, not raw HTML');
  await testPlainTextResultStillCopiesItsText();
  console.log('  ok - plain-text result still copies its text unchanged');
  await testFailedStatusLineIsCopiedTooAndChromeIsNot();
  console.log('  ok - status line copied, HTML tags still absent');
  await testEmptyAndImageOnlySummariesNeverCopyRawHtml();
  console.log('  ok - empty/image-only summaries never copy raw HTML');
  testFormattedTextFromNodeBranches();
  console.log('  ok - formattedTextFromNode branch behaviours');
  testNestedListAsAnItemsFirstChildStartsItsOwnLine();
  console.log('  ok - a nested list first in its item starts its own line');
  console.log('resultPanelFormattedCopy.test.js: all tests passed');
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
