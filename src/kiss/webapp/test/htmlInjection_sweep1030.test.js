// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for model-controlled markup reaching the chat
// page (media/main.js):
// * kissSanitize drops SVG <script>/<style> (an SVG element's tagName
//   keeps its written case, so an upper-case block list missed them) and
//   the SVG <animate>/<set> elements that rewrite a link's href to a
//   javascript: URL at run time;
// * a tool_call's `lang` (the raw file extension when the printer has no
//   name for it) cannot close the code element's class attribute and add
//   an event handler.

'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');

function answerPage() {
  const {win} = makePage({
    beforeScripts: stubRaf,
    extraScripts: ['marked.min.js'],
  });
  return {win, tab: win._testApi.getActiveTabId()};
}

/** Stream *text* as the agent's answer in the active tab and end it. */
function streamAnswer(win, tab, text) {
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {type: 'text_delta', text, tabId: tab});
  send(win, {type: 'text_end', tabId: tab});
  const txt = win.document.querySelectorAll('#output .txt');
  return txt[txt.length - 1];
}

function testSvgScriptAndStyleDropped() {
  const {win, tab} = answerPage();
  const el = streamAnswer(
    win,
    tab,
    'Here:\n\n<svg><script>window.pwned = 1</script>' +
      '<style>body{display:none}</style><text x="5" y="5">kept</text></svg>',
  );
  assert.ok(el, 'the answer was rendered');
  assert.strictEqual(
    el.querySelector('svg text').textContent,
    'kept',
    'the harmless SVG content stays',
  );
  assert.strictEqual(
    el.querySelectorAll('script, style').length,
    0,
    'no SVG <script> or <style> survives: ' + el.innerHTML,
  );
  win.close();
  console.log('  ok - SVG <script> and <style> are stripped from an answer');
}

function testSvgAnimationHrefDropped() {
  const {win, tab} = answerPage();
  const el = streamAnswer(
    win,
    tab,
    'Look:\n\n<svg><a><animate attributeName="href" ' +
      'values="javascript:alert(1)"/><set attributeName="xlink:href" ' +
      'to="javascript:alert(2)"/><text x="20" y="20">Click me</text></a></svg>',
  );
  assert.strictEqual(
    el.querySelector('svg a text').textContent,
    'Click me',
    'the link text stays',
  );
  assert.strictEqual(
    el.querySelectorAll('animate, set').length,
    0,
    'no <animate>/<set> can retarget the link: ' + el.innerHTML,
  );
  assert.ok(!/javascript:/i.test(el.innerHTML), 'no javascript: URL left');
  win.close();
  console.log('  ok - SVG <animate>/<set> href rewrites are stripped');
}

function testHtmlScriptStillDropped() {
  const {win, tab} = answerPage();
  const el = streamAnswer(
    win,
    tab,
    'x\n\n<div><script>window.pwned = 1</script><b onclick="x()">bold</b></div>',
  );
  assert.strictEqual(el.querySelectorAll('script').length, 0);
  assert.strictEqual(el.querySelector('b').textContent, 'bold');
  assert.ok(!el.querySelector('b').hasAttribute('onclick'));
  win.close();
  console.log('  ok - HTML <script> and on* attributes are still stripped');
}

function testToolCallLangCannotAddAttributes() {
  const {win, tab} = answerPage();
  const lang = 'x" onmouseover="window.pwned=1';
  send(win, {type: 'status', running: true, tabId: tab});
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: '/tmp/a.' + lang,
    lang,
    content: 'hi',
    tabId: tab,
  });
  const code = win.document.querySelector('#output .ev.tc .tc-b pre code');
  assert.ok(code, 'the content row was rendered');
  assert.strictEqual(code.textContent, 'hi');
  assert.ok(
    !code.hasAttribute('onmouseover'),
    'the lang added an attribute: ' + code.outerHTML,
  );
  assert.strictEqual(code.getAttribute('class'), 'language-' + lang);
  win.close();
  console.log("  ok - a tool_call's lang stays inside the class attribute");
}

function main() {
  console.log('htmlInjection_sweep1030');
  testSvgScriptAndStyleDropped();
  testSvgAnimationHrefDropped();
  testHtmlScriptStillDropped();
  testToolCallLangCannotAddAttributes();
}

main();
