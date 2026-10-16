// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// DOM helpers shared by the run_parallel / run_agent sub-agent tab suites
// (page built by simplify2_harness.makePage, or by any bootstrap that
// evals media/main.js into chat.html).

const assert = require('assert');
const {send} = require('./simplify2_harness');

/** Every sub-agent tab rendered in the tab bar, in DOM order. */
function subagentTabEls(win) {
  return Array.from(
    win.document.querySelectorAll('#tab-list .chat-tab.subagent-tab'),
  );
}

/** The first tool-call panel whose header names run_parallel, or null. */
function runParallelPanel(win) {
  const headers = win.document.querySelectorAll('#output .ev.tc .tc-h');
  for (const h of headers) {
    const txt = (h.textContent || '').replace(/^[^A-Za-z]+/, '').trim();
    if (txt.startsWith('run_parallel')) return h.closest('.ev.tc');
  }
  return null;
}

/** Click *panel*'s header chevron (collapse or expand it). */
function togglePanel(win, panel) {
  panel
    .querySelector('.tc-h')
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** The user clicks the close button of tab *tabId*. */
function clickClose(win, tabId) {
  const btn = win.document.querySelector(
    `#tab-list .chat-tab[data-tab-id="${tabId}"] .chat-tab-close`,
  );
  assert.ok(btn, 'sub-agent tab ' + tabId + ' must render a close button');
  btn.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

/** The daemon reports every sub-agent in *tabIds* finished. */
function finishFanOut(win, tabIds) {
  for (const id of tabIds) send(win, {type: 'subagentDone', tab_id: id});
}

module.exports = {
  subagentTabEls,
  runParallelPanel,
  togglePanel,
  clickClose,
  finishFanOut,
};
