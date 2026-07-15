// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Regression tests for the slide-up panels (Settings / Frequent tasks /
// Tricks) in media/main.js.  Locks in the exact open/close behavior —
// 'open' class on panel + overlay, the backend request each open
// triggers — so the duplicated open*/close* helpers can be unified
// without any visible change.
//
//     node src/kiss/agents/vscode/test/simplify2_panels.test.js
'use strict';

const assert = require('assert');
const {makeWebview} = require('./simplify2_harness');

function isOpen(win, id) {
  return win.document.getElementById(id).classList.contains('open');
}

function run() {
  // The Frequent-tasks toggle button is provided by the remote
  // webapp's HTML (not by chat.html); add it before main.js boots so
  // the same wiring path is exercised.
  const {win, posted} = makeWebview({
    beforeScripts(w) {
      const btn = w.document.createElement('button');
      btn.id = 'frequent-tasks-btn';
      w.document.body.appendChild(btn);
    },
  });
  const doc = win.document;

  // --- Tricks panel: toggle button opens, overlay click closes. ---
  win.__TRICKS__ = ['do the thing'];
  const tricksBtn = doc.getElementById('tricks-btn');
  tricksBtn.click();
  assert.ok(isOpen(win, 'tricks-panel'), 'tricks panel opens');
  assert.ok(isOpen(win, 'tricks-overlay'), 'tricks overlay opens');
  assert.ok(
    doc.getElementById('tricks-list').textContent.includes('do the thing'),
    'tricks list renders window.__TRICKS__',
  );
  doc.getElementById('tricks-overlay').click();
  assert.ok(!isOpen(win, 'tricks-panel'), 'overlay click closes tricks panel');
  assert.ok(!isOpen(win, 'tricks-overlay'), 'tricks overlay closes');
  tricksBtn.click();
  assert.ok(isOpen(win, 'tricks-panel'), 'tricks re-opens');
  tricksBtn.click();
  assert.ok(!isOpen(win, 'tricks-panel'), 'toggle button closes tricks panel');

  // --- Frequent tasks panel: open requests tasks from the backend. ---
  const frequentBtn = doc.getElementById('frequent-tasks-btn');
  frequentBtn.click();
  assert.ok(isOpen(win, 'frequent-panel'), 'frequent panel opens');
  assert.ok(isOpen(win, 'frequent-overlay'), 'frequent overlay opens');
  assert.ok(
    posted.some(m => m.type === 'getFrequentTasks' && m.limit === 50),
    'opening the frequent panel requests {getFrequentTasks, limit: 50}',
  );
  doc.getElementById('frequent-panel-close').click();
  assert.ok(!isOpen(win, 'frequent-panel'), 'close button closes panel');
  assert.ok(!isOpen(win, 'frequent-overlay'), 'frequent overlay closes');

  // --- Settings panel: gear tab opens (requests config), X closes. ---
  const gear = doc.querySelector('.chat-tab-settings');
  assert.ok(gear, 'tab bar renders the settings gear button');
  gear.click();
  assert.ok(isOpen(win, 'settings-panel'), 'settings panel opens');
  assert.ok(isOpen(win, 'settings-overlay'), 'settings overlay opens');
  assert.ok(
    posted.some(m => m.type === 'getConfig'),
    'opening the settings panel requests {getConfig}',
  );
  const before = posted.length;
  doc.getElementById('settings-panel-close').click();
  assert.ok(!isOpen(win, 'settings-panel'), 'X closes settings panel');
  assert.ok(!isOpen(win, 'settings-overlay'), 'settings overlay closes');
  assert.ok(
    !posted.slice(before).some(m => m.type === 'saveConfig'),
    'closing before the config form is populated must not saveConfig',
  );

  win.close();
  console.log('  ok - settings/frequent/tricks panels open and close');
}

run();
console.log('simplify2_panels.test.js: all tests passed');
