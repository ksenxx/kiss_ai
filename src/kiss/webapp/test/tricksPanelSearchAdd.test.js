// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// End-to-end (JSDOM) tests for the Inject promptlet panel's search box
// and its "new promptlet" box + Add button: filtering of #tricks-list,
// the `addTrick` command posted to the host, and the `tricksData`
// reply that reloads the list.

const assert = require('assert');
const {makePage, send, typeInto: input} = require('./simplify2_harness');

function rows(doc) {
  return Array.from(doc.getElementById('tricks-list').children).map(
    el => el.textContent,
  );
}

function run() {
  const {win, posted} = makePage();
  const doc = win.document;
  win.__TRICKS__ = ['Alpha promptlet', 'Beta promptlet', 'gamma ALPHA'];

  doc.getElementById('tricks-btn').click();
  const search = doc.getElementById('tricks-search');
  const clear = doc.getElementById('tricks-search-clear');
  const addInput = doc.getElementById('tricks-add-input');
  const addBtn = doc.getElementById('tricks-add-btn');
  const list = doc.getElementById('tricks-list');
  assert.ok(search && clear && addInput && addBtn, 'panel controls exist');
  assert.ok(
    search.compareDocumentPosition(addInput) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the add box sits below the search box',
  );
  assert.ok(
    addInput.compareDocumentPosition(addBtn) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the Add button follows (sits right of) the add box',
  );
  assert.ok(
    addBtn.compareDocumentPosition(list) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the list comes after the add row',
  );
  assert.strictEqual(addBtn.textContent.trim(), 'Add');
  assert.deepStrictEqual(rows(doc), win.__TRICKS__, 'all promptlets listed');
  assert.strictEqual(clear.style.display, 'none', 'clear hidden while empty');

  // --- search filters case-insensitively and shows the clear button
  input(win, search, '  alpha');
  assert.deepStrictEqual(rows(doc), ['Alpha promptlet', 'gamma ALPHA']);
  assert.strictEqual(clear.style.display, '', 'clear button shown');
  input(win, search, 'zzz');
  assert.strictEqual(list.textContent.trim(), 'No matching promptlets');
  assert.strictEqual(list.querySelectorAll('.sidebar-empty').length, 1);
  clear.click();
  assert.strictEqual(search.value, '');
  assert.strictEqual(clear.style.display, 'none');
  assert.deepStrictEqual(rows(doc), win.__TRICKS__, 'clear restores list');
  assert.strictEqual(doc.activeElement, search, 'clear refocuses search');

  // --- a filtered row still injects into the composer on click
  input(win, search, 'beta');
  list.children[0].click();
  assert.ok(
    doc.getElementById('task-input').value.includes('Beta promptlet'),
    'clicking a filtered row injects it into the prompt',
  );
  clear.click();

  // --- Add button enabled only with non-blank text
  assert.ok(addBtn.disabled, 'Add disabled while empty');
  input(win, addInput, '   ');
  assert.ok(addBtn.disabled, 'Add disabled for whitespace');
  const before = posted.length;
  addBtn.click();
  assert.strictEqual(posted.length, before, 'disabled Add posts nothing');
  addInput.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
  assert.strictEqual(posted.length, before, 'Enter on blank posts nothing');

  // --- Add click posts addTrick with the trimmed text
  input(win, addInput, '  Delta promptlet  ');
  assert.ok(!addBtn.disabled, 'Add enabled with text');
  addBtn.click();
  const added = posted.filter(m => m.type === 'addTrick');
  // JSON round-trip: posted objects come from the JSDOM realm, whose
  // Object.prototype differs from Node's, which deepStrictEqual rejects.
  assert.deepStrictEqual(JSON.parse(JSON.stringify(added)), [
    {type: 'addTrick', text: 'Delta promptlet'},
  ]);
  // The box keeps its text until the daemon's tricksData lists the new
  // promptlet, then clears: ui_antipattern_drafts.test.js.
  assert.strictEqual(doc.activeElement, addInput, 'focus stays in the box');

  // --- Enter in the box also submits
  input(win, addInput, 'Epsilon');
  addInput.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
  );
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'addTrick').map(m => m.text),
    ['Delta promptlet', 'Epsilon'],
  );
  // Other keys do not submit.
  input(win, addInput, 'Zeta');
  addInput.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: 'a', bubbles: true}),
  );
  assert.strictEqual(posted.filter(m => m.type === 'addTrick').length, 2);
  assert.strictEqual(addInput.value, 'Zeta', 'non-Enter key keeps text');

  // --- tricksData reloads the list (honouring the current search)
  input(win, search, 'delta');
  assert.strictEqual(list.textContent.trim(), 'No matching promptlets');
  const reloaded = [...win.__TRICKS__, 'Delta promptlet', 'Epsilon'];
  send(win, {type: 'tricksData', tricks: reloaded});
  assert.deepStrictEqual(
    JSON.parse(JSON.stringify(win.__TRICKS__)),
    reloaded,
    'window.__TRICKS__ replaced',
  );
  assert.deepStrictEqual(rows(doc), ['Delta promptlet'], 'filter applied to reload');
  clear.click();
  assert.deepStrictEqual(rows(doc), reloaded, 'full reloaded list shown');

  // Re-opening the panel renders from the reloaded list, not the
  // page-load one.
  doc.getElementById('tricks-overlay').click();
  doc.getElementById('tricks-btn').click();
  assert.deepStrictEqual(rows(doc), reloaded);

  // A malformed tricksData emptying the list: tricksPanelCopyDelete.test.js.

  // --- the daemon's error reply surfaces in the error banner
  send(win, {type: 'error', text: 'Promptlet must not be empty'});
  assert.ok(
    doc.body.textContent.includes('Promptlet must not be empty'),
    'error event text is shown to the user',
  );

  // Close the page so no timer main.js armed keeps node alive.
  win.close();
  console.log('tricksPanelSearchAdd.test.js passed');
}

run();
