// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// End-to-end (JSDOM) tests for the Inject promptlet panel's edit
// (pencil) button: the first `window.__MY_TRICKS_COUNT__` rows (the
// user's own, from ~/.kiss/MY_INJECTION.md) carry it between the copy
// and delete buttons; pressing it turns the row into a textarea with
// Save / Cancel, and saving posts `editTrick` {text, newText} while
// showing the new text at once.  Also covers the keyboard (Enter saves,
// Shift+Enter breaks a line, Escape cancels), the no-op paths (unchanged
// or emptied text), the resync after a rejected edit, and that an open
// editor is closed when the list changes under it.

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness');

function rows(doc) {
  return Array.from(doc.getElementById('tricks-list').children);
}

function texts(doc) {
  return rows(doc).map(el => {
    const t = el.querySelector('.sidebar-item-text');
    return t ? t.textContent : el.querySelector('.tricks-edit-input').value;
  });
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true, cancelable: true}));
}

function key(win, el, keyName, init) {
  el.dispatchEvent(
    new win.KeyboardEvent('keydown', {key: keyName, bubbles: true, cancelable: true, ...init}),
  );
}

function input(win, el, value) {
  el.value = value;
  el.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function editor(doc, row) {
  return row.querySelector('.tricks-edit-input');
}

function editsPosted(posted) {
  return JSON.parse(JSON.stringify(posted.filter(m => m.type === 'editTrick')));
}

function runPanel() {
  const {win, posted} = makeWebview();
  const doc = win.document;
  win.__TRICKS__ = ['Mine one', 'Mine two', 'Bundled alpha', 'Bundled beta'];
  win.__MY_TRICKS_COUNT__ = 2;

  doc.getElementById('tricks-btn').click();
  const list = doc.getElementById('tricks-list');
  const panel = doc.getElementById('tricks-panel');
  const composer = doc.getElementById('task-input');
  assert.deepStrictEqual(texts(doc), win.__TRICKS__, 'all promptlets listed');

  // --- only the user-owned rows have a pencil, between copy and delete
  assert.deepStrictEqual(
    rows(doc).map(r => r.querySelectorAll('.sidebar-item-edit').length),
    [1, 1, 0, 0],
    'edit only on the first userCount rows',
  );
  const first = rows(doc)[0];
  const editBtn = first.querySelector('.sidebar-item-edit');
  assert.strictEqual(editBtn.getAttribute('aria-label'), 'Edit promptlet');
  assert.strictEqual(editBtn.dataset.tooltip, 'Edit promptlet');
  assert.strictEqual(editBtn.type, 'button');
  assert.ok(editBtn.querySelector('svg'), 'edit button shows the pencil icon');
  const copyBtn = first.querySelector('.sidebar-item-copy');
  const delBtn = first.querySelector('.sidebar-item-delete');
  assert.ok(
    copyBtn.compareDocumentPosition(editBtn) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the pencil sits right of the copy button',
  );
  assert.ok(
    editBtn.compareDocumentPosition(delBtn) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the pencil sits left of (next to) the delete button',
  );

  // --- pressing the pencil turns the row into an in-place editor
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  assert.strictEqual(composer.value, '', 'the pencil does not inject the promptlet');
  assert.ok(panel.classList.contains('open'), 'the panel stays open');
  let row = rows(doc)[1];
  assert.ok(row.classList.contains('editing'));
  let ta = editor(doc, row);
  assert.ok(ta, 'the row holds a textarea');
  assert.strictEqual(ta.tagName, 'TEXTAREA');
  assert.strictEqual(ta.value, 'Mine two', 'prefilled with the promptlet');
  assert.strictEqual(ta.getAttribute('aria-label'), 'Edit promptlet');
  assert.strictEqual(doc.activeElement, ta, 'the textarea takes focus');
  assert.strictEqual(ta.selectionStart, 'Mine two'.length, 'caret at the end');
  assert.strictEqual(row.querySelector('.tricks-edit-save').textContent, 'Save');
  assert.strictEqual(row.querySelector('.tricks-edit-cancel').textContent, 'Cancel');
  assert.strictEqual(row.querySelectorAll('.sidebar-item-text').length, 0);
  assert.strictEqual(row.querySelectorAll('.sidebar-item-copy').length, 0);
  assert.strictEqual(row.querySelectorAll('.sidebar-item-edit').length, 0);
  assert.strictEqual(row.querySelectorAll('.sidebar-item-delete').length, 0);
  assert.strictEqual(
    list.querySelectorAll('.tricks-edit-input').length,
    1,
    'the other rows are untouched',
  );
  assert.deepStrictEqual(texts(doc), ['Mine one', 'Mine two', 'Bundled alpha', 'Bundled beta']);

  // --- clicking inside the editor does not inject the promptlet
  click(win, ta);
  click(win, row);
  assert.strictEqual(composer.value, '');
  assert.ok(panel.classList.contains('open'));
  assert.ok(rows(doc)[1].classList.contains('editing'), 'still editing');

  // --- Save posts editTrick {text, newText} and shows the new text now
  input(win, ta, '  Mine two, edited  ');
  click(win, row.querySelector('.tricks-edit-save'));
  assert.deepStrictEqual(editsPosted(posted), [
    {type: 'editTrick', text: 'Mine two', newText: 'Mine two, edited'},
  ]);
  assert.deepStrictEqual(texts(doc), [
    'Mine one',
    'Mine two, edited',
    'Bundled alpha',
    'Bundled beta',
  ]);
  assert.deepStrictEqual(JSON.parse(JSON.stringify(win.__TRICKS__)), [
    'Mine one',
    'Mine two, edited',
    'Bundled alpha',
    'Bundled beta',
  ]);
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0, 'editor closed');
  assert.strictEqual(win.__MY_TRICKS_COUNT__, 2, 'an edit keeps the user count');
  assert.deepStrictEqual(
    rows(doc).map(r => r.querySelectorAll('.sidebar-item-edit').length),
    [1, 1, 0, 0],
    'the edited row keeps its pencil',
  );
  assert.strictEqual(rows(doc)[1].dataset.tooltip, 'Mine two, edited');
  assert.strictEqual(composer.value, '', 'save does not inject');
  assert.ok(panel.classList.contains('open'), 'save keeps the panel open');
  assert.strictEqual(posted.filter(m => m.type === 'deleteTrick').length, 0);
  assert.strictEqual(posted.filter(m => m.type === 'addTrick').length, 0);

  // --- the daemon's answer (unstamped tricksData) repaints, editor stays closed
  send(win, {
    type: 'tricksData',
    tricks: ['Mine one', 'Mine two, edited', 'Bundled alpha', 'Bundled beta'],
    userCount: 2,
  });
  assert.deepStrictEqual(texts(doc), [
    'Mine one',
    'Mine two, edited',
    'Bundled alpha',
    'Bundled beta',
  ]);
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- Enter saves; the new text is what the daemon later broadcasts
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[0]);
  input(win, ta, 'Mine one, by Enter');
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted).slice(1), [
    {type: 'editTrick', text: 'Mine one', newText: 'Mine one, by Enter'},
  ]);
  assert.deepStrictEqual(texts(doc)[0], 'Mine one, by Enter');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- Shift+Enter breaks a line instead of saving
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[0]);
  const shiftEnter = new win.KeyboardEvent('keydown', {
    key: 'Enter',
    shiftKey: true,
    bubbles: true,
    cancelable: true,
  });
  ta.dispatchEvent(shiftEnter);
  assert.ok(!shiftEnter.defaultPrevented, 'Shift+Enter is left to the textarea');
  assert.strictEqual(editsPosted(posted).length, 2, 'nothing posted');
  assert.ok(rows(doc)[0].classList.contains('editing'), 'still editing');
  input(win, ta, 'Line one\nLine two');
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted)[2], {
    type: 'editTrick',
    text: 'Mine one, by Enter',
    newText: 'Line one\nLine two',
  });
  assert.strictEqual(texts(doc)[0], 'Line one\nLine two');

  // --- Escape cancels: nothing posted, the old text stays
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[0]);
  input(win, ta, 'thrown away');
  key(win, ta, 'Escape');
  assert.strictEqual(editsPosted(posted).length, 3, 'Escape posts nothing');
  assert.strictEqual(texts(doc)[0], 'Line one\nLine two');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- the Cancel button does the same
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[0]);
  input(win, ta, 'also thrown away');
  click(win, rows(doc)[0].querySelector('.tricks-edit-cancel'));
  assert.strictEqual(editsPosted(posted).length, 3, 'Cancel posts nothing');
  assert.strictEqual(texts(doc)[0], 'Line one\nLine two');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  assert.strictEqual(composer.value, '');

  // --- saving unchanged (or only re-spaced) text posts nothing
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[1]);
  input(win, ta, '  Mine two, edited\n');
  click(win, rows(doc)[1].querySelector('.tricks-edit-save'));
  assert.strictEqual(editsPosted(posted).length, 3, 'unchanged text posts nothing');
  assert.strictEqual(texts(doc)[1], 'Mine two, edited');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- saving emptied text posts nothing and keeps the promptlet
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[1]);
  input(win, ta, '   ');
  key(win, ta, 'Enter');
  assert.strictEqual(editsPosted(posted).length, 3, 'empty text posts nothing');
  assert.strictEqual(texts(doc)[1], 'Mine two, edited');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- typing in the search box repaints the list but keeps the open
  // editor, its draft and the search box's focus
  const search = doc.getElementById('tricks-search');
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[0]);
  input(win, ta, 'My unsaved draft');
  search.focus();
  input(win, search, 'line');
  assert.deepStrictEqual(texts(doc), ['My unsaved draft'], 'filtered on the stored text');
  assert.strictEqual(editor(doc, rows(doc)[0]), ta, 'the same editor element is re-attached');
  assert.strictEqual(ta.value, 'My unsaved draft', 'the draft survives the repaint');
  assert.strictEqual(doc.activeElement, search, 'focus stays in the search box');
  assert.strictEqual(editsPosted(posted).length, 3, 'nothing posted');
  input(win, search, 'bundled');
  assert.deepStrictEqual(texts(doc), ['Bundled alpha', 'Bundled beta'], 'editor row filtered out');
  input(win, search, '');
  assert.strictEqual(editor(doc, rows(doc)[0]), ta, 'the editor comes back with its draft');
  assert.strictEqual(ta.value, 'My unsaved draft');
  key(win, ta, 'Escape');
  assert.strictEqual(texts(doc)[0], 'Line one\nLine two');

  // --- Enter / Escape confirming an IME composition are left to the IME
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[0]);
  input(win, ta, '変換中');
  for (const init of [{isComposing: true}, {keyCode: 229}]) {
    const composingEnter = new win.KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
      ...init,
    });
    ta.dispatchEvent(composingEnter);
    assert.ok(!composingEnter.defaultPrevented, 'composition Enter is not swallowed');
    key(win, ta, 'Escape', init);
    assert.ok(rows(doc)[0].classList.contains('editing'), 'still editing');
  }
  assert.strictEqual(editsPosted(posted).length, 3, 'composition keys post nothing');
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted)[3], {
    type: 'editTrick',
    text: 'Line one\nLine two',
    newText: '変換中',
  });
  send(win, {
    type: 'tricksData',
    tricks: ['Line one\nLine two', 'Mine two, edited', 'Bundled alpha', 'Bundled beta'],
    userCount: 2,
  });
  posted.splice(posted.findIndex(m => m.type === 'editTrick' && m.newText === '変換中'), 1);
  assert.strictEqual(editsPosted(posted).length, 3);

  // --- opening a second editor closes the first (one at a time)
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  assert.ok(rows(doc)[0].classList.contains('editing'));
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  assert.ok(!rows(doc)[0].classList.contains('editing'));
  assert.ok(rows(doc)[1].classList.contains('editing'));
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 1);
  key(win, editor(doc, rows(doc)[1]), 'Escape');

  // --- a rejected edit: the daemon answers `error` + the list on disk,
  // stamped for this window, and the old text comes back
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  input(win, editor(doc, rows(doc)[1]), 'Line one\nLine two');
  click(win, rows(doc)[1].querySelector('.tricks-edit-save'));
  assert.deepStrictEqual(editsPosted(posted)[3], {
    type: 'editTrick',
    text: 'Mine two, edited',
    newText: 'Line one\nLine two',
  });
  assert.strictEqual(texts(doc)[1], 'Line one\nLine two', 'shown optimistically');
  send(win, {
    type: 'error',
    text: 'That promptlet is already in ~/.kiss/MY_INJECTION.md',
    connId: 'c1',
  });
  send(win, {
    type: 'tricksData',
    tricks: ['Line one\nLine two', 'Mine two, edited', 'Bundled alpha', 'Bundled beta'],
    userCount: 2,
    connId: 'c1',
  });
  assert.deepStrictEqual(texts(doc), [
    'Line one\nLine two',
    'Mine two, edited',
    'Bundled alpha',
    'Bundled beta',
  ]);
  assert.deepStrictEqual(
    rows(doc).map(r => r.querySelectorAll('.sidebar-item-edit').length),
    [1, 1, 0, 0],
  );

  // --- a list change from elsewhere closes an open editor: its index
  // could now point at another promptlet
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  input(win, editor(doc, rows(doc)[1]), 'typing...');
  send(win, {
    type: 'tricksData',
    tricks: ['Mine two, edited', 'Bundled alpha', 'Bundled beta'],
    userCount: 1,
  });
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  assert.deepStrictEqual(texts(doc), ['Mine two, edited', 'Bundled alpha', 'Bundled beta']);
  assert.deepStrictEqual(
    rows(doc).map(r => r.querySelectorAll('.sidebar-item-edit').length),
    [1, 0, 0],
  );
  // ...and so does deleting another row from this window
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B', 'Bundled alpha'],
    userCount: 2,
  });
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  assert.ok(rows(doc)[1].classList.contains('editing'));
  click(win, rows(doc)[0].querySelector('.sidebar-item-delete'));
  assert.deepStrictEqual(texts(doc), ['Mine B', 'Bundled alpha']);
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  assert.strictEqual(rows(doc)[0].querySelectorAll('.sidebar-item-edit').length, 1);
  assert.strictEqual(editsPosted(posted).length, 4, 'closing posts no edit');

  // --- a filtered list edits the right promptlet (index into the full list)
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B', 'Bundled alpha'],
    userCount: 2,
  });
  input(win, doc.getElementById('tricks-search'), 'mine b');
  assert.deepStrictEqual(texts(doc), ['Mine B']);
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(doc, rows(doc)[0]);
  assert.strictEqual(ta.value, 'Mine B');
  input(win, ta, 'Mine B2');
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted)[4], {
    type: 'editTrick',
    text: 'Mine B',
    newText: 'Mine B2',
  });
  assert.deepStrictEqual(texts(doc), ['Mine B2']);
  doc.getElementById('tricks-search-clear').click();
  assert.deepStrictEqual(texts(doc), ['Mine A', 'Mine B2', 'Bundled alpha']);

  // --- clicking a non-editing row still injects the promptlet
  click(win, rows(doc)[2]);
  assert.strictEqual(composer.value, 'Bundled alpha');
  assert.ok(!panel.classList.contains('open'));

  // --- a page loaded without the count global shows no pencils
  const bare = makeWebview();
  bare.win.__TRICKS__ = ['a', 'b'];
  bare.win.document.getElementById('tricks-btn').click();
  const bareList = bare.win.document.getElementById('tricks-list');
  assert.strictEqual(bareList.querySelectorAll('.sidebar-item-copy').length, 2);
  assert.strictEqual(bareList.querySelectorAll('.sidebar-item-edit').length, 0);

  console.log('tricksPanelEdit.test.js passed');
}

runPanel();
