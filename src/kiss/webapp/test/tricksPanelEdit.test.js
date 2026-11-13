// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// End-to-end (JSDOM) tests for the Inject promptlet panel's edit
// (pencil) button: the first `window.__MY_TRICKS_COUNT__` rows (the
// user's own, from the brand home's MY_INJECTION.md) carry it between the copy
// and delete buttons; pressing it turns the row into a textarea with
// Save / Cancel, and saving posts `editTrick` {text, newText} while
// showing the new text at once.  Also covers the keyboard (Enter saves,
// Shift+Enter breaks a line, Escape cancels), the no-op paths (unchanged
// or emptied text), the resync after a rejected edit, and that an open
// editor is closed when the list changes under it.

const assert = require('assert');
const {
  makePage,
  send,
  click,
  typeInto: input,
  promptletRows: rows,
} = require('./simplify2_harness');
const {key, IME_FORMS} = require('./ui_antipattern_harness');

function texts(doc) {
  return rows(doc).map(el => {
    const t = el.querySelector('.sidebar-item-text');
    return t ? t.textContent : el.querySelector('.tricks-edit-input').value;
  });
}

function editor(row) {
  return row.querySelector('.tricks-edit-input');
}

function editsPosted(posted) {
  return JSON.parse(JSON.stringify(posted.filter(m => m.type === 'editTrick')));
}

function runPanel() {
  const {win, posted} = makePage();
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
    copyBtn.compareDocumentPosition(editBtn) &
      win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the pencil sits right of the copy button',
  );
  assert.ok(
    editBtn.compareDocumentPosition(delBtn) &
      win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the pencil sits left of (next to) the delete button',
  );

  // --- pressing the pencil turns the row into an in-place editor
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  assert.strictEqual(
    composer.value,
    '',
    'the pencil does not inject the promptlet',
  );
  assert.ok(panel.classList.contains('open'), 'the panel stays open');
  const row = rows(doc)[1];
  assert.ok(row.classList.contains('editing'));
  let ta = editor(row);
  assert.ok(ta, 'the row holds a textarea');
  assert.strictEqual(ta.tagName, 'TEXTAREA');
  assert.strictEqual(ta.value, 'Mine two', 'prefilled with the promptlet');
  assert.strictEqual(ta.getAttribute('aria-label'), 'Edit promptlet');
  assert.strictEqual(doc.activeElement, ta, 'the textarea takes focus');
  assert.strictEqual(ta.selectionStart, 'Mine two'.length, 'caret at the end');
  assert.strictEqual(
    row.querySelector('.tricks-edit-save').textContent,
    'Save',
  );
  assert.strictEqual(
    row.querySelector('.tricks-edit-cancel').textContent,
    'Cancel',
  );
  assert.strictEqual(row.querySelectorAll('.sidebar-item-text').length, 0);
  assert.strictEqual(row.querySelectorAll('.sidebar-item-copy').length, 0);
  assert.strictEqual(row.querySelectorAll('.sidebar-item-edit').length, 0);
  assert.strictEqual(row.querySelectorAll('.sidebar-item-delete').length, 0);
  assert.strictEqual(
    list.querySelectorAll('.tricks-edit-input').length,
    1,
    'the other rows are untouched',
  );
  assert.deepStrictEqual(texts(doc), [
    'Mine one',
    'Mine two',
    'Bundled alpha',
    'Bundled beta',
  ]);

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
  assert.strictEqual(
    list.querySelectorAll('.tricks-edit-input').length,
    0,
    'editor closed',
  );
  assert.strictEqual(
    win.__MY_TRICKS_COUNT__,
    2,
    'an edit keeps the user count',
  );
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
  ta = editor(rows(doc)[0]);
  input(win, ta, 'Mine one, by Enter');
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted).slice(1), [
    {type: 'editTrick', text: 'Mine one', newText: 'Mine one, by Enter'},
  ]);
  assert.deepStrictEqual(texts(doc)[0], 'Mine one, by Enter');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  send(win, {
    type: 'tricksData',
    tricks: [
      'Mine one, by Enter',
      'Mine two, edited',
      'Bundled alpha',
      'Bundled beta',
    ],
    userCount: 2,
  });

  // --- Shift+Enter breaks a line instead of saving
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[0]);
  const shiftEnter = new win.KeyboardEvent('keydown', {
    key: 'Enter',
    shiftKey: true,
    bubbles: true,
    cancelable: true,
  });
  ta.dispatchEvent(shiftEnter);
  assert.ok(
    !shiftEnter.defaultPrevented,
    'Shift+Enter is left to the textarea',
  );
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
  ta = editor(rows(doc)[0]);
  input(win, ta, 'thrown away');
  key(win, ta, 'Escape');
  assert.strictEqual(editsPosted(posted).length, 3, 'Escape posts nothing');
  assert.strictEqual(texts(doc)[0], 'Line one\nLine two');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- the Cancel button does the same
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[0]);
  input(win, ta, 'also thrown away');
  click(win, rows(doc)[0].querySelector('.tricks-edit-cancel'));
  assert.strictEqual(editsPosted(posted).length, 3, 'Cancel posts nothing');
  assert.strictEqual(texts(doc)[0], 'Line one\nLine two');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  assert.strictEqual(composer.value, '');

  // --- saving unchanged (or only re-spaced) text posts nothing
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[1]);
  input(win, ta, '  Mine two, edited\n');
  click(win, rows(doc)[1].querySelector('.tricks-edit-save'));
  assert.strictEqual(
    editsPosted(posted).length,
    3,
    'unchanged text posts nothing',
  );
  assert.strictEqual(texts(doc)[1], 'Mine two, edited');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- saving emptied text posts nothing and keeps the promptlet
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[1]);
  input(win, ta, '   ');
  key(win, ta, 'Enter');
  assert.strictEqual(editsPosted(posted).length, 3, 'empty text posts nothing');
  assert.strictEqual(texts(doc)[1], 'Mine two, edited');
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- typing in the search box repaints the list but keeps the open
  // editor, its draft and the search box's focus
  const search = doc.getElementById('tricks-search');
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[0]);
  input(win, ta, 'My unsaved draft');
  search.focus();
  input(win, search, 'line');
  assert.deepStrictEqual(
    texts(doc),
    ['My unsaved draft'],
    'filtered on the stored text',
  );
  assert.strictEqual(
    editor(rows(doc)[0]),
    ta,
    'the same editor element is re-attached',
  );
  assert.strictEqual(
    ta.value,
    'My unsaved draft',
    'the draft survives the repaint',
  );
  assert.strictEqual(
    doc.activeElement,
    search,
    'focus stays in the search box',
  );
  assert.strictEqual(editsPosted(posted).length, 3, 'nothing posted');
  input(win, search, 'bundled');
  assert.deepStrictEqual(
    texts(doc),
    ['Bundled alpha', 'Bundled beta'],
    'editor row filtered out',
  );
  input(win, search, '');
  assert.strictEqual(
    editor(rows(doc)[0]),
    ta,
    'the editor comes back with its draft',
  );
  assert.strictEqual(ta.value, 'My unsaved draft');
  key(win, ta, 'Escape');
  assert.strictEqual(texts(doc)[0], 'Line one\nLine two');

  // --- Enter / Escape confirming an IME composition are left to the IME
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[0]);
  input(win, ta, '変換中');
  for (const init of IME_FORMS) {
    const composingEnter = new win.KeyboardEvent('keydown', {
      key: 'Enter',
      bubbles: true,
      cancelable: true,
      ...init,
    });
    ta.dispatchEvent(composingEnter);
    assert.ok(
      !composingEnter.defaultPrevented,
      'composition Enter is not swallowed',
    );
    key(win, ta, 'Escape', init);
    assert.ok(rows(doc)[0].classList.contains('editing'), 'still editing');
  }
  assert.strictEqual(
    editsPosted(posted).length,
    3,
    'composition keys post nothing',
  );
  key(win, ta, 'Enter');
  // The row's two-line edit is not answered yet: the composed text is
  // shown, and sent once the daemon says what the row holds.
  assert.strictEqual(texts(doc)[0], '変換中');
  assert.strictEqual(editsPosted(posted).length, 3, 'one edit at a time');
  // The daemon answers each edit in order (the two-line one, then the
  // composed one); another window then puts the two-line text back.
  for (const first of ['Line one\nLine two', '変換中', 'Line one\nLine two']) {
    send(win, {
      type: 'tricksData',
      tricks: [first, 'Mine two, edited', 'Bundled alpha', 'Bundled beta'],
      userCount: 2,
    });
    assert.deepStrictEqual(editsPosted(posted)[3], {
      type: 'editTrick',
      text: 'Line one\nLine two',
      newText: '変換中',
    });
  }
  assert.strictEqual(editsPosted(posted).length, 4);

  // --- opening a second editor closes the first (one at a time)
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  assert.ok(rows(doc)[0].classList.contains('editing'));
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  assert.ok(!rows(doc)[0].classList.contains('editing'));
  assert.ok(rows(doc)[1].classList.contains('editing'));
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 1);
  key(win, editor(rows(doc)[1]), 'Escape');

  // --- a rejected edit: the daemon answers `error` + the list on disk,
  // stamped for this window, and the old text comes back
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  input(win, editor(rows(doc)[1]), 'Line one\nLine two');
  click(win, rows(doc)[1].querySelector('.tricks-edit-save'));
  assert.deepStrictEqual(editsPosted(posted).at(-1), {
    type: 'editTrick',
    text: 'Mine two, edited',
    newText: 'Line one\nLine two',
  });
  assert.strictEqual(
    texts(doc)[1],
    'Line one\nLine two',
    'shown optimistically',
  );
  send(win, {
    type: 'error',
    text: 'That promptlet is already in ~/.kiss/MY_INJECTION.md',
    connId: 'c1',
  });
  send(win, {
    type: 'tricksData',
    tricks: [
      'Line one\nLine two',
      'Mine two, edited',
      'Bundled alpha',
      'Bundled beta',
    ],
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

  // --- 2026-11-04 W6 F2: a rejected edit whose new text is nowhere
  // else (a line starting with "## ", say) keeps the draft: the editor
  // reopens on the promptlet with the refused text and the reason.
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  input(win, editor(rows(doc)[1]), 'Unique draft\n## Context\nmore');
  click(win, rows(doc)[1].querySelector('.tricks-edit-save'));
  send(win, {
    type: 'error',
    text: "Promptlet must not contain a line starting with '## '",
    connId: 'c1',
  });
  send(win, {
    type: 'tricksData',
    tricks: [
      'Line one\nLine two',
      'Mine two, edited',
      'Bundled alpha',
      'Bundled beta',
    ],
    userCount: 2,
    connId: 'c1',
  });
  assert.strictEqual(
    list.querySelectorAll('.tricks-edit-input').length,
    1,
    'the editor reopens',
  );
  assert.ok(rows(doc)[1].classList.contains('editing'));
  ta = editor(rows(doc)[1]);
  assert.strictEqual(ta.value, 'Unique draft\n## Context\nmore', 'the draft');
  assert.strictEqual(doc.activeElement, ta);
  const note = rows(doc)[1].querySelector('.config-field-note-error');
  assert.ok(
    note && note.textContent.includes("line starting with '## '"),
    'with the reason',
  );
  // Saved again, the edit names the stored text; a success closes it.
  input(win, ta, 'Unique draft, fixed');
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted).at(-1), {
    type: 'editTrick',
    text: 'Mine two, edited',
    newText: 'Unique draft, fixed',
  });
  send(win, {
    type: 'tricksData',
    tricks: [
      'Line one\nLine two',
      'Unique draft, fixed',
      'Bundled alpha',
      'Bundled beta',
    ],
    userCount: 2,
  });
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  assert.deepStrictEqual(texts(doc), [
    'Line one\nLine two',
    'Unique draft, fixed',
    'Bundled alpha',
    'Bundled beta',
  ]);

  // --- a list change from elsewhere closes an open editor: its index
  // could now point at another promptlet
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  input(win, editor(rows(doc)[1]), 'typing...');
  send(win, {
    type: 'tricksData',
    tricks: ['Mine two, edited', 'Bundled alpha', 'Bundled beta'],
    userCount: 1,
  });
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  assert.deepStrictEqual(texts(doc), [
    'Mine two, edited',
    'Bundled alpha',
    'Bundled beta',
  ]);
  assert.deepStrictEqual(
    rows(doc).map(r => r.querySelectorAll('.sidebar-item-edit').length),
    [1, 0, 0],
  );
  // ...but deleting ANOTHER promptlet from this window keeps the editor
  // and its draft, as the same deletion from another window does (below):
  // the Delete confirm authorised deleting A, not discarding B's edit.
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B', 'Mine C', 'Bundled alpha'],
    userCount: 3,
  });
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  assert.ok(rows(doc)[1].classList.contains('editing'));
  ta = editor(rows(doc)[1]);
  input(win, ta, 'DRAFT B\nsecond line');
  // The trash icon reveals an inline Delete confirm first
  // (ui_antipattern_destructive_confirm.test.js).  Deleting the row
  // AFTER the edited one leaves it where it is...
  click(win, rows(doc)[2].querySelector('.sidebar-item-delete'));
  click(win, rows(doc)[2].querySelector('.sidebar-confirm-yes'));
  assert.deepStrictEqual(texts(doc), [
    'Mine A',
    'DRAFT B\nsecond line',
    'Bundled alpha',
  ]);
  assert.strictEqual(
    editor(rows(doc)[1]),
    ta,
    'BUG - deleting a later promptlet closed the editor and lost the draft',
  );
  // ...and deleting the row BEFORE it moves the editor up with the draft.
  click(win, rows(doc)[0].querySelector('.sidebar-item-delete'));
  click(win, rows(doc)[0].querySelector('.sidebar-confirm-yes'));
  assert.deepStrictEqual(texts(doc), ['DRAFT B\nsecond line', 'Bundled alpha']);
  assert.strictEqual(
    editor(rows(doc)[0]),
    ta,
    'BUG - deleting an earlier promptlet closed the editor and lost the draft',
  );
  assert.strictEqual(ta.value, 'DRAFT B\nsecond line', 'the draft survives');
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'deleteTrick').map(m => m.text),
    ['Mine C', 'Mine A'],
  );
  // The daemon's answers find the editor on its promptlet, and the save
  // names that promptlet.
  send(win, {
    type: 'tricksData',
    tricks: ['Mine B', 'Bundled alpha'],
    userCount: 1,
  });
  assert.strictEqual(editor(rows(doc)[0]), ta);
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted).at(-1), {
    type: 'editTrick',
    text: 'Mine B',
    newText: 'DRAFT B\nsecond line',
  });
  send(win, {
    type: 'tricksData',
    tricks: ['DRAFT B\nsecond line', 'Bundled alpha'],
    userCount: 1,
  });
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  assert.strictEqual(editsPosted(posted).length, 8, 'deleting posts no edit');
  // (The editor row itself has Save / Cancel instead of a Delete button,
  // so the edited promptlet is never the one deleted from this window.)

  // --- another window deleting a PRECEDING promptlet leaves the edited
  // one in place: the editor follows it to its new row with the draft,
  // and the save names the promptlet, not the row it had
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B', 'Bundled alpha'],
    userCount: 2,
  });
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[1]);
  input(win, ta, 'UNSAVED B\nsecond line');
  send(win, {
    type: 'tricksData',
    tricks: ['Mine B', 'Bundled alpha'],
    userCount: 1,
  });
  assert.strictEqual(
    editor(rows(doc)[0]),
    ta,
    'the editor follows its promptlet to its new row',
  );
  assert.strictEqual(ta.value, 'UNSAVED B\nsecond line', 'the draft survives');
  assert.deepStrictEqual(texts(doc), [
    'UNSAVED B\nsecond line',
    'Bundled alpha',
  ]);
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted).slice(8), [
    {type: 'editTrick', text: 'Mine B', newText: 'UNSAVED B\nsecond line'},
  ]);
  assert.deepStrictEqual(texts(doc), [
    'UNSAVED B\nsecond line',
    'Bundled alpha',
  ]);
  // The daemon's answer settles the edit.
  send(win, {
    type: 'tricksData',
    tricks: ['UNSAVED B\nsecond line', 'Bundled alpha'],
    userCount: 1,
  });
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- another window changing an UNRELATED promptlet repaints the
  // list: the editor keeps the keyboard focus and the caret
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B', 'Bundled alpha'],
    userCount: 2,
  });
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[0]);
  input(win, ta, 'Mine A draft');
  ta.focus();
  ta.setSelectionRange(4, 4);
  assert.strictEqual(doc.activeElement, ta, 'setup: the editor has focus');
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B2', 'Bundled alpha'],
    userCount: 2,
  });
  assert.strictEqual(editor(rows(doc)[0]), ta, 'the editor is kept');
  assert.strictEqual(ta.value, 'Mine A draft');
  assert.strictEqual(doc.activeElement, ta, 'the editor keeps the focus');
  assert.strictEqual(ta.selectionStart, 4, 'and the caret');
  assert.strictEqual(ta.selectionEnd, 4);
  key(win, ta, 'Escape');
  assert.strictEqual(editsPosted(posted).length, 9, 'nothing more posted');

  // --- a filtered list edits the right promptlet (index into the full list)
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B', 'Bundled alpha'],
    userCount: 2,
  });
  input(win, doc.getElementById('tricks-search'), 'mine b');
  assert.deepStrictEqual(texts(doc), ['Mine B']);
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  ta = editor(rows(doc)[0]);
  assert.strictEqual(ta.value, 'Mine B');
  input(win, ta, 'Mine B2');
  key(win, ta, 'Enter');
  assert.deepStrictEqual(editsPosted(posted).at(-1), {
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

  // --- 2026-11-04 R4: two edits in flight, answered in order.  The
  // daemon refuses both: the first draft reopens its editor, and the
  // second refusal neither closes that editor nor loses its own draft,
  // which takes the editor once the first is done with it.
  const onDisk = ['Mine A', 'Mine B2', 'Bundled alpha'];
  send(win, {type: 'tricksData', tricks: onDisk.slice(), userCount: 2});
  doc.getElementById('tricks-btn').click();
  const drafts = ['Draft A\n## Context\nA', 'Draft B\n## Context\nB'];
  for (let i = 0; i < 2; i++) {
    click(win, rows(doc)[i].querySelector('.sidebar-item-edit'));
    input(win, editor(rows(doc)[i]), drafts[i]);
    click(win, rows(doc)[i].querySelector('.tricks-edit-save'));
  }
  assert.deepStrictEqual(
    editsPosted(posted)
      .slice(-2)
      .map(e => [e.text, e.newText]),
    [
      ['Mine A', drafts[0]],
      ['Mine B2', drafts[1]],
    ],
  );
  const refusal = {
    type: 'error',
    text: "Promptlet must not contain a line starting with '## '",
    connId: 'c1',
  };
  send(win, refusal);
  send(win, {
    type: 'tricksData',
    tricks: onDisk.slice(),
    userCount: 2,
    connId: 'c1',
  });
  assert.ok(rows(doc)[0].classList.contains('editing'), "A's editor reopens");
  assert.strictEqual(editor(rows(doc)[0]).value, drafts[0]);
  send(win, refusal);
  send(win, {
    type: 'tricksData',
    tricks: onDisk.slice(),
    userCount: 2,
    connId: 'c1',
  });
  assert.ok(
    rows(doc)[0].classList.contains('editing'),
    "A's editor is still open",
  );
  assert.strictEqual(
    editor(rows(doc)[0]).value,
    drafts[0],
    "A's draft survives B's refusal",
  );
  assert.strictEqual(
    list.querySelectorAll('.tricks-edit-input').length,
    1,
    'one editor',
  );
  // A success list for another edit, from any window, leaves the
  // editor on its unchanged row alone.
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B2', 'Bundled gamma'],
    userCount: 2,
  });
  assert.strictEqual(
    editor(rows(doc)[0]).value,
    drafts[0],
    'the editor outlives an unrelated change',
  );
  // A done with (cancelled): B's draft takes the editor, with its reason.
  key(win, editor(rows(doc)[0]), 'Escape');
  assert.ok(rows(doc)[1].classList.contains('editing'), "B's editor reopens");
  assert.strictEqual(
    editor(rows(doc)[1]).value,
    drafts[1],
    "B's draft, intact",
  );
  assert.ok(
    rows(doc)[1]
      .querySelector('.config-field-note-error')
      .textContent.includes("'## '"),
    "with B's reason",
  );
  assert.strictEqual(doc.activeElement, editor(rows(doc)[1]));
  // The promptlet it edits replaced from elsewhere: the editor closes.
  send(win, {
    type: 'tricksData',
    tricks: ['Mine A', 'Mine B5', 'Bundled gamma'],
    userCount: 2,
  });
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);

  // --- the same row saved twice before the daemon answers (the second
  // edit waits for the first one's answer, then names the text the row
  // holds on disk, not the one it showed), both refused: the first
  // draft, which the user replaced, is offered nowhere; the second
  // reopens its editor on the stored promptlet, intact, with its reason.
  const stored = ['Mine A', 'Mine B5', 'Bundled gamma'];
  const same = ['Same A\n## Context\nA', 'Same B\n## Context\nB'];
  const values = () =>
    Array.from(doc.querySelectorAll('textarea, input')).map(el => el.value);
  const refuse = (tricks, userCount = 2, error = refusal) => {
    send(win, error);
    send(win, {type: 'tricksData', tricks, userCount, connId: 'c1'});
  };
  const before = editsPosted(posted).length;
  for (const draft of same) {
    click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
    input(win, editor(rows(doc)[0]), draft);
    click(win, rows(doc)[0].querySelector('.tricks-edit-save'));
  }
  assert.strictEqual(editsPosted(posted).length, before + 1, 'one at a time');
  refuse(stored.slice());
  assert.deepStrictEqual(
    editsPosted(posted)
      .slice(before)
      .map(e => [e.text, e.newText]),
    [
      [stored[0], same[0]],
      [stored[0], same[1]],
    ],
  );
  refuse(stored.slice());
  assert.ok(rows(doc)[0].classList.contains('editing'), "B's editor reopens");
  assert.strictEqual(editor(rows(doc)[0]).value, same[1], 'B, intact');
  assert.strictEqual(doc.activeElement, editor(rows(doc)[0]));
  assert.ok(
    rows(doc)[0]
      .querySelector('.config-field-note-error')
      .textContent.includes("'## '"),
    "with B's reason",
  );
  key(win, editor(rows(doc)[0]), 'Escape');
  assert.ok(
    !values().some(v => v.includes('Same')),
    'A is offered nowhere, B no more',
  );

  // --- a refused draft whose promptlet is gone meanwhile: a one-line
  // draft goes to the empty Add box with its reason; a multi-line one
  // (the box would strip its newlines) waits for the promptlet to be
  // back, and reopens its editor then.
  const addBox = doc.getElementById('tricks-add-input');
  click(win, rows(doc)[1].querySelector('.sidebar-item-edit'));
  input(win, editor(rows(doc)[1]), same[1]);
  key(win, editor(rows(doc)[1]), 'Enter');
  refuse([stored[0], 'Bundled gamma'], 1);
  assert.strictEqual(list.querySelectorAll('.tricks-edit-input').length, 0);
  assert.strictEqual(addBox.value, '', 'the Add box is no place for it');
  click(win, rows(doc)[0].querySelector('.sidebar-item-edit'));
  input(win, editor(rows(doc)[0]), 'One line');
  key(win, editor(rows(doc)[0]), 'Enter');
  const taken = {type: 'error', text: 'That promptlet is taken', connId: 'c1'};
  refuse(['Bundled gamma'], 0, taken);
  assert.strictEqual(addBox.value, 'One line', 'in the Add box');
  assert.strictEqual(
    doc.getElementById('tricks-add-error').textContent,
    taken.text,
    'with its reason',
  );
  send(win, {type: 'tricksData', tricks: stored.slice(1), userCount: 1});
  assert.ok(
    rows(doc)[0].classList.contains('editing'),
    'the promptlet is back',
  );
  assert.strictEqual(editor(rows(doc)[0]).value, same[1], 'B, intact');
  key(win, editor(rows(doc)[0]), 'Escape');

  // A page loaded without the count global shows no pencils:
  // tricksPanelCopyDelete.test.js (its bare page).

  // Close the page so no timer main.js armed keeps node alive.
  win.close();
  console.log('tricksPanelEdit.test.js passed');
}

// A row changed again (saved, or deleted) before the daemon answered
// its rename is not named by the text it shows: the daemon may refuse
// that rename as a duplicate of ANOTHER row, which the follow-up would
// then change instead.  The follow-up waits for the answer and names
// the text the row holds on disk.
function runRowChangedBeforeItsAnswer() {
  const editors = p => p.doc.querySelectorAll('.tricks-edit-input').length;
  for (const followUp of ['edit', 'delete']) {
    const p = panelOf(['Original A', 'Existing B']);
    p.save(0, 'Existing B');
    if (followUp === 'edit') p.save(0, 'Replacement C');
    else p.del(0);
    const shown = followUp === 'edit' ? ['Replacement C'] : [];
    const answer = shown.concat('Existing B');
    assert.deepStrictEqual(texts(p.doc), answer);
    assert.deepStrictEqual(
      p.sent(),
      ['Original A > Existing B'],
      'BUG: the ' + followUp + ' went out under the text of another row',
    );
    // The daemon refuses the duplicate and sends the list on disk: the
    // follow-up goes out now, for the row the user changed.
    p.refuse(['Original A', 'Existing B']);
    assert.deepStrictEqual(p.sent().slice(1), [
      followUp === 'edit' ? 'Original A > Replacement C' : 'delete Original A',
    ]);
    // The row shows the follow-up again, before and after its answer;
    // the refused text, which the other row reads, is offered nowhere.
    assert.deepStrictEqual(texts(p.doc), answer);
    p.list(answer);
    assert.deepStrictEqual(texts(p.doc), answer, 'the other row is untouched');
    assert.strictEqual(editors(p), 0);
    assert.strictEqual(p.doc.getElementById('tricks-add-input').value, '');
    p.win.close();
  }

  // A follow-up still being typed when the answer arrives stays in the
  // editor of its own row, and is saved for it.  Refused, that row
  // holds its old text (the one it showed meanwhile is the other
  // row's); accepted, the new one, wherever the list has it now.
  for (const [saved, refused, at] of [
    ['Existing B', true, 0],
    ['Renamed A', false, 1],
  ]) {
    const p = panelOf(['Original A', 'Existing B']);
    p.save(0, saved);
    click(p.win, rows(p.doc)[0].querySelector('.sidebar-item-edit'));
    input(p.win, editor(rows(p.doc)[0]), 'Replacement C');
    if (refused) p.refuse(['Original A', 'Existing B']);
    else p.list(['Added elsewhere', 'Renamed A', 'Existing B']);
    const followed = editor(rows(p.doc)[at]);
    assert.ok(followed, 'BUG: the open editor moved to another row');
    key(p.win, followed, 'Enter');
    assert.deepStrictEqual(p.sent().slice(1), [
      (refused ? 'Original A' : 'Renamed A') + ' > Replacement C',
    ]);
    p.win.close();
  }

  // An edit awaiting its answer stays with its row while the rows
  // around it are deleted, and leaves with it: the row saved or deleted
  // next is still the one named, and the row moving up is not it.
  let p = panelOf(['A', 'B', 'C', 'D']);
  p.save(1, 'B2');
  p.del(0);
  p.del(2);
  p.save(0, 'B3');
  p.del(0);
  p.save(0, 'C2');
  assert.deepStrictEqual(p.sent(), ['B > B2', 'delete A', 'delete D', 'C > C2']);
  p.list(['B2', 'C2']);
  assert.deepStrictEqual(p.sent().slice(4), ['delete B2']);
  assert.deepStrictEqual(texts(p.doc), ['C2']);
  p.win.close();

  // A refused draft that waited for the editor is not put on a row the
  // user deleted since (the delete waited for that row's later edit).
  p = panelOf(['A', 'B']);
  p.save(0, 'Draft X');
  click(p.win, rows(p.doc)[1].querySelector('.sidebar-item-edit'));
  p.refuse(['A', 'B']);
  p.save(0, 'U');
  p.del(0);
  p.list(['U', 'B']);
  assert.deepStrictEqual(p.sent().slice(1), ['A > U', 'delete U']);
  assert.deepStrictEqual(texts(p.doc), ['B']);
  assert.strictEqual(editors(p), 0, 'BUG: the draft is on another row');
  p.win.close();

  // A draft refused after the user changed its row again is offered
  // nowhere (W3 audit 2026-11-12): saved from a reopened editor, it
  // would overwrite the row's newer text.  The row saved again before
  // the answer: the newer text goes out in its place, once ...
  const addBox = q => q.doc.getElementById('tricks-add-input').value;
  p = panelOf(['A', 'B']);
  p.save(0, 'X');
  p.save(0, 'Y');
  p.refuse(['A', 'B']);
  assert.strictEqual(editors(p), 0, 'BUG: the replaced draft is back');
  p.list(['Y', 'B']);
  assert.deepStrictEqual(p.sent(), ['A > X', 'A > Y']);
  assert.deepStrictEqual(texts(p.doc), ['Y', 'B']);
  p.win.close();
  // ... or after it, from the editor the refused draft waited for ...
  p = panelOf(['A', 'B']);
  p.save(0, 'X');
  click(p.win, rows(p.doc)[0].querySelector('.sidebar-item-edit'));
  input(p.win, editor(rows(p.doc)[0]), 'Z');
  p.refuse(['A', 'B']);
  key(p.win, editor(rows(p.doc)[0]), 'Enter');
  p.list(['Z', 'B']);
  assert.deepStrictEqual(texts(p.doc), ['Z', 'B'], 'BUG: the replaced draft is back');
  p.win.close();
  // ... or the row deleted while the draft waited for another row's editor.
  p = panelOf(['A', 'B']);
  p.save(0, 'X');
  click(p.win, rows(p.doc)[1].querySelector('.sidebar-item-edit'));
  p.refuse(['A', 'B']);
  p.del(0);
  key(p.win, editor(rows(p.doc)[0]), 'Escape');
  assert.deepStrictEqual(texts(p.doc), ['B']);
  assert.strictEqual(addBox(p), '', 'BUG: the draft of a deleted promptlet is back');
  p.win.close();

  // The row gone from the answer's list (deleted from another window):
  // a waiting edit still goes out, for the daemon to refuse and its
  // draft to come back; a waiting delete has nothing left to delete,
  // nor a draft to give back.  A row put back to its stored text needs
  // no second edit.  The refused X is offered nowhere.
  for (const [second, gone, sent, back] of [
    ['Y', true, ['A > X', 'A > Y'], 'Y'],
    [null, true, ['A > X'], ''],
    ['A', false, ['A > X'], ''],
  ]) {
    p = panelOf(['A', 'B']);
    p.save(0, 'X');
    if (second) p.save(0, second);
    else p.del(0);
    p.refuse(gone ? ['B'] : ['A', 'B']);
    assert.deepStrictEqual(p.sent(), sent);
    assert.strictEqual(editors(p), 0);
    assert.strictEqual(addBox(p), '');
    if (back) p.refuse(['B']);
    assert.strictEqual(addBox(p), back);
    p.win.close();
  }
}

/**
 * A promptlets sheet listing *stored* (all the user's own), with the
 * user's and the daemon's moves as one-liners; `sent` lists the edits
 * and deletes posted so far, as "text > newText" and "delete text".
 */
function panelOf(stored) {
  const {win, posted} = makePage();
  const doc = win.document;
  const list = tricks =>
    send(win, {type: 'tricksData', tricks: tricks.slice(), userCount: tricks.length});
  list(stored);
  doc.getElementById('tricks-btn').click();
  return {
    win,
    doc,
    list,
    refuse(tricks) {
      send(win, {type: 'error', command: 'editTrick', text: 'Refused'});
      list(tricks);
    },
    save(row, text) {
      click(win, rows(doc)[row].querySelector('.sidebar-item-edit'));
      input(win, editor(rows(doc)[row]), text);
      key(win, editor(rows(doc)[row]), 'Enter');
    },
    del(row) {
      click(win, rows(doc)[row].querySelector('.sidebar-item-delete'));
      click(win, rows(doc)[row].querySelector('.sidebar-confirm-yes'));
    },
    sent: () =>
      posted
        .filter(m => /^(edit|delete)Trick$/.test(m.type))
        .map(m => (m.newText ? m.text + ' > ' + m.newText : 'delete ' + m.text)),
  };
}

runRowChangedBeforeItsAnswer();
runPanel();
