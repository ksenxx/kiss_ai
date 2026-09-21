// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Frontend test for the SEA slash-command autocomplete popup.
//
// The daemon pushes a ``seaCommands`` event on every ready and on
// every registry rescan.  Typing ``/`` (or ``/xxx``) at the very start
// of the composer must show a dropdown of matching commands; selecting
// one must replace the composer text with ``/<name> `` (a trailing
// space) so the user can type the task text right after.
//
// Also pins that a plain ``/`` with no matches hides the dropdown, and
// that a slash typed on a later line is NOT treated as a command.

'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness.js');

function picker(win) {
  return win.document.getElementById('autocomplete');
}

function isPickerOpen(win) {
  return picker(win).style.display === 'block';
}

function fireInput(inp) {
  inp.dispatchEvent(new inp.ownerDocument.defaultView.Event('input'));
}

function setInput(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.setSelectionRange(text.length, text.length);
  fireInput(inp);
  return inp;
}

function seaCommandItems(win) {
  return Array.from(picker(win).querySelectorAll('.ac-item'));
}

function seaCommandSectionLabels(win) {
  return Array.from(picker(win).querySelectorAll('.ac-section')).map(
    s => s.textContent,
  );
}

function main() {
  // The daemon-pushed command list -----------------------------------
  {
    const {win} = makeWebview();
    send(win, {
      type: 'seaCommands',
      commands: ['slack', 'gmail', 'gcal', 'github'],
    });
    // Composer is empty: no popup shown.
    assert.strictEqual(isPickerOpen(win), false);

    // Typing ``/`` alone -> popup with ALL commands.
    setInput(win, '/');
    assert.ok(isPickerOpen(win), 'popup opens on lone /');
    assert.deepStrictEqual(seaCommandSectionLabels(win), ['Commands']);
    const allItems = seaCommandItems(win);
    assert.strictEqual(allItems.length, 4);
    // The first item is preselected and carries the "tab" hint.
    assert.ok(allItems[0].classList.contains('sel'));
    assert.ok(allItems[0].querySelector('.ac-hint'));
  }

  // Prefix filtering + accept via click -------------------------------
  {
    const {win} = makeWebview();
    send(win, {
      type: 'seaCommands',
      commands: ['slack', 'gmail', 'gcal', 'github'],
    });
    const inp = setInput(win, '/g');
    const items = seaCommandItems(win);
    const texts = items.map(el => el.dataset.text);
    // Only commands whose name contains "g" survive.
    assert.deepStrictEqual(texts.sort(), ['gcal', 'github', 'gmail']);
    // Each rendered label starts with the ``/`` prefix.
    items.forEach(el => {
      const label = el.querySelector('.ac-text').textContent;
      assert.ok(
        label.startsWith('/'),
        `label ${JSON.stringify(label)} missing / prefix`,
      );
    });
    // Click the "gmail" item: composer becomes "/gmail " with the
    // cursor placed AFTER the trailing space.
    const gmail = items.find(el => el.dataset.text === 'gmail');
    gmail.click();
    assert.strictEqual(inp.value, '/gmail ');
    assert.strictEqual(inp.selectionStart, inp.value.length);
    assert.strictEqual(isPickerOpen(win), false);
  }

  // Prefix that matches nothing -> popup hides -----------------------
  {
    const {win} = makeWebview();
    send(win, {type: 'seaCommands', commands: ['slack', 'gmail']});
    setInput(win, '/zzz');
    assert.strictEqual(
      isPickerOpen(win),
      false,
      'popup hides with no matches',
    );
  }

  // A slash on a later line is NOT a command ------------------------
  {
    const {win} = makeWebview();
    send(win, {type: 'seaCommands', commands: ['slack']});
    setInput(win, 'plain text\n/slack');
    assert.strictEqual(
      isPickerOpen(win),
      false,
      'slash on line 2 does not trigger command popup',
    );
  }

  // A rescan pushed while the popup is open re-flows it --------------
  {
    const {win} = makeWebview();
    send(win, {type: 'seaCommands', commands: ['alpha']});
    setInput(win, '/');
    assert.strictEqual(seaCommandItems(win).length, 1);
    send(win, {type: 'seaCommands', commands: ['alpha', 'beta']});
    // The popup MUST reflect the fresh list without needing another
    // keystroke.
    assert.strictEqual(seaCommandItems(win).length, 2);
  }

  // A stale ``files`` reply must NOT hide an open slash popup -------
  {
    const {win} = makeWebview();
    send(win, {type: 'seaCommands', commands: ['alpha']});
    // Open a slash popup.
    setInput(win, '/al');
    assert.ok(isPickerOpen(win), 'slash popup opens on /al');
    // A stale ``files`` reply from a prior ``@`` mention (no @-context
    // in the composer any more) MUST NOT hide the slash popup.
    send(win, {type: 'files', prefix: 'old', files: []});
    assert.ok(
      isPickerOpen(win),
      'stale @-files reply does not close the slash popup',
    );
  }

  // A ``ghost`` reply arriving while the slash popup is open --------
  // must NOT paint an inline ghost suggestion over the composer.
  {
    const {win} = makeWebview();
    send(win, {type: 'seaCommands', commands: ['gmail', 'gcal']});
    setInput(win, '/g');
    assert.ok(isPickerOpen(win));
    send(win, {type: 'ghost', query: '/g', suggestion: 'host-ghost'});
    const ghost = win.document.getElementById('ghost-overlay');
    // The overlay stays hidden / empty while a slash popup is up.
    assert.ok(
      !ghost || !ghost.textContent || ghost.style.display === 'none',
      'ghost overlay must not paint under slash popup',
    );
  }

  console.log('  ok - SEA slash-command autocomplete');
}

main();
