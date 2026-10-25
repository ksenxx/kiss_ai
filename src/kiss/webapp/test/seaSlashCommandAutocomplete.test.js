// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Frontend test for the SEA slash-command autocomplete popup.
//
// The daemon pushes a ``seaCommands`` event on every ready, on every
// registry rescan and in reply to the page's own ``getSeaCommands``.
// Typing ``/`` (or ``/xxx``) at the start of a word -- the start of
// the composer, or after a space or a line break anywhere in it --
// must show a dropdown of matching commands; selecting one must put
// ``/<name> `` (a trailing space) at the slash so the user can type
// the task text right after.
//
// The dropdown lists the event's ``recent`` commands first (the three
// most recently run), then its ``frequent`` ones, then the rest in
// name order; opening it asks the daemon for a fresh list once.
//
// Also pins that a plain ``/`` with no matches hides the dropdown, and
// that a slash inside a path (``src/``) is NOT treated as a command.

'use strict';

const assert = require('assert');
const {makePage: makeChatPage, send} = require('./simplify2_harness.js');

// Every page built, closed together at the end so no timer of theirs
// keeps the process alive.
const windows = [];

function makePage() {
  const page = makeChatPage();
  windows.push(page.win);
  return page;
}

function keydown(win, inp, key) {
  inp.dispatchEvent(
    new win.KeyboardEvent('keydown', {key, bubbles: true, cancelable: true}),
  );
}

function selectedText(win) {
  const sel = picker(win).querySelector('.ac-item.sel');
  return sel ? sel.dataset.text : null;
}

function seaCommandRequests(posted) {
  return posted.filter(m => m.type === 'getSeaCommands').length;
}

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
    const {win} = makePage();
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
    const {win} = makePage();
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
    const {win} = makePage();
    send(win, {type: 'seaCommands', commands: ['slack', 'gmail']});
    setInput(win, '/zzz');
    assert.strictEqual(isPickerOpen(win), false, 'popup hides with no matches');
  }

  // A slash anywhere in the text opens the popup -----------------------
  // ... after a space, and accepting keeps the text before the slash.
  {
    const {win} = makePage();
    send(win, {type: 'seaCommands', commands: ['slack', 'gmail']});
    const inp = setInput(win, 'please fix this /gm');
    assert.ok(isPickerOpen(win), 'slash after a space opens the popup');
    assert.deepStrictEqual(
      seaCommandItems(win).map(el => el.dataset.text),
      ['gmail'],
    );
    seaCommandItems(win)[0].click();
    assert.strictEqual(inp.value, 'please fix this /gmail ');
    assert.strictEqual(inp.selectionStart, inp.value.length);
    assert.strictEqual(isPickerOpen(win), false);
  }
  // ... on a later line, with text after the cursor left untouched.
  {
    const {win} = makePage();
    send(win, {type: 'seaCommands', commands: ['slack']});
    const inp = win.document.getElementById('task-input');
    inp.value = 'plain text\n/sl tail';
    inp.setSelectionRange(14, 14);
    fireInput(inp);
    assert.ok(isPickerOpen(win), 'slash on line 2 opens the popup');
    seaCommandItems(win)[0].click();
    // The text after the cursor already starts with a space: no
    // second one is added.
    assert.strictEqual(inp.value, 'plain text\n/slack tail');
    assert.strictEqual(inp.selectionStart, 'plain text\n/slack'.length);
  }
  // ... but a slash inside a path or an @-mention is NOT a command.
  {
    const {win} = makePage();
    send(win, {type: 'seaCommands', commands: ['slack', 'src']});
    setInput(win, 'look at src/');
    assert.strictEqual(isPickerOpen(win), false, 'src/ is a path');
    setInput(win, 'look at src/s');
    assert.strictEqual(isPickerOpen(win), false, 'src/s is a path');
    setInput(win, 'see @docs/s');
    assert.strictEqual(isPickerOpen(win), false, '@docs/s is a mention');
  }

  // Recent, then frequent, then the rest --------------------------------
  {
    const {win} = makePage();
    send(win, {
      type: 'seaCommands',
      // The daemon sends the names sorted; the rest keeps that order.
      commands: ['alpha', 'beta', 'delta', 'eps', 'gamma', 'zeta'],
      // ``gone`` is no longer registered and must be ignored; ``zeta``
      // is both recent and frequent and must be listed once.
      recent: ['zeta', 'gone', 'beta'],
      frequent: ['delta', 'zeta', 'alpha'],
    });
    setInput(win, '/');
    assert.deepStrictEqual(seaCommandSectionLabels(win), [
      'Recent',
      'Frequent',
      'Commands',
    ]);
    assert.deepStrictEqual(
      seaCommandItems(win).map(el => el.dataset.text),
      ['zeta', 'beta', 'delta', 'alpha', 'eps', 'gamma'],
    );
    // The most recent command is preselected.
    assert.strictEqual(selectedText(win), 'zeta');
    // Filtering applies to every group; empty groups drop their header.
    setInput(win, '/a');
    assert.deepStrictEqual(seaCommandSectionLabels(win), [
      'Recent',
      'Frequent',
      'Commands',
    ]);
    assert.deepStrictEqual(
      seaCommandItems(win).map(el => el.dataset.text),
      ['zeta', 'beta', 'delta', 'alpha', 'gamma'],
    );
    setInput(win, '/eps');
    assert.deepStrictEqual(seaCommandSectionLabels(win), ['Commands']);
    assert.deepStrictEqual(
      seaCommandItems(win).map(el => el.dataset.text),
      ['eps'],
    );
  }

  // Opening the popup asks the daemon for a fresh list ONCE -----------
  {
    const {win, posted} = makePage();
    send(win, {type: 'seaCommands', commands: ['slack', 'gmail']});
    const before = seaCommandRequests(posted);
    const inp = setInput(win, '/');
    assert.strictEqual(seaCommandRequests(posted), before + 1);
    // Narrowing the same word does not ask again, nor does the
    // identical reply to the request.
    setInput(win, '/g');
    send(win, {type: 'seaCommands', commands: ['slack', 'gmail']});
    assert.strictEqual(seaCommandRequests(posted), before + 1);
    assert.ok(isPickerOpen(win));
    // Escape dismisses the popup; typing on reopens it and asks again.
    keydown(win, inp, 'Escape');
    assert.strictEqual(isPickerOpen(win), false);
    setInput(win, '/gm');
    assert.ok(isPickerOpen(win));
    assert.strictEqual(seaCommandRequests(posted), before + 2);
    // A word with no matches asks once, not on every keystroke.
    setInput(win, '/zz');
    assert.strictEqual(isPickerOpen(win), false);
    setInput(win, '/zzz');
    assert.strictEqual(seaCommandRequests(posted), before + 2);
  }

  // A reply with the same lists keeps the arrow-key selection; -------
  // a reply with a new ranking reflows the popup.
  {
    const {win, posted} = makePage();
    const event = {
      type: 'seaCommands',
      commands: ['alpha', 'beta', 'gamma'],
      recent: ['gamma'],
      frequent: [],
    };
    send(win, event);
    const inp = setInput(win, '/');
    keydown(win, inp, 'ArrowDown');
    assert.strictEqual(selectedText(win), 'alpha');
    send(win, event);
    assert.strictEqual(selectedText(win), 'alpha', 'same reply: no reset');
    send(win, Object.assign({}, event, {recent: ['beta']}));
    assert.deepStrictEqual(
      seaCommandItems(win).map(el => el.dataset.text),
      ['beta', 'alpha', 'gamma'],
    );
    assert.strictEqual(selectedText(win), 'beta');
    // A reply landing after Escape must not reopen the dismissed
    // popup (the slash word is still in the composer) nor ask again.
    keydown(win, inp, 'Escape');
    assert.strictEqual(isPickerOpen(win), false);
    const requests = seaCommandRequests(posted);
    send(win, Object.assign({}, event, {recent: ['alpha']}));
    assert.strictEqual(isPickerOpen(win), false, 'dismissed stays dismissed');
    assert.strictEqual(seaCommandRequests(posted), requests);
    // The next keystroke shows the new ranking (alpha is now recent).
    setInput(win, '/a');
    assert.deepStrictEqual(seaCommandSectionLabels(win), [
      'Recent',
      'Commands',
    ]);
    assert.deepStrictEqual(
      seaCommandItems(win).map(el => el.dataset.text),
      ['alpha', 'beta', 'gamma'],
    );
  }

  // A rescan pushed while the popup is open re-flows it --------------
  {
    const {win} = makePage();
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
    const {win} = makePage();
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
    const {win} = makePage();
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

try {
  main();
} finally {
  windows.forEach(w => w.close());
}
