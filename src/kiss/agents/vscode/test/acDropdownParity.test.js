// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Pins the shared behavior of the two autocomplete dropdown renderers
// (the @-mention file picker and the main-input completion picker) so
// their historically duplicated rendering code can be collapsed into a
// single helper without drift: grouped sections, one icon per item, a
// 'tab' hint on the very first item only, the kbd footer, and initial
// selection on the first item.

'use strict';

const assert = require('assert');
const {makeWebview, send} = require('./simplify2_harness.js');

function picker(win) {
  return win.document.getElementById('autocomplete');
}

function setInput(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.setSelectionRange(text.length, text.length);
  return inp;
}

function checkDropdownStructure(win, expectedSections, expectedItems) {
  const p = picker(win);
  assert.strictEqual(p.style.display, 'block', 'picker opens');
  const sections = Array.from(p.querySelectorAll('.ac-section')).map(
    e => e.textContent,
  );
  assert.deepStrictEqual(sections, expectedSections);
  const items = Array.from(p.querySelectorAll('.ac-item'));
  assert.strictEqual(items.length, expectedItems);
  assert.strictEqual(
    p.querySelectorAll('.ac-icon').length,
    expectedItems,
    'every item carries an icon',
  );
  const hints = p.querySelectorAll('.ac-hint');
  assert.strictEqual(hints.length, 1, "exactly one 'tab' hint");
  assert.ok(
    items[0].querySelector('.ac-hint'),
    "the 'tab' hint sits on the FIRST item",
  );
  assert.ok(
    items[0].classList.contains('sel'),
    'first item is initially selected',
  );
  const footer = p.querySelector('.ac-footer');
  assert.ok(footer, 'kbd footer rendered');
  assert.ok(/navigate/.test(footer.textContent));
  assert.ok(/accept/.test(footer.textContent));
  assert.ok(/dismiss/.test(footer.textContent));
  return items;
}

function main() {
  // --- @-mention file picker -------------------------------------------
  {
    const {win} = makeWebview();
    const inp = setInput(win, 'open @src');
    send(win, {
      type: 'files',
      prefix: 'src',
      files: [
        {type: 'frequent', text: 'src/main.js'},
        {type: 'file', text: 'src/api.js'},
        {type: 'file', text: 'src/tips.js'},
      ],
    });
    const items = checkDropdownStructure(win, ['Frequent', 'Files'], 3);
    items[1].click();
    assert.ok(
      inp.value.includes('src/api.js'),
      'clicking a file item inserts the @-mention',
    );
    assert.strictEqual(picker(win).style.display, 'none', 'picker closes');
  }

  // --- main-input completion picker ------------------------------------
  {
    const {win} = makeWebview();
    const inp = setInput(win, 'fix');
    send(win, {
      type: 'completions',
      query: 'fix',
      completions: [
        {type: 'task', text: 'fix the parser bug'},
        {type: 'frequent', text: 'fix the tests'},
        {type: 'identifier', text: 'fix_arguments'},
      ],
    });
    const items = checkDropdownStructure(
      win,
      ['History', 'Frequent', 'From editor'],
      3,
    );
    items[0].click();
    assert.strictEqual(
      inp.value.trim(),
      'fix the parser bug',
      'clicking a completion accepts it into the input',
    );
    assert.strictEqual(picker(win).style.display, 'none', 'picker closes');
  }

  console.log('  ok - both autocomplete dropdowns share identical structure');
}

main();
