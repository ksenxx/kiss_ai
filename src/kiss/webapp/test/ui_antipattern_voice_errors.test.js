// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// UI anti-pattern fix A9 / A3 for the voice trigger (media/voice.js): a
// microphone / audio / server failure is no longer visible only as a
// hover tooltip on a button hidden inside the collapsed "More actions"
// menu.  It also raises a sticky, dismissible toast in the page's
// notification container, worded in plain language with a remedy, and a
// later successful start closes that toast again.

const assert = require('assert');
const {settle} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {makeVoicePage, micBtn, startMic} = require('./voice_page_helpers');
const runner = makeRunner();
const report = runner.report;

// Every page a test opened is closed afterwards, pass or fail.
async function test(name, fn) {
  try {
    await runner.test(name, fn);
  } finally {
    while (windows.length) windows.pop().close();
  }
}

// Every page a test builds; closed after the test so no voice or toast
// timer keeps node alive.
const windows = [];

// The chat page over voice_page_helpers' scriptable microphone, with the
// mic remembered OFF: a test turns it on with a click.
function makePage() {
  const page = makeVoicePage();
  windows.push(page.win);
  return page;
}

function toast(win) {
  return win.document.querySelector(
    '.kiss-notification[data-notification-id="voice-trigger-error"]',
  );
}

/**
 * Let the failed start land: voice.js raises the toast and paints the
 * button's state in one synchronous step, so once the toast exists the
 * class, tooltip and aria-pressed are set too.
 */
async function settledToast(win) {
  await settle();
  const t = toast(win);
  assert.ok(t, 'no voice error toast appeared');
  return t;
}

function toastMessage(win) {
  return toast(win).querySelector('.kiss-notification-message').textContent;
}

async function main() {
  await test('a denied microphone raises a sticky error toast with a remedy', async () => {
    const page = makePage();
    const {win} = page;
    assert.strictEqual(toast(win), null, 'no toast before the failure');
    micBtn(win).click();
    await settle();
    page.mic.calls[0].refuse('Permission denied');
    const t = await settledToast(win);
    assert.ok(
      micBtn(win).classList.contains('voice-error'),
      micBtn(win).className,
    );
    assert.ok(t.classList.contains('kiss-notification-error'), t.className);
    assert.strictEqual(
      t.dataset.notificationSticky,
      'true',
      'errors never auto-dismiss',
    );
    const text = toastMessage(win);
    assert.strictEqual(
      text,
      'Microphone access was denied. Allow the microphone for this site in ' +
        'the browser settings, then turn the voice trigger on again.',
    );
    assert.ok(!/Permission denied/.test(text), 'no raw error jargon');
    // The tooltip carries the same plain-language text.
    assert.ok(
      /Microphone access was denied/.test(
        micBtn(win).getAttribute('data-tooltip'),
      ),
    );
    // The toast is dismissible.
    t.querySelector('.kiss-notification-close').click();
    assert.strictEqual(toast(win), null, 'the close button removes the toast');
  });

  await test('a missing microphone gets its own remedy', async () => {
    const page = makePage();
    micBtn(page.win).click();
    await settle();
    page.mic.calls[0].refuse('Requested device not found', 'NotFoundError');
    await settledToast(page.win);
    const text = toastMessage(page.win);
    assert.ok(
      /No microphone was found\. Connect a microphone/.test(text),
      text,
    );
  });

  await test('an unknown failure names the cause and what to do next', async () => {
    const page = makePage();
    micBtn(page.win).click();
    await settle();
    page.mic.calls[0].refuse('Starting audio failed', 'AbortError');
    await settledToast(page.win);
    assert.strictEqual(
      toastMessage(page.win),
      'The voice trigger could not start (Starting audio failed). Check the ' +
        'microphone, then turn the voice trigger on again.',
    );
  });

  await test('a repeated failure updates the same toast, and a later success closes it', async () => {
    const page = makePage();
    const {win} = page;
    micBtn(win).click();
    await settle();
    page.mic.calls[0].refuse('microphone busy (code 1)', 'AbortError');
    const t = await settledToast(win);
    assert.ok(t.classList.contains('kiss-notification-error'));
    assert.ok(/microphone busy \(code 1\)/.test(t.textContent));
    assert.ok(/turn the voice trigger on again/.test(t.textContent));
    assert.strictEqual(micBtn(win).getAttribute('aria-pressed'), 'false');
    // A repeated failure updates the same toast instead of stacking.
    micBtn(win).click();
    await settle();
    page.mic.calls[1].refuse('microphone busy (code 2)', 'AbortError');
    await settle();
    assert.ok(/code 2/.test(toast(win).textContent), 'the toast was updated');
    assert.strictEqual(
      win.document.querySelectorAll('.kiss-notification').length,
      1,
      'one toast, updated in place',
    );
    assert.strictEqual(micBtn(win).getAttribute('aria-pressed'), 'false');
    // The third attempt gets the microphone: the stale error goes away.
    micBtn(win).click();
    await startMic(page);
    assert.strictEqual(
      toast(win),
      null,
      'a working voice trigger clears the stale error',
    );
    assert.strictEqual(
      micBtn(win).getAttribute('aria-pressed'),
      'true',
      'toggle state is exposed',
    );
  });
}

main().then(() => {
  report('ui_antipattern_voice_errors');
});
