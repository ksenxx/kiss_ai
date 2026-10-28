// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for the Recommendations subpanel renderer
// (media/main.js renderRecommendations): like every other subpanel
// renderer it re-measures the task-info panel after it changed the
// status line (one line "Suggested 3h ago" vs. the two-to-three-line
// "No recommendations yet" / error text), since --meta-body-share is
// measured from the space left by the headers and status lines.  jsdom
// has no layout, so the re-measured share is 0.00px (the same probe
// metaPanelSections.test.js uses): a planted 99px is replaced only when
// the renderer republished it.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  stubRaf,
  fakeMatchMedia,
} = require('./simplify2_harness');

function makeMetaPage() {
  return makePage({
    beforeScripts: win => {
      stubRaf(win);
      win.ResizeObserver = class {
        observe() {}
      };
      fakeMatchMedia(win, true);
    },
  });
}

function shareAfter(win, ev) {
  const panel = win.document.getElementById('meta-panel');
  panel.style.setProperty('--meta-body-share', '99px');
  send(win, ev);
  return panel.style.getPropertyValue('--meta-body-share');
}

function main() {
  const {win} = makeMetaPage();
  const status = win.document.getElementById('meta-recommendations-status');
  assert.ok(status, 'the recommendations status line exists');

  assert.strictEqual(
    shareAfter(win, {type: 'recommendations', tasks: [], generatedAt: 0}),
    '0.00px',
    'the empty-state text changes the measured height: the share is republished',
  );
  assert.ok(/No recommendations yet/.test(status.textContent));

  assert.strictEqual(
    shareAfter(win, {
      type: 'recommendations',
      tasks: ['Write the release notes'],
      generatedAt: Date.now() - 3600000,
    }),
    '0.00px',
    'a suggestion list changes the status line: the share is republished',
  );
  assert.ok(/^Suggested /.test(status.textContent));
  assert.strictEqual(
    win.document.querySelectorAll(
      '#meta-recommendations-list .recommendation-row',
    ).length,
    1,
  );

  assert.strictEqual(
    shareAfter(win, {type: 'recommendations', tasks: [], error: 'cron is off'}),
    '0.00px',
    'an error line changes the measured height: the share is republished',
  );
  assert.strictEqual(status.textContent, 'cron is off');
  win.close();
  console.log('recommendationsLayout_sweep1026.test.js: all tests passed');
}

main();
