// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) test for the Recommendations subpanel renderer
// (media/main.js renderRecommendations): like every other subpanel
// renderer it re-measures the task-info panel after a reply.  The
// section is its card alone and is hidden while it has no task, so a
// reply brings it on show or takes it away, and --meta-body-share is
// measured from the space the shown sections leave.  jsdom has no
// layout, so the re-measured share is 0.00px (the same probe
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
  const section = win.document.getElementById('meta-recommendations');

  assert.strictEqual(
    shareAfter(win, {type: 'recommendations', tasks: [], generatedAt: 0}),
    '0.00px',
    'a reply without a task: the share is republished',
  );
  assert.ok(section.hidden, 'no task: the section stays hidden');

  assert.strictEqual(
    shareAfter(win, {
      type: 'recommendations',
      tasks: ['Write the release notes'],
      generatedAt: Date.now() - 3600000,
    }),
    '0.00px',
    'a task brings the card on show: the share is republished',
  );
  assert.ok(!section.hidden);
  assert.strictEqual(
    win.document.querySelectorAll(
      '#meta-recommendations-list .recommendation-row',
    ).length,
    1,
  );

  assert.strictEqual(
    shareAfter(win, {type: 'recommendations', tasks: [], error: 'cron is off'}),
    '0.00px',
    'an error reply takes the card away: the share is republished',
  );
  assert.ok(section.hidden);
  win.close();
  console.log('recommendationsLayout_sweep1026.test.js: all tests passed');
}

main();
