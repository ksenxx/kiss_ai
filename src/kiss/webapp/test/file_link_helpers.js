// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Shared readers of the file-path links main.js renders into #output
// (clickFilePath*.test.js, *FileLinks.test.js, worktree*Recheck.test.js):
// a verified path becomes a `[data-path]` link, an unverified one is
// demoted to a `[data-path-missing]` span.

'use strict';

const fs = require('fs');
const harness = require('./simplify2_harness');

/**
 * The verified file links in `#output` whose `data-path` is exactly `p`.
 *
 * @returns {Array<Element>} The matching link elements, in document order.
 */
function findLinks(win, p) {
  return Array.from(
    win.document.querySelectorAll('#output [data-path]'),
  ).filter(el => el.dataset.path === p);
}

/**
 * The demoted (not clickable) path spans in `#output` whose
 * `data-path-missing` is exactly `p`.
 *
 * @returns {Array<Element>} The matching span elements, in document order.
 */
function findMissing(win, p) {
  return Array.from(
    win.document.querySelectorAll('#output [data-path-missing]'),
  ).filter(el => el.getAttribute('data-path-missing') === p);
}

/**
 * A chat page whose working directory is `workDir` and whose launch
 * animation is over, so path checks go out at once.
 *
 * @returns {{win: Window, posted: Array<object>}} The harness page.
 */
function makeWorkDirPage(workDir) {
  const page = harness.makePage();
  harness.send(page.win, {
    type: 'configData',
    config: {work_dir: workDir},
    apiKeys: {},
  });
  page.win._testApi.endLaunch();
  return page;
}

/**
 * The page's pending file-link bookkeeping as `{spans, checks}`.  The
 * probe's object is built in the JSDOM realm; it is copied into this one
 * so deepStrictEqual compares values rather than prototypes.
 */
function pendingCounts(win) {
  const c = win._testApi.pendingFileLinkCounts();
  return {spans: c.spans, checks: c.checks};
}

/** Whether `p` is an existing file or directory on the real file system. */
function existsOnRealFs(p) {
  try {
    const st = fs.statSync(p);
    return st.isFile() || st.isDirectory();
  } catch {
    return false;
  }
}

module.exports = {
  findLinks,
  findMissing,
  makeWorkDirPage,
  pendingCounts,
  existsOnRealFs,
};
