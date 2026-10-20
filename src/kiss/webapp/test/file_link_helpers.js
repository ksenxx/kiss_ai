// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Shared readers of the file-path links main.js renders into #output
// (clickFilePath*.test.js, *FileLinks.test.js, worktree*Recheck.test.js):
// a verified path becomes a `[data-path]` link, an unverified one is
// demoted to a `[data-path-missing]` span.

'use strict';

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

module.exports = {findLinks, findMissing};
