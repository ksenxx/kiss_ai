// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

/** The worktree action bar, or null when no branch awaits a decision. */
function bar(win) {
  return win.document.querySelector('.wt-bar');
}

/** The worktree action buttons in display order. */
function barButtons(win) {
  return Array.from(win.document.querySelectorAll('.wt-bar .wt-btns .wt-btn'));
}

module.exports = {bar, barButtons};
