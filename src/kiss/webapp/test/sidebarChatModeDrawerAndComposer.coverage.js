// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires every byte of the code that decides which
// chat a file screen stands for (the regions of media/main.js fenced by
// `// filechat-coverage:start` / `// filechat-coverage:end`) to run in
// test/sidebarChatModeDrawerAndComposer.test.js: the chat that Stop,
// Share chat and voice act on while a content tab is on screen, the
// chat whose workspace that tab belongs to, and the Stop button shown
// for it.
//
// One outcome cannot be reached: reportSurvivingChatTab() runs after
// its callers (closeTab, reconcileTabs) put a tab on screen, so there
// the active tab always exists.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'sidebarChatModeDrawerAndComposer.test.js',
  mark: 'filechat',
  label: "file screen's chat",
  everyByte: true,
  targets: [{file: 'main.js', sourceUrl: 'filechat-main.js', regions: 4}],
});
