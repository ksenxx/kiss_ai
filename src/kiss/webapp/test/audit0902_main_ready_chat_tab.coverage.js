// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the on-screen chat
// tab resolution in media/main.js (the regions fenced by
// `// readychat-coverage:start` / `// readychat-coverage:end`) when
// running the functional jsdom suite in
// test/audit0902_main_ready_chat_tab.test.js.  That suite
// drives chatTabIdForHost() through every branch: an active chat tab,
// an active content tab with a live owner, one whose owner is gone
// (fallback to any visible chat), and one with no visible chat left
// ('').

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'audit0902_main_ready_chat_tab.test.js',
  mark: 'readychat',
  label: 'ready chat-tab',
  targets: [
    {file: 'main.js', sourceUrl: 'audit0902-readychat-main.js', minRegions: 2},
  ],
});
