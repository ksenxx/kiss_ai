// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Coverage gate: requires 100% line coverage of the right sidebar's
// subpanel code in media/main.js -- Schedule, Machines, Apps,
// Recommendations and Spend, plus the chat-machine drop-down
// (`renderChatMachine`, `tabChatMachine`) that shares their machine
// state; the region fenced by `// sidebarpanels-coverage:start` /
// `// sidebarpanels-coverage:end` -- when running the functional jsdom
// suite in test/sidebarPanels.test.js.  A line counts as covered only
// when every byte of its code ran, so an untaken `return` or `||`
// operand on a line that otherwise ran fails the gate too.

'use strict';

const {gateRegions} = require('./coverage_gate');

gateRegions({
  testFile: 'sidebarPanels.test.js',
  mark: 'sidebarpanels',
  label: 'sidebar panels',
  everyByte: true,
  targets: [
    {file: 'main.js', sourceUrl: 'sidebarpanels-main.js'},
  ],
});
