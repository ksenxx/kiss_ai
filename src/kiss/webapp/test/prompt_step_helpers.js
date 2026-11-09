// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const {send} = require('./simplify2_harness');

/** Send one thinking/tool step before checking prompt-panel collapse behavior. */
function sendStep(win, i) {
  send(win, {type: 'thinking_start'});
  send(win, {type: 'thinking_delta', text: 'thought ' + i});
  send(win, {type: 'thinking_end'});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'echo ' + i,
    description: 'step ' + i,
  });
  send(win, {
    type: 'tool_result',
    name: 'Bash',
    output: String(i),
    success: true,
  });
}

module.exports = {sendStep};
