// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (JSDOM) tests for the settings panel's Claude Code
// subpanel, driving the real chat.html + api.js + main.js through the
// shared agent-subpanel body (settings_agent_subpanel.js): layout and
// defaults, the `claudeCodeStatus` wording for every reply shape,
// `claude_code_use_subscription` load/save with the setup token, and
// replies for other machines.  The subpanel order is pinned in
// appearanceMenuAndMachineSettings.test.js.
'use strict';

const {runAgentSubpanelTests} = require('./settings_agent_subpanel');

const BASE = {
  installed: true,
  loggedIn: false,
  authMethod: 'none',
  useSubscription: true,
  hasToken: false,
};

const CLAUDE_CODE = {
  subpanel: 'claude-code',
  cli: 'claude',
  statusCmd: 'getClaudeCodeStatus',
  statusReply: 'claudeCodeStatus',
  tokenKey: 'CLAUDE_CODE_OAUTH_TOKEN',
  configKey: 'claude_code_use_subscription',
  loggedInAuth: 'claude.ai',
  statusCases: [
    [
      {
        ...BASE,
        installed: false,
        error: "Claude Code CLI ('claude') not found on PATH.",
      },
      /not installed on this machine/,
      'warn',
    ],
    [
      {...BASE, error: '`claude auth status` (exit 1) returned no JSON: boom'},
      /Could not read the claude login: .*boom/,
      'warn',
    ],
    [BASE, /Not logged in: run `claude login`/, 'warn'],
    [
      {...BASE, useSubscription: false},
      /Not logged in; cc\/ models will use ANTHROPIC_API_KEY/,
      'warn',
    ],
    [
      {
        ...BASE,
        loggedIn: true,
        authMethod: 'claude.ai',
        subscriptionType: 'max',
        email: 'me@example.com',
      },
      /^Logged in as me@example.com \(max\): cc\/ models bill your Claude subscription\.$/,
      'ok',
    ],
    [
      {...BASE, loggedIn: true, authMethod: 'oauth_token', hasToken: true},
      /^Logged in: cc\/ models bill your Claude subscription\.$/,
      'ok',
    ],
    [
      {...BASE, loggedIn: true, authMethod: 'api_key', useSubscription: false},
      /authenticates with api_key: cc\/ models are billed per token/,
      'warn',
    ],
    [
      {...BASE, loggedIn: true, authMethod: '', useSubscription: false},
      /authenticates with an unknown method/,
      'warn',
    ],
  ],
};

runAgentSubpanelTests(CLAUDE_CODE);
console.log('settingsClaudeCode: all tests passed');
