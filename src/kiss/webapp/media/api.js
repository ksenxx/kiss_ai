// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/* global module */
(function (global) {
  'use strict';

  // The daemon commands the chat page sends.  Commands the daemon
  // accepts from other clients only (the Python ``run`` API,
  // ``getTabsState``, the ``ready`` fan-out's ``getModels``) are not
  // here.
  const SORCAR_API_COMMANDS = [
    'submit',
    'appendUserMessage',
    'stop',
    'interruptTool',
    'userAnswer',
    'newChat',
    'openTab',
    'closeTab',
    'resumeSession',
    'ready',
    'getHistory',
    'getAdjacentTask',
    'setFavorite',
    'setChatSummary',
    'deleteTask',
    'deleteChat',
    'openExternal',
    'getInputHistory',
    'getSeaCommands',
    'getWelcomeInfo',
    'selectModel',
    'getConfig',
    'saveConfig',
    'getMyModels',
    'getClaudeCodeStatus',
    'getCodexStatus',
    'startCodexLogin',
    'saveMyModel',
    'deleteMyModel',
    'addTrick',
    'deleteTrick',
    'editTrick',
    'setWorkDir',
    'setMachine',
    'getFiles',
    'recordFileUsage',
    'openFile',
    'saveFile',
    'checkPaths',
    'getTaskUpdate',
    'getCronJobs',
    'cronAction',
    'getMachines',
    'machineAction',
    'getMachineWorkDirs',
    'getAppsStatus',
    'getSpendReport',
    'getRecommendations',
    'listDir',
    'gitStatus',
    'gitLog',
    'gitShow',
    'gitAction',
    'fsAction',
    'shareChat',
    'shareChatTasks',
    'complete',
    'worktreeAction',
    'autocommitAction',
    'runUpdate',
    'updateModels',
    'snoozeUpdate',
    'updateWhenIdle',
    'serverReset',
    'voiceTranscribe',
    'tipsOptOut',
    'dismissNotice',
    'signinLogin',
    'signinPaste',
    'signinCancel',
    'browserOpen',
    'browserClose',
    'browserNavigate',
    'browserInput',
    'browserViewport',
    'terminalOpen',
    'terminalInput',
    'terminalResize',
    'terminalClose',
  ];

  function createSorcarApi(post) {
    const api = {
      send: function (msg) {
        if (!msg || SORCAR_API_COMMANDS.indexOf(msg.type) < 0) {
          throw new Error('SorcarApi: unknown command ' + (msg && msg.type));
        }
        post(msg);
      },
    };
    SORCAR_API_COMMANDS.forEach(name => {
      api[name] = function (fields) {
        const msg = {};
        if (fields) {
          Object.keys(fields).forEach(k => {
            msg[k] = fields[k];
          });
        }
        msg.type = name;
        post(msg);
      };
    });
    return api;
  }

  global.createSorcarApi = createSorcarApi;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      createSorcarApi: createSorcarApi,
      SORCAR_API_COMMANDS: SORCAR_API_COMMANDS,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
