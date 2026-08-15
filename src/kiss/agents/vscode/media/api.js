// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/* global module */
(function (global) {
  'use strict';

  const SORCAR_API_COMMANDS = [
    'run',
    'submit',
    'appendUserMessage',
    'stop',
    'userAnswer',
    'newChat',
    'openTab',
    'closeTab',
    'resumeSession',
    'ready',
    'getHistory',
    'getAdjacentTask',
    'getFrequentTasks',
    'deleteFrequentTask',
    'setFavorite',
    'getInputHistory',
    'getWelcomeSuggestions',
    'activeTasksQuery',
    'getModels',
    'selectModel',
    'getConfig',
    'saveConfig',
    'setWorkDir',
    'getFiles',
    'recordFileUsage',
    'openFile',
    'checkPaths',
    'complete',
    'worktreeAction',
    'generateCommitMessage',
    'autocommitAction',
    'auth',
    'runUpdate',
    'serverReset',
    'voiceTranscribe',
    'voiceToggle',
    'voiceSensitivity',
    'voiceAck',
    'voiceDropped',
    'focusEditor',
    'webviewFocusChanged',
    'activeTabChanged',
    'notificationAction',
    'sizeReport',
    'resolveDroppedPaths',
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

  global.SORCAR_API_COMMANDS = SORCAR_API_COMMANDS;
  global.createSorcarApi = createSorcarApi;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      createSorcarApi: createSorcarApi,
      SORCAR_API_COMMANDS: SORCAR_API_COMMANDS,
    };
  }
})(typeof window !== 'undefined' ? window : globalThis);
