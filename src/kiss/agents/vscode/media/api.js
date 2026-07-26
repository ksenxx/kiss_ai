// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * SorcarApi — the client facade of the Sorcar server API.
 *
 * The ONLY channel through which the chat UI (media/main.js, in both
 * the VS Code webview and the remote webapp) talks to the KISS Sorcar
 * server.  Each method maps 1:1 onto a command of the server API
 * catalog defined in ``src/kiss/server/sorcar.py`` (the single source
 * of truth); the daemon validates every command against that catalog
 * and answers invalid ones with an ``error`` event.
 *
 * Usage (main.js owns the single ``acquireVsCodeApi()`` handle):
 *
 *   const api = createSorcarApi(msg => vscode.postMessage(msg));
 *   api.stop({tabId: activeTabId});
 *   api.getConfig();
 *   api.send(prebuiltMessage);   // validated generic escape hatch
 */
/* global module */
(function (global) {
  'use strict';

  /**
   * Command names of the Sorcar server API (mirrors the ``API``
   * catalog in ``src/kiss/server/sorcar.py``) plus the VS Code
   * host-only webview messages that never reach the daemon.
   */
  const SORCAR_API_COMMANDS = [
    // session / task lifecycle
    'run',
    'submit',
    'appendUserMessage',
    'stop',
    'userAnswer',
    'newChat',
    'closeTab',
    'resumeSession',
    'ready',
    // history / metadata
    'getHistory',
    'getAdjacentTask',
    'getFrequentTasks',
    'deleteTask',
    'deleteFrequentTask',
    'setFavorite',
    'getInputHistory',
    'getWelcomeSuggestions',
    'activeTasksQuery',
    // models / configuration
    'getModels',
    'selectModel',
    'getConfig',
    'saveConfig',
    'setWorkDir',
    // files / autocomplete
    'getFiles',
    'recordFileUsage',
    'openFile',
    'complete',
    // worktree / merge / commit flows
    'mergeAction',
    'worktreeAction',
    'autocommitAction',
    'generateCommitMessage',
    // daemon administration
    'auth',
    'runUpdate',
    'serverReset',
    // voice
    'voiceTranscribe',
    'voiceToggle',
    'voiceSensitivity',
    'voiceAck',
    // VS Code host-only webview messages
    'focusEditor',
    'webviewFocusChanged',
    'notificationAction',
    'sizeReport',
    'resolveDroppedPaths',
  ];

  /**
   * Build the API client.
   *
   * @param {function(Object)} post Transport function delivering one
   *     command object to the server (``vscode.postMessage`` in the
   *     webview, the WebSocket shim in the remote webapp).
   * @returns {Object} An object with one method per API command —
   *     ``api.stop({tabId})`` posts ``{type: 'stop', tabId}`` — plus
   *     ``api.send(msg)``, which posts a prebuilt command after
   *     checking its ``type`` is part of the API.
   */
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
        // Set ``type`` last: the method's identity always wins, so a
        // stray ``type`` field in ``fields`` can never rebrand the
        // command into a different (or out-of-catalog) one.
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
