// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The Projects page: a "New project" button and, under it, one settings
// panel per project.  A project is a named group of chats whose
// settings override the product's (the daemon half is
// kiss/agents/sorcar/projects.py).  A panel has the look of the
// Settings sheet: name, description, default working directory, max
// budget, the check boxes, the API keys, then Cancel and Save, and a
// pin button at its top right corner.
//
//   const view = ProjectsView.create(host, {
//     api,                 // saveProject / deleteProject / setProjectPinned
//     wireWorkDirSelect,   // (select, value, noneLabel): fill the folder menu
//     newFolder,           // (select): create a folder and select it
//     dressSecretInput,    // (input, noun): mask a key box, add its eye toggle
//     close,               // (): close the page's tab, content window too
//   });
//   view.update({projects, config, apiKeys})  -> repaint
//   view.openNewProject()    -> show the New project panel
//   view.focusProject(id)    -> scroll to a project's panel
//   view.onSaved(ev)         -> a projectSaved reply
//   view.onError(ev)         -> a projectError reply
//   view.disconnected()      -> the daemon connection dropped
//   view.refuseSaves(text)   -> no answer comes to the saves that are out
//   view.destroy()           -> remove the listeners
//
// The New project panel ends the page's visit: once its project is
// saved, or its Cancel is pressed, the page asks to be closed (`close`).
// Its form is locked while its create is on its way (lockForm).
//
// A panel is pre-filled with the product's settings overlaid with what
// the project overrides.  Save sends what the project overrides and
// what the user edited (collectSettings); the daemon keeps of it only
// what differs from the product's settings.  A field the user left
// alone is not sent, so it keeps following them: such a field is
// filled again whenever the settings change, unsaved edits elsewhere
// in the panel or not (paintSettings).

(function (global) {
  'use strict';

  const PRODUCT =
    (global.__BRAND__ && global.__BRAND__.productName) || 'KISS Sorcar';

  // The check boxes of the Settings sheet, by the settings key each holds.
  const CHECKBOXES = [
    ['use_web_browser', 'Use web tools'],
    ['use_memory', 'Use persistent memory'],
    ['compaction', 'Compaction'],
    ['chrome_bridge', 'Drive my Chrome'],
    ['victory_sound', 'Play a sound when a task finishes'],
    [
      'claude_code_use_subscription',
      'Bill my Claude subscription for cc/ models (hide ANTHROPIC_API_KEY from the claude CLI)',
    ],
    [
      'codex_use_subscription',
      'Bill my ChatGPT subscription for codex/ models (hide OPENAI_API_KEY and CODEX_API_KEY from the codex CLI)',
    ],
  ];

  // The key fields of the Settings sheet, in its collapsible groups:
  // [variable, label, placeholder].
  const KEY_GROUPS = [
    [
      'API Keys',
      [
        ['ANTHROPIC_API_KEY', 'Anthropic API Key', 'Enter Anthropic API key'],
        [
          'ANTHROPIC_WORKSPACE_ID',
          'Anthropic Workspace ID (identity-linked keys only)',
          'wrkspc_...',
        ],
        ['OPENAI_API_KEY', 'OpenAI API Key', 'Enter OpenAI API key'],
        ['GEMINI_API_KEY', 'Gemini API Key', 'Enter Gemini API key'],
        [
          'OPENROUTER_API_KEY',
          'OpenRouter API Key',
          'Enter OpenRouter API key',
        ],
      ],
    ],
    [
      'Amazon Bedrock',
      [
        [
          'AWS_BEARER_TOKEN_BEDROCK',
          'Amazon Bedrock API Key (bearer token)',
          'Enter Bedrock API key',
        ],
        [
          'AWS_ACCESS_KEY_ID',
          'AWS Access Key ID (Bedrock, if no bearer token)',
          'AKIA...',
        ],
        [
          'AWS_SECRET_ACCESS_KEY',
          'AWS Secret Access Key (Bedrock)',
          'Enter AWS secret access key',
        ],
        [
          'AWS_SESSION_TOKEN',
          'AWS Session Token (Bedrock, temporary credentials only)',
          'Enter AWS session token',
        ],
        ['AWS_REGION', 'AWS Region (Bedrock)', 'us-east-1'],
      ],
    ],
    [
      'Claude Code',
      [
        [
          'CLAUDE_CODE_OAUTH_TOKEN',
          'Claude subscription token (optional)',
          'sk-ant-oat01-...',
        ],
      ],
    ],
    [
      'Codex',
      [
        [
          'CODEX_ACCESS_TOKEN',
          'Codex access token (optional)',
          'Codex access token',
        ],
      ],
    ],
  ];

  // The pin of a project's page in Claude Code Desktop: a pushpin
  // tilted 45 degrees, an outline while the project is not pinned and
  // solid while it is (main.css shows one of the two paths).  Claude's
  // own glyph is not free to copy; these are the "push-pin" drawings
  // of Phosphor Icons, regular and fill (@phosphor-icons/core 2.1.1),
  // the free ones nearest to it.  The viewBox sets them in the middle
  // 70% of the icon's square and the outline's stroke brings its line
  // to 1.25px at 20px, as Claude draws its pin.
  //
  // Phosphor Icons: MIT License, Copyright (c) 2023 Phosphor Icons.
  // Permission is hereby granted, free of charge, to any person
  // obtaining a copy of this software and associated documentation
  // files (the "Software"), to deal in the Software without
  // restriction, including without limitation the rights to use, copy,
  // modify, merge, publish, distribute, sublicense, and/or sell copies
  // of the Software, and to permit persons to whom the Software is
  // furnished to do so, subject to the following conditions: The above
  // copyright notice and this permission notice shall be included in
  // all copies or substantial portions of the Software.  THE SOFTWARE
  // IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  // MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  // NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  // HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
  // WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  // DEALINGS IN THE SOFTWARE.
  const PIN_SVG =
    '<svg width="20" height="20" viewBox="-12.57 -28.54 297.08 297.08" fill="currentColor" aria-hidden="true">' +
    '<path class="project-pin-outline" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" d="M235.32,81.37,174.63,20.69a16,16,0,0,0-22.63,0L98.37,74.49c-10.66-3.34-35-7.37-60.4,13.14a16,16,0,0,0-1.29,23.78L85,159.71,42.34,202.34a8,8,0,0,0,11.32,11.32L96.29,171l48.29,48.29A16,16,0,0,0,155.9,224c.38,0,.75,0,1.13,0a15.93,15.93,0,0,0,11.64-6.33c19.64-26.1,17.75-47.32,13.19-60L235.33,104A16,16,0,0,0,235.32,81.37ZM224,92.69h0l-57.27,57.46a8,8,0,0,0-1.49,9.22c9.46,18.93-1.8,38.59-9.34,48.62L48,100.08c12.08-9.74,23.64-12.31,32.48-12.31A40.13,40.13,0,0,1,96.81,91a8,8,0,0,0,9.25-1.51L163.32,32,224,92.68Z"/>' +
    '<path class="project-pin-solid" d="M235.33,104l-53.47,53.65c4.56,12.67,6.45,33.89-13.19,60A15.93,15.93,0,0,1,157,224c-.38,0-.75,0-1.13,0a16,16,0,0,1-11.32-4.69L96.29,171,53.66,213.66a8,8,0,0,1-11.32-11.32L85,159.71l-48.3-48.3A16,16,0,0,1,38,87.63c25.42-20.51,49.75-16.48,60.4-13.14L152,20.7a16,16,0,0,1,22.63,0l60.69,60.68A16,16,0,0,1,235.33,104Z"/></svg>';

  const CHEVRON_SVG =
    '<svg class="config-subpanel-chevron" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="9 18 15 12 9 6"/></svg>';

  /** Markup of one check box of a panel (the labels are constants). */
  function checkboxMarkup(box) {
    return (
      '<label class="config-label config-checkbox"><input type="checkbox" data-setting="' +
      box[0] +
      '"> ' +
      box[1] +
      '</label>'
    );
  }

  /** Markup of one key field: masked, but for the region. */
  function keyMarkup(key) {
    return (
      '<label class="config-label">' +
      key[1] +
      '<input type="' +
      (key[0] === 'AWS_REGION' ? 'text' : 'password') +
      '" autocomplete="off" data-key="' +
      key[0] +
      '" placeholder="' +
      key[2] +
      '"></label>'
    );
  }

  /** Markup of one collapsible group of key fields, folded. */
  function keyGroupMarkup(group) {
    return (
      '<div class="config-subpanel"><button type="button" class="config-subpanel-toggle" aria-expanded="false">' +
      CHEVRON_SVG +
      '<span>' +
      group[0] +
      '</span></button><div class="config-subpanel-body" hidden>' +
      group[1].map(keyMarkup).join('') +
      '</div></div>'
    );
  }

  const PANEL_MARKUP =
    '<button type="button" class="project-pin-btn" aria-pressed="false">' +
    PIN_SVG +
    '</button>' +
    '<label class="config-label">Project name' +
    '<input type="text" class="project-name" maxlength="200" autocomplete="off" placeholder="Name of the project"></label>' +
    '<label class="config-label">Description' +
    '<textarea class="project-description" rows="2" maxlength="2000" placeholder="What the project is about"></textarea></label>' +
    '<div class="config-label">Default working directory' +
    '<div class="project-workdir-row"><select class="project-workdir" aria-label="Default working directory"></select>' +
    '<button type="button" class="config-update-btn project-new-folder">New folder</button></div></div>' +
    '<label class="config-label">Max budget per task ($)' +
    '<input type="number" class="project-max-budget" min="0" step="1"></label>' +
    CHECKBOXES.map(checkboxMarkup).join('') +
    '<div class="config-divider"></div>' +
    KEY_GROUPS.map(keyGroupMarkup).join('') +
    '<div class="project-error" role="alert" hidden></div>' +
    '<div class="project-confirm" hidden><span>Delete this project? Its chats are kept.</span>' +
    '<button type="button" class="config-update-btn project-delete-yes">Delete</button>' +
    '<button type="button" class="config-update-btn project-delete-no">Keep</button></div>' +
    '<div class="project-actions">' +
    '<button type="button" class="config-update-btn project-delete">Delete</button>' +
    '<span class="project-actions-gap"></span>' +
    '<button type="button" class="config-update-btn project-cancel">Cancel</button>' +
    '<button type="button" class="config-update-btn project-save">Save</button></div>';

  const PAGE_MARKUP =
    '<div class="projects-page-inner"><div class="projects-header">' +
    '<h2 class="projects-title">Projects</h2>' +
    '<button type="button" class="config-update-btn projects-new-btn">New project</button></div>' +
    '<p class="projects-intro">A project groups chats and gives them settings of their own. ' +
    'A setting you leave as it is keeps following the settings of ' +
    PRODUCT +
    '.</p><div class="projects-new-slot"></div>' +
    '<p class="projects-empty">No projects yet.</p><div class="projects-list"></div></div>';

  // What the lock of the New project panel disables: all but Cancel.
  const LOCKED = 'input, textarea, select, button:not(.project-cancel)';

  // Shown in a panel whose save was out when the connection dropped.
  const UNCONFIRMED =
    'The server connection dropped before this save was confirmed. ' +
    'Once it is back, save again if the project is not listed as you saved it.';

  // The saves sent since the page loaded, by every Projects page it has
  // had: an answer still on its way to a page whose tab was closed
  // must not pass for the answer to a save of the page opened after it.
  let saves = 0;

  /**
   * A settings panel: the New project panel (`project` null) or the
   * panel of an existing project.  `dirty`: the user edited a field
   * since the last paint or the last save; `edited`: the fields the
   * user edited since the panel was last painted whole; `requestId`:
   * the save that is out.  Its masked key boxes get the eye toggle of
   * the Settings sheet's (`dressSecretInput` of main.js), named as
   * there: the workspace id is an identifier, not a key.
   */
  function makePanel(project, dressSecretInput) {
    const el = document.createElement('section');
    el.className = 'project-panel';
    el.innerHTML = PANEL_MARKUP;
    for (const input of el.querySelectorAll('input[type="password"]')) {
      dressSecretInput(
        input,
        input.dataset.key === 'ANTHROPIC_WORKSPACE_ID'
          ? 'workspace ID'
          : 'API key',
      );
    }
    return {el, project, dirty: false, edited: new Set(), requestId: ''};
  }

  /** The one element of *panel* that matches *selector*. */
  function field(panel, selector) {
    return panel.el.querySelector(selector);
  }

  /** Whether *panel*'s pin button shows the project as pinned. */
  function pinShown(panel) {
    return (
      field(panel, '.project-pin-btn').getAttribute('aria-pressed') === 'true'
    );
  }

  /** Show *panel*'s pin button as pinned or not. */
  function paintPin(panel, pinned) {
    const button = field(panel, '.project-pin-btn');
    const label = pinned ? 'Unpin project' : 'Pin project';
    button.setAttribute('aria-pressed', String(pinned));
    button.setAttribute('aria-label', label);
    button.title = label;
  }

  /** Bring *panel* into view and put the keyboard in its name box. */
  function reveal(panel) {
    panel.el.scrollIntoView({block: 'nearest'});
    field(panel, '.project-name').focus();
  }

  /** Show *text* as *panel*'s error line; '' hides the line. */
  function showError(panel, text) {
    const line = field(panel, '.project-error');
    line.textContent = text;
    line.hidden = !text;
  }

  /** Fold or unfold the key group headed by *button*. */
  function toggleSubpanel(button) {
    const body = button.nextElementSibling;
    body.hidden = !body.hidden;
    button.classList.toggle('expanded', !body.hidden);
    button.setAttribute('aria-expanded', String(!body.hidden));
  }

  /**
   * Fill *panel*'s settings fields (the folder, the budget, the check
   * boxes and the keys) with the product's settings overlaid with what
   * its project overrides.  A field the user edited keeps the edit; the
   * others follow the settings as the page last heard of them.
   */
  function paintSettings(panel, page) {
    const own = panel.project ? panel.project.settings : {};
    const settings = Object.assign({}, page.config, own);
    const keys = Object.assign({}, page.apiKeys, own.api_keys);
    const workDir = field(panel, '.project-workdir');
    if (!panel.edited.has(workDir)) {
      page.deps.wireWorkDirSelect(
        workDir,
        settings.default_work_dir || '',
        'Same as ' + PRODUCT,
      );
    }
    const budget = field(panel, '.project-max-budget');
    if (!panel.edited.has(budget)) {
      budget.value =
        settings.max_budget === undefined ? '' : String(settings.max_budget);
    }
    for (const box of panel.el.querySelectorAll('input[data-setting]')) {
      if (!panel.edited.has(box)) box.checked = !!settings[box.dataset.setting];
    }
    for (const input of panel.el.querySelectorAll('input[data-key]')) {
      if (!panel.edited.has(input)) input.value = keys[input.dataset.key] || '';
    }
  }

  /**
   * Fill *panel*'s fields with the product's settings overlaid with
   * what its project overrides, and drop the user's unsaved edits.
   */
  function paintPanel(panel, page) {
    const project = panel.project;
    panel.el.dataset.projectId = project ? project.id : '';
    field(panel, '.project-name').value = project ? project.name : '';
    field(panel, '.project-description').value = project
      ? project.description
      : '';
    panel.edited.clear();
    paintSettings(panel, page);
    paintPin(panel, !!project && project.isPinned);
    field(panel, '.project-delete').hidden = !project;
    field(panel, '.project-confirm').hidden = true;
    showError(panel, '');
    panel.dirty = false;
  }

  /**
   * Repaint *panel* now that the page's data changed.  One whose save
   * is out is left as it is (what it shows is on its way to the
   * daemon); one the user has unsaved edits in keeps them, and only the
   * settings fields the user did not edit follow; any other is painted
   * whole.
   */
  function repaintPanel(panel, page) {
    if (panel.requestId) return;
    if (panel.dirty) paintSettings(panel, page);
    else paintPanel(panel, page);
  }

  /**
   * Lock or unlock the form of the New project panel.  It is locked
   * while its create is out: the answer closes the page, so an edit
   * made meanwhile would go with it, and a second Save would create a
   * second project.  Cancel stays.
   */
  function lockForm(panel, locked) {
    for (const control of panel.el.querySelectorAll(LOCKED)) {
      control.disabled = locked;
    }
  }

  /**
   * The settings *panel*'s project is to have: what it overrides
   * already, with the fields the user edited as the form shows them.
   * A field the user left alone is not read.  It shows this page's
   * copy of the product's settings, which is older than the daemon's
   * once another page has changed them (or blank, before the daemon
   * has sent any), and the daemon keeps what differs from its own as
   * the project's.
   */
  function collectSettings(panel, page) {
    const own = panel.project ? panel.project.settings : {};
    const settings = Object.assign({}, own, {
      api_keys: Object.assign({}, own.api_keys),
    });
    const workDir = field(panel, '.project-workdir');
    if (panel.edited.has(workDir)) settings.default_work_dir = workDir.value;
    // A number box holds a number or nothing, which follows the product.
    const budget = field(panel, '.project-max-budget');
    if (panel.edited.has(budget)) {
      delete settings.max_budget;
      if (budget.value !== '') settings.max_budget = Number(budget.value);
    }
    for (const box of panel.el.querySelectorAll('input[data-setting]')) {
      if (panel.edited.has(box)) settings[box.dataset.setting] = box.checked;
    }
    // The sound is a preference of this browser, not a setting of the
    // product: the project keeps it only while it differs.
    if (settings.victory_sound === !!page.config.victory_sound)
      delete settings.victory_sound;
    for (const input of panel.el.querySelectorAll('input[data-key]')) {
      if (panel.edited.has(input))
        settings.api_keys[input.dataset.key] = input.value.trim();
    }
    return settings;
  }

  /** The Projects page built in one host element. */
  class ProjectsPage {
    /**
     * @param {HTMLElement} host The empty element the page fills.
     * @param {{api: object, wireWorkDirSelect: Function, newFolder: Function, dressSecretInput: Function, close: Function}} deps
     *   What the page needs from main.js (see the head of this file).
     */
    constructor(host, deps) {
      this.host = host;
      this.deps = deps;
      this.config = {};
      this.apiKeys = {};
      this.panels = new Map(); // project id -> its panel
      host.classList.add('projects-page');
      host.innerHTML = PAGE_MARKUP;
      this.newPanel = makePanel(null, deps.dressSecretInput);
      this.newPanel.el.hidden = true;
      host.querySelector('.projects-new-slot').appendChild(this.newPanel.el);
      for (const type of ['click', 'input', 'change'])
        host.addEventListener(type, this);
    }

    /**
     * Repaint the page.  A panel the user has unsaved edits in keeps
     * them, and so does one whose save is out (what it shows is on its
     * way to the daemon): its pin is repainted, and in the first the
     * settings fields the user left as they were (repaintPanel).
     *
     * @param {{projects: Array<object>, config: object, apiKeys: object}} data
     *   The daemon's projects (oldest first), the product's settings
     *   and its key values.
     */
    update(data) {
      this.config = data.config;
      this.apiKeys = data.apiKeys;
      const list = this.host.querySelector('.projects-list');
      const kept = new Map();
      for (const [index, project] of data.projects.entries()) {
        const panel =
          this.panels.get(project.id) ||
          makePanel(project, this.deps.dressSecretInput);
        panel.project = project;
        kept.set(project.id, panel);
        // Moved only when out of place: a move takes the focus away.
        const at = list.children[index] || null;
        if (at !== panel.el) list.insertBefore(panel.el, at);
        if (panel.dirty || panel.requestId) paintPin(panel, project.isPinned);
        repaintPanel(panel, this);
      }
      for (const [id, panel] of this.panels) {
        if (!kept.has(id)) panel.el.remove();
      }
      this.panels = kept;
      this.host.querySelector('.projects-empty').hidden = kept.size > 0;
      repaintPanel(this.newPanel, this);
    }

    /** Show the New project panel, filled with the product's settings. */
    openNewProject() {
      this.newPanel.el.hidden = false;
      reveal(this.newPanel);
    }

    /**
     * Bring the panel of a project into view.
     *
     * @param {string} id The project's id; an unknown one does nothing.
     */
    focusProject(id) {
      const panel = this.panels.get(id);
      if (panel) reveal(panel);
    }

    /**
     * The panel whose save is out under *requestId*, if any.
     *
     * @param {string} requestId The `requestId` of a reply.
     * @returns {object|undefined} The panel.
     */
    panelOfRequest(requestId) {
      for (const panel of [this.newPanel, ...this.panels.values()]) {
        if (panel.requestId && panel.requestId === requestId) return panel;
      }
      return undefined;
    }

    /**
     * A `projectSaved` reply: the save of the panel that asked is over,
     * and a new project's closes the page.  What the user typed into a
     * project's panel since the save was sent stays an unsaved edit.
     *
     * @param {{requestId: string}} ev The reply.
     */
    onSaved(ev) {
      const panel = this.panelOfRequest(ev.requestId);
      if (!panel) return;
      panel.requestId = '';
      if (panel === this.newPanel) this.deps.close();
    }

    /**
     * A `projectError` reply: the save of the panel that asked is
     * over, and the reply's text says why (refuse).
     *
     * @param {{requestId: string, text: string}} ev The reply.
     */
    onError(ev) {
      const panel = this.panelOfRequest(ev.requestId);
      if (panel) this.refuse(panel, ev.text);
    }

    /**
     * The daemon connection dropped.  The answer to a save that is out
     * never comes: the daemon sends it on the connection the save came
     * in on, and a save the transport had yet to send is not sent (its
     * `_GIVEN_UP_AT_DROP`).  Every such save is over, stored or not:
     * the list the page is sent once it is connected again tells.
     */
    disconnected() {
      this.refuseSaves(UNCONFIRMED);
    }

    /**
     * Every save that is out is over (refuse): none gets an answer
     * that names its panel, and *text* says why.
     *
     * @param {string} text The reason, shown in each such panel.
     */
    refuseSaves(text) {
      for (const panel of [this.newPanel, ...this.panels.values()]) {
        if (panel.requestId) this.refuse(panel, text);
      }
    }

    /**
     * End the save *panel* has out without an answer that it was
     * stored: its form is an unsaved edit again, the New project
     * panel's unlocked for another try, and *text* says why.  The
     * settings fields the user left alone follow what the page was
     * sent while the save was out, which repaintPanel kept off the
     * panel.
     *
     * @param {object} panel The panel whose save is over.
     * @param {string} text The reason, shown in the panel.
     */
    refuse(panel, text) {
      panel.requestId = '';
      panel.dirty = true;
      if (!panel.project) lockForm(panel, false);
      paintSettings(panel, this);
      showError(panel, text);
    }

    /** Remove the page's listeners; the caller empties the host. */
    destroy() {
      for (const type of ['click', 'input', 'change'])
        this.host.removeEventListener(type, this);
    }

    /**
     * The page's one listener: an edit marks its field as edited and its
     * panel as having unsaved edits, a click runs its button, unless the
     * button is disabled (a locked form's).
     *
     * @param {Event} ev A click, input or change event of the host.
     */
    handleEvent(ev) {
      const panelEl = ev.target.closest('.project-panel');
      const panel =
        panelEl === this.newPanel.el
          ? this.newPanel
          : this.panels.get(panelEl && panelEl.dataset.projectId);
      if (ev.type !== 'click') {
        if (panel) {
          panel.dirty = true;
          panel.edited.add(ev.target);
        }
        return;
      }
      const button = ev.target.closest('button');
      if (!button || button.disabled) return;
      if (button.classList.contains('projects-new-btn')) this.openNewProject();
      else if (panel) this.pressed(panel, button);
    }

    /**
     * Run a button of a panel.
     *
     * @param {object} panel The panel the button is in.
     * @param {HTMLButtonElement} button The button pressed.
     */
    pressed(panel, button) {
      const kind = button.classList;
      if (kind.contains('config-subpanel-toggle')) toggleSubpanel(button);
      else if (kind.contains('project-pin-btn')) this.togglePin(panel);
      else if (kind.contains('project-new-folder'))
        this.deps.newFolder(field(panel, '.project-workdir'));
      else if (kind.contains('project-save')) this.save(panel);
      else if (kind.contains('project-cancel')) this.cancel(panel);
      else if (kind.contains('project-delete'))
        field(panel, '.project-confirm').hidden = false;
      else if (kind.contains('project-delete-no'))
        field(panel, '.project-confirm').hidden = true;
      else if (kind.contains('project-delete-yes')) {
        this.deps.api.deleteProject({projectId: panel.project.id});
      }
    }

    /**
     * The pin button: a project's pin is stored at once; the New
     * project panel's is an edit of its form and travels with its save.
     *
     * @param {object} panel The panel whose pin was pressed.
     */
    togglePin(panel) {
      if (panel.project) {
        this.deps.api.setProjectPinned({
          projectId: panel.project.id,
          pinned: !panel.project.isPinned,
        });
      } else {
        paintPin(panel, !pinShown(panel));
        panel.dirty = true;
      }
    }

    /**
     * Send *panel*'s form to the daemon; a form without a name is not
     * sent.  What is sent is no unsaved edit: only what is typed from
     * now on is one.  The New project panel's form is locked until the
     * daemon answers (lockForm).
     *
     * @param {object} panel The panel whose Save was pressed.
     */
    save(panel) {
      const name = field(panel, '.project-name').value.trim();
      if (!name) {
        showError(panel, 'A project needs a name.');
        field(panel, '.project-name').focus();
        return;
      }
      showError(panel, '');
      saves += 1;
      panel.requestId = 'project-save-' + saves;
      panel.dirty = false;
      const project = {
        name,
        description: field(panel, '.project-description').value.trim(),
        settings: collectSettings(panel, this),
      };
      if (panel.project) project.id = panel.project.id;
      else project.isPinned = pinShown(panel);
      this.deps.api.saveProject({project, requestId: panel.requestId});
      if (!panel.project) lockForm(panel, true);
    }

    /**
     * Drop *panel*'s unsaved edits; the New project panel's Cancel
     * closes the page.  A save that is out is let go of: its answer is
     * no longer the panel's.
     *
     * @param {object} panel The panel whose Cancel was pressed.
     */
    cancel(panel) {
      panel.requestId = '';
      if (panel.project) paintPanel(panel, this);
      else this.deps.close();
    }
  }

  /**
   * Build the Projects page inside *host*.
   *
   * @param {HTMLElement} host The empty element the page fills.
   * @param {object} deps `api`, `wireWorkDirSelect`, `newFolder`,
   *   `dressSecretInput` and `close`.
   * @returns {ProjectsPage} The page.
   */
  function create(host, deps) {
    return new ProjectsPage(host, deps);
  }

  global.ProjectsView = {create};
})(window);
