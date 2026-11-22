// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the right sidebar's Skills subpanel (the
// skillspanel block of media/main.js, #meta-skills in media/chat.html):
//
// * the section sits right before Connectors, with a search field and
//   button under its header;
// * a `skills` reply lists the skills the agent can use (name, source
//   badge, description), the count line, the empty-state text or the
//   failure; the refresh button asks again and turns until the reply;
// * Enter in the field or its button sends `searchSkills` with the
//   trimmed text and a fresh `generation`, and shows a busy state
//   until the `skillSearch` echoing that generation arrives (any
//   other is dropped: it answers an earlier search); the found skills
//   are listed in the order received, each with a link to its page,
//   its source, its installs and a + (or "Added"); Clear removes them;
// * + sends `installSkill` and the row waits ("Checking…") for the
//   `skillInstall` of its id: installed, review (findings, "Add
//   anyway", which sends `force: true`, and "Cancel", which forgets
//   the verdict), blocked or error; a skill
//   the newest search says is there is Added whatever a check said;
// * nothing a reply carries is ever parsed as HTML, and a link takes
//   only an http(s) address;
// * every command names the folder the side views browse (`workDir`,
//   once one is known), and a change of that folder asks for its
//   skills;
// * a machine switch empties the subpanel and asks the new machine, a
//   list or search of another machine is dropped while a check keeps
//   its machine's row waiting and is answered wherever the subpanel
//   is, and a lost daemon connection releases every wait;
// * a check the transport has yet to send when the connection is
//   back (its + was pressed as the socket was closing) is awaited
//   again: on the real transport (media/transport.js).
//
// The moments the subpanel asks for its data (connect, poll, machine
// switch) are pinned with the other subpanels' in sidebarPanels.test.js;
// its place in the sections' layout in metaPanelSections.test.js.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  stubRaf,
  fakeMatchMedia,
  fakeIntervals,
  pickMachine,
  sendWorkDir,
  withTeardown,
} = require('./simplify2_harness');
const {makeRunner, key, IME_FORMS} = require('./ui_antipattern_harness');
const {trackWindow, closeWindows} = require('./workdir_helpers');
const {
  installFakeWebSocket,
  readShimJs,
  frames,
  settle,
} = require('./transport_page_helpers');

const {test: runTest, report} = makeRunner();
// Each test's windows are closed once it has settled.
const test = withTeardown(runTest, closeWindows);

/**
 * Boot a remote chat page whose daemon is up.  `desktop` (default
 * true) is the (min-width: 900px) media query: false boots the phone
 * layout, where the task-info panel is a drawer.
 */
function makePage(opts) {
  const {desktop = true} = opts || {};
  let intervals;
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    sourceUrl: 'skillspanel-main.js',
    beforeScripts(win) {
      stubRaf(win);
      intervals = fakeIntervals(win).intervals;
      fakeMatchMedia(win, desktop);
    },
  });
  trackWindow(win);
  send(win, {type: 'daemonStatus', connected: true});
  return {win, posted, intervals};
}

/** deepStrictEqual across realms: jsdom objects have foreign prototypes. */
function eq(actual, expected, message) {
  assert.deepStrictEqual(JSON.parse(JSON.stringify(actual)), expected, message);
}

function el(win, id) {
  return win.document.getElementById(id);
}

/** The Skills commands the page posted from index *from* on. */
function skillPosts(posted, from) {
  return posted
    .slice(from || 0)
    .filter(m =>
      ['getSkills', 'searchSkills', 'installSkill'].includes(m.type),
    );
}

const SKILLS = [
  {
    name: 'docx',
    description: 'Create, edit and analyze Word documents.',
    source: 'bundled',
  },
  {name: 'pdf-forms', description: 'Fill in PDF forms.', source: 'user'},
  {name: 'release-notes', description: '', source: ''},
];

const FOUND = [
  {
    id: 'acme/skills/pdf',
    name: 'pdf',
    source: 'acme/skills',
    installs: 233480,
    url: 'https://skills.example/acme/skills/pdf',
    installed: false,
  },
  {
    id: 'acme/skills/pdf-forms',
    name: 'pdf-forms',
    source: 'acme/skills',
    installs: 1200,
    url: 'http://skills.example/acme/skills/pdf-forms',
    installed: true,
  },
  {
    id: 'solo/kit/pdf-ocr',
    name: 'pdf-ocr',
    source: 'solo/kit',
    installs: 1,
    url: '',
    installed: false,
  },
];

const FINDINGS = [
  {
    severity: 'medium',
    rule: 'network',
    message: 'Downloads a script at run time',
    file: 'scripts/setup.sh',
    line: 12,
  },
  {
    severity: 'low',
    rule: 'docs',
    message: 'Asks for broad file access',
    file: 'SKILL.md',
    line: 0,
  },
  {severity: 'info', rule: 'note', message: 'No licence', file: '', line: 0},
];

/** The available skills as [{name, badge, sub, tip}], in list order. */
function skillRows(win) {
  return Array.from(
    win.document.querySelectorAll('#meta-skills-list .skill-row'),
    li => {
      const badge = li.querySelector('.sidebar-panel-badge');
      const sub = li.querySelector('.sidebar-panel-sub');
      return {
        name: li.querySelector('.sidebar-panel-name').textContent,
        badge: badge ? badge.textContent : null,
        sub: sub ? sub.textContent : null,
        tip: li.dataset.tooltip || null,
      };
    },
  );
}

function foundRow(win, id) {
  return Array.from(
    win.document.querySelectorAll('#meta-skills-results .skill-found-row'),
  ).find(li => li.dataset.skillId === id);
}

/** What the check's verdict block of a found row shows; null without one. */
function verdictShape(li) {
  const box = li.querySelector('.skill-verdict');
  if (!box) return null;
  return {
    message: box.querySelector('.skill-verdict-message').textContent,
    findings: Array.from(box.querySelectorAll('.skill-findings > li'), f => {
      const where = f.querySelector('.skill-finding-where');
      return [
        f.querySelector('.skill-severity').className,
        f.querySelector('.skill-severity').textContent,
        f.querySelector('.skill-finding-text').textContent,
        where ? where.textContent : null,
      ];
    }),
    buttons: Array.from(box.querySelectorAll('button'), b => b.textContent),
  };
}

/**
 * One found row as plain data: its name, the link's address (null for
 * a plain name), the second line, the state mark, and the + button
 * ('enabled', 'disabled' or null when the row has none).
 */
function foundShape(li) {
  const name = li.querySelector('.sidebar-panel-name');
  const state = li.querySelector('.skill-state');
  const add = li.querySelector('button.skill-add');
  return {
    name: name.textContent,
    href: name.tagName === 'A' ? name.getAttribute('href') : null,
    sub: li.querySelector('.sidebar-panel-sub').textContent,
    state: state ? state.textContent : null,
    add: add ? (add.disabled ? 'disabled' : 'enabled') : null,
  };
}

function foundShapes(win) {
  return Array.from(
    win.document.querySelectorAll('#meta-skills-results .skill-found-row'),
    foundShape,
  );
}

/** Type *text* into the search field and press Enter. */
function search(win, text) {
  const field = el(win, 'meta-skills-query');
  field.value = text;
  return key(win, field, 'Enter');
}

/**
 * Answer the newest search the page sent the way the daemon does: the
 * `skillSearch` echoes its `query` and `generation`, and a machine's
 * is stamped `forwarded` with the machine's name.  *fields* are the
 * `results` or the `error`, or replace what is echoed.
 */
function answerSearch(win, posted, fields) {
  const cmd = posted.filter(m => m.type === 'searchSkills').pop();
  const reply = {
    type: 'skillSearch',
    query: cmd.query,
    results: [],
    error: '',
    generation: cmd.generation,
  };
  if (cmd.machine) {
    reply.forwarded = true;
    reply.machine = cmd.machine;
  }
  send(win, Object.assign(reply, fields));
  return reply;
}

/** A page showing the results of a search for "pdf". */
function pageWithResults(opts) {
  const page = makePage(opts);
  search(page.win, 'pdf');
  answerSearch(page.win, page.posted, {results: FOUND});
  return page;
}

function addButton(win, id) {
  return foundRow(win, id).querySelector('button.skill-add');
}

/**
 * The daemon's side of a handshake on the fake socket *sock*:
 * `auth_ok`, then the `pong` that confirms what the transport flushed
 * on it and lets the frames held behind that go out.
 */
function authenticate(sock) {
  sock.fireOpen();
  sock.fireMessage({type: 'auth_ok'});
  sock.fireMessage({type: 'pong'});
}

/**
 * The page on the real transport (media/transport.js) over a fake
 * WebSocket, its first socket authenticated.  `sent(type)` is every
 * frame of *type* the daemon was sent so far, over all sockets.
 */
async function makeShimPage() {
  const sockets = [];
  const {win} = makeChatPage({
    bodyClass: 'remote-chat',
    sourceUrl: 'skillspanel-main.js',
    beforeScripts(win) {
      stubRaf(win);
      fakeIntervals(win);
      fakeMatchMedia(win, true);
      installFakeWebSocket(win, sockets, {connecting: true});
      win.eval(readShimJs()); // in place of makePage's recording transport
    },
  });
  trackWindow(win);
  authenticate(sockets[0]);
  // The shim hands main.js its events once jsdom has ended the page load.
  await settle();
  return {
    win,
    sockets,
    sent: type => sockets.flatMap(frames).filter(f => f.type === type),
  };
}

async function main() {
  await test('Skills sits right before Connectors: a header, a refresh button, the search field and its button', () => {
    const {win} = makePage();
    const section = el(win, 'meta-skills');
    const ids = Array.from(
      win.document.querySelectorAll('#meta-panel > .meta-section'),
      s => s.id,
    );
    assert.strictEqual(ids[ids.indexOf('meta-skills') + 1], 'meta-apps');
    assert.strictEqual(ids[ids.indexOf('meta-skills') - 1], 'meta-machines');
    assert.strictEqual(
      section.querySelector('.meta-section-toggle').textContent,
      'Skills',
    );
    assert.strictEqual(section.getAttribute('aria-label'), 'Skills');
    assert.strictEqual(
      el(win, 'meta-skills-refresh').getAttribute('aria-label'),
      'Refresh skills',
    );
    const field = el(win, 'meta-skills-query');
    assert.strictEqual(field.placeholder, 'Search skills on the internet');
    const button = el(win, 'meta-skills-search-btn');
    assert.strictEqual(button.getAttribute('aria-label'), 'Search skills');
    assert.ok(button.querySelector('svg'), 'the button is a magnifier icon');
    // The field and its button come before the scrolling body; the
    // results, the status line and the list are inside it.
    const body = el(win, 'meta-skills-body');
    assert.ok(
      el(win, 'meta-skills-search').compareDocumentPosition(body) &
        win.Node.DOCUMENT_POSITION_FOLLOWING,
    );
    for (const id of [
      'meta-skills-found',
      'meta-skills-status',
      'meta-skills-list',
    ])
      assert.ok(body.contains(el(win, id)), id + ' scrolls with the body');
    // Its separator names it, as every section's does.
    assert.strictEqual(
      section.nextElementSibling.getAttribute('aria-label'),
      'Resize the Skills section',
    );
    assert.ok(el(win, 'meta-skills-found').hidden, 'no search yet');
    assert.strictEqual(
      el(win, 'meta-skills-status').textContent,
      'Loading\u2026',
    );
  });

  await test('a `skills` reply lists the available skills: name, source badge and description', () => {
    const {win} = makePage();
    send(win, {type: 'skills', skills: SKILLS, error: ''});
    eq(skillRows(win), [
      {
        name: 'docx',
        badge: 'bundled',
        sub: 'Create, edit and analyze Word documents.',
        tip: 'Create, edit and analyze Word documents.',
      },
      {
        name: 'pdf-forms',
        badge: 'user',
        sub: 'Fill in PDF forms.',
        tip: 'Fill in PDF forms.',
      },
      // No source: no badge.  No description: no second line.
      {name: 'release-notes', badge: null, sub: null, tip: null},
    ]);
    const status = el(win, 'meta-skills-status');
    assert.strictEqual(status.textContent, '3 skills available');
    assert.ok(!status.classList.contains('is-error'));
    // The next reply replaces the rows.
    send(win, {type: 'skills', skills: SKILLS.slice(0, 1), error: ''});
    eq(
      skillRows(win).map(r => r.name),
      ['docx'],
    );
    assert.strictEqual(status.textContent, '1 skill available');
    // None at all: the line says how to get one.
    send(win, {type: 'skills', skills: [], error: ''});
    eq(skillRows(win), []);
    assert.strictEqual(
      status.textContent,
      'No skills yet. Search the internet above to add one.',
    );
    // A failure is the line, in the failure colour; a reply without a
    // list is an empty one.
    send(win, {type: 'skills', error: 'The skills folder cannot be read.'});
    eq(skillRows(win), []);
    assert.strictEqual(status.textContent, 'The skills folder cannot be read.');
    assert.ok(status.classList.contains('is-error'));
    send(win, {type: 'skills', skills: SKILLS, error: ''});
    assert.ok(
      !status.classList.contains('is-error'),
      'the next reply clears it',
    );
  });

  await test('the refresh button asks for the skills again and turns until the reply', () => {
    const {win, posted} = makePage();
    send(win, {type: 'skills', skills: SKILLS, error: ''});
    const refresh = el(win, 'meta-skills-refresh');
    const from = posted.length;
    click(win, refresh);
    eq(skillPosts(posted, from), [{type: 'getSkills'}]);
    assert.ok(refresh.disabled && refresh.classList.contains('spinning'));
    send(win, {type: 'skills', skills: SKILLS, error: ''});
    assert.ok(!refresh.disabled && !refresh.classList.contains('spinning'));
  });

  await test('Enter in the field searches with the trimmed text; the reply lists what was found in the order received', () => {
    const {win, posted} = makePage();
    const from = posted.length;
    const field = el(win, 'meta-skills-query');
    const button = el(win, 'meta-skills-search-btn');
    const found = el(win, 'meta-skills-found');
    const title = el(win, 'meta-skills-found-title');
    const clear = el(win, 'meta-skills-clear');
    field.value = '  pdf  ';
    // A key that is not Enter, and the Enter that picks an IME
    // candidate, search nothing.
    assert.ok(key(win, field, 'a'), 'another key is the field\u2019s own');
    for (const ime of IME_FORMS) key(win, field, 'Enter', ime);
    eq(skillPosts(posted, from), []);
    assert.ok(!key(win, field, 'Enter'), 'the Enter is consumed');
    eq(skillPosts(posted, from), [
      {type: 'searchSkills', query: 'pdf', generation: 1},
    ]);
    // Busy until the reply: the button is off, the block says so.
    assert.ok(button.disabled);
    assert.strictEqual(button.getAttribute('aria-busy'), 'true');
    assert.ok(!found.hidden);
    assert.strictEqual(title.textContent, 'Searching\u2026');
    assert.ok(clear.disabled, 'nothing to clear yet');
    eq(foundShapes(win), []);
    // A second Enter while the search is under way sends nothing.
    key(win, field, 'Enter');
    assert.strictEqual(skillPosts(posted, from).length, 1);

    // A reply that names no search, or another one, is not the answer.
    send(win, {type: 'skillSearch', query: 'pdf', results: FOUND, error: ''});
    answerSearch(win, posted, {results: FOUND, generation: 2});
    assert.strictEqual(title.textContent, 'Searching\u2026');
    assert.ok(button.disabled);
    eq(foundShapes(win), []);
    answerSearch(win, posted, {results: FOUND});
    assert.ok(!button.disabled);
    assert.strictEqual(button.getAttribute('aria-busy'), 'false');
    assert.ok(!clear.disabled);
    assert.strictEqual(title.textContent, 'Results for "pdf"');
    assert.ok(!title.classList.contains('is-error'));
    eq(foundShapes(win), [
      {
        name: 'pdf',
        href: 'https://skills.example/acme/skills/pdf',
        sub: 'acme/skills \u00b7 233.5K installs',
        state: null,
        add: 'enabled',
      },
      // Already on the machine: marked, with no + to press.
      {
        name: 'pdf-forms',
        href: 'http://skills.example/acme/skills/pdf-forms',
        sub: 'acme/skills \u00b7 1.2K installs',
        state: 'Added',
        add: null,
      },
      // No page on the registry: the name is plain text.
      {
        name: 'pdf-ocr',
        href: null,
        sub: 'solo/kit \u00b7 1 install',
        state: null,
        add: 'enabled',
      },
    ]);
    const link = foundRow(win, 'acme/skills/pdf').querySelector('a');
    assert.strictEqual(link.target, '_blank');
    assert.strictEqual(link.rel, 'noopener');
    assert.strictEqual(
      addButton(win, 'acme/skills/pdf').getAttribute('aria-label'),
      'Add pdf',
    );
  });

  await test('the search button searches too; an empty field asks for the most installed skills', () => {
    // On a phone the panel is a drawer: the subpanel works the same.
    const {win, posted} = makePage({desktop: false});
    const from = posted.length;
    const title = el(win, 'meta-skills-found-title');
    click(win, el(win, 'meta-skills-search-btn'));
    eq(skillPosts(posted, from), [
      {type: 'searchSkills', query: '', generation: 1},
    ]);
    answerSearch(win, posted, {
      results: [
        Object.assign({}, FOUND[0], {installs: 1500000}),
        Object.assign({}, FOUND[2], {installs: 999, source: ''}),
        {id: 'x/y/z', name: 'z'},
      ],
    });
    assert.strictEqual(title.textContent, 'Most installed skills');
    eq(
      foundShapes(win).map(r => r.sub),
      // No source: the installs alone.  No count: none.
      ['acme/skills \u00b7 1.5M installs', '999 installs', '0 installs'],
    );
    // Nothing found.
    search(win, 'zzz');
    answerSearch(win, posted);
    assert.strictEqual(title.textContent, 'No skills found');
    eq(foundShapes(win), []);
    // The registry cannot be reached: the sentence, in the failure
    // colour; a reply without a list is an empty one.
    search(win, 'pdf');
    answerSearch(win, posted, {
      results: undefined,
      error: 'The skills registry timed out.',
    });
    assert.strictEqual(title.textContent, 'The skills registry timed out.');
    assert.ok(title.classList.contains('is-error'));
    eq(foundShapes(win), []);
    // Searching again takes the failure colour off the busy line.
    search(win, 'pdf');
    assert.strictEqual(title.textContent, 'Searching\u2026');
    assert.ok(!title.classList.contains('is-error'));
  });

  await test('Clear removes the results and the searched text', () => {
    const {win} = pageWithResults();
    const field = el(win, 'meta-skills-query');
    assert.strictEqual(foundShapes(win).length, 3);
    click(win, el(win, 'meta-skills-clear'));
    assert.ok(el(win, 'meta-skills-found').hidden);
    eq(foundShapes(win), []);
    assert.strictEqual(el(win, 'meta-skills-found-title').textContent, '');
    assert.strictEqual(field.value, '');
    assert.strictEqual(
      win.document.activeElement,
      field,
      'the keyboard goes back to the field',
    );
  });

  await test('+ has the daemon check the skill: the row waits, then says Added', () => {
    const {win, posted} = pageWithResults();
    const from = posted.length;
    const add = addButton(win, 'acme/skills/pdf');
    add.focus();
    click(win, add);
    eq(skillPosts(posted, from), [
      {type: 'installSkill', id: 'acme/skills/pdf'},
    ]);
    eq(foundShape(foundRow(win, 'acme/skills/pdf')), {
      name: 'pdf',
      href: 'https://skills.example/acme/skills/pdf',
      sub: 'acme/skills \u00b7 233.5K installs',
      state: 'Checking\u2026',
      add: 'disabled',
    });
    assert.strictEqual(verdictShape(foundRow(win, 'acme/skills/pdf')), null);
    // The row was rebuilt under the keyboard: the focus stays in it.
    assert.strictEqual(
      win.document.activeElement,
      foundRow(win, 'acme/skills/pdf'),
    );
    // The other rows are as they were.
    assert.strictEqual(
      foundShape(foundRow(win, 'solo/kit/pdf-ocr')).add,
      'enabled',
    );

    send(win, {
      type: 'skillInstall',
      id: 'acme/skills/pdf',
      status: 'installed',
      name: 'pdf',
      message: 'pdf was added.',
      findings: [],
    });
    const row = foundRow(win, 'acme/skills/pdf');
    eq(foundShape(row), {
      name: 'pdf',
      href: 'https://skills.example/acme/skills/pdf',
      sub: 'acme/skills \u00b7 233.5K installs',
      state: 'Added',
      add: null,
    });
    assert.strictEqual(verdictShape(row), null, 'nothing more to read');
    assert.strictEqual(win.document.activeElement, row);
    // The daemon follows up with the new list.
    send(win, {
      type: 'skills',
      skills: [{name: 'pdf', description: 'Read PDFs.', source: 'user'}],
      error: '',
    });
    eq(
      skillRows(win).map(r => r.name),
      ['pdf'],
    );
    assert.strictEqual(foundShape(row).state, 'Added');
  });

  await test('a skill that needs a review shows the findings, "Add anyway", which forces the install, and "Cancel", which forgets the verdict', () => {
    const {win, posted} = pageWithResults();
    click(win, addButton(win, 'acme/skills/pdf'));
    send(win, {
      type: 'skillInstall',
      id: 'acme/skills/pdf',
      status: 'review',
      name: 'pdf',
      message: 'The check found something to look at before adding pdf.',
      findings: FINDINGS,
    });
    let row = foundRow(win, 'acme/skills/pdf');
    assert.strictEqual(foundShape(row).state, 'Needs review');
    assert.strictEqual(
      foundShape(row).add,
      null,
      'the + gave way to the verdict',
    );
    eq(verdictShape(row), {
      message: 'The check found something to look at before adding pdf.',
      findings: [
        [
          'skill-severity medium',
          'medium',
          'Downloads a script at run time',
          'scripts/setup.sh:12',
        ],
        // No line: the file alone.  No file: no place.
        ['skill-severity low', 'low', 'Asks for broad file access', 'SKILL.md'],
        ['skill-severity info', 'info', 'No licence', null],
      ],
      buttons: ['Add anyway', 'Cancel'],
    });
    // Read out or copied, a finding is words: the badge, then the sentence.
    assert.ok(
      row
        .querySelector('.skill-findings > li')
        .textContent.startsWith('medium Downloads a script'),
    );
    // A search that says a skill of that name is there (another
    // surface added it, or the folder browsed now has one): the row
    // is Added, without the check's findings and their "Add anyway".
    search(win, 'pdf');
    answerSearch(win, posted, {
      results: [Object.assign({}, FOUND[0], {installed: true})],
    });
    row = foundRow(win, 'acme/skills/pdf');
    assert.strictEqual(foundShape(row).state, 'Added');
    assert.strictEqual(verdictShape(row), null);
    assert.strictEqual(row.querySelectorAll('button').length, 0);
    // Where it is not there, what the check came to stands.
    search(win, 'pdf');
    answerSearch(win, posted, {results: FOUND});
    row = foundRow(win, 'acme/skills/pdf');
    assert.strictEqual(foundShape(row).state, 'Needs review');
    // Cancel forgets the verdict and sends nothing: the row offers its
    // + again, which takes the keyboard, and a press has the skill
    // checked afresh.
    let from = posted.length;
    click(win, row.querySelector('.skill-cancel'));
    row = foundRow(win, 'acme/skills/pdf');
    eq(
      [foundShape(row).state, foundShape(row).add, verdictShape(row)],
      [null, 'enabled', null],
    );
    assert.strictEqual(
      win.document.activeElement,
      addButton(win, 'acme/skills/pdf'),
    );
    eq(skillPosts(posted, from), []);
    click(win, addButton(win, 'acme/skills/pdf'));
    eq(skillPosts(posted, from), [
      {type: 'installSkill', id: 'acme/skills/pdf'},
    ]);
    send(win, {
      type: 'skillInstall',
      id: 'acme/skills/pdf',
      status: 'review',
      name: 'pdf',
      message: 'Look first.',
      findings: [],
    });
    row = foundRow(win, 'acme/skills/pdf');
    from = posted.length;
    click(win, row.querySelector('.skill-add-anyway'));
    eq(skillPosts(posted, from), [
      {type: 'installSkill', id: 'acme/skills/pdf', force: true},
    ]);
    row = foundRow(win, 'acme/skills/pdf');
    assert.strictEqual(foundShape(row).state, 'Checking\u2026');
    assert.strictEqual(verdictShape(row), null);
    // Added, with the notes the check still has.
    send(win, {
      type: 'skillInstall',
      id: 'acme/skills/pdf',
      status: 'installed',
      name: 'pdf',
      message: 'pdf was added.',
      findings: FINDINGS.slice(1),
    });
    row = foundRow(win, 'acme/skills/pdf');
    assert.strictEqual(foundShape(row).state, 'Added');
    eq(verdictShape(row), {
      message: 'pdf was added.',
      findings: [
        ['skill-severity low', 'low', 'Asks for broad file access', 'SKILL.md'],
        ['skill-severity info', 'info', 'No licence', null],
      ],
      buttons: [],
    });
  });

  await test('a blocked skill shows why and cannot be added; a failed check leaves the + to try again', () => {
    const {win, posted} = pageWithResults();
    click(win, addButton(win, 'acme/skills/pdf'));
    send(win, {
      type: 'skillInstall',
      id: 'acme/skills/pdf',
      status: 'blocked',
      name: 'pdf',
      message: 'pdf was not added: it sends your files to another server.',
      findings: [
        {
          severity: 'critical',
          rule: 'exfiltration',
          message: 'Uploads the work folder',
          file: 'run.py',
          line: 3,
        },
      ],
    });
    const blocked = foundRow(win, 'acme/skills/pdf');
    assert.strictEqual(foundShape(blocked).state, 'Blocked');
    assert.strictEqual(
      blocked.querySelectorAll('button').length,
      0,
      'neither a + nor an "Add anyway"',
    );
    eq(verdictShape(blocked), {
      message: 'pdf was not added: it sends your files to another server.',
      findings: [
        [
          'skill-severity critical',
          'critical',
          'Uploads the work folder',
          'run.py:3',
        ],
      ],
      buttons: [],
    });

    // The check could not run: the sentence, and the + again.
    click(win, addButton(win, 'solo/kit/pdf-ocr'));
    send(win, {
      type: 'skillInstall',
      id: 'solo/kit/pdf-ocr',
      status: 'error',
      name: '',
      message: 'The skill could not be downloaded.',
      findings: [],
    });
    let failed = foundRow(win, 'solo/kit/pdf-ocr');
    assert.strictEqual(foundShape(failed).state, null);
    assert.strictEqual(foundShape(failed).add, 'enabled');
    eq(verdictShape(failed), {
      message: 'The skill could not be downloaded.',
      findings: [],
      buttons: [],
    });
    assert.strictEqual(
      failed.querySelector('.skill-findings'),
      null,
      'no empty list',
    );
    const from = posted.length;
    click(win, addButton(win, 'solo/kit/pdf-ocr'));
    eq(skillPosts(posted, from), [
      {type: 'installSkill', id: 'solo/kit/pdf-ocr'},
    ]);
    // A verdict this page does not know, with nothing else in the
    // reply, reads as a failed check.
    send(win, {type: 'skillInstall', id: 'solo/kit/pdf-ocr', status: 'held'});
    failed = foundRow(win, 'solo/kit/pdf-ocr');
    assert.strictEqual(foundShape(failed).add, 'enabled');
    eq(verdictShape(failed), {message: '', findings: [], buttons: []});
    // "A skill of that name already exists": the next search says so,
    // and the row is Added without the stale sentence.
    search(win, 'pdf');
    answerSearch(win, posted, {
      results: [Object.assign({}, FOUND[2], {installed: true})],
    });
    failed = foundRow(win, 'solo/kit/pdf-ocr');
    assert.strictEqual(foundShape(failed).state, 'Added');
    assert.strictEqual(verdictShape(failed), null);
  });

  await test('nothing a reply carries is parsed as HTML, and only an http(s) address becomes a link', () => {
    const {win, posted} = makePage();
    const html = '<img src=x onerror="window.pwned = 1"><b>bold</b>';
    send(win, {
      type: 'skills',
      skills: [{name: html, description: html, source: html}],
      error: '',
    });
    eq(skillRows(win), [{name: html, badge: html, sub: html, tip: html}]);
    search(win, html);
    answerSearch(win, posted, {
      results: [
        {
          id: 'a/b/script',
          name: html,
          source: html,
          installs: 5,
          url: 'javascript:window.pwned = 1',
          installed: false,
        },
        {
          id: 'a/b/data',
          name: 'data',
          source: 'a/b',
          installs: 5,
          url: 'data:text/html,<script>window.pwned = 1</script>',
          installed: false,
        },
        {
          id: 'a/b/upper',
          name: 'upper',
          source: 'a/b',
          installs: 5,
          url: 'HTTPS://skills.example/a/b/upper',
          installed: false,
        },
      ],
    });
    assert.strictEqual(
      el(win, 'meta-skills-found-title').textContent,
      'Results for "' + html + '"',
    );
    eq(
      foundShapes(win).map(r => [r.name, r.href, r.sub]),
      [
        [html, null, html + ' \u00b7 5 installs'],
        ['data', null, 'a/b \u00b7 5 installs'],
        ['upper', 'HTTPS://skills.example/a/b/upper', 'a/b \u00b7 5 installs'],
      ],
    );
    click(win, addButton(win, 'a/b/script'));
    send(win, {
      type: 'skillInstall',
      id: 'a/b/script',
      status: 'review',
      name: html,
      message: html,
      findings: [
        // A severity out of the list takes no class of its own making.
        {
          severity: 'high" onclick="window.pwned = 1',
          rule: html,
          message: html,
          file: html,
          line: 7,
        },
        {severity: 'high', rule: 'r', message: 'm', file: 'f.py', line: 1},
      ],
    });
    eq(verdictShape(foundRow(win, 'a/b/script')), {
      message: html,
      findings: [
        ['skill-severity info', 'info', html, html + ':7'],
        ['skill-severity high', 'high', 'm', 'f.py:1'],
      ],
      buttons: ['Add anyway', 'Cancel'],
    });
    const section = el(win, 'meta-skills');
    assert.strictEqual(section.querySelectorAll('img, b, script').length, 0);
    assert.strictEqual(section.querySelectorAll('[onclick]').length, 0);
    assert.strictEqual(win.pwned, undefined);
  });

  await test('a machine switch keeps the listed skills out of reach, lets go of the search and asks the new machine; another machine\u2019s list and search are dropped, its check answers its own row', () => {
    const {win, posted} = pageWithResults();
    send(win, {type: 'skills', skills: SKILLS, error: ''});
    send(win, {type: 'configData', config: {}, machine: 'laptop'});
    send(win, {
      type: 'machines',
      machines: [
        {name: 'laptop', local: true, status: 'reachable'},
        {name: 'gpu-box', local: false, status: 'reachable'},
      ],
      checkedAt: 1,
    });
    click(win, addButton(win, 'acme/skills/pdf'));
    el(win, 'meta-skills-query').value = 'pdf';
    const from = posted.length;
    const listed = () => skillRows(win).map(r => r.name);
    const held = () => {
      const list = el(win, 'meta-skills-list');
      return list.hasAttribute('inert') && list.classList.contains('is-held');
    };
    assert.ok(!held());
    pickMachine(win, 'gpu-box');
    // The skills stay on screen, out of reach, until gpu-box answers
    // (the subpanel does not blink to "Loading…"); the search and its
    // check were the other machine's.
    eq(
      listed(),
      SKILLS.map(s => s.name),
    );
    assert.ok(held());
    assert.strictEqual(
      el(win, 'meta-skills-status').textContent,
      '3 skills available',
    );
    eq(foundShapes(win), []);
    assert.ok(el(win, 'meta-skills-found').hidden);
    assert.strictEqual(
      el(win, 'meta-skills-query').value,
      'pdf',
      'the typed text stays',
    );
    eq(skillPosts(posted, from), [{type: 'getSkills', machine: 'gpu-box'}]);
    // This daemon's own replies (answered late) are not gpu-box's.
    send(win, {type: 'skills', skills: SKILLS.slice(0, 2), error: ''});
    send(win, {
      type: 'skillSearch',
      query: 'pdf',
      results: FOUND,
      error: '',
      generation: 1,
    });
    eq(
      listed(),
      SKILLS.map(s => s.name),
    );
    assert.ok(held());
    eq(foundShapes(win), []);
    // gpu-box's are painted, in reach again, and every command names it.
    send(win, {
      type: 'skills',
      skills: SKILLS.slice(0, 1),
      error: '',
      forwarded: true,
      machine: 'gpu-box',
    });
    eq(listed(), ['docx']);
    assert.ok(!held());
    const again = posted.length;
    click(win, el(win, 'meta-skills-refresh'));
    search(win, 'pdf');
    // This daemon's answer to a search of that number is not gpu-box's.
    send(win, {
      type: 'skillSearch',
      query: 'pdf',
      results: [],
      error: '',
      generation: 2,
    });
    assert.ok(el(win, 'meta-skills-search-btn').disabled, 'still awaited');
    answerSearch(win, posted, {results: FOUND});
    // The check left on the first machine is not this row's.
    assert.strictEqual(
      foundShape(foundRow(win, 'acme/skills/pdf')).add,
      'enabled',
    );
    click(win, addButton(win, 'acme/skills/pdf'));
    eq(skillPosts(posted, again), [
      {type: 'getSkills', machine: 'gpu-box'},
      {type: 'searchSkills', query: 'pdf', generation: 2, machine: 'gpu-box'},
      {type: 'installSkill', id: 'acme/skills/pdf', machine: 'gpu-box'},
    ]);
    // The first machine's verdict for the same id is not gpu-box's.
    send(win, {
      type: 'skillInstall',
      id: 'acme/skills/pdf',
      status: 'blocked',
      name: 'pdf',
      message: 'no',
      findings: [],
    });
    assert.strictEqual(
      foundShape(foundRow(win, 'acme/skills/pdf')).state,
      'Checking\u2026',
    );
    send(win, {
      type: 'skillInstall',
      id: 'acme/skills/pdf',
      status: 'installed',
      name: 'pdf',
      message: 'pdf was added.',
      findings: [],
      forwarded: true,
      machine: 'gpu-box',
    });
    assert.strictEqual(
      foundShape(foundRow(win, 'acme/skills/pdf')).state,
      'Added',
    );
    // It answered the check this daemon was asked for: back on this
    // daemon, the row of that skill says what the check came to.
    pickMachine(win, '');
    eq(foundShapes(win), []);
    search(win, 'pdf');
    answerSearch(win, posted, {results: FOUND});
    const mine = foundRow(win, 'acme/skills/pdf');
    assert.strictEqual(foundShape(mine).state, 'Blocked');
    eq(verdictShape(mine), {message: 'no', findings: [], buttons: []});
  });

  await test('after a machine round-trip a repeated search is answered by its own reply, and a check under way still holds its row', () => {
    const {win, posted} = makePage();
    send(win, {type: 'configData', config: {}, machine: 'laptop'});
    send(win, {
      type: 'machines',
      machines: [
        {name: 'laptop', local: true, status: 'reachable'},
        {name: 'gpu-box', local: false, status: 'reachable'},
      ],
      checkedAt: 1,
    });
    pickMachine(win, 'gpu-box');
    const button = el(win, 'meta-skills-search-btn');
    const title = el(win, 'meta-skills-found-title');
    search(win, 'pdf');
    answerSearch(win, posted, {results: FOUND});
    click(win, addButton(win, 'acme/skills/pdf'));
    // A second search of the same text is on its way when the chat
    // header leaves gpu-box and comes back.
    search(win, 'pdf');
    const lost = posted.filter(m => m.type === 'searchSkills').pop();
    pickMachine(win, '');
    pickMachine(win, 'gpu-box');
    assert.ok(!button.disabled, 'the subpanel let go of that search');
    const from = posted.length;
    search(win, 'pdf');
    eq(skillPosts(posted, from), [
      {type: 'searchSkills', query: 'pdf', generation: 3, machine: 'gpu-box'},
    ]);
    // The daemon gives up on the earlier search first: same machine,
    // same text, another search.  It neither ends the wait nor shows.
    send(win, {
      type: 'skillSearch',
      query: 'pdf',
      results: [],
      error: 'gpu-box did not answer: timed out',
      generation: lost.generation,
      forwarded: true,
      machine: 'gpu-box',
    });
    assert.strictEqual(title.textContent, 'Searching\u2026');
    assert.ok(button.disabled);
    answerSearch(win, posted, {results: FOUND});
    assert.strictEqual(title.textContent, 'Results for "pdf"');
    assert.ok(!title.classList.contains('is-error'));
    assert.ok(!button.disabled);
    // The check sent before the round-trip is still gpu-box's to
    // answer: its row waits, with no + to ask for a second one.
    eq(foundShape(foundRow(win, 'acme/skills/pdf')), {
      name: 'pdf',
      href: 'https://skills.example/acme/skills/pdf',
      sub: 'acme/skills \u00b7 233.5K installs',
      state: 'Checking\u2026',
      add: 'disabled',
    });
    send(win, {
      type: 'skillInstall',
      id: 'acme/skills/pdf',
      status: 'review',
      name: 'pdf',
      message: 'Look at this first.',
      findings: FINDINGS.slice(0, 1),
      forwarded: true,
      machine: 'gpu-box',
    });
    assert.strictEqual(
      foundShape(foundRow(win, 'acme/skills/pdf')).state,
      'Needs review',
    );
    assert.strictEqual(skillPosts(posted, from).length, 1);
  });

  await test('every command names the folder the side views browse, and a change of folder asks for its skills', () => {
    const {win, posted} = makePage();
    // No folder is known yet: none is named (the daemon uses its own).
    eq(skillPosts(posted), [{type: 'getSkills'}]);
    let from = posted.length;
    sendWorkDir(win, '/proj');
    eq(skillPosts(posted, from), [{type: 'getSkills', workDir: '/proj'}]);
    // The same folder again asks nothing.
    sendWorkDir(win, '/proj');
    assert.strictEqual(skillPosts(posted, from).length, 1);
    from = posted.length;
    click(win, el(win, 'meta-skills-refresh'));
    search(win, 'pdf');
    answerSearch(win, posted, {results: FOUND});
    click(win, addButton(win, 'acme/skills/pdf'));
    eq(skillPosts(posted, from), [
      {type: 'getSkills', workDir: '/proj'},
      {type: 'searchSkills', query: 'pdf', generation: 1, workDir: '/proj'},
      {type: 'installSkill', id: 'acme/skills/pdf', workDir: '/proj'},
    ]);
    // A machine's commands name the folder that machine reported,
    // never this daemon's.
    send(win, {
      type: 'machines',
      machines: [
        {name: 'laptop', local: true, status: 'reachable'},
        {name: 'gpu-box', local: false, status: 'reachable'},
      ],
      checkedAt: 1,
    });
    from = posted.length;
    pickMachine(win, 'gpu-box');
    eq(skillPosts(posted, from), [{type: 'getSkills', machine: 'gpu-box'}]);
    send(win, {
      type: 'machineWorkDirs',
      machine: 'gpu-box',
      workDir: '/gpu/ws',
      recentWorkDirs: [],
      error: '',
    });
    eq(skillPosts(posted, from).slice(1), [
      {type: 'getSkills', workDir: '/gpu/ws', machine: 'gpu-box'},
    ]);
    // While the daemon is away a new folder asks nothing: the
    // reconnect asks for every subpanel, the skills of that folder.
    send(win, {type: 'daemonStatus', connected: false});
    from = posted.length;
    send(win, {
      type: 'machineWorkDirs',
      machine: 'gpu-box',
      workDir: '/gpu/other',
      recentWorkDirs: [],
      error: '',
    });
    eq(skillPosts(posted, from), []);
    send(win, {type: 'daemonStatus', connected: true});
    eq(skillPosts(posted, from), [
      {type: 'getSkills', workDir: '/gpu/other', machine: 'gpu-box'},
    ]);
    // A file-system root is no folder to name (a daemon takes none
    // for one, and its list says so): the list of a machine that
    // reports one is that of its daemon's own folder, and is shown.
    from = posted.length;
    send(win, {
      type: 'machineWorkDirs',
      machine: 'gpu-box',
      workDir: '/',
      recentWorkDirs: [],
      error: '',
    });
    send(win, {
      type: 'skills',
      skills: SKILLS.slice(0, 1),
      error: '',
      workDir: '',
      forwarded: true,
      machine: 'gpu-box',
    });
    eq(
      skillRows(win).map(r => r.name),
      ['docx'],
    );
    eq(skillPosts(posted, from), [{type: 'getSkills', machine: 'gpu-box'}]);
  });

  await test('a change of folder lets go of the search: a found skill is Added where the folder searched in has it', () => {
    const {win, posted} = makePage();
    const found = el(win, 'meta-skills-found');
    // `pdf` is a skill of project A alone.
    const inA = [Object.assign({}, FOUND[0], {installed: true})];
    sendWorkDir(win, '/project-a');
    search(win, 'pdf');
    answerSearch(win, posted, {results: inA});
    eq(
      foundShapes(win).map(r => [r.state, r.add]),
      [['Added', null]],
    );
    // The search on show goes with its folder; the text typed stays.
    sendWorkDir(win, '/project-b');
    assert.ok(found.hidden);
    eq(foundShapes(win), []);
    assert.strictEqual(el(win, 'meta-skills-query').value, 'pdf');
    // So does a search still awaited: its answer is about project A.
    sendWorkDir(win, '/project-a');
    search(win, 'pdf');
    sendWorkDir(win, '/project-b');
    assert.ok(found.hidden && !el(win, 'meta-skills-search-btn').disabled);
    answerSearch(win, posted, {results: inA});
    assert.ok(found.hidden);
    // The same search made in project B is answered for project B.
    search(win, 'pdf');
    eq(skillPosts(posted).pop(), {
      type: 'searchSkills',
      query: 'pdf',
      generation: 3,
      workDir: '/project-b',
    });
    answerSearch(win, posted, {results: FOUND.slice(0, 1)});
    eq(
      foundShapes(win).map(r => [r.state, r.add]),
      [[null, 'enabled']],
    );
  });

  await test('a list of another folder than the one browsed is dropped: the one that follows a skill added in the folder left behind', () => {
    // The folder left behind: a named one, and the daemon's own (the
    // page knew none yet), which its lists name as ''.
    ['/project-a', ''].forEach(behind => {
      const {win, posted} = makePage();
      const listed = () => skillRows(win).map(r => r.name);
      // The daemon's list of *skills*, naming the folder it is of.
      const sendList = (skills, workDir) =>
        send(win, {type: 'skills', skills, error: '', workDir});
      if (behind) sendWorkDir(win, behind);
      eq(
        skillPosts(posted).pop(),
        behind ? {type: 'getSkills', workDir: behind} : {type: 'getSkills'},
      );
      sendList(SKILLS, behind);
      eq(listed(), ['docx', 'pdf-forms', 'release-notes']);
      search(win, 'pdf');
      answerSearch(win, posted, {results: FOUND});
      click(win, addButton(win, 'acme/skills/pdf'));
      // The folder changes while the daemon checks the skill.
      sendWorkDir(win, '/project-b');
      sendList([], '/project-b');
      eq(listed(), []);
      // The skill is added: the list that follows is of the folder
      // left behind.
      send(win, {
        type: 'skillInstall',
        id: 'acme/skills/pdf',
        status: 'installed',
        name: 'pdf',
        message: 'Added pdf.',
        findings: [],
      });
      sendList(SKILLS, behind);
      eq(listed(), [], 'the list on show is still project B\u2019s');
      assert.strictEqual(
        el(win, 'meta-skills-status').textContent,
        'No skills yet. Search the internet above to add one.',
      );
      // Project B's own list, and one that says nothing of a folder
      // (the reply of an older daemon), are the list.
      sendList(SKILLS.slice(0, 1), '/project-b');
      eq(listed(), ['docx']);
      send(win, {type: 'skills', skills: SKILLS, error: ''});
      eq(listed(), ['docx', 'pdf-forms', 'release-notes']);
    });
  });

  await test('while the side views browse another machine than the subpanel shows, no folder is named', () => {
    // The stacked layout: there a content tab outlives its owner chat.
    const {win, posted, intervals} = makePage({desktop: false});
    send(win, {
      type: 'machines',
      machines: [
        {name: 'laptop', local: true, status: 'reachable'},
        {name: 'gpu-box', local: false, status: 'reachable'},
      ],
      checkedAt: 1,
    });
    const home = win._testApi.getActiveTabId();
    pickMachine(win, 'gpu-box');
    send(win, {
      type: 'machineWorkDirs',
      machine: 'gpu-box',
      workDir: '/gpu/ws',
      recentWorkDirs: [],
      error: '',
    });
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: home, chatId: 'c-home', title: 'home', workDir: '/ws'}],
    });
    // A file of the gpu-box chat opens as a content tab owned by it.
    // While the file is shown, another surface moves the global pick
    // home (the chat, still new, follows it and the subpanels with
    // it) and then closes the chat.  The side views fall back to the
    // file's own machine, and the subpanels follow at their next poll
    // only.
    send(win, {
      type: 'fileContent',
      tabId: home,
      machine: 'gpu-box',
      path: '/gpu/ws/notes.txt',
      name: 'notes.txt',
      content: 'hi\n',
      version: 'v1',
    });
    const file = win._testApi.openTabs().find(t => t.isContentTab);
    assert.strictEqual(win._testApi.getActiveTabId(), file.id);
    send(win, {type: 'machineChanged', machine: ''});
    send(win, {type: 'tabs_state', tabs: []});
    assert.strictEqual(win._testApi.getActiveTabId(), file.id);
    // Until then the subpanel shows this daemon's skills: its commands
    // go to this daemon, without gpu-box's folder.
    let from = posted.length;
    click(win, el(win, 'meta-skills-refresh'));
    search(win, 'pdf');
    eq(skillPosts(posted, from), [
      {type: 'getSkills'},
      {type: 'searchSkills', query: 'pdf', generation: 1},
    ]);
    from = posted.length;
    intervals.find(i => i.ms === 30000).cb();
    eq(skillPosts(posted, from), [
      {type: 'getSkills', workDir: '/gpu/ws', machine: 'gpu-box'},
    ]);
  });

  await test('a lost daemon connection releases the waits, and nothing is sent until it is back', () => {
    const {win, posted} = pageWithResults();
    const refresh = el(win, 'meta-skills-refresh');
    const button = el(win, 'meta-skills-search-btn');
    // A check that came back, a check under way, a reload and a search.
    click(win, addButton(win, 'solo/kit/pdf-ocr'));
    send(win, {
      type: 'skillInstall',
      id: 'solo/kit/pdf-ocr',
      status: 'blocked',
      name: 'pdf-ocr',
      message: 'pdf-ocr was not added.',
      findings: [],
    });
    click(win, addButton(win, 'acme/skills/pdf'));
    click(win, refresh);
    search(win, 'pdf');
    assert.ok(refresh.disabled && button.disabled);
    send(win, {type: 'daemonStatus', connected: false});
    assert.ok(!refresh.disabled && !refresh.classList.contains('spinning'));
    assert.ok(!button.disabled);
    // The results on show are back, the waiting row with its +; the
    // verdict that had arrived stands.
    assert.strictEqual(
      el(win, 'meta-skills-found-title').textContent,
      'Results for "pdf"',
    );
    eq(
      foundShapes(win).map(r => [r.name, r.state, r.add]),
      [
        ['pdf', null, 'enabled'],
        ['pdf-forms', 'Added', null],
        ['pdf-ocr', 'Blocked', null],
      ],
    );
    // No reply would come: the controls send nothing.
    const from = posted.length;
    click(win, refresh);
    click(win, button);
    click(win, addButton(win, 'acme/skills/pdf'));
    eq(skillPosts(posted, from), []);
    assert.ok(!refresh.classList.contains('spinning') && !button.disabled);
    assert.strictEqual(
      foundShape(foundRow(win, 'acme/skills/pdf')).add,
      'enabled',
    );
    // The reconnect asks for the skills with the other subpanels.
    send(win, {type: 'daemonStatus', connected: true});
    el(win, 'meta-skills-query').value = 'docx';
    click(win, button);
    eq(skillPosts(posted, from), [
      {type: 'getSkills'},
      {type: 'searchSkills', query: 'docx', generation: 3},
    ]);
    // The answer to the search the lost connection let go of arrives
    // late: it is not the awaited one, whichever of the two comes first.
    const late = {
      type: 'skillSearch',
      query: 'pdf',
      results: [],
      error: '',
      generation: 2,
    };
    send(win, late);
    assert.strictEqual(
      el(win, 'meta-skills-found-title').textContent,
      'Searching\u2026',
    );
    assert.ok(button.disabled);
    answerSearch(win, posted, {results: FOUND.slice(0, 1)});
    send(win, late);
    assert.strictEqual(
      el(win, 'meta-skills-found-title').textContent,
      'Results for "docx"',
    );
    eq(
      foundShapes(win).map(r => r.name),
      ['pdf'],
    );
    assert.ok(!button.disabled);
  });

  for (const machine of ['', 'gpu-box']) {
    const where = machine || 'this daemon';
    await test(`a + pressed as the connection goes is checked once it is back, its row waiting again with no + for a second check (${where})`, async () => {
      const {win, sockets, sent} = await makeShimPage();
      // A machine's commands name it and its replies are stamped with it.
      const to = machine ? {machine: machine} : {};
      const from = machine ? {forwarded: true, machine: machine} : {};
      if (machine) {
        sockets[0].fireMessage({
          type: 'machines',
          machines: [
            {name: 'laptop', local: true, status: 'reachable'},
            {name: machine, local: false, status: 'reachable'},
          ],
          checkedAt: 1,
        });
        pickMachine(win, machine);
      }
      search(win, 'pdf');
      sockets[0].fireMessage(
        Object.assign(
          {
            type: 'skillSearch',
            query: 'pdf',
            results: FOUND,
            error: '',
            generation: sent('searchSkills').pop().generation,
          },
          from,
        ),
      );
      // The server began the close handshake; onclose has not fired, so
      // the page has not heard of the drop.
      sockets[0].readyState = win.WebSocket.CLOSING;
      click(win, addButton(win, 'acme/skills/pdf'));
      eq(sent('installSkill'), [], 'it waits in the transport');
      sockets[0].fireClose();
      assert.strictEqual(
        foundShape(foundRow(win, 'acme/skills/pdf')).state,
        null,
        'no check is under way while the daemon is away',
      );
      win.dispatchEvent(new win.Event('focus')); // the user is back
      sockets[1].fireOpen();
      sockets[1].fireMessage({type: 'auth_ok'});
      // The transport still owes the daemon that check, which will be
      // answered: the row waits for it, with no + for a second one.
      const waiting = foundShape(foundRow(win, 'acme/skills/pdf'));
      eq([waiting.state, waiting.add], ['Checking\u2026', 'disabled']);
      eq(sent('installSkill'), []);
      sockets[1].fireMessage({type: 'pong'});
      eq(sent('installSkill'), [
        Object.assign({type: 'installSkill', id: 'acme/skills/pdf'}, to),
      ]);
      sockets[1].fireMessage(
        Object.assign(
          {
            type: 'skillInstall',
            id: 'acme/skills/pdf',
            status: 'installed',
            name: 'pdf',
            message: 'pdf was added.',
            findings: [],
          },
          from,
        ),
      );
      assert.strictEqual(
        foundShape(foundRow(win, 'acme/skills/pdf')).state,
        'Added',
      );
    });
  }

  report('skillsPanel');
}

main();
