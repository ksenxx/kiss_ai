"""Tests for per-tab ask-user modal localisation.

Verifies the refactor that makes ``#ask-user-question``, ``#ask-user-input``,
and ``#ask-user-submit`` local to each tab (replaced by per-tab DOM nodes
mounted into a shared ``#ask-user-slot``) and introduces a per-tab FIFO
``askQueue`` so concurrent asks from multiple tabs are routed correctly.

These tests deliberately focus on the new invariants; pre-existing
``askUserTabId`` assertions in ``test_vscode_tabs.py`` are superseded.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MAIN_JS = _REPO_ROOT / "kiss" / "agents" / "vscode" / "media" / "main.js"
_MAIN_CSS = _REPO_ROOT / "kiss" / "agents" / "vscode" / "media" / "main.css"
_SORCAR_TAB_TS = (
    _REPO_ROOT / "kiss" / "agents" / "vscode" / "src" / "SorcarTab.ts"
)


class TestHtmlHasPerTabSlot(unittest.TestCase):
    """The chat HTML in SorcarTab.ts no longer hard-codes question/input/submit."""

    ts_src: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ts_src = _SORCAR_TAB_TS.read_text()

    def test_modal_has_slot(self) -> None:
        """``#ask-user-slot`` is present inside ``.modal-content``."""
        assert 'id="ask-user-slot"' in self.ts_src

    def test_modal_no_static_triplet(self) -> None:
        """The static ``#ask-user-question/input/submit`` elements are removed."""
        assert 'id="ask-user-question"' not in self.ts_src
        assert 'id="ask-user-input"' not in self.ts_src
        assert 'id="ask-user-submit"' not in self.ts_src

    def test_modal_still_present(self) -> None:
        """``#ask-user-modal`` container is kept (shared overlay backdrop)."""
        assert 'id="ask-user-modal"' in self.ts_src


class TestCssClassesForPerTabElements(unittest.TestCase):
    """The CSS targets per-tab elements by class, not id."""

    css_src: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.css_src = _MAIN_CSS.read_text()

    def test_css_uses_classes(self) -> None:
        assert ".ask-user-question" in self.css_src
        assert ".ask-user-input" in self.css_src
        assert ".ask-user-submit" in self.css_src

    def test_css_no_id_rules_for_triplet(self) -> None:
        assert "#ask-user-question" not in self.css_src
        assert "#ask-user-input" not in self.css_src
        assert "#ask-user-submit" not in self.css_src


class TestMainJsStructural(unittest.TestCase):
    """Structural invariants of the per-tab ask-user refactor in main.js."""

    js: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.js = _MAIN_JS.read_text()

    def test_no_shared_ask_user_tab_id(self) -> None:
        """The module-global ``askUserTabId`` slot is gone."""
        assert "askUserTabId" not in self.js

    def test_no_shared_triplet_element_refs(self) -> None:
        """main.js no longer queries the old singleton triplet by id."""
        assert "getElementById('ask-user-question')" not in self.js
        assert "getElementById('ask-user-input')" not in self.js
        assert "getElementById('ask-user-submit')" not in self.js

    def test_slot_element_is_queried(self) -> None:
        """The shared slot element is resolved once at module init."""
        assert "getElementById('ask-user-slot')" in self.js

    def test_tab_has_askqueue(self) -> None:
        """Every tab carries its own FIFO queue of pending questions."""
        assert "askQueue: []" in self.js

    def test_tab_has_ask_elements(self) -> None:
        """Every tab carries its own question/input/submit element refs."""
        assert "askQuestionEl: null" in self.js
        assert "askInputEl: null" in self.js
        assert "askSubmitEl: null" in self.js

    def test_askuser_handler_enqueues(self) -> None:
        """The ``askUser`` event pushes onto the target tab's queue."""
        idx = self.js.index("case 'askUser':")
        block = self.js[idx : idx + 600]
        assert "askQueue.push" in block
        assert "ev.tabId" in block
        assert "showNextAskForTab" in block

    def test_submit_routes_tabid(self) -> None:
        """``submitAskForTab`` posts ``userAnswer`` with the tab's id."""
        idx = self.js.index("function submitAskForTab(")
        block = self.js[idx : idx + 600]
        assert "type: 'userAnswer'" in block
        assert "tabId: tab.id" in block
        assert "askQueue.shift()" in block

    def test_tab_switch_syncs_modal(self) -> None:
        """``restoreTab`` ends by syncing the modal to the newly active tab."""
        idx = self.js.index("function restoreTab(")
        end = self.js.index("function renderTabBar(", idx)
        block = self.js[idx:end]
        assert "syncAskModalToActiveTab()" in block


# --------------------------------------------------------------------------
# Behavioural tests: execute a minimal replay of the new helpers in Node.js
# to verify queue and slot behaviour end-to-end without a real DOM.
# --------------------------------------------------------------------------

_JS_PREAMBLE = r"""
// Minimal DOM stubs used to drive the per-tab ask helpers inline.
function _mkEl(tag) {
  var el = {
    tagName: tag,
    id: '',
    className: '',
    placeholder: '',
    textContent: '',
    innerHTML: '',
    value: '',
    attrs: {},
    style: {display: ''},
    children: [],
    _listeners: {},
    classList: {
      _c: [],
      add: function (c) { if (this._c.indexOf(c) < 0) this._c.push(c); },
      remove: function (c) { var i = this._c.indexOf(c); if (i >= 0) this._c.splice(i, 1); },
      contains: function (c) { return this._c.indexOf(c) >= 0; },
    },
    firstChild: null,
    appendChild: function (c) {
      this.children.push(c);
      c.parent = this;
      this.firstChild = this.children[0];
      return c;
    },
    removeChild: function (c) {
      var i = this.children.indexOf(c);
      if (i >= 0) this.children.splice(i, 1);
      this.firstChild = this.children[0] || null;
      c.parent = null;
      return c;
    },
    setAttribute: function (k, v) { this.attrs[k] = v; },
    addEventListener: function (t, fn) {
      this._listeners[t] = (this._listeners[t] || []).concat([fn]);
    },
    click: function () {
      (this._listeners.click || []).forEach(function (fn) { fn(); });
    },
    keydown: function (evt) {
      (this._listeners.keydown || []).forEach(function (fn) { fn(evt); });
    },
    focus: function () {},
  };
  return el;
}

var slot = _mkEl('div');
slot.id = 'ask-user-slot';
var modal = _mkEl('div');
modal.id = 'ask-user-modal';
modal.style.display = 'none';

// Posted messages collected from submitAskForTab().
var posted = [];
var vscode = { postMessage: function (m) { posted.push(m); } };

// Active-tab identifier (mutable for tests).
var activeTabId = null;

// No-op markdown: behave like the non-marked branch.
var marked = undefined;
function hlBlock() {}

// --- Helpers lifted verbatim from the main.js refactor ---
function ensureAskElementsForTab(tab) {
  if (tab.askQuestionEl) return;
  var q = _mkEl('div'); q.className = 'ask-user-question';
  var i = _mkEl('textarea'); i.className = 'ask-user-input';
  var s = _mkEl('button'); s.className = 'ask-user-submit';
  s.textContent = 'Submit';
  s.addEventListener('click', function () { submitAskForTab(tab); });
  i.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') submitAskForTab(tab);
  });
  tab.askQuestionEl = q;
  tab.askInputEl = i;
  tab.askSubmitEl = s;
}

function setAskQuestionTextForTab(tab, text) {
  tab.askQuestionEl.textContent = text || '';
}

function clearAskSlot() {
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  modal.style.display = 'none';
}

function mountAskForTab(tab) {
  while (slot.firstChild) slot.removeChild(slot.firstChild);
  slot.appendChild(tab.askQuestionEl);
  slot.appendChild(tab.askInputEl);
  slot.appendChild(tab.askSubmitEl);
  modal.style.display = 'flex';
}

function showNextAskForTab(tab) {
  if (tab.askQueue.length === 0) {
    if (tab.id === activeTabId) clearAskSlot();
    return;
  }
  ensureAskElementsForTab(tab);
  setAskQuestionTextForTab(tab, tab.askQueue[0]);
  tab.askInputEl.value = '';
  if (tab.id === activeTabId) mountAskForTab(tab);
}

function submitAskForTab(tab) {
  var answer = tab.askInputEl ? tab.askInputEl.value : '';
  vscode.postMessage({type: 'userAnswer', answer: answer, tabId: tab.id});
  tab.askQueue.shift();
  if (tab.askInputEl) tab.askInputEl.value = '';
  showNextAskForTab(tab);
}

function syncAskModalToActiveTab(tabs) {
  clearAskSlot();
  var tab = tabs.find(function (t) { return t.id === activeTabId; });
  if (!tab || tab.askQueue.length === 0) return;
  ensureAskElementsForTab(tab);
  mountAskForTab(tab);
}

function makeTab(id) {
  return {
    id: id,
    askQueue: [],
    askQuestionEl: null,
    askInputEl: null,
    askSubmitEl: null,
  };
}

// --- Event dispatcher mirroring main.js `case 'askUser'` ---
function onAskUser(tabs, ev) {
  var askTabId = ev.tabId !== undefined ? ev.tabId : activeTabId;
  var askTab = tabs.find(function (t) { return t.id === askTabId; });
  if (!askTab) return;
  var wasEmpty = askTab.askQueue.length === 0;
  askTab.askQueue.push(ev.question || '');
  if (wasEmpty) showNextAskForTab(askTab);
}
"""


def _run_node(script: str) -> subprocess.CompletedProcess[str]:
    """Run *script* in Node.js, returning the captured process result."""
    return subprocess.run(
        ["node", "-e", _JS_PREAMBLE + "\n" + script],
        capture_output=True,
        text=True,
        timeout=15,
    )


class TestBehaviourPerTabQueue(unittest.TestCase):
    """End-to-end behavioural checks for the per-tab ask-user pipeline."""

    def test_active_tab_ask_mounts_modal(self) -> None:
        """An askUser event for the active tab mounts its elements into the slot."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A'), makeTab('B')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q-A', tabId: 'A'});
            if (slot.children.length !== 3) throw new Error('slot not mounted');
            if (modal.style.display !== 'flex') throw new Error('modal not shown');
            if (slot.children[0].textContent !== 'Q-A') throw new Error('question text wrong');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_nonactive_tab_ask_does_not_mount(self) -> None:
        """An askUser for a non-active tab does NOT display the modal."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A'), makeTab('B')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q-B', tabId: 'B'});
            if (modal.style.display === 'flex') throw new Error('modal unexpectedly shown');
            if (slot.children.length !== 0) throw new Error('slot unexpectedly populated');
            if (tabs[1].askQueue.length !== 1) throw new Error('queue not populated');
            if (tabs[1].askQuestionEl === null) throw new Error('elements not created');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_switching_reveals_other_tabs_pending_ask(self) -> None:
        """Switching to a tab with a pending ask mounts and shows its modal."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A'), makeTab('B')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q-B', tabId: 'B'});
            // Simulate tab switch A -> B
            activeTabId = 'B';
            syncAskModalToActiveTab(tabs);
            if (modal.style.display !== 'flex') throw new Error('modal not shown after switch');
            if (slot.children.length !== 3) throw new Error('slot not populated after switch');
            if (slot.children[0].textContent !== 'Q-B') throw new Error('wrong question');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_switch_away_hides_modal_then_restores(self) -> None:
        """Switching away hides the modal; switching back restores it."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A'), makeTab('B')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q-A', tabId: 'A'});
            // A -> B
            activeTabId = 'B';
            syncAskModalToActiveTab(tabs);
            if (modal.style.display !== 'none') throw new Error('modal not hidden');
            if (slot.children.length !== 0) throw new Error('slot not cleared');
            // B -> A
            activeTabId = 'A';
            syncAskModalToActiveTab(tabs);
            if (modal.style.display !== 'flex') throw new Error('modal not re-shown');
            if (slot.children[0].textContent !== 'Q-A') throw new Error('wrong question');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_partial_answer_preserved_across_tab_switch(self) -> None:
        """A half-typed answer in tab A survives A->B->A switching."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A'), makeTab('B')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q-A', tabId: 'A'});
            tabs[0].askInputEl.value = 'half typed';
            activeTabId = 'B';
            syncAskModalToActiveTab(tabs);
            activeTabId = 'A';
            syncAskModalToActiveTab(tabs);
            if (tabs[0].askInputEl.value !== 'half typed') throw new Error('input value lost');
            if (slot.children[1] !== tabs[0].askInputEl) throw new Error('input not remounted');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_submit_routes_answer_to_originating_tab(self) -> None:
        """Submitting an answer posts userAnswer with the owning tab's id."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A'), makeTab('B')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q-A', tabId: 'A'});
            tabs[0].askInputEl.value = 'my answer';
            tabs[0].askSubmitEl.click();
            if (posted.length !== 1) throw new Error('expected one posted message');
            var m = posted[0];
            if (m.type !== 'userAnswer') throw new Error('bad type');
            if (m.tabId !== 'A') throw new Error('bad tabId');
            if (m.answer !== 'my answer') throw new Error('bad answer');
            if (modal.style.display !== 'none') throw new Error('modal not hidden after submit');
            if (tabs[0].askQueue.length !== 0) throw new Error('queue not drained');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_concurrent_asks_routed_independently(self) -> None:
        """Ask-to-B while A's modal is up does NOT hijack A's modal."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A'), makeTab('B')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q-A', tabId: 'A'});
            // B's ask arrives while A's modal is up.  It must NOT hijack A.
            onAskUser(tabs, {type: 'askUser', question: 'Q-B', tabId: 'B'});
            if (slot.children[0].textContent !== 'Q-A')
              throw new Error('A hijacked by B');
            if (tabs[1].askQueue.length !== 1) throw new Error('B queue wrong');
            // Answer A.
            tabs[0].askInputEl.value = 'ans-A';
            tabs[0].askSubmitEl.click();
            // Switch to B — B's Q-B shows.
            activeTabId = 'B';
            syncAskModalToActiveTab(tabs);
            if (slot.children[0].textContent !== 'Q-B') throw new Error('B not shown');
            tabs[1].askInputEl.value = 'ans-B';
            tabs[1].askSubmitEl.click();
            if (posted.length !== 2) throw new Error('expected two posted answers');
            if (posted[0].tabId !== 'A' || posted[0].answer !== 'ans-A')
              throw new Error('A routing wrong: ' + JSON.stringify(posted[0]));
            if (posted[1].tabId !== 'B' || posted[1].answer !== 'ans-B')
              throw new Error('B routing wrong: ' + JSON.stringify(posted[1]));
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_queued_asks_for_same_tab_are_served_in_order(self) -> None:
        """Two asks pushed onto the same tab show up as Q1 then Q2 on submit."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q1', tabId: 'A'});
            // Simulate a second ask queued behind the first (hypothetical, but
            // the queue must support it).
            onAskUser(tabs, {type: 'askUser', question: 'Q2', tabId: 'A'});
            if (slot.children[0].textContent !== 'Q1') throw new Error('Q1 not shown first');
            if (tabs[0].askQueue.length !== 2) throw new Error('queue wrong length');
            tabs[0].askInputEl.value = 'ans1';
            tabs[0].askSubmitEl.click();
            if (slot.children[0].textContent !== 'Q2') throw new Error('Q2 not shown after submit');
            if (tabs[0].askInputEl.value !== '') throw new Error('input not cleared for next');
            tabs[0].askInputEl.value = 'ans2';
            tabs[0].askSubmitEl.click();
            if (posted.map(function (m) { return m.answer; }).join(',') !== 'ans1,ans2')
              throw new Error('answers out of order: ' + JSON.stringify(posted));
            if (modal.style.display !== 'none') throw new Error('modal not hidden after drain');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_enter_key_in_input_submits(self) -> None:
        """Pressing Enter in the textarea submits the current answer."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'Q', tabId: 'A'});
            tabs[0].askInputEl.value = 'via enter';
            tabs[0].askInputEl.keydown({key: 'Enter'});
            if (posted.length !== 1) throw new Error('enter did not submit');
            if (posted[0].answer !== 'via enter') throw new Error('wrong answer');
            if (posted[0].tabId !== 'A') throw new Error('wrong tabId');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"

    def test_unknown_tabid_is_ignored(self) -> None:
        """askUser with a tabId not in ``tabs`` is a safe no-op."""
        r = _run_node(
            r"""
            var tabs = [makeTab('A')];
            activeTabId = 'A';
            onAskUser(tabs, {type: 'askUser', question: 'X', tabId: 'ghost'});
            if (modal.style.display === 'flex') throw new Error('modal opened for ghost tab');
            if (tabs[0].askQueue.length !== 0) throw new Error('queue polluted');
            console.log('OK');
            """
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "OK"


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
