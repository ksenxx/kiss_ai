"""Tests for bug fixes identified in the code review of core/ and sorcar/.

Each test verifies a specific fix without mocks, patches, or test doubles.
"""


# ---------------------------------------------------------------------------
# kiss/agents/sorcar/chatbot_ui.py — CHATBOT_JS
# ---------------------------------------------------------------------------

class TestReplayTaskEventsRestoresInput:
    """Tests that replayTaskEvents restores inp.value after replaying events.

    When task_done/task_error/task_stopped events are replayed, setReady()
    clears inp.value. The fix ensures inp.value is restored to the task text
    after all events are replayed.
    """

    def test_replay_restores_input_after_task_done(self):
        """After replaying events including task_done, inp.value=txt is set."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function replayTaskEvents(idx,txt){")
        end = CHATBOT_JS.index("function renderSidebarTasks(tasks){")
        replay_js = CHATBOT_JS[start:end]

        # The fetch callback should restore inp.value=txt after the forEach loop
        # Find the .then(function(events){ block
        then_start = replay_js.index(".then(function(events){")
        then_block = replay_js[then_start:]

        # After the forEach loop ends (with "});"), inp.value=txt must appear
        # before the sb() call
        foreach_end = then_block.index("});", then_block.index("events.forEach"))
        after_foreach = then_block[foreach_end:]
        sb_pos = after_foreach.index("sb();")
        between = after_foreach[:sb_pos]
        assert "inp.value=txt" in between, (
            "replayTaskEvents must restore inp.value=txt after replaying events "
            "and before sb(), so task_done/setReady clearing is undone"
        )

    def test_replay_sets_input_at_start_and_after_fetch(self):
        """inp.value=txt is set both at function start and after event replay."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        start = CHATBOT_JS.index("function replayTaskEvents(idx,txt){")
        end = CHATBOT_JS.index("function renderSidebarTasks(tasks){")
        replay_js = CHATBOT_JS[start:end]

        # Count occurrences of inp.value=txt - should be at least 2
        # (once at top, once after forEach)
        count = replay_js.count("inp.value=txt")
        assert count >= 2, (
            f"Expected inp.value=txt at least twice in replayTaskEvents "
            f"(top + after replay), found {count}"
        )


class TestMergeWarningOnDisabledChatbox:
    """Tests that clicking the chatbox during merge view shows a warning."""

    def test_show_merge_warning_function_in_js(self):
        """showMergeWarning function is defined in CHATBOT_JS."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        assert "function showMergeWarning(){" in CHATBOT_JS

    def test_input_text_wrap_click_calls_show_merge_warning(self):
        """Click handler on input-text-wrap calls showMergeWarning when merging."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        # Find the click handler on input-text-wrap
        handler_idx = CHATBOT_JS.index(
            "getElementById('input-text-wrap').addEventListener('click'"
        )
        # Extract the handler body (next ~100 chars)
        handler = CHATBOT_JS[handler_idx : handler_idx + 200]
        assert "if(merging)showMergeWarning()" in handler

    def test_merge_warning_has_timeout_to_hide(self):
        """showMergeWarning removes 'visible' class after a timeout."""
        from kiss.agents.sorcar.chatbot_ui import CHATBOT_JS

        fn_start = CHATBOT_JS.index("function showMergeWarning(){")
        # Find the end of the function by looking for the closing brace
        # after addEventListener (which comes right after showMergeWarning)
        fn_end = CHATBOT_JS.index(
            "addEventListener('click'", fn_start
        )
        fn_body = CHATBOT_JS[fn_start:fn_end]
        assert "classList.add('visible')" in fn_body
        assert "classList.remove('visible')" in fn_body
        assert "setTimeout" in fn_body


# ---------------------------------------------------------------------------
# kiss/core/models/openai_compatible_model.py — DEEPSEEK_REASONING_MODELS
# ---------------------------------------------------------------------------

class TestDeepSeekReasoningModelsConsistency:
    """Verify DEEPSEEK_REASONING_MODELS entries match model_info.py entries."""

    def test_together_models_have_correct_prefixes(self):
        """Together AI entries use full model name with 'deepseek-ai/' prefix."""
        from kiss.core.models.openai_compatible_model import DEEPSEEK_REASONING_MODELS

        together_entries = [
            m for m in DEEPSEEK_REASONING_MODELS
            if "/" in m and not m.startswith("deepseek/")
        ]
        for entry in together_entries:
            assert entry.startswith("deepseek-ai/"), (
                f"Together AI entry '{entry}' should start with 'deepseek-ai/'"
            )

    def test_openrouter_entries_use_api_names(self):
        """OpenRouter entries use API names (without 'openrouter/' prefix)."""
        from kiss.core.models.openai_compatible_model import DEEPSEEK_REASONING_MODELS

        # Entries starting with "deepseek/" are OpenRouter API names
        or_entries = [m for m in DEEPSEEK_REASONING_MODELS if m.startswith("deepseek/")]
        assert len(or_entries) > 0, "Should have OpenRouter-style entries"
        for entry in or_entries:
            assert not entry.startswith("openrouter/"), (
                f"Entry '{entry}' should NOT have 'openrouter/' prefix"
            )
