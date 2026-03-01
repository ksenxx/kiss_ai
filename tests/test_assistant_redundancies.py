"""Tests verifying redundancies in the assistant package have been removed."""

from kiss.agents.assistant.chatbot_ui import CHATBOT_CSS, CHATBOT_JS, _build_html


def test_no_duplicate_css_min_height():
    """The old min-height:24px line should not be present (was overridden by 96px)."""
    assert "min-height:24px" not in CHATBOT_CSS
    assert "min-height:96px" in CHATBOT_CSS


def test_no_duplicate_textarea_rows():
    """HTML should have only one textarea with rows=4, not the old rows=1."""
    html = _build_html("Test", "", "/tmp")
    assert html.count('rows="4"') == 1
    assert 'rows="1"' not in html


def test_accept_ghost_no_duplicate_lines():
    """acceptGhost should append ghostSuggest once and call clearGhost once."""
    assert CHATBOT_JS.count("inp.value+=ghostSuggest;") == 1
    # Find acceptGhost function body
    start = CHATBOT_JS.index("function acceptGhost(){")
    end = CHATBOT_JS.index("}", start) + 1
    body = CHATBOT_JS[start:end]
    assert body.count("clearGhost()") == 1
    assert body.count("inp.focus()") == 1


def test_config_import_not_in_assistant_agent():
    """assistant_agent.py should not redundantly import config (already in __init__)."""
    import inspect
    import kiss.agents.assistant.assistant_agent as module
    source = inspect.getsource(module)
    assert "import kiss.agents.assistant.config" not in source


def test_config_registered_via_init():
    """Config should be registered via __init__.py import, accessible on DEFAULT_CONFIG."""
    from kiss.core import config as config_module
    cfg = config_module.DEFAULT_CONFIG
    assert hasattr(cfg, "assistant")
    assert hasattr(cfg.assistant, "assistant_agent")
    assert hasattr(cfg.assistant, "relentless_agent")


def test_assistant_agent_imports_work():
    """AssistantAgent should be importable and functional after removing redundant import."""
    from kiss.agents.assistant.assistant_agent import AssistantAgent
    agent = AssistantAgent("test")
    assert agent.name == "test"
