"""Minimal sorcar server for shutdown integration test."""

import shutil
import sys
import traceback
import webbrowser
from typing import Any


def main() -> None:
    port_file = sys.argv[1]
    work_dir = sys.argv[2]

    _original_which = shutil.which

    def _no_code_server(cmd: str, mode: int = 0, path: Any = None) -> str | None:
        if cmd == "code-server":
            return None
        return _original_which(cmd, mode=mode, path=path)

    shutil.which = _no_code_server  # type: ignore[assignment]

    webbrowser.open = lambda url: None  # type: ignore[assignment,misc]

    from kiss.agents.sorcar import browser_ui
    from kiss.agents.sorcar import sorcar as sorcar_module
    from kiss.core.relentless_agent import RelentlessAgent

    class DummyAgent(RelentlessAgent):
        def __init__(self, name: str) -> None:
            pass

        def run(self, **kwargs: Any) -> str:  # type: ignore[override]
            return "done"

    _orig_find_free_port = browser_ui.find_free_port

    def _patched_find_free_port() -> int:
        port = _orig_find_free_port()
        with open(port_file, "w") as f:
            f.write(str(port))
        return port

    sorcar_module.find_free_port = _patched_find_free_port  # type: ignore[attr-defined]

    sorcar_module.run_chatbot(
        agent_factory=DummyAgent, title="Test", work_dir=work_dir
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
