"""Minimal sorcar server for shutdown integration test."""

import shutil
import sys
import tempfile
import traceback
import webbrowser
from pathlib import Path
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

    import kiss.agents.sorcar.task_history as th
    from kiss.agents.sorcar import browser_ui
    from kiss.agents.sorcar import sorcar as sorcar_module
    from kiss.core.relentless_agent import RelentlessAgent

    # Redirect to a temp dir so we don't pollute ~/.kiss and so no
    # stale ui-port file prevents find_free_port from being called.
    tmp_kiss = Path(tempfile.mkdtemp()) / ".kiss"
    tmp_kiss.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = tmp_kiss
    th._DB_PATH = tmp_kiss / "history.db"
    th._db_conn = None
    # sorcar.py binds _KISS_DIR locally via `from ... import _KISS_DIR`,
    # so we must also patch the sorcar module's own binding.
    sorcar_module._KISS_DIR = tmp_kiss

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
