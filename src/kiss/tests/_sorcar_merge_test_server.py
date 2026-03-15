"""Sorcar test server whose agent modifies a file so merge view opens."""

import atexit
import shutil
import sys
import traceback
import webbrowser
from pathlib import Path
from typing import Any

import coverage


def main() -> None:
    port_file = sys.argv[1]
    work_dir = sys.argv[2]
    cov_data_file = sys.argv[3] if len(sys.argv) > 3 else ".coverage.merge_server"

    cov = coverage.Coverage(
        source=["kiss.agents.sorcar"],
        branch=True,
        data_file=cov_data_file,
    )
    cov.start()
    def _save_cov() -> None:
        cov.stop()
        cov.save()

    atexit.register(_save_cov)

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

    class FileModifyingAgent(RelentlessAgent):
        def __init__(self, name: str) -> None:
            self._work_dir = work_dir

        def run(self, **kwargs: Any) -> str:  # type: ignore[override]
            p = Path(self._work_dir) / "file.txt"
            p.write_text("MODIFIED line1\nline2\nnew line3\n")
            return "done"

    _orig_find_free_port = browser_ui.find_free_port

    def _patched_find_free_port() -> int:
        port = _orig_find_free_port()
        with open(port_file, "w") as f:
            f.write(str(port))
        return port

    sorcar_module.find_free_port = _patched_find_free_port  # type: ignore[attr-defined]
    sorcar_module.run_chatbot(
        agent_factory=FileModifyingAgent, title="MergeTest", work_dir=work_dir
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
