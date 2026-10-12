# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt2 (core agents): genuine bugs in the sorcar core agent files.

Bug 2 — ``set_model`` never swaps the live model mid-run
--------------------------------------------------------
``SorcarAgent`` is a :class:`RelentlessAgent`: the model that actually
performs LLM calls lives on the inner per-session executor
(``self._current_executor``, a fresh ``KISSAgent`` per sub-session).
``RelentlessAgent._reset`` never sets ``self.model`` on the parent, so
the ``set_model`` tool's ``getattr(self, "model", None)`` was ALWAYS
``None`` during a real run — the tool silently took the "deferred"
branch (only updating ``self.model_name``, which is read again only at
the NEXT sub-session, which usually never starts).  The documented
contract — "swaps the agent's live LLM model instance in place so the
very next LLM call goes to the changed model" — was never met in
production.  Fixed by targeting ``self._current_executor`` (when live)
so the swap lands on the model instance the ReAct loop reads on each
step.

No mocks / no patches: a real local HTTP server plays the LLM (always
returning a ``finish`` tool call) and persistence is redirected to a
temp directory.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.tests.server.conftest import redirect_persistence, restore_persistence


class _SetModelHandler(BaseHTTPRequestHandler):
    """First request returns a ``set_model`` tool call; later ones ``finish``.

    Records the ``model`` field of every incoming request body on the
    server object so the test can assert which model each LLM call was
    addressed to.
    """

    def do_POST(self) -> None:  # noqa: N802
        cl = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(cl) if cl else b"{}"
        try:
            req_model = json.loads(raw).get("model", "")
        except Exception:
            req_model = ""
        server: Any = self.server
        with server.state_lock:
            server.request_models.append(req_model)
            first = len(server.request_models) == 1
        if first:
            tool = {
                "id": "call_sm",
                "type": "function",
                "function": {
                    "name": "set_model",
                    "arguments": json.dumps({"model_name": "swapped-model"}),
                },
            }
        else:
            tool = {
                "id": "call_fin",
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": json.dumps(
                        {"success": "true", "summary": "done"}
                    ),
                },
            }
        body = json.dumps(
            {
                "id": "chatcmpl-x",
                "object": "chat.completion",
                "model": req_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [tool],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class TestSetModelSwapsLiveModel:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt2-setmodel-")
        self.saved = redirect_persistence(self.tmpdir)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _SetModelHandler)
        self.srv.request_models = []  # type: ignore[attr-defined]
        self.srv.state_lock = threading.Lock()  # type: ignore[attr-defined]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"
        agent_state.agent_states.clear()

    def teardown_method(self) -> None:
        self.srv.shutdown()
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        agent_state.agent_states.clear()

    def test_next_llm_call_uses_swapped_model(self) -> None:
        """After the agent calls ``set_model('swapped-model')`` mid-run, the
        very next LLM request must be addressed to ``swapped-model``."""
        from kiss.agents.sorcar.sorcar_agent import SorcarAgent

        work = Path(self.tmpdir) / "w"
        work.mkdir()
        agent = SorcarAgent("set-model-live")
        result = agent.run(
            prompt_template="please switch models then finish",
            model_name="gpt-4o-mini",
            model_config={"base_url": self.url, "api_key": "test-key"},
            work_dir=str(work),
            web_tools=False,
            max_steps=5,
        )
        assert "success" in result
        models = list(self.srv.request_models)  # type: ignore[attr-defined]
        assert len(models) >= 2
        assert models[0] == "gpt-4o-mini"
        assert models[1] == "swapped-model"
        assert agent.model_name == "swapped-model"
