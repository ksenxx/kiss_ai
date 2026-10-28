# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A ``run_agent`` job whose thread cannot start must not stay registered.

``agent_dispatch.start_agent_job`` registers the job and then starts
its thread under one lock.  Before the fix a ``Thread.start()`` that
raised (``RuntimeError: can't start new thread`` under thread
exhaustion) propagated out of the tool call but left the job in the
registry with ``finished`` unset and a thread that would never run:
``agent_job(id, "wait")`` and ``"kill"`` answered "is still running"
for ever, the job's explicit budget stayed reserved against every later
dispatch of the same caller, and the finish gate named a job the model
could not wait for or kill.

End-to-end: a child interpreter exhausts its own thread quota for real
(``RLIMIT_NPROC`` lowered to 1, which makes ``pthread_create`` fail with
``EAGAIN``), calls ``start_agent_job`` and reports the registry.  No
test double stands in for ``Thread.start``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

_CHILD = r"""
import json, resource, threading, sys
from kiss.agents.sorcar import agent_dispatch as ad

class Owner:
    pass

owner = Owner()
kwargs = {"timeout": 5.0, "budget": 1.25, "prompt": "p", "sea_path": "",
          "work_dir": "", "model_name": "", "alias": ""}
# From here on no thread can be created in this process (root is exempt
# from RLIMIT_NPROC, which the probe below reports).
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
try:
    threading.Thread(target=lambda: None).start()
    print(json.dumps({"limit_enforced": False}))
    sys.exit(0)
except RuntimeError:
    pass
report = {"limit_enforced": True, "raised": ""}
try:
    ad.start_agent_job("x", kwargs, owner)
except RuntimeError as exc:
    report["raised"] = str(exc)
live = ad.live_agent_jobs(owner)
report["registered"] = list(ad.agent_jobs_of(owner))
report["reserved"] = ad.reserved_budget(live)
if live:
    tool = ad.make_agent_job_tool(owner)
    report["wait"] = tool(live[0].job_id, "wait", "1")
    report["kill"] = tool(live[0].job_id, "kill")
print(json.dumps(report))
"""


@posix_only("RLIMIT_NPROC is a POSIX resource limit")
def test_failed_thread_start_leaves_no_registered_job(tmp_path: Path) -> None:
    """``start_agent_job`` re-raises the thread error and the registry stays empty."""
    env = {**os.environ, "KISS_HOME": str(tmp_path / "home")}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120,
        env=env, cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    if not report["limit_enforced"]:
        pytest.skip("RLIMIT_NPROC is not enforced for this user (root)")
    assert "can't start new thread" in report["raised"], report
    assert report["registered"] == [], (
        f"ghost job left in the registry after a failed Thread.start(): {report}"
    )
    assert report["reserved"] == 0.0
