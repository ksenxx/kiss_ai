"""Optimize an agent using genetic Pareto frontier optimization.

Pseudocode
----------

OPTIMIZE(Tasks, Folder, Program, Metrics-Description,
         Generations, Frontier-Size, Max-Frontier, Sample-Size, Runs, P-mutate):

  Frontier := empty

  PHASE 1 — Seed:
    First variant is an unmodified copy of Folder (the baseline).
    Create up to Frontier-Size additional variants by calling IMPROVE on the baseline.
    Evaluate each; add to Frontier via UPDATE-PARETO.

  PHASE 2 — Evolve:
    Repeat Generations times:
      With probability P-mutate (or if |Frontier| < 2): MUTATION
        Pick a random member of Frontier as parent.
        Child := IMPROVE(parent, Tasks, parent's metrics and history)
      Otherwise: CROSSOVER
        Pick two members; let Primary be the one with lower Score.
        Child := IMPROVE(Primary, Tasks, combined histories of both parents)
      Evaluate Child; add to Frontier via UPDATE-PARETO.

  Return the Frontier member with the lowest Score.
  Copy its files back to Folder.

EVALUATE(Variant, Program, Tasks, Sample-Size, Runs):
  Pick Sample-Size random tasks.  Run each Runs times.
  Each run: load the agent from Variant/Program, execute it on the task
    (up to 15 steps, $0.50 budget, using Claude Sonnet 4.5).
  Record: success/failure, tokens, time, cost.
  Return: failure-rate (arithmetic mean), tokens/time/cost (geometric means).

IMPROVE(Source, Tasks, Metrics-Description, Current-Metrics, History):
  Copy Source to a new folder.
  Invoke an LLM coding agent (up to 15 steps) with a prompt describing
    the code, the tasks, current metrics, past history, and optimization rules.
  The agent edits the code to reduce the target metrics.

UPDATE-PARETO(Frontier, Candidate, Max-Frontier):
  Reject if any member dominates Candidate (all metrics <= and at least one <).
  Remove members that Candidate dominates.  Add Candidate.
  If |Frontier| > Max-Frontier, trim by crowding distance.

SCORE(Variant):
  Weighted sum of metrics: failure-rate x 1M, tokens x 1, time x 1K, cost x 100K.

"""

import importlib.util
import inspect
import math
import os
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiss.agents.coding_agents.claude_coding_agent import ClaudeCodingAgent
from kiss.core.base import Base

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

IMPROVE_PROMPT = """You are optimizing an AI agent to minimize: {metrics_description}

## Agent Source
The agent code is under: {work_dir}/
The main entry point is: {work_dir}/{program_path}
You may read and modify any file under {work_dir}/.

## KISS Framework Reference
- {kiss_folder}/API.md
- {kiss_folder}/README.md
- {kiss_folder}/src/kiss/agents/coding_agents/ for example agents

## Tasks the Agent Must Handle
{task_descriptions}

## Current Performance
{current_metrics}

## Previous Improvement History
{improvement_history}

## Rules
1. Read all files under {work_dir}/ to understand the agent
2. Make targeted improvements to reduce the metrics
3. You may modify any file under {work_dir}/
4. PRESERVE: class name, __init__ signature, run() method signature, streaming mechanism
5. Do NOT change the agent's interface or streaming mechanism
6. The agent MUST still work correctly on ALL tasks above
7. Do NOT use: caching, multiprocessing, async/await, docker
8. Do NOT set max_thinking_tokens below 1024
9. Do NOT remove required imports or break module structure
10. Verify the file imports cleanly:
    python3 -c "import importlib.util; \
    spec=importlib.util.spec_from_file_location('t','{work_dir}/{program_path}'); \
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"

## Strategies
- IMPORTANT: Optimizations must be GENERAL across ALL tasks, not task-specific
- Do NOT add task-specific instructions to the system prompt
- Shorter system prompts preserving meaning
- Remove redundant instructions
- Reduce allowed_tools to essentials for bash-heavy tasks
- Minimize conversation turns
- Optimize permission_handler
"""

TASKS = [
"""**Task:** Create a robust key-value database engine using only Bash scripts.

**Requirements:**
1. Create `db.sh` operating on `./my_db` directory.
2. **Basic Operations:** `db.sh set <key> <value>`, `db.sh get <key>`, `db.sh delete <key>`.
3. **Atomicity:** Transaction support:
   - `db.sh begin` starts a session (writes cached, invisible to others)
   - `db.sh commit` atomically applies cached changes
   - `db.sh rollback` discards pending changes
4. **Concurrency:** Simultaneous processes must never corrupt data (use `mkdir`-based mutex).
5. **Validation:** Write `test_stress.sh` launching 10 concurrent processes, verifying no data loss.

**Constraints:** No sqlite3/python. Standard Linux utilities only. \
Operate within `./my_db`. No docs.""",

"""**Task:** Build a task scheduler with dependency resolution in Bash.

**Requirements:**
1. Create `scheduler.sh` with:
   - `scheduler.sh add <name> <command> [--priority <1-10>] [--depends <t1,t2>]` — enqueue task
   - `scheduler.sh run` — execute tasks respecting priority and dependencies
   - `scheduler.sh status` — show task states (pending/running/done/failed)
   - `scheduler.sh log <name>` — show stdout/stderr of completed task
2. **Dependencies:** Tasks wait for dependencies. Detect circular dependencies.
3. **Parallel Execution:** Run up to 3 independent tasks concurrently via background processes.
4. **Failure Handling:** Failed task blocks dependents; other independent tasks continue.
5. **Validation:** Write `test_scheduler.sh` that creates a diamond dependency graph \
(A->B, A->C, B->D, C->D), verifies execution order, tests circular detection and \
failure propagation.

**Constraints:** Pure Bash, standard utilities. State in `./scheduler_data/`. No docs.""",

"""**Task:** Implement a file version control system in Bash.

**Requirements:**
1. Create `vcs.sh` with:
   - `vcs.sh init` — create `.vcs/` repository
   - `vcs.sh add <file>` — stage a file
   - `vcs.sh commit <message>` — snapshot staged files with message and timestamp
   - `vcs.sh log` — show commit history (hash, message, timestamp, changed files)
   - `vcs.sh diff <file>` — unified diff between working copy and last commit
   - `vcs.sh checkout <hash>` — restore files to a commit's state
   - `vcs.sh status` — show modified, staged, untracked files
2. **Commit Hashing:** Unique hash per commit via `sha256sum`.
3. **Storage:** One copy per unique file content.
4. **Validation:** Write `test_vcs.sh` that creates files, makes 3+ commits, verifies \
log, diff, checkout restore, and status accuracy.

**Constraints:** Pure Bash (diff, sha256sum, cp). Data in `.vcs/`. No docs.""",

"""**Task:** Build a log file analyzer and statistics reporter in Bash.

**Requirements:**
1. Create `gen_logs.sh` generating `access.log` with 10000 lines:
   `YYYY-MM-DD HH:MM:SS <IP> <METHOD> <PATH> <STATUS> <RESPONSE_MS> <BYTES>`
   with realistic distributions.
2. Create `analyze.sh` producing:
   - **Summary:** Total requests, unique IPs, date range, total bytes
   - **Status breakdown:** Count and percentage per status code
   - **Top 10 IPs** by request count with percentage
   - **Top 10 paths** by request count
   - **Hourly histogram** (text bar chart with `#`)
   - **P50, P90, P99 response times**
   - **Error rate per hour** (4xx+5xx / total)
3. **Performance:** Process 10000 lines in under 30 seconds.
4. **Validation:** Write `test_analyze.sh` verifying all sections exist, numbers are \
reasonable, percentiles correct on small dataset.

**Constraints:** Pure Bash (awk, sort, uniq). No python/perl. No docs.""",
    """**Task:** Create a make-like build system in Bash with incremental builds.

**Requirements:**
1. Create `build.sh` reading a `Buildfile`:
   ```
   target: dep1 dep2
       command1
       command2
   ```
   (4-space indented commands)
2. **Commands:**
   - `build.sh <target>` — build target and dependencies
   - `build.sh clean` — remove build artifacts
   - `build.sh list` — show all targets and dependencies
3. **Incremental:** Only rebuild if dependencies are newer. Track in `.build_cache/`.
4. **Dependency Resolution:** Build deps before dependents. Detect circular dependencies.
5. **Error Handling:** Stop on command failure, report which target/command failed.
6. **Validation:** Write `test_build.sh`:
   - Multi-level chain (sources -> objects -> library -> executable)
   - Verify initial build runs all steps
   - Verify re-build skips up-to-date targets
   - Modify source, verify partial rebuild
   - Test circular detection and error handling

**Constraints:** Pure Bash. Use `cp`/`cat` as fake compile commands. \
Metadata in `.build_cache/`. No docs.""",
]


@dataclass
class AgentVariant:
    folder_path: str
    metrics: dict[str, float]
    parent_ids: list[int]
    id: int = 0
    generation: int = 0
    improvement_history: str = ""

    def dominates(self, other: "AgentVariant") -> bool:
        all_metrics = set(self.metrics.keys()) | set(other.metrics.keys())
        strictly_better = False
        for metric in all_metrics:
            self_val = self.metrics.get(metric, sys.maxsize)
            other_val = other.metrics.get(metric, sys.maxsize)
            if self_val > other_val:
                return False
            if self_val < other_val:
                strictly_better = True
        return strictly_better

    def score(self, weights: dict[str, float] | None = None) -> float:
        if weights is None:
            weights = {
                "unsuccess_rate": 1_000_000,
                "execution_time": 1000,
                "cost": 100_000,
            }
        return sum(self.metrics.get(m, 0) * w for m, w in weights.items())


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in metrics.items()
    )


def _load_module(file_path: str) -> tuple[Any, str]:
    mod_name = f"opt_variant_{random.randint(0, 999999)}"
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        return None, mod_name
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        return None, mod_name
    return module, mod_name


def _find_agent_class(module: Any) -> type | None:
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            obj.__module__ == module.__name__
            and issubclass(obj, Base)
            and obj is not Base
        ):
            return obj
    return None


def _create_agent(agent_cls: type) -> Any:
    sig = inspect.signature(agent_cls)
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "name":
            kwargs["name"] = "OptEval"
        elif param.default is inspect.Parameter.empty:
            kwargs[name] = "OptEval"
    return agent_cls(**kwargs)


EVAL_MAX_STEPS = 15
EVAL_TIMEOUT = 300
IMPROVE_MAX_STEPS = 15


def _evaluate_on_task(
    variant_folder: str,
    program_path: str,
    task: str,
) -> tuple[dict[str, float], str]:
    fail_metrics = {"success": 1, "execution_time": 0.0, "cost": 0.0}
    agent_file = str(Path(variant_folder) / program_path)
    if not Path(agent_file).exists():
        return fail_metrics, "Agent file not found"

    module, mod_name = _load_module(agent_file)
    if module is None:
        print(f"Failed to load {agent_file}")
        return fail_metrics, "Failed to load module"

    agent_cls = _find_agent_class(module)
    if agent_cls is None:
        print(f"No agent class in {agent_file}")
        sys.modules.pop(mod_name, None)
        return fail_metrics, "No agent class extending Base found"

    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    start = time.time()
    agent = None
    error_msg = ""
    old_global_budget_used = Base.global_budget_used
    try:
        agent = _create_agent(agent_cls)
        os.chdir(work_dir)
        run_sig = inspect.signature(agent.run)
        run_kwargs: dict[str, Any] = {
            "prompt_template": task,
            "max_steps": EVAL_MAX_STEPS,
            "max_budget": 0.50,
        }
        if "work_dir" in run_sig.parameters:
            run_kwargs["work_dir"] = work_dir
        if "model_name" in run_sig.parameters:
            run_kwargs["model_name"] = "claude-sonnet-4-5"
        if "subtasker_model_name" in run_sig.parameters:
            run_kwargs["subtasker_model_name"] = "claude-sonnet-4-5"
        agent.run(**run_kwargs)
        success = 0
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"Eval error: {e}")
        success = 1
    finally:
        os.chdir(old_cwd)
        sys.modules.pop(mod_name, None)
        shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.time() - start
    if elapsed > EVAL_TIMEOUT:
        print(f"Task took {elapsed:.0f}s (limit {EVAL_TIMEOUT}s)")
    return {
        "success": success,
        "execution_time": elapsed,
        "cost": Base.global_budget_used - old_global_budget_used,
    }, error_msg


def _geometric_mean(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def evaluate_variant(
    variant_folder: str,
    program_path: str,
    tasks: list[str],
    eval_sample_size: int,
    eval_runs: int,
) -> tuple[dict[str, float], str]:
    sample = random.sample(tasks, min(eval_sample_size, len(tasks)))
    all_failures: list[float] = []
    all_times: list[float] = []
    all_costs: list[float] = []
    errors: list[str] = []

    for i, task in enumerate(sample):
        for run in range(eval_runs):
            label = f"  Task {i + 1}/{len(sample)} run {run + 1}/{eval_runs}"
            print(f"{label}: {task.strip()[:60]}...")
            metrics, error = _evaluate_on_task(
                variant_folder, program_path, task
            )
            all_failures.append(metrics["success"])
            all_times.append(max(metrics["execution_time"], 0.01))
            all_costs.append(max(metrics["cost"], 0.0001))
            if error:
                errors.append(f"Task {i + 1} run {run + 1}: {error}")
            print(f"    {_format_metrics(metrics)}")

    n = len(all_failures)
    return {
        "unsuccess_rate": sum(all_failures) / n,
        "execution_time": _geometric_mean(all_times),
        "cost": _geometric_mean(all_costs),
    }, "; ".join(errors)


def improve_variant(
    source_folder: str,
    target_folder: str,
    program_path: str,
    tasks: list[str],
    metrics_description: str,
    current_metrics: str,
    improvement_history: str,
) -> bool:
    shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)

    resolved_target = str(Path(target_folder).resolve())

    task_descriptions = "\n".join(
        f"### Task {i + 1}\n{t.strip()}" for i, t in enumerate(tasks)
    )

    agent = ClaudeCodingAgent("Improver")
    try:
        agent.run(
            prompt_template=IMPROVE_PROMPT,
            arguments={
                "program_path": program_path,
                "task_descriptions": task_descriptions,
                "metrics_description": metrics_description,
                "current_metrics": current_metrics or "No metrics yet (first run)",
                "improvement_history": improvement_history or "No previous improvements",
                "work_dir": resolved_target,
                "kiss_folder": str(PROJECT_ROOT),
            },
            model_name="claude-sonnet-4-5",
            max_steps=IMPROVE_MAX_STEPS,
            work_dir=resolved_target,
            readable_paths=[resolved_target, str(PROJECT_ROOT)],
            writable_paths=[resolved_target],
        )
        return True
    except Exception as e:
        print(f"Improvement failed: {e}")
        return False


def update_pareto_frontier(
    frontier: list[AgentVariant],
    new: AgentVariant,
    max_size: int,
) -> tuple[list[AgentVariant], bool]:
    for v in frontier:
        if v.dominates(new):
            return frontier, False
    frontier = [v for v in frontier if not new.dominates(v)]
    frontier.append(new)
    if len(frontier) > max_size:
        frontier = _trim_frontier(frontier, max_size)
    return frontier, True


def _trim_frontier(frontier: list[AgentVariant], max_size: int) -> list[AgentVariant]:
    n = len(frontier)
    all_metrics: set[str] = set()
    for v in frontier:
        all_metrics.update(v.metrics.keys())
    crowding = [0.0] * n
    for metric in all_metrics:
        values = [v.metrics.get(metric, 0) for v in frontier]
        vrange = max(values) - min(values) or 1
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        crowding[sorted_idx[0]] = crowding[sorted_idx[-1]] = float("inf")
        for i in range(1, n - 1):
            idx = sorted_idx[i]
            diff = values[sorted_idx[i + 1]] - values[sorted_idx[i - 1]]
            crowding[idx] += diff / vrange
    sorted_idx = sorted(range(n), key=lambda i: crowding[i], reverse=True)
    return [frontier[i] for i in sorted_idx[:max_size]]


def optimize(
    tasks: list[str],
    folder: str,
    metrics_description: str,
    program_path: str,
    max_generations: int = 3,
    initial_frontier_size: int = 2,
    max_frontier_size: int = 4,
    eval_sample_size: int = 2,
    eval_runs: int = 2,
    mutation_probability: float = 0.8,
) -> AgentVariant:

    work_dir = Path(tempfile.mkdtemp())
    variant_counter = 0
    frontier: list[AgentVariant] = []
    folder = str(Path(folder).resolve())

    print("=== Optimize Agent (Pareto Frontier) ===")
    print(f"Folder: {folder}, Program: {program_path}")
    print(f"Tasks: {len(tasks)}, Metrics: {metrics_description}")
    print(
        f"Generations: {max_generations}, Frontier: {initial_frontier_size}-"
        f"{max_frontier_size}, Eval sample: {eval_sample_size}, "
        f"Eval runs: {eval_runs}"
    )

    # Phase 1: Build initial frontier
    max_init_attempts = initial_frontier_size + 2
    for _attempt in range(max_init_attempts):
        if len(frontier) >= initial_frontier_size:
            break
        variant_counter += 1
        vid = variant_counter
        vdir = str(work_dir / f"variant_{vid}")

        if vid == 1:
            shutil.copytree(folder, vdir)
            print(f"Variant {vid}: baseline")
        else:
            baseline_metrics = frontier[0].metrics if frontier else {}
            failure_notes = "\n".join(
                f"- Variant {v.id}: {v.improvement_history}"
                for v in frontier
                if v.improvement_history.startswith("FAILED")
            )
            print(f"Variant {vid}: improving from baseline...")
            improve_variant(
                str(work_dir / "variant_1"),
                vdir,
                program_path,
                tasks,
                metrics_description,
                _format_metrics(baseline_metrics) if baseline_metrics else "",
                failure_notes or "",
            )

        print(f"Evaluating variant {vid}...")
        metrics, eval_error = evaluate_variant(
            vdir, program_path, tasks, eval_sample_size, eval_runs
        )
        history = f"FAILED: {eval_error}" if eval_error else ""
        if eval_error:
            print(f"Variant {vid} issues: {eval_error}")
        variant = AgentVariant(
            folder_path=vdir,
            metrics=metrics,
            parent_ids=[],
            id=vid,
            generation=0,
            improvement_history=history,
        )
        frontier, added = update_pareto_frontier(frontier, variant, max_frontier_size)
        print(
            f"Variant {vid}: {_format_metrics(metrics)}, "
            f"{'added' if added else 'rejected'} (frontier: {len(frontier)})"
        )

    if not frontier:
        raise RuntimeError("Failed to create any valid initial variants")

    # Phase 2: Evolution
    for gen in range(1, max_generations + 1):
        print(f"\n=== Generation {gen}/{max_generations} ===")
        failure_notes = "\n".join(
            f"- Variant {v.id}: {v.improvement_history}"
            for v in frontier
            if v.improvement_history.startswith("FAILED")
        )

        if random.random() < mutation_probability or len(frontier) < 2:
            parent = random.choice(frontier)
            print(f"Mutation from variant {parent.id}")
            variant_counter += 1
            vid = variant_counter
            vdir = str(work_dir / f"variant_{vid}")
            history = parent.improvement_history
            if failure_notes:
                history += f"\n\nPrevious failures:\n{failure_notes}"
            ok = improve_variant(
                parent.folder_path,
                vdir,
                program_path,
                tasks,
                metrics_description,
                _format_metrics(parent.metrics),
                history,
            )
            if not ok:
                continue
            parent_ids = [parent.id]
        else:
            v1, v2 = random.sample(frontier, 2)
            primary, secondary = (
                (v1, v2) if v1.score() <= v2.score() else (v2, v1)
            )
            print(
                f"Crossover: variant {primary.id} x {secondary.id}"
            )
            variant_counter += 1
            vid = variant_counter
            vdir = str(work_dir / f"variant_{vid}")
            history = (
                f"From variant {primary.id}:\n{primary.improvement_history}\n"
                f"From variant {secondary.id}:\n{secondary.improvement_history}"
            )
            if failure_notes:
                history += f"\n\nPrevious failures:\n{failure_notes}"
            ok = improve_variant(
                primary.folder_path,
                vdir,
                program_path,
                tasks,
                metrics_description,
                _format_metrics(primary.metrics),
                history,
            )
            if not ok:
                continue
            parent_ids = [primary.id, secondary.id]

        print(f"Evaluating variant {vid}...")
        metrics, eval_error = evaluate_variant(
            vdir, program_path, tasks, eval_sample_size, eval_runs
        )
        variant = AgentVariant(
            folder_path=vdir,
            metrics=metrics,
            parent_ids=parent_ids,
            id=vid,
            generation=gen,
            improvement_history=f"FAILED: {eval_error}" if eval_error else "",
        )
        frontier, added = update_pareto_frontier(frontier, variant, max_frontier_size)
        best = min(frontier, key=lambda v: v.score())
        print(
            f"Variant {vid}: {_format_metrics(metrics)} "
            f"{'added' if added else 'rejected'} | Best: {best.id}"
        )

    best = min(frontier, key=lambda v: v.score())
    print("\n=== Optimization Complete ===")
    for v in frontier:
        print(f"  variant {v.id}: {_format_metrics(v.metrics)}")
    print(f"Best: variant {best.id}, score={best.score():.2f}")

    shutil.copytree(best.folder_path, folder, dirs_exist_ok=True)
    print(f"Best variant copied back to: {folder}")

    shutil.rmtree(work_dir, ignore_errors=True)
    return best


def main() -> None:
    folder = str(Path(__file__).parent)
    program_path = "relentless_coding_agent.py"

    best = optimize(
        tasks=TASKS,
        folder=folder,
        metrics_description=(
            "unsuccess_rate (highest priority), "
            "running_time (second), "
            "cost (third), "
        ),
        program_path=program_path,
        max_generations=10,
        initial_frontier_size=3,
        max_frontier_size=6,
        eval_sample_size=2,
        eval_runs=2,
    )

    print(f"\nBest variant: {best.id}")
    print(f"Metrics: {_format_metrics(best.metrics)}")
    print(f"Score: {best.score():.2f}")
    print(f"Modified files in: {folder}")


if __name__ == "__main__":
    main()
