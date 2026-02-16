# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Single-agent coding system with smart continuation for long tasks."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import yaml

import kiss.agents.coding_agents.config as _coding_config  # noqa: F401  # register coding_agent config
from kiss.core import config as config_module
from kiss.core.base import Base
from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import get_max_context_length
from kiss.core.printer import Printer
from kiss.core.useful_tools import UsefulTools
from kiss.core.utils import resolve_path
from kiss.docker.docker_manager import DockerManager

TASK_PROMPT = """# Task
{task_description}

# CRITICAL Rules
- Use Write() for new/full files. Edit() only for tiny fixes.
- Bash(): set timeout_seconds=120 for test runs with sleeps/waits.
- Bash scripts with background jobs: use bounded poll loops (max iterations), never unbounded waits.
- IMMEDIATELY call finish(success=True, summary="done") once tests pass. NO extra verification.
- At step {step_threshold}: call finish(success=False, summary={{"done":[...], "next":[...]}})
- Work dir: {work_dir}
{previous_progress}"""

CONTINUATION_PROMPT = """# CONTINUATION

{progress_text}

Fix remaining issues then call finish. Don't redo completed work."""


def finish(success: bool, summary: str) -> str:
    """Finish execution with status and summary.

    Args:
        success: True if successful, False otherwise.
        summary: Summary of work done and remaining work (JSON for continuation).
    """
    if isinstance(success, str):
        success = success.strip().lower() not in ("false", "0", "no", "")
    return yaml.dump({"success": bool(success), "summary": summary}, indent=2, sort_keys=False)


class RelentlessCodingAgent(Base):
    """Single-agent coding system with auto-continuation for infinite tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _reset(
        self,
        model_name: str | None,
        max_sub_sessions: int | None,
        max_steps: int | None,
        max_budget: float | None,
        work_dir: str | None,
        base_dir: str | None,
        readable_paths: list[str] | None,
        writable_paths: list[str] | None,
        docker_image: str | None,
    ) -> None:
        global_cfg = config_module.DEFAULT_CONFIG
        cfg = global_cfg.coding_agent.relentless_coding_agent
        default_work_dir = str(Path(global_cfg.agent.artifact_dir).resolve() / "kiss_workdir")

        actual_base_dir = base_dir if base_dir is not None else default_work_dir
        actual_work_dir = work_dir if work_dir is not None else default_work_dir

        Path(actual_base_dir).mkdir(parents=True, exist_ok=True)
        Path(actual_work_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = str(Path(actual_base_dir).resolve())
        self.work_dir = str(Path(actual_work_dir).resolve())
        self.readable_paths = [resolve_path(p, self.base_dir) for p in readable_paths or []]
        self.writable_paths = [resolve_path(p, self.base_dir) for p in writable_paths or []]
        self.readable_paths.append(Path(self.work_dir))
        self.writable_paths.append(Path(self.work_dir))
        self.is_agentic = True

        self.max_sub_sessions = (
            max_sub_sessions if max_sub_sessions is not None else cfg.max_sub_sessions
        )
        self.max_steps = max_steps if max_steps is not None else cfg.max_steps
        self.max_budget = max_budget if max_budget is not None else cfg.max_budget
        self.model_name = (
            model_name
            if model_name is not None
            else cfg.model_name
        )
        self.max_tokens = get_max_context_length(self.model_name)

        self.budget_used: float = 0.0
        self.total_tokens_used: int = 0

        self.docker_image = docker_image
        self.docker_manager: DockerManager | None = None

        self.useful_tools = UsefulTools(
            base_dir=self.base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
        )

    def _docker_bash(self, command: str, description: str) -> str:
        if self.docker_manager is None:
            raise KISSError("Docker manager not initialized")
        return self.docker_manager.run_bash_command(command, description)

    def _parse_progress(self, summary: str) -> tuple[list[str], list[str]]:
        try:
            progress = json.loads(summary)
            done = progress.get("done", [])
            next_items = progress.get("next", [])
            if isinstance(done, list) and isinstance(next_items, list):
                done = list(dict.fromkeys(str(d) for d in done if d))
                next_items = list(dict.fromkeys(str(n) for n in next_items if n))
                return done, next_items
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return [], []

    def _format_progress(self, done_items: list[str], next_items: list[str]) -> str:
        if not done_items:
            return ""
        progress = "## Done\n"
        for item in done_items[-10:]:
            progress += f"- {item}\n"
        if next_items:
            progress += "\n## TODO\n"
            for item in next_items:
                progress += f"- {item}\n"
        return progress

    def _build_continuation_section(
        self, done_items: list[str], next_items: list[str]
    ) -> str:
        progress_text = self._format_progress(done_items, next_items)
        return "\n\n" + CONTINUATION_PROMPT.format(
            progress_text=progress_text,
        )

    def perform_task(self) -> str:
        print(f"Executing task: {self.task_description}")
        bash_tool = self._docker_bash if self.docker_manager else self.useful_tools.Bash

        done_items: list[str] = []
        next_items: list[str] = []

        for trial in range(self.max_sub_sessions):
            step_threshold = self.max_steps - 2

            if trial == 0:
                progress_section = ""
            else:
                progress_section = self._build_continuation_section(
                    done_items, next_items
                )

            executor = KISSAgent(f"{self.name} Trial-{trial}")
            try:
                result = executor.run(
                    model_name=self.model_name,
                    prompt_template=TASK_PROMPT,
                    arguments={
                        "task_description": self.task_description,
                        "previous_progress": progress_section,
                        "step_threshold": str(step_threshold),
                        "work_dir": self.work_dir,
                    },
                    tools=[
                        finish,
                        bash_tool,
                        self.useful_tools.Read,
                        self.useful_tools.Edit,
                        self.useful_tools.Write,
                    ],
                    max_steps=self.max_steps,
                    max_budget=self.max_budget,
                    printer=self.printer,
                )
            except Exception:
                last_msgs = executor.messages[-2:] if hasattr(executor, "messages") else []
                context = " ".join(
                    str(m.get("content", ""))
                    for m in last_msgs
                    if isinstance(m, dict)
                )
                result = yaml.dump(
                    {
                        "success": False,
                        "summary": json.dumps(
                            {"done": done_items, "next": [f"Continue: {context}"]}
                        ),
                    },
                    sort_keys=False,
                )

            self.budget_used += executor.budget_used  # type: ignore
            self.total_tokens_used += executor.total_tokens_used  # type: ignore

            ret = yaml.safe_load(result)
            payload = ret if isinstance(ret, dict) else {}

            if payload.get("success", False):
                return result

            summary = payload.get("summary", "")
            trial_done, trial_next = self._parse_progress(summary)

            if trial_done:
                for item in trial_done:
                    if item not in done_items:
                        done_items.append(item)
                next_items = trial_next
            elif summary and summary not in done_items:
                done_items.append(summary)

        raise KISSError(f"Task failed after {self.max_sub_sessions} sub-sessions")

    def run(
        self,
        model_name: str | None = None,
        prompt_template: str = "",
        arguments: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_budget: float | None = None,
        work_dir: str | None = None,
        base_dir: str | None = None,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
        printer: Printer | None = None,
        max_sub_sessions: int | None = None,
        docker_image: str | None = None,
        print_to_console: bool | None = None,
        print_to_browser: bool | None = None,
    ) -> str:
        """Run the coding agent."""
        self._reset(
            model_name,
            max_sub_sessions,
            max_steps,
            max_budget,
            work_dir,
            base_dir,
            readable_paths,
            writable_paths,
            docker_image,
        )
        self.prompt_template = prompt_template
        self.arguments = arguments or {}
        self.task_description = prompt_template.format(**self.arguments)
        self.set_printer(
            printer,
            print_to_console=print_to_console,
            print_to_browser=print_to_browser,
        )

        if self.docker_image:
            with DockerManager(self.docker_image) as docker_mgr:
                self.docker_manager = docker_mgr
                try:
                    return self.perform_task()
                finally:
                    self.docker_manager = None
        else:
            return self.perform_task()


def main() -> None:
    import time as time_mod

    agent = RelentlessCodingAgent("Example Multi-Agent")
    task_description = """
**Task:** Build a complete in-memory relational database engine in C with SQL parsing, \
query execution, indexing, and transactions.

**Requirements:**

### Part 1: Storage Engine (`storage.c` / `storage.h`)
1. Implement a page-based storage manager:
   - Fixed 4096-byte pages. Each table is a collection of pages.
   - Rows are stored as length-prefixed byte sequences within pages. \
Pages use a slotted-page layout: a header with slot count and free-space offset, \
a slot directory at the top growing downward, and row data at the bottom growing upward.
   - Support column types: `INT` (4 bytes, signed 32-bit), \
`TEXT` (variable length, max 255 bytes), `FLOAT` (8 bytes, IEEE 754 double).
   - Implement a buffer pool of 64 pages using LRU eviction. Pages are pinned while in use; \
eviction must never discard a pinned or dirty page.
   - Implement `page_alloc()`, `page_read(page_id)`, \
`page_write(page_id, data)`, `page_free(page_id)`.
2. Implement a B+ tree index (`btree.c` / `btree.h`):
   - Order-32 B+ tree (max 31 keys per internal node, max 31 key-value pairs per leaf).
   - Leaf nodes are linked in a doubly-linked list for range scans.
   - Support `btree_insert(key, row_id)`, `btree_delete(key)`, `btree_search(key)`, \
`btree_range(low, high)` returning an iterator.
   - Keys are 64-bit signed integers. Duplicate keys allowed (secondary indexes).
   - Handle node splits and merges (rebalancing) correctly. After deletion, merge \
underflowing nodes (less than half full) with siblings or redistribute keys.

### Part 2: SQL Parser (`parser.c` / `parser.h`)
1. Implement a recursive-descent SQL parser supporting:
   - `CREATE TABLE name (col1 INT, col2 TEXT, col3 FLOAT, PRIMARY KEY(col1))`
   - `DROP TABLE name`
   - `INSERT INTO name VALUES (1, 'hello', 3.14)` \
and `INSERT INTO name (col1, col3) VALUES (1, 3.14)`
   - `SELECT col1, col2 FROM t1 WHERE col1 > 10 AND col2 = 'foo' ORDER BY col1 DESC LIMIT 20`
   - `SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.fk_id WHERE t1.val > 5`
   - `UPDATE name SET col2 = 'new' WHERE col1 = 42`
   - `DELETE FROM name WHERE col1 < 10`
   - `CREATE INDEX idx_name ON table(column)`
   - `BEGIN`, `COMMIT`, `ROLLBACK`
2. The parser must produce an AST (abstract syntax tree) using structs. Each node type \
has its own struct: `CreateTableNode`, `SelectNode`, `InsertNode`, `UpdateNode`, `DeleteNode`, \
`WhereClause` (supporting `AND`, `OR`, `NOT`, comparisons `=`, `!=`, `<`, `>`, `<=`, `>=`), \
`JoinClause`, `OrderByClause`, `LimitClause`.
3. The tokenizer must handle: identifiers, single-quoted strings \
(with `''` escape for embedded quotes), integer literals, float literals, \
parentheses, commas, operators, and SQL keywords (case-insensitive).

### Part 3: Query Executor (`executor.c` / `executor.h`)
1. Walk the AST and execute queries:
   - `CREATE TABLE`: allocate metadata, store schema (column names, types, primary key).
   - `INSERT`: validate types, enforce primary key uniqueness \
(if PK exists), write row to table pages, update all indexes.
   - `SELECT`: full table scan or index scan (use index when \
WHERE filters on an indexed column with `=` or range). Implement a simple \
nested-loop join for INNER JOIN queries. Apply WHERE filter, \
ORDER BY (in-memory quicksort), and LIMIT.
   - `UPDATE`: find matching rows, modify in place \
(or delete+reinsert if row size changes), update indexes.
   - `DELETE`: remove rows, compact page slots, update indexes.
   - `CREATE INDEX`: scan existing rows and bulk-load them into a new B+ tree.
2. Query results are returned as an array of result rows. Each row is an array of column values.
3. Implement a simple query planner that chooses between full scan and index scan based on \
whether a usable index exists for the WHERE predicate.

### Part 4: Transaction Manager (`txn.c` / `txn.h`)
1. Implement MVCC-style transactions with snapshot isolation:
   - Each row version has `created_by_txn` and `deleted_by_txn` fields.
   - `BEGIN` assigns an incrementing transaction ID and takes a snapshot of active transaction IDs.
   - A transaction can only see row versions where \
`created_by_txn` is committed and not in the snapshot, and `deleted_by_txn` \
is either null or an uncommitted/in-snapshot transaction.
   - `COMMIT` marks the transaction as committed in a global transaction table.
   - `ROLLBACK` marks it as aborted; all its row versions become invisible.
2. Detect write-write conflicts: if two concurrent transactions modify the same row, \
the second to commit must abort with an error message.
3. Implement a write-ahead log (WAL):
   - Before any page modification, write a log record: \
`(txn_id, page_id, offset, old_data, new_data)`.
   - On `COMMIT`, flush the WAL to disk (an in-memory buffer representing disk).
   - Implement `recover()` that replays committed transactions and undoes aborted ones from the WAL.

### Part 5: Interactive REPL and Test Suite
1. Create `main.c` with an interactive REPL:
   - Prompt `db> `, read SQL statements (semicolon-terminated, may span multiple lines).
   - Print results in aligned columnar format with headers.
   - Print row count after each query. Print execution time in milliseconds.
   - Handle `.quit`, `.tables` (list tables), `.schema <table>` (show CREATE statement), \
`.indexes <table>` (list indexes on table).
2. Create `Makefile`:
   - `make` or `make all` — compile with `gcc -Wall -Wextra -Werror -std=c11 -O2`
   - `make debug` — compile with `-g -fsanitize=address -fsanitize=undefined`
   - `make test` — compile and run the test suite
   - `make clean` — remove binaries and objects
3. Create `test_db.c` with comprehensive tests \
(using a simple assertion macro, no test framework needed):
   - **Schema tests:** CREATE TABLE, DROP TABLE, duplicate table error, type validation.
   - **CRUD tests:** INSERT rows, SELECT with WHERE, UPDATE, DELETE. \
Verify correct row counts and values.
   - **Index tests:** CREATE INDEX, verify index scan is used (check a flag or counter), \
insert 1000 rows and verify point lookup and range query return correct results.
   - **Join test:** Two tables with foreign key relationship, \
INNER JOIN returns correct combined rows.
   - **B+ tree stress test:** Insert 10000 sequential keys, \
then 10000 random keys. Delete half randomly. Verify all remaining keys \
are findable. Verify range scans return correct sorted results.
   - **Transaction tests:**
     - Begin, insert, rollback — row must not be visible.
     - Begin T1, begin T2, T1 inserts row, T2 cannot see it, T1 commits, T2 still cannot see it \
(snapshot isolation). New T3 can see it.
     - Write-write conflict: T1 updates row, T2 updates same row, T2 commit must fail.
   - **WAL recovery test:** Begin transaction, insert rows, commit, \
simulate crash (discard buffer pool), call `recover()`, verify rows are \
present. Also test: uncommitted transaction's changes are rolled back.
   - **Edge cases:** Empty table SELECT, DELETE from empty table, \
INSERT with wrong number of columns, INSERT with type mismatch, \
SELECT from nonexistent table, ORDER BY on TEXT column, \
NULL-like behavior for missing columns in partial INSERT.
   - **Concurrency simulation:** Run 100 transactions sequentially \
(simulating concurrent interleaving), each inserting a unique row. \
Verify final table has exactly 100 rows.
   - All tests print `PASS` or `FAIL` with test name. Final summary: `X/Y tests passed`.

**Constraints:** Pure C11. No external libraries beyond the C standard \
library (stdio, stdlib, string, stdint, stdbool, assert, time). \
Compile with `gcc`. All source files in the current directory. \
No docs. No comments longer than one line. Memory: all allocations must be freed (no leaks under \
normal operation — test with address sanitizer).
"""

    work_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        result = agent.run(
            prompt_template=task_description,
            model_name="claude-sonnet-4-5",
            max_steps=15,
            work_dir=work_dir,
            print_to_browser=True,
            print_to_console=True,
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time

    print("FINAL RESULT:")
    result_data = yaml.safe_load(result)
    print("Completed successfully: " + str(result_data["success"]))
    print(result_data["summary"])
    print("Work directory was: " + work_dir)
    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()
