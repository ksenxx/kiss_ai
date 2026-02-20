# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Single-agent coding system with smart continuation for long tasks."""

from __future__ import annotations

import os
import tempfile

import yaml

import kiss.agents.coding_agents.config as _coding_config  # noqa: F401
from kiss.agents.assistant.relentless_agent import RelentlessAgent
from kiss.core.printer import Printer
from kiss.core.useful_tools import UsefulTools


class RelentlessCodingAgent(RelentlessAgent):
    """Single-agent coding system with auto-continuation for infinite tasks."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _get_tools(self) -> list:
        useful_tools = UsefulTools(
            base_dir=self.base_dir,
            readable_paths=[str(p) for p in self.readable_paths],
            writable_paths=[str(p) for p in self.writable_paths],
        )
        bash_tool = self._docker_bash if self.docker_manager else useful_tools.Bash
        return [bash_tool, useful_tools.Read, useful_tools.Edit, useful_tools.Write]

    def run(  # type: ignore[override]
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
        return super().run(
            model_name=model_name,
            prompt_template=prompt_template,
            arguments=arguments,
            max_steps=max_steps,
            max_budget=max_budget,
            work_dir=work_dir,
            base_dir=base_dir,
            readable_paths=readable_paths,
            writable_paths=writable_paths,
            printer=printer,
            max_sub_sessions=max_sub_sessions,
            docker_image=docker_image,
            print_to_console=print_to_console,
            print_to_browser=print_to_browser,
            tools_factory=self._get_tools,
            config_path="coding_agent.relentless_coding_agent",
        )


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
