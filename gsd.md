# Plan: Incorporating GSD Ideas into KISS Sorcar

## Executive Summary

After deep analysis of [GSD (Get Shit Done)](https://github.com/gsd-build/get-shit-done) — a 48k-star meta-prompting and context engineering system — this document extracts its most useful ideas and proposes concrete ways to incorporate them into KISS Sorcar's agent system. The focus is on ideas that will make the Sorcar agent more reliable and consistently complete tasks well, not on adding enterprise ceremony.

GSD's core insight is that **agent quality is heavily shaped by the context it receives, not just the model's raw ability**. Everything GSD does — structured planning docs, fresh-context execution, quality gates, revision loops, and deep-work rules — serves one goal: giving the AI the right context at the right time.

---

## Part 1: GSD's Best Ideas (Ranked by Impact for Sorcar)

### 1. Quality Gates (Pre-flight, Revision, Escalation, Abort)

**What GSD does:** Every validation checkpoint in every workflow is classified into one of 4 gate types:
- **Pre-flight gate** — Check preconditions before starting (does the file exist? is the config valid?). Cheap, deterministic, prevents wasted work.
- **Revision gate** — After producing output, check quality and loop back with specific feedback if insufficient. Bounded by iteration cap (max 3). Includes stall detection (if issue count doesn't decrease, escalate).
- **Escalation gate** — When automated resolution fails, surface the issue to the human with clear options.
- **Abort gate** — Stop immediately when continuing would cause damage (context window critically low, error state).

**What Sorcar does today:** Current KISS prompts and developer instructions already include several strong checks such as reading lessons, running `uv run check --full`, running `uv run pytest -v`, and avoiding superficial fixes. However, these checks are mostly instruction-level rather than an explicit gate model with named phases, bounded retries, and stall detection.

**Proposed change:** Add a lightweight gate model to the execution flow, mostly by tightening prompt structure first.

### 2. Fresh Context Per Sub-Task (Context Rot Prevention)

**What GSD does:** When executing a multi-plan phase, each plan runs in a fresh context window. The orchestrator spawns a new subagent for each plan, giving it only the files it needs (`<read_first>` list). This prevents context rot — the quality degradation that can happen as conversation history accumulates.

**What Sorcar does today:** `RelentlessAgent` already creates continuation sub-sessions when needed, which partially addresses context accumulation. KISS also has worktree isolation and stateful chat history. What is missing is deliberate decomposition of large tasks into independently scoped fresh-context executions based on task structure rather than only on exhaustion limits.

**Proposed change:** For very complex tasks, decompose into sub-tasks and execute each in a fresh agent session.

### 3. Deep Work Rules (Concrete Plans, Not Vague Instructions)

**What GSD does:** Every task must include:
- `<read_first>` — Files the executor MUST read before modifying anything.
- `<acceptance_criteria>` — Verifiable conditions provable with grep/test/CLI output. Never subjective.
- `<action>` — Concrete values, not references. Never say "align X with Y" without specifying the exact target state.

**What Sorcar does today:** The system and developer prompts already impose strong coding, testing, and verification rules, but they do not consistently force a compact explicit plan with exact target states, required reads, and concrete acceptance checks before execution.

**Proposed change:** Add a small set of deep-work planning rules to the prompt for complex tasks.

### 4. Discuss → Research → Plan → Execute → Verify Pipeline

**What GSD does:** A strict 5-phase pipeline for every significant piece of work:
1. **Discuss** — Identify gray areas, capture user preferences
2. **Research** — Investigate domain, patterns, dependencies
3. **Plan** — Create atomic task plans with acceptance criteria
4. **Execute** — Run plans in dependency order, often with fresh context each
5. **Verify** — Check deliverables against criteria, run tests, and do final review

**What Sorcar does today:** For many tasks the agent can move quickly from request to execution, and the current prompts do require verification. What is missing is a lightweight explicit pipeline that becomes mandatory only for sufficiently complex work.

**Proposed change:** For complex tasks such as multi-file changes or architectural work, require a compact planning phase before execution.

### 5. Revision Loops with Stall Detection

**What GSD does:** After the planner produces plans, a checker reviews them. If issues are found, it loops back with specific feedback, capped at a small number of iterations. If the issue count stops decreasing, the system escalates rather than looping forever.

**What Sorcar does today:** The agent does retry and continue, but there is no first-class bounded revision loop with explicit issue tracking and stall detection.

**Proposed change:** Add a bounded self-review loop, first via prompt guidance and later in code if needed.

### 6. Requirements Traceability

**What GSD does:** Every requirement gets an ID. Plans must reference the requirements they address. A coverage gate checks that every requirement is covered before execution begins.

**What Sorcar does today:** The user request is often the only requirements artifact. For simple tasks that is fine, but larger tasks would benefit from explicit sub-requirements and a coverage check before declaring success.

**Proposed change:** For complex tasks, decompose the request into checkable sub-requirements and verify that all are satisfied.

### 7. Dependency-Ordered Execution

**What GSD does:** Plans are grouped into dependency-ordered waves. Independent plans can run in parallel inside a wave, while waves execute sequentially.

**What Sorcar does today:** Work is mostly sequential inside a single agent session.

**Proposed change:** Keep Sorcar sequential, but adopt the dependency-ordering idea for task decomposition and planning.

### 8. Self-Improvement Loop (LESSONS.md)

**What GSD does:** The system maintains state across sessions and reuses learned context.

**What Sorcar does today:** KISS already has `LESSONS.md` and explicit instructions to read it at the start and update it only when a generally useful lesson is learned.

**No major change needed** — this area is already aligned well with GSD's philosophy.

---

## Part 2: Concrete Implementation Plan

### Phase 1: Tighten Prompt-Level Gates (Low effort, High impact)

Most of this behavior already exists across the current instructions. The improvement should be to consolidate and sharpen it in `SYSTEM.md` rather than duplicating similar rules in multiple places.

Example prompt block:

```markdown
## Pre-flight Checks
- Read every file you will modify before changing it.
- If the task depends on architecture or existing behavior, read the relevant source of truth files first.
- If the task references files, commands, or assumptions that do not exist, stop and report instead of guessing.
- Establish the baseline before changing behavior when that baseline matters.
```

**Where:** Refine `SYSTEM.md` so the gate language is centralized and non-redundant.

**Why:** Consolidation reduces missed checks and makes the execution loop easier for the agent to follow consistently.

### Phase 2: Strengthen Explicit Self-Verification (Low effort, High impact)

KISS already requires verification before finishing. The improvement is to make the verification checklist more explicit and easier for the agent to execute consistently before calling `finish(success=True)`.

Suggested checklist:
1. Re-read every file that was modified.
2. Run `uv run check --full`.
3. Run `uv run pytest -v`.
4. Verify the task requirements and acceptance checks explicitly.

Example prompt block:

```markdown
## Post-Execution Verification
Before calling finish(success=True), you MUST:
1. Run the required checks and fix any failures.
2. Re-read the modified files and verify the changes are correct and complete.
3. Check each user requirement or sub-requirement explicitly.
4. If any verification fails, continue working instead of finishing.
```

**Where:** Refine `SYSTEM.md` and avoid duplicating rules that are already enforced elsewhere unless the duplication is intentional and minimal.

### Phase 3: Add Deep-Work Planning Rules (Low effort, High impact)

Add GSD's anti-shallow-execution rules in a KISS-sized form.

Example prompt block:

```markdown
## Deep Work Rules
- Before modifying a file, read it first.
- When a task says "align" or "match", determine the exact target state before editing.
- Every meaningful change should have a concrete verification method.
- Do not act on assumptions when the current state can be checked.
- For multi-part work, list the concrete planned changes before executing them.
```

**Where:** Add to `SYSTEM.md`.

### Phase 4: Structured Planning for Complex Tasks (Medium effort, High impact)

For tasks involving multiple files or architectural changes, require a short planning step before code changes.

Example prompt block:

```markdown
## Planning for Complex Tasks
For tasks involving 3+ files, cross-module behavior, or architectural changes:
1. List the files that need to change and why.
2. State the exact intended change in each file.
3. Identify dependencies and execution order.
4. State how each change will be verified.
```

**Where:** Add to `SYSTEM.md` as a conditional workflow for complex tasks only.

### Phase 5: Bounded Revision Loop (Medium effort, Medium impact)

If code changes are desired later, enhance `RelentlessAgent.perform_task()` to support bounded self-revision.

**Concept:** After a task attempt fails verification, retry with specific error context up to a small limit. Track whether the failure set is shrinking. If it stalls, stop retrying blindly and surface the issue clearly.

Possible behaviors:
- Track repeated test or lint failures across continuations.
- If the same failures persist, add an explicit rethink prompt.
- After a small retry limit, return a clear incomplete result rather than spinning.

**Where:** `src/kiss/core/relentless_agent.py`, likely by refining continuation prompt logic rather than adding a large new subsystem.

### Phase 6: Optional Task Decomposition with Fresh Context (High effort, High impact)

For very complex tasks, consider decomposing work into sub-tasks executed in fresh agent sessions.

**Concept:**
1. A planning step produces sub-tasks.
2. Each sub-task runs in a fresh session with only the relevant context.
3. Results are collected.
4. A final verification pass checks the overall result.

**Where:** Prefer extending the existing agent stack rather than introducing a large parallel hierarchy immediately. If implemented, this should likely build on `RelentlessAgent`, `StatefulSorcarAgent`, and existing worktree behavior.

**Note:** This could be highly impactful, but it is also the riskiest architectural change and should come only after prompt-level improvements prove insufficient.

### Phase 7: Reuse Existing Persistence Before Adding New State Files (Low effort, Medium impact)

Before adding new project-state files, evaluate whether existing persistence and history mechanisms already cover the use case. If additional state is needed, keep it minimal and avoid duplicating information stored elsewhere.

Possible minimal state to capture:
- recent completed tasks
- known blockers
- notable architecture decisions
- recently modified files

**Where:** Any implementation should fit into the existing persistence layer carefully and avoid introducing another always-on artifact unless there is a clear benefit.

---

## Part 3: Priority Order

| Priority | Phase | Effort | Impact | Description |
|----------|-------|--------|--------|-------------|
| **P0** | 1 | Low | High | Tighten prompt-level gates |
| **P0** | 2 | Low | High | Strengthen explicit self-verification |
| **P0** | 3 | Low | High | Add deep-work planning rules |
| **P1** | 4 | Medium | High | Structured planning for complex tasks |
| **P1** | 5 | Medium | Medium | Bounded revision loop with stall detection |
| **P2** | 6 | High | High | Optional task decomposition with fresh context |
| **P2** | 7 | Low | Medium | Reuse existing persistence before adding state |

**P0 items:** Mostly prompt and instruction cleanup. These are the least risky changes and align best with the current architecture.

**P1 items:** Small workflow or code refinements that improve performance on more complex tasks.

**P2 items:** Larger architectural changes that should be attempted only if prompt-level improvements are not enough.

---

## Part 4: What Not to Adopt from GSD

Some GSD features are not a good fit for KISS Sorcar:

1. **Enterprise workflow ceremony** — Sorcar tasks are typically direct request-to-result interactions. Avoid heavy project-management overhead.
2. **Large command surface area** — Sorcar benefits from keeping the workflow simple rather than splitting behavior across many slash commands.
3. **Many specialized agent roles** — The concepts of research, planning, execution, and verification are useful, but they should remain lightweight phases unless proven otherwise.
4. **Heavy document state** — Maintaining many always-on phase documents would add complexity without clear benefit for most Sorcar tasks.
5. **Domain-specific gates** — UI-specific, schema-specific, or security-specific gate systems should not be generalized into the core unless the project truly needs them.
6. **Parallel execution by default** — Parallelism adds complexity and is not obviously beneficial for Sorcar's common task pattern.

---

## Part 5: GSD Ideas Already Present in KISS

These GSD ideas already have strong analogs in KISS:

| GSD Concept | KISS Equivalent |
|-------------|----------------|
| Self-improvement | `LESSONS.md` read/write loop |
| Continuation on exhaustion | `RelentlessAgent` continuation sessions |
| Fresh context on continuation | New sub-session with progress summary |
| Project-specific rules | `SORCAR.md` loaded into the agent context |
| Git worktree isolation | `WorktreeSorcarAgent` |
| Chat session history | `StatefulSorcarAgent` with persisted chat/task history |

---

## Summary

The best GSD ideas for KISS are mostly lightweight process improvements rather than heavy architecture changes:

1. **Prompt-level gates** — centralize prerequisite checks such as read-before-write.
2. **Deep-work rules** — require exact target states and concrete verification.
3. **Explicit self-verification** — make the existing verification loop easier to execute consistently.
4. **Structured planning for complex tasks** — require planning only when complexity justifies it.

These changes align with KISS's current architecture and should improve reliability without importing GSD's heavier workflow ceremony.