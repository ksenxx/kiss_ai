# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Local tree-sitter code knowledge graph (graphify features #1 + #3 + #5).

Builds a deterministic, zero-LLM-cost knowledge graph of a worktree
(classes, functions, methods, imports, call edges across the major
languages) so agents can *query structure instead of grepping files*:

* **#1 — graph tool**: :func:`build_graph` (once per worktree, then
  incrementally), :meth:`CodeGraph.query` (term-matched BFS subgraph),
  :meth:`CodeGraph.path` (shortest path between concepts) and
  :meth:`CodeGraph.explain` (a node's connections + source location +
  degree), exposed to the agent via :func:`make_code_graph_tool`.
* **#3 — query-before-grep interception**: :func:`grep_hint` inspects a
  ``grep``/``rg`` Bash command and, when the built graph knows the
  searched identifier, returns an inline ``[code_graph]`` answer instead
  of spawning grep ("the deny message IS the answer"). Prefix a command
  with ``command`` when a literal grep must intentionally bypass it.
* **#5 — staleness-proof freshness**: a per-file SHA256 cache makes
  rebuilds incremental (only changed files re-extracted, deleted files
  pruned), and :func:`install_post_commit_hook` installs an idempotent
  git post-commit hook that updates the graph for the committed files
  as a fully detached background job (graphify lesson: rebuild only
  from git hooks, never per tool call).

Every edge carries a confidence tag: ``EXTRACTED`` for relationships
explicit in the source (imports, definitions, same-file calls) and
``INFERRED`` for cross-file call edges deduced in the second
call-graph pass.  Edges are never invented.

The module is deliberately standalone (stdlib + optional
``tree_sitter_language_pack``): the rest of Sorcar touches it through
exactly two seams — ``make_code_graph_tool`` in the agent's tool list
and :func:`intercept_grep_hint` inside the agent's ``Bash`` wrappers —
both behind ``try/except`` so a missing tree-sitter install can never
break the agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STORAGE_SUBDIR = Path(".kiss") / "code_graph"

_CACHE_VERSION = 1

_QUERY_DEPTH = 3

_UPDATE_LOCK = ".update.lock"

_HOOK_LOCK = ".hook.lock"

_STALE_LOCK_SECONDS = 60 * 60

_BUILD_LOCK_TIMEOUT_S = 30.0

_HOOK_LOCK_TIMEOUT_S = 10.0


_HOOK_BEGIN = "# >>> kiss code_graph hook >>>"
_HOOK_END = "# <<< kiss code_graph hook <<<"

_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".kiss",
    ".kiss-worktrees",
}

_EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
}

_DEF_TYPES: dict[str, dict[str, tuple[str, str]]] = {
    "python": {
        "class_definition": ("class", "field:name"),
        "function_definition": ("function", "field:name"),
    },
    "javascript": {
        "class_declaration": ("class", "field:name"),
        "function_declaration": ("function", "field:name"),
        "method_definition": ("method", "field:name"),
    },
    "typescript": {
        "class_declaration": ("class", "field:name"),
        "function_declaration": ("function", "field:name"),
        "method_definition": ("method", "field:name"),
        "interface_declaration": ("interface", "field:name"),
    },
    "tsx": {
        "class_declaration": ("class", "field:name"),
        "function_declaration": ("function", "field:name"),
        "method_definition": ("method", "field:name"),
        "interface_declaration": ("interface", "field:name"),
    },
    "go": {
        "function_declaration": ("function", "field:name"),
        "method_declaration": ("method", "field:name"),
        "type_declaration": ("type", "type_spec"),
    },
    "rust": {
        "function_item": ("function", "field:name"),
        "struct_item": ("struct", "field:name"),
        "enum_item": ("enum", "field:name"),
        "trait_item": ("trait", "field:name"),
    },
    "java": {
        "class_declaration": ("class", "field:name"),
        "interface_declaration": ("interface", "field:name"),
        "method_declaration": ("method", "field:name"),
    },
    "ruby": {
        "class": ("class", "field:name"),
        "module": ("module", "field:name"),
        "method": ("method", "field:name"),
    },
    "c": {
        "function_definition": ("function", "c_declarator"),
        "struct_specifier": ("struct", "field:name"),
    },
    "cpp": {
        "function_definition": ("function", "c_declarator"),
        "class_specifier": ("class", "field:name"),
        "struct_specifier": ("struct", "field:name"),
    },
}

_CALL_TYPES: dict[str, dict[str, str]] = {
    "python": {"call": "function"},
    "javascript": {"call_expression": "function"},
    "typescript": {"call_expression": "function"},
    "tsx": {"call_expression": "function"},
    "go": {"call_expression": "function"},
    "rust": {"call_expression": "function"},
    "java": {"method_invocation": "name"},
    "ruby": {"call": "method"},
    "c": {"call_expression": "function"},
    "cpp": {"call_expression": "function"},
}

_IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "go": {"import_spec"},
    "rust": {"use_declaration"},
    "java": {"import_declaration"},
    "c": {"preproc_include"},
    "cpp": {"preproc_include"},
}

_IMPORT_NAME_RE = re.compile(
    r"^(?:import|from|use|include|require|#include)(?:\s+|$)"
)


def graph_dir(work_dir: str) -> Path:
    """Return the storage directory for *work_dir*'s code graph.

    Args:
        work_dir: Root of the worktree the graph indexes.

    Returns:
        ``<work_dir>/.kiss/code_graph`` (not created).
    """
    return Path(work_dir).resolve() / _STORAGE_SUBDIR


def _parser_for(lang: str) -> Any | None:
    """Return a tree-sitter parser for *lang*, or ``None`` when
    tree-sitter-language-pack is missing or the grammar cannot be
    loaded (e.g. offline on-demand download failure)."""
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser(lang)  # type: ignore[arg-type]
    except Exception:
        logger.debug("no tree-sitter parser for %s", lang, exc_info=True)
        return None


def _node_text(node: Any) -> str:
    """Decode a tree-sitter node's source slice lossily and safely."""
    return str(node.text.decode("utf-8", errors="replace"))


def _c_declarator_name(node: Any) -> str | None:
    """Walk a C/C++ ``function_definition`` declarator chain to the
    function identifier."""
    decl = node.child_by_field_name("declarator")
    while decl is not None:
        if decl.type in ("identifier", "field_identifier", "qualified_identifier"):
            return _node_text(decl)
        nxt = decl.child_by_field_name("declarator")
        if nxt is None:
            for child in decl.named_children:
                if child.type in ("identifier", "field_identifier"):
                    return _node_text(child)
            return None
        decl = nxt
    return None


def _def_name(node: Any, strategy: str) -> str | None:
    """Extract a definition's name per its language strategy."""
    if strategy == "field:name":
        name = node.child_by_field_name("name")
        return _node_text(name) if name is not None else None
    if strategy == "c_declarator":
        return _c_declarator_name(node)
    for child in node.named_children:
        if child.type == "type_spec":
            name = child.child_by_field_name("name")
            if name is not None:
                return _node_text(name)
    return None


def _import_label(node: Any) -> str | None:
    """Best-effort module name from an import-ish node's text."""
    text = " ".join(_node_text(node).split())
    text = _IMPORT_NAME_RE.sub("", text)
    if not text:
        return None
    if "from" in text.split():
        after = text.split("from", 1)[1].strip()
        if after:
            text = after
        else:
            text = text.split()[0]
    else:
        text = text.split()[0]
    text = text.strip("\"'<>;(){} ")
    return text or None


def _extract_file(rel_path: str, source: bytes, lang: str) -> dict[str, Any] | None:
    """Extract nodes / raw edges / call sites from one file.

    Returns a JSON-able per-file record::

        {"defs": [{key, name, kind, line, parent}],
         "imports": [module, ...],
         "calls": [{caller, callee}]}

    ``key`` is a per-file definition identity, while ``parent`` and
    ``caller`` refer to that key (``None`` means module level).  Keys,
    rather than labels, preserve scope when two classes both define a
    method such as ``run``.  ``None`` is returned when no parser is
    available for *lang*.
    """
    parser = _parser_for(lang)
    if parser is None:
        return None
    tree = parser.parse(source)
    def_types = _DEF_TYPES.get(lang, {})
    call_types = _CALL_TYPES.get(lang, {})
    import_types = _IMPORT_TYPES.get(lang, set())
    defs: list[dict[str, Any]] = []
    imports: list[str] = []
    calls: list[dict[str, Any]] = []

    def walk(node: Any, enclosing: int | None) -> None:
        for child in node.children:
            new_enclosing = enclosing
            if child.type in def_types:
                kind, strategy = def_types[child.type]
                name = _def_name(child, strategy)
                if name:
                    key = len(defs)
                    defs.append(
                        {
                            "key": key,
                            "name": name,
                            "kind": kind,
                            "line": child.start_point[0] + 1,
                            "parent": enclosing,
                        }
                    )
                    new_enclosing = key
            elif child.type in import_types:
                label = _import_label(child)
                if label:
                    imports.append(label)
            elif child.type in call_types:
                callee_field = call_types[child.type]
                callee = child.child_by_field_name(callee_field)
                if callee is not None:
                    name = re.split(r"[.:>]", _node_text(callee))[-1]
                    name = name.split("(")[0].strip()
                    if name and re.fullmatch(r"\w+", name):
                        calls.append({"caller": enclosing, "callee": name})
            walk(child, new_enclosing)

    walk(tree.root_node, None)
    return {"defs": defs, "imports": imports, "calls": calls}


class CodeGraph:
    """An in-memory code knowledge graph with query / path / explain.

    Attributes:
        nodes: Mapping of node id → ``{id, label, kind, file, line}``.
        edges: List of ``{source, target, relation, confidence}``.
        stats: Build statistics (``files`` scanned, ``reextracted``).
    """

    def __init__(
        self,
        nodes: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
        stats: dict[str, int] | None = None,
    ) -> None:
        """Initialise the graph.

        Args:
            nodes: Node-id → node-record mapping.
            edges: Edge records.
            stats: Optional build statistics.
        """
        self.nodes = nodes
        self.edges = edges
        self.stats = stats or {}
        self._adj: dict[str, list[tuple[str, dict[str, Any]]]] | None = None

    def _adjacency(self) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        """Return (and cache) the undirected adjacency list."""
        if self._adj is None:
            adj: dict[str, list[tuple[str, dict[str, Any]]]] = {
                nid: [] for nid in self.nodes
            }
            for edge in self.edges:
                s, t = edge["source"], edge["target"]
                if s in adj and t in adj:
                    adj[s].append((t, edge))
                    adj[t].append((s, edge))
            self._adj = adj
        return self._adj

    def find_node(self, label: str) -> dict[str, Any] | None:
        """Return the best node whose label matches *label*.

        Exact match wins; otherwise a unique case-insensitive match is
        accepted.

        Args:
            label: The node label to look up (e.g. ``"Application"``).

        Returns:
            The node record, or ``None`` when nothing matches.
        """
        exact = [n for n in self.nodes.values() if n["label"] == label]
        if exact:
            return exact[0]
        folded = [
            n for n in self.nodes.values() if n["label"].lower() == label.lower()
        ]
        return folded[0] if folded else None

    def find_edge(
        self, source: str, target: str, relation: str
    ) -> dict[str, Any] | None:
        """Return the first edge (source, target, relation), else ``None``.

        Args:
            source: Source node id.
            target: Target node id.
            relation: Edge relation (``calls``/``imports``/...).
        """
        for edge in self.edges:
            if (
                edge["source"] == source
                and edge["target"] == target
                and edge["relation"] == relation
            ):
                return edge
        return None

    def _node_line(self, node: dict[str, Any]) -> str:
        loc = f"{node['file']}:{node['line']}" if node.get("file") else "?"
        return f"NODE {node['label']} ({node['kind']}) {loc}"

    def _edge_line(self, edge: dict[str, Any]) -> str:
        src = self.nodes[edge["source"]]["label"]
        dst = self.nodes[edge["target"]]["label"]
        return f"EDGE {src} --{edge['relation']} [{edge['confidence']}]--> {dst}"

    def query(self, question: str, max_chars: int = 8000) -> str:
        """Answer *question* with the relevant subgraph (graphify format).

        Question terms are matched against node labels; the top
        matching nodes seed a BFS (depth 3) whose nodes and edges are
        rendered as ``NODE``/``EDGE`` lines under a character budget.

        Args:
            question: Free-text question or identifier list.
            max_chars: Output budget (~4 chars/token).

        Returns:
            ``NODE``/``EDGE`` lines, or a no-match message.
        """
        terms = [t.lower() for t in re.findall(r"\w+", question) if len(t) > 1]
        exact_question = question.strip().casefold()
        scored: list[tuple[int, str]] = []
        for nid, node in self.nodes.items():
            label = node["label"].casefold()
            score = sum(1 for t in terms if t in label)
            if terms and label == exact_question:
                score += len(terms) + 1
            if score:
                scored.append((score, nid))
        if not scored:
            return "No matching nodes in the code graph."
        scored.sort(key=lambda pair: (-pair[0], self.nodes[pair[1]]["label"]))
        seeds = [nid for _, nid in scored[:3]]
        seed_rank = {nid: rank for rank, nid in enumerate(seeds)}
        adj = self._adjacency()
        seen: set[str] = set(seeds)
        distance = {nid: 0 for nid in seeds}
        frontier = deque((nid, 0) for nid in seeds)
        sub_edges: list[dict[str, Any]] = []
        seen_edges: set[int] = set()
        while frontier:
            nid, depth = frontier.popleft()
            if depth >= _QUERY_DEPTH:
                continue
            for neighbor, edge in adj.get(nid, []):
                if id(edge) not in seen_edges:
                    seen_edges.add(id(edge))
                    sub_edges.append(edge)
                if neighbor not in seen:
                    seen.add(neighbor)
                    distance[neighbor] = depth + 1
                    frontier.append((neighbor, depth + 1))
        ordered_nodes = sorted(
            seen,
            key=lambda nid: (
                distance[nid],
                seed_rank.get(nid, len(seeds)),
                self.nodes[nid]["label"].lower(),
                self.nodes[nid]["file"],
                self.nodes[nid]["line"],
                nid,
            ),
        )
        ordered_edges = sorted(
            sub_edges,
            key=lambda edge: (
                max(distance[edge["source"]], distance[edge["target"]]),
                min(distance[edge["source"]], distance[edge["target"]]),
                self.nodes[edge["source"]]["label"].lower(),
                edge["relation"],
                self.nodes[edge["target"]]["label"].lower(),
            ),
        )
        nodes_by_depth: dict[int, list[str]] = {}
        for nid in ordered_nodes:
            nodes_by_depth.setdefault(distance[nid], []).append(nid)
        edges_by_depth: dict[int, list[dict[str, Any]]] = {}
        for edge in ordered_edges:
            edge_depth = max(
                distance[edge["source"]], distance[edge["target"]]
            )
            edges_by_depth.setdefault(edge_depth, []).append(edge)

        lines = [
            self._node_line(self.nodes[nid]) for nid in nodes_by_depth.get(0, [])
        ]
        lines.extend(self._edge_line(edge) for edge in edges_by_depth.get(0, []))
        max_distance = max(distance.values(), default=0)
        for depth in range(1, max_distance + 1):
            parent_edges: dict[str, list[dict[str, Any]]] = {}
            same_ring_edges: list[dict[str, Any]] = []
            for edge in edges_by_depth.get(depth, []):
                source, target = edge["source"], edge["target"]
                if distance[source] > distance[target]:
                    parent_edges.setdefault(source, []).append(edge)
                elif distance[target] > distance[source]:
                    parent_edges.setdefault(target, []).append(edge)
                else:
                    same_ring_edges.append(edge)
            for nid in nodes_by_depth.get(depth, []):
                lines.append(self._node_line(self.nodes[nid]))
                lines.extend(
                    self._edge_line(edge) for edge in parent_edges.get(nid, [])
                )
            lines.extend(self._edge_line(edge) for edge in same_ring_edges)
        out: list[str] = []
        used = 0
        for line in lines:
            if used + len(line) + 1 > max_chars:
                out.append("... (truncated)")
                break
            out.append(line)
            used += len(line) + 1
        return "\n".join(out)

    def path(self, a: str, b: str) -> str:
        """Return the shortest path between concepts *a* and *b*.

        Args:
            a: Label of the start node.
            b: Label of the end node.

        Returns:
            One line per hop showing relation and confidence, or a
            diagnostic when an endpoint is unknown / no path exists.
        """
        start = self.find_node(a)
        end = self.find_node(b)
        if start is None:
            return f"No node matching {a!r} in the code graph."
        if end is None:
            return f"No node matching {b!r} in the code graph."
        if start["id"] == end["id"]:
            return f"{start['label']} (same node)"
        adj = self._adjacency()
        prev: dict[str, tuple[str, dict[str, Any]]] = {}
        seen = {start["id"]}
        frontier = deque([start["id"]])
        while frontier:
            nid = frontier.popleft()
            if nid == end["id"]:
                break
            for neighbor, edge in adj.get(nid, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    prev[neighbor] = (nid, edge)
                    frontier.append(neighbor)
        if end["id"] not in seen:
            return f"No path between {a!r} and {b!r}."
        hops: list[str] = []
        nid = end["id"]
        while nid != start["id"]:
            parent, edge = prev[nid]
            parent_label = self.nodes[parent]["label"]
            node_label = self.nodes[nid]["label"]
            if edge["source"] == parent:
                hops.append(
                    f"{parent_label} --{edge['relation']} "
                    f"[{edge['confidence']}]--> {node_label}"
                )
            else:
                hops.append(
                    f"{parent_label} <--{edge['relation']} "
                    f"[{edge['confidence']}]-- {node_label}"
                )
            nid = parent
        return "\n".join(reversed(hops))

    def explain(self, name: str) -> str:
        """Explain node *name*: kind, location, degree and neighbors.

        Args:
            name: Node label (case-insensitive fallback).

        Returns:
            A multi-line description, or a no-match diagnostic.
        """
        node = self.find_node(name)
        if node is None:
            return f"No node matching {name!r} in the code graph."
        neighbors = self._adjacency().get(node["id"], [])
        loc = f"{node['file']}:{node['line']}" if node.get("file") else "?"
        lines = [
            f"{node['label']} ({node['kind']}) at {loc}, degree {len(neighbors)}",
        ]
        for neighbor_id, edge in neighbors:
            other = self.nodes[neighbor_id]
            direction = "->" if edge["source"] == node["id"] else "<-"
            lines.append(
                f"  {direction} {edge['relation']} [{edge['confidence']}] "
                f"{other['label']} ({other['kind']})"
            )
        return "\n".join(lines)


def _iter_source_files(root: Path) -> list[Path]:
    """List indexable source files under *root* (skip vendor/hidden dirs)."""
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _EXT_TO_LANG:
            continue
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        files.append(path)
    return files


def _record_to_graph_parts(
    rel: str, record: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert one file's extraction record into nodes + local edges.

    Returns:
        ``(nodes, edges, pending_calls)`` where *pending_calls* are the
        raw call sites for the global (INFERRED) call-graph pass.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    file_id = f"file:{rel}"
    nodes[file_id] = {
        "id": file_id,
        "label": Path(rel).name,
        "kind": "file",
        "file": rel,
        "line": 1,
    }
    key_to_id: dict[int, str] = {}
    defs_by_key: dict[int, dict[str, Any]] = {}
    name_to_keys: dict[str, list[int]] = {}
    for d in record["defs"]:
        key = int(d["key"])
        nid = f"def:{rel}:{d['name']}:{d['line']}"
        if nid in nodes:
            # Two same-named definitions on one physical line (legal in
            # e.g. JavaScript) must not silently share one node.
            nid = f"{nid}:{key}"
        nodes[nid] = {
            "id": nid,
            "label": d["name"],
            "kind": d["kind"],
            "file": rel,
            "line": d["line"],
        }
        key_to_id[key] = nid
        defs_by_key[key] = d
        name_to_keys.setdefault(d["name"], []).append(key)
    for d in record["defs"]:
        parent_key = d["parent"]
        parent_id = key_to_id.get(int(parent_key)) if parent_key is not None else None
        nid = key_to_id[int(d["key"])]
        edges.append(
            {
                "source": parent_id or file_id,
                "target": nid,
                "relation": "contains" if parent_id else "defines",
                "confidence": "EXTRACTED",
            }
        )
    for module in record["imports"]:
        mid = f"mod:{module}"
        nodes.setdefault(
            mid,
            {"id": mid, "label": module, "kind": "module", "file": "", "line": 0},
        )
        edges.append(
            {
                "source": file_id,
                "target": mid,
                "relation": "imports",
                "confidence": "EXTRACTED",
            }
        )
    pending: list[dict[str, Any]] = []
    for call in record["calls"]:
        caller_key = call["caller"]
        caller_id = (
            key_to_id.get(int(caller_key)) if caller_key is not None else file_id
        )
        candidates = name_to_keys.get(call["callee"], [])
        local_callee_id: str | None = None
        scopes: list[int | None] = []
        requested_caller = int(caller_key) if caller_key is not None else None
        cursor = requested_caller if requested_caller in defs_by_key else None
        while cursor is not None:
            scopes.append(cursor)
            parent = defs_by_key[cursor]["parent"]
            cursor = int(parent) if parent is not None else None
        scopes.append(None)
        for scope in scopes:
            scoped = [
                key for key in candidates if defs_by_key[key]["parent"] == scope
            ]
            if len(scoped) == 1:
                local_callee_id = key_to_id[scoped[0]]
                break
            if len(scoped) > 1:
                break
        pending.append(
            {
                "file": rel,
                "callee": call["callee"],
                "caller_id": caller_id,
                "local_callee_id": local_callee_id,
            }
        )
    return nodes, edges, pending


def _assemble(records: dict[str, dict[str, Any]], stats: dict[str, int]) -> CodeGraph:
    """Assemble the full graph from per-file extraction *records*.

    Same-file call edges are ``EXTRACTED``; cross-file callee
    resolution happens here in the second pass and is ``INFERRED``.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for rel, record in records.items():
        file_nodes, file_edges, file_pending = _record_to_graph_parts(rel, record)
        nodes.update(file_nodes)
        edges.extend(file_edges)
        pending.extend(file_pending)
    label_index: dict[str, list[str]] = {}
    for nid, node in nodes.items():
        if node["kind"] not in ("file", "module"):
            label_index.setdefault(node["label"], []).append(nid)
    seen_calls: set[tuple[str, str]] = set()
    for call in pending:
        caller_id = call["caller_id"]
        if caller_id is None or caller_id not in nodes:
            continue
        if call["local_callee_id"] is not None:
            callee_id = call["local_callee_id"]
            confidence = "EXTRACTED"
        else:
            candidates = label_index.get(call["callee"], [])
            if len(candidates) != 1:
                continue
            callee_id = candidates[0]
            confidence = "INFERRED"
        key = (caller_id, callee_id)
        if key in seen_calls:
            continue
        seen_calls.add(key)
        edges.append(
            {
                "source": caller_id,
                "target": callee_id,
                "relation": "calls",
                "confidence": confidence,
            }
        )
    return CodeGraph(nodes, edges, stats)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON beside *path* and atomically replace its old version."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            json.dump(data, stream, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _ensure_graph_git_excluded(work_dir: str) -> None:
    """Keep generated graph artifacts out of git status, best-effort."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "info/exclude"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        exclude = Path(result.stdout.strip())
        if not exclude.is_absolute():
            exclude = Path(work_dir).resolve() / exclude
        existing = ""
        if exclude.is_file():
            existing = exclude.read_bytes().decode("utf-8", errors="surrogateescape")
        pattern = ".kiss/code_graph/"
        if pattern not in existing.splitlines():
            if existing and not existing.endswith("\n"):
                existing += "\n"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_bytes(
                (existing + pattern + "\n").encode("utf-8", errors="surrogateescape")
            )
    except (OSError, ValueError, subprocess.CalledProcessError):
        logger.debug("could not add code_graph to git info/exclude", exc_info=True)


def _save(
    work_dir: str,
    records: dict[str, dict[str, Any]],
    hashes: dict[str, str],
    graph: CodeGraph,
) -> None:
    """Atomically persist graph.json and its versioned SHA cache."""
    storage = graph_dir(work_dir)
    storage.mkdir(parents=True, exist_ok=True)
    _ensure_graph_git_excluded(work_dir)
    _atomic_write_json(
        storage / "cache.json",
        {"version": _CACHE_VERSION, "hashes": hashes, "records": records},
    )
    _atomic_write_json(
        storage / "graph.json",
        {
            "version": _CACHE_VERSION,
            "nodes": list(graph.nodes.values()),
            "edges": graph.edges,
        },
    )


def _load_cache(work_dir: str) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Load a compatible SHA256 cache; empty on absence/corruption."""
    try:
        data = json.loads((graph_dir(work_dir) / "cache.json").read_text())
        if data.get("version") != _CACHE_VERSION:
            return {}, {}
        return dict(data["hashes"]), dict(data["records"])
    except Exception:
        return {}, {}


def load_graph(work_dir: str) -> CodeGraph | None:
    """Load the persisted graph for *work_dir*.

    Args:
        work_dir: Worktree root whose graph to load.

    Returns:
        The graph, or ``None`` when absent or unreadable.
    """
    try:
        data = json.loads((graph_dir(work_dir) / "graph.json").read_text())
        if data.get("version") != _CACHE_VERSION:
            return None
        nodes = {n["id"]: n for n in data["nodes"]}
        return CodeGraph(nodes, list(data["edges"]))
    except Exception:
        return None


def build_graph(
    work_dir: str,
    incremental: bool = True,
    only_files: list[str] | None = None,
) -> CodeGraph:
    """Build (or incrementally update) the code graph for *work_dir*.

    A per-file SHA256 cache keeps rebuilds incremental: unchanged files
    reuse their cached extraction, changed/new files are re-parsed, and
    files gone from disk are pruned (graphify feature #5).

    Args:
        work_dir: Worktree root to index.
        incremental: When ``True`` (default) reuse the SHA256 cache;
            ``False`` re-extracts every file.
        only_files: Optional worktree-relative paths — restrict change
            detection to these files (used by the git hook, which knows
            exactly what a commit touched); all other cached files are
            kept as-is.

    Returns:
        The assembled :class:`CodeGraph` (``stats`` reports ``files``
        scanned and ``reextracted``).
    """
    lock = _acquire_update_lock_blocking(work_dir, timeout=_BUILD_LOCK_TIMEOUT_S)
    if lock is None:
        # Building without the lock could silently overwrite a
        # concurrent builder's newer cache/graph (the S2-10 lost
        # update), so never enter the critical section unserialized.
        existing = load_graph(work_dir)
        if existing is not None:
            logger.warning(
                "code_graph update lock in %s busy for %.0fs; "
                "returning the existing graph unmodified",
                work_dir, _BUILD_LOCK_TIMEOUT_S,
            )
            return existing
        raise RuntimeError(
            f"code_graph update lock in {work_dir} is held by another "
            "builder; try again later"
        )
    try:
        return _build_graph_locked(work_dir, incremental, only_files)
    finally:
        _release_update_lock(lock)


def _build_graph_locked(
    work_dir: str,
    incremental: bool = True,
    only_files: list[str] | None = None,
) -> CodeGraph:
    """The :func:`build_graph` body; caller holds the update lock.

    The cache read-modify-write below is not safe against a concurrent
    builder (two writers can each read the same old records and each
    atomically overwrite the other's newer cache/graph pair), so every
    entry point must serialize through the ``.update.lock`` file.
    """
    root = Path(work_dir).resolve()
    old_hashes, old_records = _load_cache(work_dir) if incremental else ({}, {})
    hashes: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    reextracted = 0
    files = _iter_source_files(root)
    on_disk = {str(p.relative_to(root)) for p in files}
    for path in files:
        rel = str(path.relative_to(root))
        if only_files is not None and rel not in only_files:
            if rel in old_records:
                hashes[rel] = old_hashes.get(rel, "")
                records[rel] = old_records[rel]
            continue
        try:
            source = path.read_bytes()
        except OSError:
            logger.debug("unreadable file skipped: %s", path, exc_info=True)
            continue
        digest = hashlib.sha256(source).hexdigest()
        if rel in old_records and old_hashes.get(rel) == digest:
            hashes[rel] = digest
            records[rel] = old_records[rel]
            continue
        record = _extract_file(rel, source, _EXT_TO_LANG[path.suffix])
        if record is None:
            continue
        reextracted += 1
        hashes[rel] = digest
        records[rel] = record
    records = {rel: rec for rel, rec in records.items() if rel in on_disk}
    hashes = {rel: h for rel, h in hashes.items() if rel in on_disk}
    graph = _assemble(records, {"files": len(records), "reextracted": reextracted})
    _save(work_dir, records, hashes, graph)
    return graph


_GREP_COMMANDS = {"grep", "rg", "egrep", "fgrep"}

_FLAGS_WITH_VALUE = {
    "-e", "-f", "-m", "-A", "-B", "-C", "-d", "-D",
    "--include", "--exclude", "--exclude-dir", "--regexp",
    "-g", "-t", "-T", "--type", "--glob", "--max-count",
}
_PATTERN_FLAGS = {"-e", "--regexp"}
_PATTERN_FILE_FLAGS = {"-f", "--file"}


def _grep_pattern(command: str) -> str | None:
    """Extract the search pattern from a ``grep``/``rg`` command line.

    Returns:
        The first non-flag argument after the grep executable, or
        ``None`` when *command* is not a grep-family invocation or has
        no pattern (including ``-f``/``--file`` invocations, whose
        patterns live in a file and whose operands are search targets).
    """
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv or Path(argv[0]).name not in _GREP_COMMANDS:
        return None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("-"):
            if arg == "--":
                return argv[i + 1] if i + 1 < len(argv) else None
            flag, separator, value = arg.partition("=")
            if flag in _PATTERN_FILE_FLAGS or (
                arg.startswith("-f") and not arg.startswith("--")
            ):
                return None
            if flag in _PATTERN_FLAGS:
                if separator:
                    return value or None
                return argv[i + 1] if i + 1 < len(argv) else None
            if arg.startswith("-e") and len(arg) > 2:
                return arg[2:]
            if flag in _FLAGS_WITH_VALUE and not separator:
                i += 1
        else:
            return arg
        i += 1
    return None


def grep_hint(command: str, work_dir: str | None) -> str | None:
    """Return an inline ``[code_graph]`` answer for a grep-like command.

    Query-before-grep interception (graphify feature #3): when *command*
    greps for an identifier the built graph knows, the graph's answer is
    returned *instead of spawning grep*.  The Bash wrapper returns this
    answer directly ("the deny message IS the answer"), collapsing both
    the search→read model round-trip and the underlying filesystem scan.

    Args:
        command: The Bash command about to run.
        work_dir: The agent's worktree (graph location).

    Returns:
        The hint block, or ``None`` when the command is not a grep, no
        graph is built, or the pattern matches nothing in the graph.
    """
    if not work_dir:
        return None
    pattern = _grep_pattern(command)
    if pattern is None:
        return None
    graph = load_graph(work_dir)
    if graph is None:
        return None
    if graph.find_node(pattern) is None:
        return None
    answer = graph.query(pattern, max_chars=1500)
    if answer.startswith("No matching nodes"):
        return None
    return (
        "[code_graph] The code graph already knows about this pattern "
        "(consider the code_graph tool instead of grep):\n"
        f"{answer}\n[/code_graph]\n"
    )


def intercept_grep_hint(
    command: str, work_dir: str | None, seen: set[str],
) -> str | None:
    """Return a code-graph answer for a grep-like *command*, at most once.

    Query-before-grep interception: when a built graph already knows the
    identifier a grep/rg command searches for, the expensive grep is
    denied and the graph answer is returned directly ("the deny message
    IS the answer").  A broken graph can never affect Bash (the lookup
    is wrapped in ``try/except``).  Each distinct hint is returned at
    most once per *seen* set, so a repeated identical grep falls through
    to the real command (the agent is explicitly seeking verification or
    context the graph did not provide).

    Shared by the plain and Docker ``Bash`` wrappers in
    ``SorcarAgent._get_tools`` so the interception logic cannot drift
    between the two.

    Args:
        command: The bash command about to be executed.
        work_dir: Working directory used to locate the built graph.
        seen: Mutable per-session dedupe set of already-returned hints.

    Returns:
        The hint string to return INSTEAD of running the command, or
        ``None`` when the command should run normally.
    """
    hint = ""
    try:
        hint = grep_hint(command, work_dir) or ""
    except Exception:
        logger.debug("code_graph grep hint failed", exc_info=True)
    if hint and hint not in seen:
        seen.add(hint)
        return hint
    return None



def _hooks_dir(work_dir: str) -> Path | None:
    """Resolve the git hooks directory for *work_dir* (or ``None``)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    path = Path(out)
    return path if path.is_absolute() else Path(work_dir).resolve() / path


def _hook_section() -> str:
    """Return the shell snippet appended to ``post-commit``.

    The update runs fully detached in the background (nohup, stdio to
    /dev/null) so committing never blocks — the graphify lesson that
    slow rebuilds belong in git hooks, as background jobs, never in
    per-tool-call hooks.
    """
    python = sys.executable
    return (
        f"{_HOOK_BEGIN}\n"
        "# Incrementally update the KISS code graph as a detached job.\n"
        "# The SHA cache hashes the worktree but reparses changed files only;\n"
        "# avoiding shell-expanded path lists preserves spaces/newlines.\n"
        f'nohup {shlex.quote(python)} -m kiss.agents.sorcar.code_graph '
        'update "$(git rev-parse --show-toplevel)" '
        ">/dev/null 2>&1 </dev/null &\n"
        f"{_HOOK_END}\n"
    )


def _atomic_write_hook(hook: Path, text: str) -> None:
    """Atomically replace *hook* with *text* and mark it executable.

    A direct ``write_text`` can be interrupted mid-write and truncate a
    user's pre-existing hook; write-to-temp + ``os.replace`` cannot.
    The temp name embeds the thread id as well as the PID — a PID-only
    name is shared by two threads of one process, and the loser's
    ``os.replace`` raises ``FileNotFoundError`` after the winner
    consumed the shared temp.
    """
    tmp = hook.with_name(
        f".{hook.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    tmp.write_text(text)
    tmp.chmod(0o755)
    os.replace(tmp, hook)


def _is_shell_hook(text: str) -> bool:
    """Return whether *text* has a shell shebang we can safely append to."""
    if not text.startswith("#!"):
        return False
    try:
        command = shlex.split(text.splitlines()[0][2:].strip())
    except ValueError:
        return False
    if not command:
        return False
    executable = Path(command[0]).name
    if executable == "env" and len(command) > 1:
        executable = Path(command[-1]).name
    return executable in {"sh", "bash", "dash", "zsh", "ksh"}


def install_post_commit_hook(work_dir: str) -> str:
    """Install the post-commit graph-update hook for *work_dir*.

    Idempotent (marker comments); appends to an existing hook without
    disturbing it, and creates a fresh executable hook otherwise.

    Args:
        work_dir: Worktree root (must be inside a git repository).

    Returns:
        A human-readable status message.
    """
    hooks = _hooks_dir(work_dir)
    if hooks is None:
        return f"Error: {work_dir} is not a git repository."
    hooks.mkdir(parents=True, exist_ok=True)
    lock = _acquire_update_lock_blocking(
        work_dir, timeout=_HOOK_LOCK_TIMEOUT_S, lock_name=_HOOK_LOCK,
    )
    if lock is None:
        return "Error: another code_graph hook update is in progress."
    try:
        return _install_post_commit_hook_locked(hooks)
    finally:
        _release_update_lock(lock)


def _install_post_commit_hook_locked(hooks: Path) -> str:
    """The hook-install read-modify-write; caller holds the hook lock.

    Two unserialized installers (or an install racing an uninstall)
    each read the same hook text and overwrite each other's edit, so
    every entry point must hold the ``.hook.lock``.
    """
    hook = hooks / "post-commit"
    if hook.is_file():
        try:
            text = hook.read_text()
        except (OSError, UnicodeError):
            return f"Error: cannot read existing post-commit hook at {hook}."
        begin_present = _HOOK_BEGIN in text
        end_present = _HOOK_END in text
        if begin_present != end_present:
            return "Error: corrupt code_graph markers in post-commit hook."
        if begin_present:
            return "code_graph post-commit hook already installed."
        if not text.strip():
            text = "#!/bin/sh\n"
        elif not _is_shell_hook(text):
            return "Error: existing post-commit hook is not a shell script."
        elif not text.endswith("\n"):
            text += "\n"
        # Insert right after the shebang line rather than appending: an
        # existing hook may unconditionally ``exit`` early, which would
        # make an appended section unreachable.  The graph update is a
        # detached nohup background job, so running it first is
        # harmless.
        shebang_end = text.index("\n") + 1
        _atomic_write_hook(
            hook, text[:shebang_end] + _hook_section() + text[shebang_end:]
        )
    else:
        _atomic_write_hook(hook, "#!/bin/sh\n" + _hook_section())
    hook.chmod(0o755)
    return f"code_graph post-commit hook installed at {hook}."


def uninstall_post_commit_hook(work_dir: str) -> str:
    """Remove our section from the post-commit hook (leave the rest).

    Args:
        work_dir: Worktree root (must be inside a git repository).

    Returns:
        A human-readable status message.
    """
    hooks = _hooks_dir(work_dir)
    if hooks is None:
        return f"Error: {work_dir} is not a git repository."
    lock = _acquire_update_lock_blocking(
        work_dir, timeout=_HOOK_LOCK_TIMEOUT_S, lock_name=_HOOK_LOCK,
    )
    if lock is None:
        return "Error: another code_graph hook update is in progress."
    try:
        return _uninstall_post_commit_hook_locked(hooks)
    finally:
        _release_update_lock(lock)


def _uninstall_post_commit_hook_locked(hooks: Path) -> str:
    """The hook-uninstall read-modify-write; caller holds the hook lock."""
    hook = hooks / "post-commit"
    if not hook.is_file():
        return "code_graph post-commit hook is not installed."
    try:
        text = hook.read_text()
    except (OSError, UnicodeError):
        return f"Error: cannot read existing post-commit hook at {hook}."
    begin_present = _HOOK_BEGIN in text
    end_present = _HOOK_END in text
    if begin_present != end_present:
        return "Error: corrupt code_graph markers in post-commit hook."
    if not begin_present:
        return "code_graph post-commit hook is not installed."
    begin = text.index(_HOOK_BEGIN)
    end = text.index(_HOOK_END) + len(_HOOK_END)
    remaining = (text[:begin] + text[end:]).strip("\n")
    if remaining in ("", "#!/bin/sh"):
        hook.unlink()
    else:
        _atomic_write_hook(hook, remaining + "\n")
    return "code_graph post-commit hook removed."



def make_code_graph_tool(work_dir: str) -> Any | None:
    """Build the ``code_graph`` agent tool for *work_dir*.

    Args:
        work_dir: The agent's worktree root.

    Returns:
        The tool callable, or ``None`` when tree-sitter support is
        unavailable (the agent then simply has no ``code_graph`` tool).
    """
    try:
        import tree_sitter_language_pack  # noqa: F401
    except Exception:
        logger.debug("tree_sitter_language_pack unavailable", exc_info=True)
        return None

    def code_graph(action: str, argument: str = "") -> str:
        """Query the local tree-sitter code knowledge graph of the worktree.

        The graph indexes classes, functions, methods, imports and call
        edges (confidence-tagged EXTRACTED/INFERRED) across the
        worktree's source files. Prefer this tool over grep/read for
        questions about code structure — it answers "what calls X",
        "how are A and B connected", and "what is Y" in a handful of
        lines instead of many file reads.

        Args:
            action: One of "build" (index/refresh the worktree; done
                automatically on first query), "query" (argument =
                free-text question; returns the relevant NODE/EDGE
                subgraph), "path" (argument = two node names separated
                by whitespace; returns the shortest connection between
                them), "explain" (argument = a node name; returns its
                kind, location, degree, and every neighbor),
                "install_hook" / "uninstall_hook" (manage the git
                post-commit hook that keeps the graph fresh in the
                background).
            argument: The action's argument (see above; unused for
                "build", "install_hook", "uninstall_hook").

        Returns:
            The graph answer or a status/diagnostic message.
        """
        if action == "build":
            graph = build_graph(work_dir)
            return (
                f"Code graph built: {len(graph.nodes)} nodes, "
                f"{len(graph.edges)} edges "
                f"({graph.stats.get('reextracted', 0)} files re-extracted)."
            )
        if action == "install_hook":
            return install_post_commit_hook(work_dir)
        if action == "uninstall_hook":
            return uninstall_post_commit_hook(work_dir)
        if action not in ("query", "path", "explain"):
            return (
                f"Error: unknown action {action!r}. Use build, query, "
                "path, explain, install_hook, or uninstall_hook."
            )
        if not argument.strip():
            return f"Error: action {action!r} requires a non-empty argument."
        graph = load_graph(work_dir) or build_graph(work_dir)
        if action == "query":
            return graph.query(argument)
        if action == "path":
            names = argument.split()
            if len(names) != 2:
                return "Error: path requires exactly two node names."
            return graph.path(names[0], names[1])
        return graph.explain(argument.strip())

    return code_graph



def _pid_is_running(pid: int) -> bool:
    """Return whether *pid* still exists (permission denied means yes)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_update_lock(work_dir: str, lock_name: str = _UPDATE_LOCK) -> Path | None:
    """Acquire the non-blocking *lock_name* lock, reclaiming stale locks."""
    storage = graph_dir(work_dir)
    storage.mkdir(parents=True, exist_ok=True)
    lock = storage / lock_name
    for _attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                raw_pid = lock.read_text().strip()
                pid = int(raw_pid)
                age = time.time() - lock.stat().st_mtime
            except (OSError, ValueError):
                pid = -1
                try:
                    age = time.time() - lock.stat().st_mtime
                except OSError:
                    continue
            if _pid_is_running(pid):
                return None
            if pid == -1 and age < _STALE_LOCK_SECONDS:
                return None
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\n")
        return lock
    return None


def _acquire_update_lock_blocking(
    work_dir: str,
    timeout: float = _BUILD_LOCK_TIMEOUT_S,
    lock_name: str = _UPDATE_LOCK,
) -> Path | None:
    """Acquire the *lock_name* lock, waiting up to *timeout* seconds.

    Serializes concurrent graph builds (hook updates vs. agent-tool
    builds) so their cache read-modify-write cycles cannot overwrite
    each other.  Returns ``None`` after *timeout*; the caller must NOT
    enter the protected critical section in that case (an unserialized
    build can silently lose another builder's newer cache/graph).
    """
    deadline = time.monotonic() + timeout
    while True:
        lock = _acquire_update_lock(work_dir, lock_name)
        if lock is not None:
            return lock
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def _release_update_lock(lock: Path) -> None:
    """Release *lock* only when it is still owned by this process."""
    try:
        if int(lock.read_text().strip()) == os.getpid():
            lock.unlink()
    except (FileNotFoundError, OSError, ValueError):
        pass


def main(argv: list[str]) -> int:
    """CLI entry: ``build <root>``, ``update <root> [files...]``,
    ``query <root> <question...>``.

    Args:
        argv: Arguments after the module name.

    Returns:
        Process exit code.
    """
    if len(argv) < 2:
        print(
            "usage: python -m kiss.agents.sorcar.code_graph "
            "{build|update|query} <root> [args...]",
            file=sys.stderr,
        )
        return 2
    verb, root = argv[0], argv[1]
    if verb == "build":
        graph = build_graph(root, incremental=False)
        print(f"built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return 0
    if verb == "update":
        lock = _acquire_update_lock(root)
        if lock is None:
            print("update skipped: another code_graph update is already running")
            return 0
        try:
            only = argv[2:] or None
            graph = _build_graph_locked(root, incremental=True, only_files=only)
            print(f"updated: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            return 0
        finally:
            _release_update_lock(lock)
    if verb == "query":
        graph = load_graph(root) or build_graph(root)
        print(graph.query(" ".join(argv[2:])))
        return 0
    print(f"unknown verb: {verb}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    raise SystemExit(main(sys.argv[1:]))
