#pragma once

// Audit-only path instrumentation for the correctness test-suite in
// projects/bespoke_tpch_x4/tests/.
//
// When the engine is compiled with -DBESPOKE_AUDIT (done ONLY by
// tests/build_audit.sh, never by bench/build.sh), AUDIT_PATH("...") prints a
// single "AUDIT <marker>" line to stderr so the tests can assert which
// execution path (precomputed fast path vs. fallback-to-baseline scan) a
// query request actually took. In normal benchmark builds the macro expands
// to a no-op and the engine binary is bit-for-bit unaffected.
#ifdef BESPOKE_AUDIT
#include <cstdio>
#define AUDIT_PATH(marker) std::fprintf(stderr, "AUDIT %s\n", marker)
#else
#define AUDIT_PATH(marker) ((void)0)
#endif
