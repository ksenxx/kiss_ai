// test_hydra.cc — end-to-end correctness tests for the HydraKV engine.
//
// Drives the engine ONLY through the public IKVStore interface with real
// workloads (real disk I/O, real threads). No mocks. Exit 0 = all pass.
//
// Usage: ./test_hydra <spill_parent_dir> [test_filter_substring] [scale_div]
//   scale_div shrinks workload sizes (for TSAN/ASAN runs), default 1.
//
// Covered bug classes (see PROGRESS.md):
//   * last-write-wins/linearizability under concurrency (stale clean
//     admissions, drop-on-land races) — audit tests with counter oracle
//   * fingerprint-alias index collisions (crafted via mix64 inverse):
//     chain walk, word steal, alias repair, multi-word reads, Delete
//   * oversized-value overflow paths + inline/oversized transitions
//   * Delete of buffered/clean/dirty/absent/oversized keys, reinsertion
//   * index-capacity overflow fallback (tiny budget)
//   * RMW atomicity, no-disk mode, buffered-fd (tmpfs) mode
//   * ReadAsync pipeline + CompletePending contract, read-your-write
//   * store destroy/recreate TLS safety, concurrent Checkpoint
#include "kvstore_interface.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern IKVStore* create_kvstore();

// ── production-hardening stats seam (implemented by hydra.cc) ────────────────
// Non-virtual side channel: the IKVStore interface is frozen (the benchmark
// harness owns it), so production observability is exported through a plain
// extern "C" function resolved at link time within this single test binary.
struct HydraProdStats {
    uint64_t durable_ok;         // 1 = no write/sync failure ever observed
    uint64_t recover_ok;         // 1 = InitExtended recovered an existing log
    uint64_t recovered_keys;     // index entries rebuilt by the last recovery
    uint64_t recover_torn_slots; // CRC-invalid slots skipped by recovery
    uint64_t write_errors;       // failed slot-log/tombstone writes (sticky ctr)
    uint64_t read_errors;        // failed or unexpectedly-short page reads
    uint64_t rejected_oversize;  // oversized upserts refused (budget cap)
    uint64_t oversize_bytes;     // bytes currently charged to the oversize map
    uint64_t compactions_run;    // completed compaction regions
    uint64_t log_bytes;          // allocated-and-unreclaimed slot-log bytes
    uint64_t live_bytes;         // indexed (live) slot bytes estimate
    uint64_t reclaimed_bytes;    // bytes reclaimed (punched and/or reusable)
    uint64_t buffered_fallback;  // 1 = O_DIRECT unavailable, buffered fd used
    uint64_t punch_unsupported;  // 1 = FALLOC_FL_PUNCH_HOLE unsupported here
    uint64_t rejected_mem;       // inline writes refused at the budget cap
    uint64_t recover_dropped_keys;  // keys dropped by the last recovery
};
extern "C" bool hydra_get_prod_stats(IKVStore* st, HydraProdStats* out);

static const char* g_argv0 = nullptr;   // for fork+exec child helpers

static std::atomic<int> g_failures{0};
static int g_scale = 1;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s : ", __FILE__, __LINE__, #cond); \
            fprintf(stderr, __VA_ARGS__);                                  \
            fprintf(stderr, "\n");                                         \
            if (g_failures.fetch_add(1) > 40) {                            \
                fprintf(stderr, "too many failures\n");                      \
                _Exit(1);                                                     \
            }                                                                 \
        }                                                                  \
    } while (0)

// ── deterministic values ─────────────────────────────────────────────────────
static void make_value(uint64_t key, uint64_t ctr, uint32_t size, GenValue& v) {
    v.size = size;
    uint8_t seed = (uint8_t)(key * 1315423911u + ctr * 2654435761u);
    if (size < 16) {                       // small: pure byte-pattern oracle
        for (uint32_t i = 0; i < size; ++i) v.data[i] = (uint8_t)(seed + i);
        return;
    }
    memcpy(v.data, &ctr, 8);
    memcpy(v.data + 8, &key, 8);
    for (uint32_t i = 16; i < size; ++i) v.data[i] = (uint8_t)(seed + i);
}
// Returns the counter if the value is well-formed for `key`, else UINT64_MAX.
static uint64_t check_value(uint64_t key, const GenValue& v, uint32_t size,
                            uint64_t expect_small = 1) {
    if (v.size != size) return UINT64_MAX;
    if (size < 16) {                       // small: verify byte pattern only
        uint8_t seed = (uint8_t)(key * 1315423911u + expect_small * 2654435761u);
        for (uint32_t i = 0; i < size; ++i)
            if (v.data[i] != (uint8_t)(seed + i)) return UINT64_MAX;
        return expect_small;
    }
    uint64_t ctr, k2 = key;
    memcpy(&ctr, v.data, 8);
    memcpy(&k2, v.data + 8, 8);
    if (k2 != key) return UINT64_MAX;
    uint8_t seed = (uint8_t)(key * 1315423911u + ctr * 2654435761u);
    for (uint32_t i = 16; i < size; ++i)
        if (v.data[i] != (uint8_t)(seed + i)) return UINT64_MAX;
    return ctr;
}

// ── engine-matching hash + inverse (for crafted alias keys) ─────────────────
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
static uint64_t unshift_xor(uint64_t y, int s) {
    uint64_t x = y;
    for (int i = 0; i < 64 / s + 1; ++i) x = y ^ (x >> s);
    return x;
}
static uint64_t mix64_inv(uint64_t y) {
    y = unshift_xor(y, 31);
    y *= 0x319642b2d24d8ec3ULL;   // modular inverse of 0x94d049bb133111eb
    y = unshift_xor(y, 27);
    y *= 0x96de1b173f119089ULL;   // modular inverse of 0xbf58476d1ce4e5b9
    y = unshift_xor(y, 30);
    return y;
}

// ── store lifecycle helpers ──────────────────────────────────────────────────
static std::string g_parent;
static IKVStore* make_store(const char* name, size_t budget, bool disk = true) {
    std::string dir = g_parent + "/" + name;
    std::string cmd = "rm -rf " + dir + " && mkdir -p " + dir;
    if (system(cmd.c_str()) != 0) { fprintf(stderr, "mkdir failed\n"); exit(1); }
    IKVStore* st = create_kvstore();
    st->InitExtended(1ull << 20, 16ull << 30, budget, disk ? dir.c_str() : nullptr);
    return st;
}
static void drop_store(IKVStore* st, const char* name) {
    delete st;
    std::string cmd = "rm -rf " + g_parent + "/" + name;
    (void)system(cmd.c_str());
}
// Reopen an existing store dir (defined below, used by earlier tests too).
static IKVStore* open_store(const char* name, size_t budget, bool wipe);

// Synchronous read via the async interface (exercises the pipeline path).
static bool read_async1(IKVStore* st, uint64_t key, GenValue& out) {
    ReadSlot slot;
    slot.key = key;
    slot.done.store(0, std::memory_order_relaxed);
    OpStatus s = st->ReadAsync(&slot);
    if (s == OpStatus::Pending) {
        while (slot.done.load(std::memory_order_acquire) != 1)
            st->CompletePending(false);
        s = slot.status;
    }
    if (s == OpStatus::Ok) out = slot.out;
    return s == OpStatus::Ok;
}

// ── T1: basic dense + sparse load / verify / NotFound ───────────────────────
static void test_basic() {
    const uint64_t N = 200000 / g_scale;
    IKVStore* st = make_store("basic", 256ull << 20);
    st->StartSession();
    GenValue v, out;
    for (uint64_t i = 0; i < N; ++i) {                    // dense keys
        make_value(i, 1, 100, v);
        st->Upsert(i, v);
    }
    for (uint64_t i = 0; i < N; ++i) {                    // sparse keys
        uint64_t k = (mix64(i) & ~(3ull << 62)) | (1ull << 63);
        make_value(k, 1, 100, v);
        st->Upsert(k, v);
    }
    uint64_t bad = 0;
    for (uint64_t i = 0; i < N; ++i) {
        if (!st->Read(i, out) || check_value(i, out, 100) != 1) bad++;
        uint64_t k = (mix64(i) & ~(3ull << 62)) | (1ull << 63);
        if (!read_async1(st, k, out) || check_value(k, out, 100) != 1) bad++;
    }
    CHECK(bad == 0, "basic: %" PRIu64 " bad reads", bad);
    for (uint64_t i = 0; i < 1000; ++i) {                 // absent keys
        uint64_t k = (mix64(i + 0x5eed) & ~(3ull << 62)) | (1ull << 62);
        CHECK(!st->Read(k, out), "absent key %" PRIu64 " found", k);
        CHECK(!read_async1(st, k, out), "absent key async %" PRIu64 " found", k);
    }
    CacheStats cs = st->GetCacheStats();
    CHECK(cs.read_hits + cs.read_misses >= 2 * N, "stats missing reads");
    CHECK(cs.budget_bytes == 256ull << 20, "budget stat wrong");
    st->Refresh();
    st->StopSession();
    drop_store(st, "basic");
    fprintf(stderr, "PASS basic\n");
}

// ── T2: last-write-wins, single thread, with Checkpoint ─────────────────────
static void test_lww() {
    const uint64_t N = 50000 / g_scale;
    IKVStore* st = make_store("lww", 128ull << 20);
    st->StartSession();
    GenValue v, out;
    for (uint64_t round = 1; round <= 5; ++round) {
        for (uint64_t i = 0; i < N; ++i) {
            make_value(i, round, 100, v);
            st->Upsert(i, v);
            if ((i & 1023) == 0) {                        // interleaved reads
                CHECK(st->Read(i, out) && check_value(i, out, 100) == round,
                      "lww read-back i=%" PRIu64 " round=%" PRIu64, i, round);
            }
        }
        if (round == 3) st->Checkpoint();
        uint64_t bad = 0;
        for (uint64_t i = 0; i < N; i += 7)
            if (!st->Read(i, out) || check_value(i, out, 100) != round) bad++;
        CHECK(bad == 0, "lww: %" PRIu64 " stale after round %" PRIu64, bad, round);
    }
    st->StopSession();
    drop_store(st, "lww");
    fprintf(stderr, "PASS lww\n");
}

// ── T3: ReadAsync pipeline, depth 256 ────────────────────────────────────────
static void test_async() {
    const uint64_t N = 300000 / g_scale;
    IKVStore* st = make_store("async", 96ull << 20);   // small: force misses
    st->StartSession();
    GenValue v;
    for (uint64_t i = 0; i < N; ++i) { make_value(i, 1, 100, v); st->Upsert(i, v); }
    const int D = 256;
    std::vector<ReadSlot> slots(D);
    std::mt19937_64 rng(42);
    uint64_t bad = 0;
    for (int batch = 0; batch < 2000 / g_scale; ++batch) {
        for (int j = 0; j < D; ++j) {
            slots[j].key = rng() % N;
            slots[j].done.store(0, std::memory_order_relaxed);
            st->ReadAsync(&slots[j]);
            if ((j & 63) == 63) st->CompletePending(false);
        }
        st->CompletePending(true);
        for (int j = 0; j < D; ++j) {
            if (slots[j].done.load(std::memory_order_acquire) != 1 ||
                slots[j].status != OpStatus::Ok ||
                check_value(slots[j].key, slots[j].out, 100) != 1) bad++;
        }
    }
    CHECK(bad == 0, "async: %" PRIu64 " bad completions", bad);
    st->StopSession();
    drop_store(st, "async");
    fprintf(stderr, "PASS async\n");
}

// ── T4/T5: concurrent linearizability audit ──────────────────────────────────
// Writers own disjoint key ranges (single writer per key). value counter must
// never be older than the last write COMPLETED before the read began.
static void audit_run(const char* name, size_t budget, int nwriters,
                      int nreaders, uint64_t nkeys, int seconds, bool use_async) {
    IKVStore* st = make_store(name, budget);
    std::vector<std::atomic<uint64_t>> issued(nkeys), completed(nkeys);
    for (auto& a : issued) a.store(0);
    for (auto& a : completed) a.store(0);
    {   // load phase: every key exists (counter 1)
        st->StartSession();
        GenValue v;
        for (uint64_t i = 0; i < nkeys; ++i) {
            make_value(i, 1, 100, v);
            st->Upsert(i, v);
            issued[i].store(1); completed[i].store(1);
        }
        st->StopSession();
    }
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> reads{0}, writes{0};
    std::vector<std::thread> ths;
    for (int w = 0; w < nwriters; ++w) {
        ths.emplace_back([&, w] {
            st->StartSession();
            std::mt19937_64 rng(1000 + w);
            uint64_t lo = nkeys * (uint64_t)w / nwriters;
            uint64_t hi = nkeys * (uint64_t)(w + 1) / nwriters;
            GenValue v;
            while (!stop.load(std::memory_order_relaxed)) {
                // zipf-ish: hammer the low end of the range
                uint64_t span = hi - lo;
                uint64_t r = rng() % span;
                uint64_t k = lo + (r < span / 2 ? r % (span / 16 + 1) : r);
                uint64_t c = issued[k].fetch_add(1, std::memory_order_acq_rel) + 1;
                make_value(k, c, 100, v);
                st->Upsert(k, v);
                // publish max completed (single writer per key => store OK,
                // but counters from this writer are monotone anyway)
                uint64_t prev = completed[k].load(std::memory_order_relaxed);
                while (prev < c && !completed[k].compare_exchange_weak(
                                       prev, c, std::memory_order_release)) {}
                writes.fetch_add(1, std::memory_order_relaxed);
            }
            st->StopSession();
        });
    }
    for (int r = 0; r < nreaders; ++r) {
        ths.emplace_back([&, r] {
            st->StartSession();
            std::mt19937_64 rng(2000 + r);
            GenValue out;
            std::vector<ReadSlot> slots(64);
            std::vector<uint64_t> lows(64);
            while (!stop.load(std::memory_order_relaxed)) {
                if (use_async) {
                    for (int j = 0; j < 64; ++j) {
                        uint64_t span = nkeys;
                        uint64_t x = rng() % span;
                        uint64_t k = (x < span / 2) ? x % (span / 16 + 1) : x;
                        lows[j] = completed[k].load(std::memory_order_acquire);
                        slots[j].key = k;
                        slots[j].done.store(0, std::memory_order_relaxed);
                        st->ReadAsync(&slots[j]);
                    }
                    st->CompletePending(true);
                    for (int j = 0; j < 64; ++j) {
                        uint64_t k = slots[j].key;
                        CHECK(slots[j].done.load(std::memory_order_acquire) == 1,
                              "audit slot not done");
                        CHECK(slots[j].status == OpStatus::Ok,
                              "audit key %" PRIu64 " lost", k);
                        uint64_t c = check_value(k, slots[j].out, 100);
                        uint64_t hi2 = issued[k].load(std::memory_order_acquire);
                        CHECK(c != UINT64_MAX, "audit corrupt value k=%" PRIu64, k);
                        CHECK(c >= lows[j] && c <= hi2,
                              "audit STALE k=%" PRIu64 " c=%" PRIu64
                              " lo=%" PRIu64 " hi=%" PRIu64, k, c, lows[j], hi2);
                        reads.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    uint64_t k = rng() % nkeys;
                    uint64_t lo = completed[k].load(std::memory_order_acquire);
                    bool ok = st->Read(k, out);
                    uint64_t hi2 = issued[k].load(std::memory_order_acquire);
                    CHECK(ok, "audit sync key %" PRIu64 " lost", k);
                    if (ok) {
                        uint64_t c = check_value(k, out, 100);
                        CHECK(c != UINT64_MAX && c >= lo && c <= hi2,
                              "audit sync STALE k=%" PRIu64 " c=%" PRIu64
                              " lo=%" PRIu64 " hi=%" PRIu64, k, c, lo, hi2);
                    }
                    reads.fetch_add(1, std::memory_order_relaxed);
                }
            }
            st->StopSession();
        });
    }
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    stop.store(true);
    for (auto& t : ths) t.join();
    fprintf(stderr, "PASS %s (reads=%" PRIu64 " writes=%" PRIu64 ")\n",
            name, reads.load(), writes.load());
    drop_store(st, name);
}
static void test_audit() {
    // Small budget => tiny cache => constant evictions/misses; hot keys are
    // upserted+flushed+dropped while reads of them are in flight (the B1
    // stale-admission window).
    audit_run("audit", 64ull << 20, 8, 8, 200000 / g_scale,
              g_scale > 1 ? 5 : 12, true);
    audit_run("audit_sync", 64ull << 20, 8, 8, 100000 / g_scale,
              g_scale > 1 ? 3 : 6, false);
}

// ── T6: oversized values / overflow map ──────────────────────────────────────
static void test_oversized() {
    IKVStore* st = make_store("oversized", 128ull << 20);
    st->StartSession();
    GenValue v, out;
    const uint64_t N = 2000;
    uint32_t sizes[] = {16, 32, 100, 101, 102, 200, 1024, 4096};
    for (uint64_t i = 0; i < N; ++i) {
        uint32_t sz = sizes[i % 8];
        make_value(i, 1, sz, v);
        st->Upsert(i, v);
    }
    for (uint64_t i = 0; i < N; ++i) {
        uint32_t sz = sizes[i % 8];
        CHECK(st->Read(i, out) && check_value(i, out, sz) == 1,
              "oversized read i=%" PRIu64 " sz=%u", i, sz);
        CHECK(read_async1(st, i, out) && check_value(i, out, sz) == 1,
              "oversized async i=%" PRIu64 " sz=%u", i, sz);
    }
    // transitions inline -> oversized -> inline (LWW must hold)
    for (uint64_t i = 0; i < N; ++i) {
        make_value(i, 2, 4096, v); st->Upsert(i, v);
        make_value(i, 3, 100, v);  st->Upsert(i, v);
        make_value(i, 4, 300, v);  st->Upsert(i, v);
        CHECK(st->Read(i, out) && check_value(i, out, 300) == 4,
              "oversized transition i=%" PRIu64, i);
    }
    // small sizes: byte-pattern oracle (engine guarantees exactly `size`
    // bytes; the old test read indeterminate bytes past size<8 values)
    for (uint64_t i = 0; i < 64; ++i) {
        uint32_t small[] = {0, 1, 4, 7, 8, 15};
        uint32_t sz = small[i % 6];
        uint64_t k = 900000 + i;
        make_value(k, 5, sz, v);
        st->Upsert(k, v);
        CHECK(st->Read(k, out) && check_value(k, out, sz, 5) == 5,
              "small size %u k=%" PRIu64, sz, k);
    }
    // concurrent readers during inline<->oversized flapping: keys always
    // exist => NotFound is a failure; counter must be within the completed/
    // issued interval for the key (linearizability, same as audit).
    std::atomic<bool> stop{false};
    std::vector<std::atomic<uint64_t>> fcomp(N), fissue(N);
    for (uint64_t i = 0; i < N; ++i) { fcomp[i].store(4); fissue[i].store(4); }
    std::thread reader([&] {
        st->StartSession();
        GenValue o;
        std::mt19937_64 rng(7);
        while (!stop.load(std::memory_order_relaxed)) {
            uint64_t k = rng() % N;
            uint64_t lo = fcomp[k].load(std::memory_order_acquire);
            bool ok = st->Read(k, o);
            uint64_t hi = fissue[k].load(std::memory_order_acquire);
            CHECK(ok, "flap NotFound k=%" PRIu64, k);
            if (!ok) continue;
            uint64_t c;
            memcpy(&c, o.data, 8);
            CHECK(check_value(k, o, o.size) == c, "flap corrupt k=%" PRIu64, k);
            CHECK(c >= lo && c <= hi,
                  "flap STALE k=%" PRIu64 " c=%" PRIu64 " lo=%" PRIu64
                  " hi=%" PRIu64, k, c, lo, hi);
        }
        st->StopSession();
    });
    for (uint64_t round = 5; round < 50; ++round) {
        for (uint64_t i = 0; i < N; ++i) {
            fissue[i].store(round, std::memory_order_release);
            make_value(i, round, (round & 1) ? 100 : 500, v);
            st->Upsert(i, v);
            fcomp[i].store(round, std::memory_order_release);
        }
    }
    stop.store(true);
    reader.join();
    st->StopSession();
    drop_store(st, "oversized");
    fprintf(stderr, "PASS oversized\n");
}

// ── T7: Delete semantics ─────────────────────────────────────────────────────
static void test_delete() {
    IKVStore* st = make_store("del", 128ull << 20);
    st->StartSession();
    GenValue v, out;
    const uint64_t N = 20000 / g_scale;
    for (uint64_t i = 0; i < N; ++i) { make_value(i, 1, 100, v); st->Upsert(i, v); }
    // delete freshly-buffered keys (pinned path) and settled keys
    for (uint64_t i = 0; i < N; i += 2) CHECK(st->Delete(i), "del existing %" PRIu64, i);
    st->Checkpoint();                          // land everything
    for (uint64_t i = 0; i < N; ++i) {
        bool found = st->Read(i, out);
        if (i % 2 == 0) CHECK(!found, "deleted key %" PRIu64 " resurrected", i);
        else CHECK(found && check_value(i, out, 100) == 1, "survivor %" PRIu64 " lost", i);
    }
    // Async path: the deleted-key early return in ReadAsync.
    for (uint64_t i = 0; i < N; i += 97) {
        bool found = read_async1(st, i, out);
        if (i % 2 == 0) CHECK(!found, "deleted %" PRIu64 " async resurrected", i);
        else CHECK(found && check_value(i, out, 100) == 1,
                   "survivor %" PRIu64 " async lost", i);
    }
    CHECK(!st->Delete(N + 12345), "delete of absent key returned true");
    // re-insert deleted keys
    for (uint64_t i = 0; i < N; i += 2) { make_value(i, 9, 100, v); st->Upsert(i, v); }
    for (uint64_t i = 0; i < N; i += 2)
        CHECK(st->Read(i, out) && check_value(i, out, 100) == 9,
              "reinserted %" PRIu64 " wrong", i);
    // oversized delete
    make_value(5, 7, 2000, v); st->Upsert(5, v);
    CHECK(st->Delete(5), "oversized delete");
    CHECK(!st->Read(5, out), "oversized key resurrected");
    // concurrent delete-vs-read: once Delete(k) returned, no reader may see
    // k again (no writers). del_done[k] is set right after Delete returns.
    {
        const uint64_t M = 30000 / (uint64_t)g_scale;
        std::vector<std::atomic<uint8_t>> del_done(M);
        for (auto& d : del_done) d.store(0);
        GenValue vv;
        for (uint64_t i = 0; i < M; ++i) {
            uint64_t k = (1ull << 59) + i;
            make_value(k, 2, 100, vv);
            st->Upsert(k, vv);
        }
        std::atomic<bool> stop{false};
        std::vector<std::thread> rd;
        for (int r = 0; r < 4; ++r) {
            rd.emplace_back([&, r] {
                st->StartSession();
                std::mt19937_64 rng(6000 + r);
                GenValue o;
                while (!stop.load(std::memory_order_relaxed)) {
                    uint64_t i = rng() % M;
                    bool was_deleted =
                        del_done[i].load(std::memory_order_acquire) != 0;
                    bool found = st->Read((1ull << 59) + i, o);
                    if (was_deleted)
                        CHECK(!found, "deleted key %" PRIu64 " RESURRECTED", i);
                    else if (found)
                        CHECK(check_value((1ull << 59) + i, o, 100) == 2,
                              "delete-race corrupt %" PRIu64, i);
                }
                st->StopSession();
            });
        }
        for (uint64_t i = 0; i < M; ++i) {
            st->Delete((1ull << 59) + i);
            del_done[i].store(1, std::memory_order_release);
        }
        // Final full pass WHILE readers still hammer: every key must stay
        // gone (catches late landings of stale in-flight copies).
        GenValue o2;
        for (uint64_t i = 0; i < M; ++i)
            CHECK(!st->Read((1ull << 59) + i, o2),
                  "deleted key %" PRIu64 " LATE-resurrected", i);
        stop.store(true);
        for (auto& th : rd) th.join();
    }
    st->StopSession();
    drop_store(st, "del");
    fprintf(stderr, "PASS delete\n");
}

// ── T7b: concurrent Upsert/Delete/Read flapping on the same keys ─────────────
// Interval oracle: writer alternates Upsert(ctr=odd phase) / Delete per key,
// publishing issue[k] before and comp[k] after each op. A reader brackets its
// Read with lo=comp[k] (acquire, before) and hi=issue[k] (acquire, after):
//  - found  => value ctr must be an ODD phase with lo <= ctr <= hi (if the
//              delete phase ctr+1 completed before the read started, i.e.
//              lo > ctr, finding ctr is a stale resurrection);
//  - !found => legal unless lo==hi and lo is odd (an upsert completed and
//              nothing newer was issued: the key MUST be visible).
static void test_updel() {
    IKVStore* st = make_store("updel", 128ull << 20);
    st->StartSession();
    const uint64_t K = 64;
    const uint64_t PHASES = 4000 / (uint64_t)g_scale;
    std::vector<std::atomic<uint64_t>> issue(K), comp(K);
    for (uint64_t k = 0; k < K; ++k) { issue[k].store(0); comp[k].store(0); }
    std::atomic<bool> stop{false};
    std::vector<std::thread> rd;
    for (int r = 0; r < 4; ++r) {
        rd.emplace_back([&, r] {
            st->StartSession();
            std::mt19937_64 rng(7000 + r);
            GenValue o;
            while (!stop.load(std::memory_order_relaxed)) {
                uint64_t k = rng() % K;
                uint64_t key = (1ull << 58) + k * 0x9E3779B97F4A7C15ULL;
                uint64_t lo = comp[k].load(std::memory_order_acquire);
                bool found = st->Read(key, o);
                uint64_t hi = issue[k].load(std::memory_order_acquire);
                if (found) {
                    uint64_t c = check_value(key, o, 100);
                    CHECK((c & 1) == 1 && c >= lo && c <= hi,
                          "updel STALE k=%" PRIu64 " c=%" PRIu64
                          " lo=%" PRIu64 " hi=%" PRIu64, k, c, lo, hi);
                } else {
                    CHECK(!(lo == hi && (lo & 1) == 1),
                          "updel LOST k=%" PRIu64 " phase=%" PRIu64, k, lo);
                }
            }
            st->StopSession();
        });
    }
    GenValue v;
    for (uint64_t p = 1; p <= PHASES; ++p) {
        for (uint64_t k = 0; k < K; ++k) {
            uint64_t key = (1ull << 58) + k * 0x9E3779B97F4A7C15ULL;
            issue[k].store(p, std::memory_order_release);
            if (p & 1) { make_value(key, p, 100, v); st->Upsert(key, v); }
            else       { st->Delete(key); }
            comp[k].store(p, std::memory_order_release);
        }
        if (p % 64 == 0) st->Checkpoint();   // force landings mid-flap
    }
    stop.store(true);
    for (auto& th : rd) th.join();
    // settle: last phase parity decides final visibility
    GenValue o;
    for (uint64_t k = 0; k < K; ++k) {
        uint64_t key = (1ull << 58) + k * 0x9E3779B97F4A7C15ULL;
        bool found = st->Read(key, o);
        if (PHASES & 1)
            CHECK(found && check_value(key, o, 100) == PHASES,
                  "updel final lost k=%" PRIu64, k);
        else
            CHECK(!found, "updel final resurrected k=%" PRIu64, k);
    }
    st->StopSession();
    drop_store(st, "updel");
    fprintf(stderr, "PASS updel\n");
}

// ── T7c: Delete of keys pinned BUFFERED in ANOTHER session's parked chunk ────
// Audit bug #1 regression: Delete used to wait for BUFFERED pins to land,
// but could only flush its OWN session's partial chunk — deleting a key
// staged in an idle foreign session livelocked (froze at a handful of ops).
// Delete must return promptly, the keys must read NotFound immediately, and
// the deletions must survive the foreign chunk's later landing AND a restart
// (the landing reseals deleted first-insert slots as tombstones).
static void test_updel2() {
    IKVStore* st = make_store("updel2", 128ull << 20);
    const uint64_t N = 512;
    const uint64_t B = 1ull << 57;
    std::atomic<bool> staged{false}, done{false};
    std::thread owner([&] {
        st->StartSession();
        GenValue vv;
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t k = B + i * 0x9E3779B97F4A7C15ULL;
            make_value(k, 1, 100, vv);
            st->Upsert(k, vv);   // first inserts stay BUFFERED in this
        }                        // session's partial chunk (512 << 8192)
        staged.store(true, std::memory_order_release);
        while (!done.load(std::memory_order_acquire))   // park IDLE — never
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        st->StopSession();       // NOW the chunk lands (tombstones deleted)
    });
    while (!staged.load(std::memory_order_acquire)) std::this_thread::yield();
    st->StartSession();          // a DIFFERENT session issues the deletes
    auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < N; i += 2)
        st->Delete(B + i * 0x9E3779B97F4A7C15ULL);
    double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    CHECK(secs < 10.0,
          "updel2: %.1fs deleting foreign-buffered keys (livelock)", secs);
    GenValue out;
    for (uint64_t i = 0; i < N; ++i) {   // visible state while chunk parked
        uint64_t k = B + i * 0x9E3779B97F4A7C15ULL;
        bool found = st->Read(k, out);
        if (i % 2 == 0) CHECK(!found, "updel2: deleted %" PRIu64 " visible", i);
        else CHECK(found && check_value(k, out, 100) == 1,
                   "updel2: survivor %" PRIu64 " lost", i);
    }
    done.store(true, std::memory_order_release);
    owner.join();                // foreign chunk lands here
    for (uint64_t i = 0; i < N; ++i) {   // state after the landing
        uint64_t k = B + i * 0x9E3779B97F4A7C15ULL;
        bool found = st->Read(k, out);
        if (i % 2 == 0) CHECK(!found, "updel2: deleted %" PRIu64 " landed back", i);
        else CHECK(found && check_value(k, out, 100) == 1,
                   "updel2: survivor %" PRIu64 " lost at landing", i);
    }
    st->Checkpoint();
    st->StopSession();
    delete st;                   // clean shutdown
    IKVStore* st2 = open_store("updel2", 128ull << 20, false);
    st2->StartSession();
    for (uint64_t i = 0; i < N; ++i) {   // deletions durable across restart
        uint64_t k = B + i * 0x9E3779B97F4A7C15ULL;
        bool found = st2->Read(k, out);
        if (i % 2 == 0)
            CHECK(!found, "updel2: deleted %" PRIu64 " resurrected", i);
        else
            CHECK(found && check_value(k, out, 100) == 1,
                  "updel2: survivor %" PRIu64 " lost across restart", i);
    }
    st2->StopSession();
    drop_store(st2, "updel2");
    fprintf(stderr, "PASS updel2\n");
}

// ── T7d: Delete foreground cost ──────────────────────────────────────────────
// Audit bug #4 regression: Delete used to do a synchronous pread+pwrite per
// on-disk version (~160x an Upsert). It now defers on-disk poisoning to the
// background reaper, so a Delete costs about as much as an Upsert.
static void test_delcost() {
    IKVStore* st = make_store("delcost", 128ull << 20);
    st->StartSession();
    GenValue v;
    const uint64_t N = 20000 / (uint64_t)g_scale;
    for (uint64_t i = 0; i < N; ++i) { make_value(i, 1, 100, v); st->Upsert(i, v); }
    st->Checkpoint();   // land everything: deletes target LANDED slots
    auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < N; ++i) { make_value(i, 2, 100, v); st->Upsert(i, v); }
    double tu = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < N; ++i) st->Delete(i);
    double td = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    // Generous bound (sanitizer/CI-safe): the audit measured 160x; the
    // deferred design is ~1x. Absolute slack absorbs tiny-N noise.
    CHECK(td <= 10.0 * tu + 0.25,
          "delcost: %" PRIu64 " deletes took %.3fs vs upserts %.3fs "
          "(audit bug: synchronous per-version tombstone I/O)", N, td, tu);
    GenValue out;
    for (uint64_t i = 0; i < N; i += 17)
        CHECK(!st->Read(i, out), "delcost: deleted %" PRIu64 " visible", i);
    st->Checkpoint();   // drains the reaper: tombstones durable
    st->StopSession();
    drop_store(st, "delcost");
    fprintf(stderr, "PASS delcost (delete %.3fs vs upsert %.3fs)\n", td, tu);
}

// ── T7e: charged + capped inline overflow absorption ────────────────────────
// Audit bug #2 regression: inline values absorbed into the overflow map by
// the index-capacity / disk-full fallbacks were UNCHARGED — the map could
// grow past the whole memory budget and the process got OOM-killed. Every
// absorption is now charged against the (default budget/8) cap; over the cap
// the store rejects honestly and counts it.
static void test_memfull() {
    IKVStore* st = make_store("memfull", 16ull << 20);   // tiny index+cap
    st->StartSession();
    GenValue v, out;
    // Fixed N (not scaled): must exceed the 16 MiB store's index capacity
    // (~250K keys) plus the 2 MiB overflow cap (~13K absorptions).
    const uint64_t N = 600000;
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t k = mix64(i);
        make_value(k, 1, 100, v);
        st->Upsert(k, v);
    }
    HydraProdStats ps{};
    CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    CHECK(ps.rejected_mem > 0,
          "memfull: no honest rejection past the absorb cap");
    CHECK(ps.oversize_bytes <= (16ull << 20) / 8 + 8192,
          "memfull: charged overflow bytes above cap: %" PRIu64,
          ps.oversize_bytes);
    // No silent damage: every key is either intact or was refused+counted.
    uint64_t missing = 0, corrupt = 0;
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t k = mix64(i);
        if (!st->Read(k, out)) missing++;
        else if (check_value(k, out, 100) != 1) corrupt++;
    }
    CHECK(corrupt == 0, "memfull: %" PRIu64 " corrupt values", corrupt);
    CHECK(missing <= ps.rejected_mem,
          "memfull: %" PRIu64 " keys missing but only %" PRIu64
          " rejections counted (silent loss)", missing, ps.rejected_mem);
    st->StopSession();
    drop_store(st, "memfull");
    fprintf(stderr, "PASS memfull (%" PRIu64 " absorbed-rejected, %" PRIu64
            " missing)\n", ps.rejected_mem, missing);
}

// ── T7f: smaller-budget reopen must surface dropped keys ─────────────────────
// Audit bug #3 regression: recovery dropped over-index-capacity keys with
// only a stderr WARNING while still reporting recover_ok=1. Drops now fail
// the recovery status and are counted.
static void test_recoverdrop() {
    {
        IKVStore* st = make_store("recdrop", 256ull << 20);
        st->StartSession();
        GenValue v;
        const uint64_t N = 2000000 / (uint64_t)g_scale;
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t k = mix64(i);
            make_value(k, 1, 100, v);
            st->Upsert(k, v);
        }
        // Oversized values too: their sidecar (written under THIS larger
        // budget's cap) must not blow past the smaller reopen's cap below.
        for (uint64_t i = 0; i < 2000; ++i) {
            make_value((1ull << 61) + i, 3, 2000, v);
            st->Upsert((1ull << 61) + i, v);
        }
        st->Checkpoint();
        st->StopSession();
        delete st;   // clean shutdown, full log on disk
    }
    // Reopen at a budget whose index cannot hold every key.
    IKVStore* st2 = open_store("recdrop", 4ull << 20, false);
    st2->StartSession();
    HydraProdStats ps{};
    CHECK(hydra_get_prod_stats(st2, &ps), "prod stats unavailable");
    // Sidecar reload respects the smaller budget's cap (bounded, counted).
    CHECK(ps.oversize_bytes <= (4ull << 20) / 8 + 4096,
          "recoverdrop: sidecar reload past cap: %" PRIu64, ps.oversize_bytes);
    CHECK(ps.recover_dropped_keys > 0,
          "recoverdrop: no drops counted (test sizing broken?)");
    CHECK(ps.recover_ok == 0,
          "recoverdrop: %" PRIu64 " keys dropped but recover_ok=1 "
          "(silent data loss reported as clean recovery)",
          ps.recover_dropped_keys);
    // What recovery kept must still be served intact.
    GenValue out;
    uint64_t kept = 0;
    for (uint64_t i = 0; i < 2000; ++i) {
        uint64_t k = mix64(i);
        if (!st2->Read(k, out)) continue;
        kept++;
        CHECK(check_value(k, out, 100) == 1,
              "recoverdrop: kept key %" PRIu64 " corrupt", i);
    }
    (void)kept;
    st2->StopSession();
    drop_store(st2, "recdrop");
    fprintf(stderr, "PASS recoverdrop (%" PRIu64 " dropped surfaced)\n",
            ps.recover_dropped_keys);
}

// ── T8: crafted fingerprint-alias keys ───────────────────────────────────────
// Keys whose mix64 hashes share the high-32 fingerprint AND the index bucket:
// exercises first_candidate aliasing, prev-chain walks, index word steal +
// alias repair, multi-word reads, and multi-version Delete.
static void test_alias() {
    for (uint64_t r = 0; r < 4096; ++r) {   // self-check the inverse
        uint64_t x = mix64(r * 0x9E3779B97F4A7C15ULL + 1);
        if (mix64_inv(mix64(x)) != x) { CHECK(false, "mix64_inv broken"); return; }
    }
    const size_t budget = 256ull << 20;
    // engine sizing: index words = largest pow2 with words*2*8 <= budget/4
    uint64_t words = 64;
    while (words * 2 * 8 <= budget / 4) words *= 2;
    // bucket(h) = (low32(h) * 8) & (words-1): adding `words/8` to low32 keeps
    // the bucket identical; high32 (the fingerprint) is untouched.
    uint64_t delta = words / 8;
    IKVStore* st = make_store("alias", budget);
    st->StartSession();
    GenValue v, out;
    const int PAIRS = 128;
    std::vector<uint64_t> A(PAIRS), B(PAIRS), C(PAIRS);
    for (int p = 0; p < PAIRS; ++p) {
        uint64_t hA = mix64(0xA11A5000 + (uint64_t)p) & ~0xFFF00000ULL;
        A[p] = mix64_inv(hA);
        B[p] = mix64_inv(hA + delta);           // same fp + bucket
        C[p] = mix64_inv(hA + 2 * delta);       // triple collision
        CHECK((mix64(A[p]) >> 32) == (mix64(B[p]) >> 32), "fp mismatch");
        make_value(A[p], 1, 100, v); st->Upsert(A[p], v);
        make_value(B[p], 1, 100, v); st->Upsert(B[p], v);
        make_value(C[p], 1, 100, v); st->Upsert(C[p], v);
    }
    // rewrite in different orders across rounds; Checkpoint forces landings
    // (index word steals + chain re-pointing). EXACT-value oracle: we know
    // the last counter written to each key.
    uint64_t expA = 1, expB = 1, expC = 1;
    for (uint64_t round = 2; round <= 8; ++round) {
        for (int p = 0; p < PAIRS; ++p) {
            uint64_t k = (round % 3 == 0) ? A[p] : (round % 3 == 1) ? B[p] : C[p];
            make_value(k, round, 100, v);
            st->Upsert(k, v);
        }
        if (round % 3 == 0) expA = round; else if (round % 3 == 1) expB = round;
        else expC = round;
        st->Checkpoint();
        for (int p = 0; p < PAIRS; ++p) {
            uint64_t ks[3] = {A[p], B[p], C[p]};
            uint64_t ex[3] = {expA, expB, expC};
            for (int j = 0; j < 3; ++j) {
                CHECK(st->Read(ks[j], out), "alias key lost r=%" PRIu64, round);
                uint64_t c = check_value(ks[j], out, 100);
                CHECK(c == ex[j], "alias WRONG r=%" PRIu64 " j=%d c=%" PRIu64
                      " want=%" PRIu64, round, j, c, ex[j]);
            }
        }
    }
    // flood the cache so alias reads must go to DISK (chain-walk paths)
    for (uint64_t i = 0; i < 3000000 / (uint64_t)g_scale; ++i) {
        uint64_t k = mix64(i) | (1ull << 61);
        make_value(k, 1, 100, v);
        st->Upsert(k, v);
    }
    st->Checkpoint();
    // The SLRU protects multi-touch keys from one-touch flood traffic (by
    // design), so the alias keys — promoted by the read rounds above — would
    // keep HITTING the cache and the disk chain-walk paths would go
    // unexercised. Deterministically displace them with crafted set-local
    // pressure: keys whose mix64 differs from the alias key's only in low
    // bits land in the SAME fastrange cache set for any nsets; promote 12
    // such keys per alias key (read x5 after landing) to churn the
    // protected ways, then insert 16 more to chew through probation.
    {
        std::vector<uint64_t> prom, fill;
        for (int p = 0; p < PAIRS; ++p) {
            uint64_t ks[3] = {A[p], B[p], C[p]};
            for (int j = 0; j < 3; ++j) {
                uint64_t h = mix64(ks[j]);
                for (uint64_t i = 1; i <= 12; ++i)
                    prom.push_back(mix64_inv(h ^ (i << 5)));
                for (uint64_t i = 40; i < 56; ++i)
                    fill.push_back(mix64_inv(h ^ (i << 5)));
            }
        }
        for (uint64_t k : prom) { make_value(k, 3, 100, v); st->Upsert(k, v); }
        st->Checkpoint();
        GenValue o;
        for (int r = 0; r < 5; ++r)
            for (uint64_t k : prom) (void)st->Read(k, o);
        for (uint64_t k : fill) { make_value(k, 3, 100, v); st->Upsert(k, v); }
        st->Checkpoint();
    }
    CacheStats cs0 = st->GetCacheStats();
    for (int p = 0; p < PAIRS; ++p) {
        uint64_t ks[3] = {A[p], B[p], C[p]};
        uint64_t ex[3] = {expA, expB, expC};
        for (int j = 0; j < 3; ++j) {
            CHECK(read_async1(st, ks[j], out) &&
                  check_value(ks[j], out, 100) == ex[j],
                  "alias flood read wrong p=%d j=%d", p, j);
        }
    }
    CacheStats cs1 = st->GetCacheStats();
    // the flood must actually push reads to disk (else the chain-walk code
    // was not exercised)
    CHECK(cs1.read_misses > cs0.read_misses, "alias reads never missed");
    // delete A: B and C must survive with correct values; A must be gone
    for (int p = 0; p < PAIRS; ++p) {
        CHECK(st->Delete(A[p]), "alias delete A");
        CHECK(!st->Read(A[p], out), "alias A resurrected p=%d", p);
        CHECK(st->Read(B[p], out) && check_value(B[p], out, 100) == expB,
              "alias B damaged by delete p=%d", p);
        CHECK(st->Read(C[p], out) && check_value(C[p], out, 100) == expC,
              "alias C damaged by delete p=%d", p);
    }
    // re-insert A after delete
    for (int p = 0; p < PAIRS; ++p) {
        make_value(A[p], 77, 100, v);
        st->Upsert(A[p], v);
        CHECK(st->Read(A[p], out) && check_value(A[p], out, 100) == 77,
              "alias A reinsert p=%d", p);
    }
    st->StopSession();
    drop_store(st, "alias");
    fprintf(stderr, "PASS alias\n");
}

// ── T9: tiny budget — index capacity overflow fallback ───────────────────────
static void test_tiny() {
    // This test intentionally drives ~3M keys through a 64 MiB store, far
    // past the index capacity, to exercise the overflow fallback. Fallback
    // absorptions are now CHARGED and CAPPED (audit: uncharged absorptions
    // could OOM the process), so give this test an explicit cap that fits
    // its whole over-capacity keyspace; test_memfull asserts the default
    // cap's honest-rejection behavior.
    setenv("HYDRA_OVERSIZE_CAP_MB", "1024", 1);
    IKVStore* st = make_store("tiny", 64ull << 20);
    unsetenv("HYDRA_OVERSIZE_CAP_MB");
    const uint64_t N = 3000000 / (uint64_t)g_scale;
    const int T = 4;
    std::vector<std::thread> ths;
    for (int t = 0; t < T; ++t) {
        ths.emplace_back([&, t] {
            st->StartSession();
            GenValue v;
            for (uint64_t i = t; i < N; i += T) {
                uint64_t k = mix64(i);
                make_value(k, 1, 100, v);
                st->Upsert(k, v);
            }
            st->StopSession();
        });
    }
    for (auto& t : ths) t.join();
    st->StartSession();
    GenValue out;
    uint64_t bad = 0;
    for (uint64_t i = 0; i < N; i += 13) {
        uint64_t k = mix64(i);
        if (!st->Read(k, out) || check_value(k, out, 100) != 1) bad++;
    }
    CHECK(bad == 0, "tiny: %" PRIu64 " bad", bad);
    // Second pass over a subset: doorkeeper-known misses now ADMIT while the
    // overflow map is non-empty, exercising the overflow_contains gate in
    // admit_read/admit_neighbor (a key sitting in overflow must never be
    // shadowed by a stale inline admission).
    for (uint64_t i = 0; i < N; i += 13) {
        uint64_t k = mix64(i);
        if (!st->Read(k, out) || check_value(k, out, 100) != 1) bad++;
    }
    CHECK(bad == 0, "tiny pass2: %" PRIu64 " bad", bad);
    st->StopSession();
    drop_store(st, "tiny");
    fprintf(stderr, "PASS tiny\n");
}

// ── T10: no-disk mode (Init 2-param path) ────────────────────────────────────
static void test_nodisk() {
    IKVStore* st = create_kvstore();
    st->Init(1ull << 16, 1ull << 30);      // forwards to InitExtended(no disk)
    st->StartSession();
    GenValue v, out;
    for (uint64_t i = 0; i < 5000; ++i) { make_value(i, 3, 100, v); st->Upsert(i, v); }
    for (uint64_t i = 0; i < 5000; ++i) {
        CHECK(st->Read(i, out) && check_value(i, out, 100) == 3, "nodisk read");
        CHECK(read_async1(st, i, out), "nodisk async");
    }
    CHECK(!st->Read(999999, out), "nodisk absent");
    st->RMW(42, v.data, 100);
    CHECK(st->Delete(7), "nodisk delete");
    CHECK(!st->Read(7, out), "nodisk deleted");
    st->Checkpoint();                       // no-op
    st->StopSession();
    delete st;
    fprintf(stderr, "PASS nodisk\n");
}

// ── T11: RMW accumulation, multi-threaded ────────────────────────────────────
static void test_rmw() {
    IKVStore* st = make_store("rmw", 128ull << 20);
    const uint64_t K = 512;
    const int T = 4, ROUNDS = 4000 / g_scale;
    std::vector<std::atomic<uint64_t>> sum(K);
    for (auto& a : sum) a.store(0);
    std::vector<std::thread> ths;
    for (int t = 0; t < T; ++t) {
        ths.emplace_back([&, t] {
            st->StartSession();
            std::mt19937_64 rng(3000 + t);
            uint8_t mod[100] = {0};
            for (int r = 0; r < ROUNDS; ++r) {
                uint64_t k = rng() % K;
                uint64_t d = 1 + (rng() % 1000);
                memcpy(mod, &d, 8);
                sum[k].fetch_add(d, std::memory_order_relaxed);
                st->RMW(k, mod, 100);
            }
            st->StopSession();
        });
    }
    for (auto& t : ths) t.join();
    st->StartSession();
    GenValue out;
    for (uint64_t k = 0; k < K; ++k) {
        uint64_t expect = sum[k].load();
        if (expect == 0) continue;
        CHECK(st->Read(k, out), "rmw key %" PRIu64 " lost", k);
        uint64_t got; memcpy(&got, out.data, 8);
        CHECK(got == expect, "rmw k=%" PRIu64 " got=%" PRIu64 " want=%" PRIu64,
              k, got, expect);
    }
    st->StopSession();
    drop_store(st, "rmw");
    fprintf(stderr, "PASS rmw\n");
}

// ── T12: destroy/recreate stores on one thread (TLS safety) ─────────────────
static void test_twostores() {
    for (int round = 0; round < 3; ++round) {
        IKVStore* a = make_store("twoA", 64ull << 20);
        a->StartSession();
        GenValue v, out;
        make_value(1, 10 + (uint64_t)round, 100, v);
        a->Upsert(1, v);
        CHECK(a->Read(1, out) && check_value(1, out, 100) == 10 + (uint64_t)round,
              "twostores A");
        // no StopSession on purpose: destructor must still be safe & durable
        drop_store(a, "twoA");
        IKVStore* b = make_store("twoB", 64ull << 20);
        b->StartSession();                    // same thread, fresh store
        make_value(2, 20 + (uint64_t)round, 100, v);
        b->Upsert(2, v);
        CHECK(b->Read(2, out) && check_value(2, out, 100) == 20 + (uint64_t)round,
              "twostores B");
        CHECK(!b->Read(1, out), "twostores leaked state across stores");
        b->StopSession();
        drop_store(b, "twoB");
    }
    fprintf(stderr, "PASS twostores\n");
}

// ── T13: Checkpoint concurrent with writers ──────────────────────────────────
static void test_ckpt() {
    IKVStore* st = make_store("ckpt", 96ull << 20);
    const uint64_t N = 100000 / g_scale;
    std::atomic<bool> stop{false};
    std::vector<std::atomic<uint64_t>> completed(N);
    for (auto& a : completed) a.store(0);
    std::vector<std::thread> ths;
    for (int t = 0; t < 4; ++t) {
        ths.emplace_back([&, t] {
            st->StartSession();
            GenValue v;
            std::mt19937_64 rng(4000 + t);
            uint64_t c = 1;
            while (!stop.load(std::memory_order_relaxed)) {
                uint64_t k = (rng() % (N / 4)) * 4 + (uint64_t)t;   // disjoint
                make_value(k, c, 100, v);
                st->Upsert(k, v);
                completed[k].store(c, std::memory_order_release);
                c++;
            }
            st->StopSession();
        });
    }
    for (int i = 0; i < 3; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        st->Checkpoint();
    }
    stop.store(true);
    for (auto& t : ths) t.join();
    st->StartSession();
    GenValue out;
    uint64_t bad = 0;
    for (uint64_t k = 0; k < N; ++k) {
        uint64_t c = completed[k].load();
        if (c == 0) continue;
        uint64_t got = st->Read(k, out) ? check_value(k, out, 100) : UINT64_MAX;
        if (got == UINT64_MAX || got < c) bad++;
    }
    CHECK(bad == 0, "ckpt: %" PRIu64 " stale/lost", bad);
    st->StopSession();
    drop_store(st, "ckpt");
    fprintf(stderr, "PASS ckpt\n");
}

// ── T14: read-your-write, incl. cross-thread handoff of fresh keys ──────────
static void test_rywrite() {
    IKVStore* st = make_store("ryw", 64ull << 20);
    const uint64_t N = 200000 / g_scale;
    std::atomic<uint64_t> published{0};
    std::atomic<bool> done{false};
    std::thread writer([&] {
        st->StartSession();
        GenValue v, out;
        for (uint64_t i = 1; i <= N; ++i) {
            uint64_t k = mix64(i) | (1ull << 60);
            make_value(k, i, 100, v);
            st->Upsert(k, v);
            CHECK(st->Read(k, out) && check_value(k, out, 100) == i,
                  "ryw own write lost i=%" PRIu64, i);
            published.store(i, std::memory_order_release);
        }
        st->StopSession();
        done.store(true);
    });
    std::thread reader([&] {
        st->StartSession();
        GenValue out;
        std::mt19937_64 rng(5);
        while (!done.load(std::memory_order_acquire)) {
            uint64_t p = published.load(std::memory_order_acquire);
            if (p == 0) continue;
            uint64_t i = 1 + rng() % p;
            uint64_t k = mix64(i) | (1ull << 60);
            GenValue o;
            bool ok = st->Read(k, o);
            CHECK(ok && check_value(k, o, 100) == i,
                  "ryw cross-thread lost i=%" PRIu64, i);
            (void)out;
        }
        st->StopSession();
    });
    writer.join();
    reader.join();
    drop_store(st, "ryw");
    fprintf(stderr, "PASS rywrite\n");
}

// ── child-process modes (crash / rlimit fault injection) ─────────────────────
// Children are spawned fork+exec so the child is a clean single-threaded
// process (fork alone from a threaded test would be undefined-behavior soup).

// --crashchild <dir> <N> <wfd>: load N keys (ctr=1), Checkpoint, signal the
// parent over pipe fd, then keep writing NEW keys (ctr=2) until SIGKILLed.
static int crash_child(const char* dir, uint64_t N, int wfd) {
    IKVStore* st = create_kvstore();
    st->InitExtended(1ull << 20, 16ull << 30, 128ull << 20, dir);
    st->StartSession();
    GenValue v;
    for (uint64_t k = 0; k < N; ++k) { make_value(k, 1, 100, v); st->Upsert(k, v); }
    st->Checkpoint();
    if (write(wfd, "R", 1) != 1) _Exit(3);
    uint64_t k = N;
    for (;;) {
        make_value(k, 2, 100, v);
        st->Upsert(k, v);
        if ((++k & 4095) == 0) st->Checkpoint();
    }
}

// --enospcchild <dir>: run the store against an 8 MB RLIMIT_FSIZE so slot-log
// writes fail with EFBIG (same fail-soft path as ENOSPC), verify the engine
// stays alive/correct/honest, then raise the limit and verify full recovery
// of durability (dirty entries were retained and reflushed).
static int enospc_child(const char* dir) {
    signal(SIGXFSZ, SIG_IGN);   // hit the pwrite EFBIG path, not a signal kill
    struct rlimit rl;
    if (getrlimit(RLIMIT_FSIZE, &rl) != 0) _Exit(3);
    struct rlimit low = rl;
    low.rlim_cur = 8ull << 20;
    if (setrlimit(RLIMIT_FSIZE, &low) != 0) _Exit(3);

    IKVStore* st = create_kvstore();
    st->InitExtended(1ull << 20, 16ull << 30, 512ull << 20, dir);   // cache >> data
    st->StartSession();
    GenValue v, out;
    HydraProdStats ps{};
    uint64_t k = 0;
    while (k < 2000000) {   // write until the disk "fills"
        make_value(k, 1, 100, v);
        st->Upsert(k, v);
        ++k;
        if ((k & 4095) == 0) {
            CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
            if (ps.write_errors > 0) break;
        }
    }
    CHECK(ps.write_errors > 0, "no write error surfaced under RLIMIT_FSIZE");
    CHECK(ps.durable_ok == 0, "durable_ok not sticky after write error");
    // Writes must keep succeeding (in cache / overflow), no crash, no hang.
    uint64_t extra_base = k;
    for (uint64_t i = 0; i < 50000; ++i) {
        make_value(extra_base + i, 1, 100, v);
        st->Upsert(extra_base + i, v);
    }
    uint64_t total = extra_base + 50000;
    // Already-written keys must still read back correctly (cache-resident).
    uint64_t bad = 0;
    for (uint64_t i = 0; i < total; i += 97)
        if (!st->Read(i, out) || check_value(i, out, 100) != 1) bad++;
    CHECK(bad == 0, "enospc: %" PRIu64 " bad reads while disk full", bad);
    st->Checkpoint();   // must not abort or hang; durability stays degraded
    CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    CHECK(ps.durable_ok == 0, "durable_ok flipped back on without cause");
    uint64_t errs_at_full = ps.write_errors;
    CHECK(errs_at_full > 0, "write_errors reset unexpectedly");

    // "Disk repaired": raise the soft limit back; retained dirty state must
    // land on the next Checkpoint and survive a clean restart.
    if (setrlimit(RLIMIT_FSIZE, &rl) != 0) _Exit(3);
    st->Checkpoint();
    struct stat sb;
    std::string f = std::string(dir) + "/hydra_slots.dat";
    CHECK(stat(f.c_str(), &sb) == 0 && (uint64_t)sb.st_size > (8ull << 20),
          "slot log did not grow after limit lift (no reflush)");
    st->StopSession();
    delete st;
    IKVStore* st2 = create_kvstore();
    st2->InitExtended(1ull << 20, 16ull << 30, 512ull << 20, dir);
    st2->StartSession();
    bad = 0;
    for (uint64_t i = 0; i < total; i += 89)
        if (!st2->Read(i, out) || check_value(i, out, 100) != 1) bad++;
    CHECK(bad == 0, "enospc: %" PRIu64 " keys lost after repair+restart", bad);
    st2->StopSession();
    delete st2;
    return g_failures.load() ? 1 : 0;
}

// Spawn "<argv0> <mode> <args...>", return child's exit status (or -1).
static int spawn_child(const std::vector<std::string>& args) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        std::vector<const char*> cargs;
        cargs.push_back(g_argv0);
        for (auto& a : args) cargs.push_back(a.c_str());
        cargs.push_back(nullptr);
        execv(g_argv0, const_cast<char* const*>(cargs.data()));
        _Exit(127);
    }
    int stat = 0;
    if (waitpid(pid, &stat, 0) != pid) return -1;
    return stat;
}

// ── T15: clean-shutdown recovery on the same path ────────────────────────────
static IKVStore* open_store(const char* name, size_t budget, bool wipe) {
    std::string dir = g_parent + "/" + name;
    if (wipe) {
        std::string cmd = "rm -rf " + dir;
        (void)system(cmd.c_str());
    }
    mkdir(dir.c_str(), 0755);
    IKVStore* st = create_kvstore();
    st->InitExtended(1ull << 20, 16ull << 30, budget, dir.c_str());
    return st;
}

static void test_recover() {
    const uint64_t N = 20000 / (uint64_t)g_scale;
    IKVStore* st = open_store("recov", 128ull << 20, true);
    st->StartSession();
    GenValue v, out;
    for (uint64_t i = 0; i < N; ++i) { make_value(i, 1, 100, v); st->Upsert(i, v); }
    for (uint64_t i = 0; i < N; i += 3) { make_value(i, 2, 100, v); st->Upsert(i, v); }
    // oversized values (must be persisted by Checkpoint and recovered)
    const uint64_t OB = 1ull << 40, ONUM = 200;
    for (uint64_t i = 0; i < ONUM; ++i) {
        make_value(OB + i, 1, 300, v);
        st->Upsert(OB + i, v);
    }
    for (uint64_t i = 0; i < ONUM; i += 2) {
        make_value(OB + i, 2, 2000, v);
        st->Upsert(OB + i, v);
    }
    // deletes: inline and oversized, must stay deleted across restart
    for (uint64_t i = 0; i < N; i += 5) st->Delete(i);
    for (uint64_t i = 0; i < ONUM; i += 4) st->Delete(OB + i);
    // size-class transitions
    uint64_t tk1 = OB + 100000, tk2 = OB + 100001;
    make_value(tk1, 5, 300, v); st->Upsert(tk1, v);
    make_value(tk1, 7, 100, v); st->Upsert(tk1, v);    // oversized -> inline
    make_value(tk2, 6, 100, v); st->Upsert(tk2, v);
    make_value(tk2, 8, 400, v); st->Upsert(tk2, v);    // inline -> oversized
    st->Checkpoint();
    st->StopSession();
    delete st;   // clean shutdown

    IKVStore* st2 = open_store("recov", 128ull << 20, false);
    st2->StartSession();
    HydraProdStats ps{};
    CHECK(hydra_get_prod_stats(st2, &ps), "prod stats unavailable");
    CHECK(ps.recover_ok == 1, "recover_ok not set after reopening a log");
    CHECK(ps.recovered_keys > 0, "recovery rebuilt no index entries");
    uint64_t bad = 0;
    for (uint64_t i = 0; i < N; ++i) {
        bool found = st2->Read(i, out);
        if (i % 5 == 0) {
            if (found) { CHECK(false, "recover: deleted %" PRIu64 " resurrected", i); bad++; }
        } else {
            uint64_t want = (i % 3 == 0) ? 2 : 1;
            if (!found || check_value(i, out, 100) != want) bad++;
        }
    }
    CHECK(bad == 0, "recover: %" PRIu64 " inline keys wrong/lost", bad);
    bad = 0;
    for (uint64_t i = 0; i < ONUM; ++i) {
        bool found = read_async1(st2, OB + i, out);
        if (i % 4 == 0) {
            if (found) { CHECK(false, "recover: deleted oversized %" PRIu64 " back", i); bad++; }
        } else if (i % 2 == 0) {
            if (!found || check_value(OB + i, out, 2000) != 2) bad++;
        } else {
            if (!found || check_value(OB + i, out, 300) != 1) bad++;
        }
    }
    CHECK(bad == 0, "recover: %" PRIu64 " oversized keys wrong/lost", bad);
    CHECK(st2->Read(tk1, out) && check_value(tk1, out, 100) == 7, "recover tk1");
    CHECK(st2->Read(tk2, out) && check_value(tk2, out, 400) == 8, "recover tk2");
    CHECK(!st2->Read(N + 424242, out), "recover: absent key found");
    // store must be fully usable post-recovery (writes, LWW, delete)
    make_value(3, 9, 100, v); st2->Upsert(3, v);
    CHECK(st2->Read(3, out) && check_value(3, out, 100) == 9, "recover LWW");
    CHECK(st2->Delete(1), "recover: delete of recovered key");
    CHECK(!st2->Read(1, out), "recover: deleted-after-recovery visible");
    st2->StopSession();
    drop_store(st2, "recov");
    fprintf(stderr, "PASS recover\n");
}

// ── T16: SIGKILL crash recovery with torn-tail detection ─────────────────────
static void test_crashrecover() {
    const uint64_t N = 30000 / (uint64_t)g_scale;
    std::string dir = g_parent + "/crashrec";
    { std::string cmd = "rm -rf " + dir; (void)system(cmd.c_str()); }
    mkdir(dir.c_str(), 0755);
    int p[2];
    CHECK(pipe(p) == 0, "pipe failed");
    pid_t pid = fork();
    if (pid == 0) {
        close(p[0]);
        char nbuf[32], fdbuf[16];
        snprintf(nbuf, sizeof nbuf, "%" PRIu64, N);
        snprintf(fdbuf, sizeof fdbuf, "%d", p[1]);
        execl(g_argv0, g_argv0, "--crashchild", dir.c_str(), nbuf, fdbuf,
              (char*)nullptr);
        _Exit(127);
    }
    CHECK(pid > 0, "fork failed");
    close(p[1]);
    char c = 0;
    CHECK(read(p[0], &c, 1) == 1 && c == 'R', "crash child never checkpointed");
    close(p[0]);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    kill(pid, SIGKILL);
    int stat = 0;
    waitpid(pid, &stat, 0);
    CHECK(WIFSIGNALED(stat) && WTERMSIG(stat) == SIGKILL, "child not SIGKILLed");

    IKVStore* st = open_store("crashrec", 128ull << 20, false);
    st->StartSession();
    HydraProdStats ps{};
    CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    CHECK(ps.recover_ok == 1, "crash recovery did not run");
    GenValue out;
    uint64_t bad = 0;
    for (uint64_t k = 0; k < N; ++k)
        if (!st->Read(k, out) || check_value(k, out, 100) != 1) bad++;
    CHECK(bad == 0, "crashrecover: %" PRIu64 " checkpointed keys lost", bad);
    // Post-checkpoint keys are best-effort, but anything visible must be an
    // EXACT complete value (torn slots must have been CRC-skipped).
    uint64_t seen = 0;
    for (uint64_t k = N; k < N + 300000; ++k) {
        if (!st->Read(k, out)) continue;
        seen++;
        CHECK(check_value(k, out, 100) == 2,
              "crashrecover: torn/corrupt value served k=%" PRIu64, k);
    }
    fprintf(stderr, "  crashrecover: %" PRIu64 " post-ckpt keys survived, "
            "%" PRIu64 " torn slots skipped\n", seen, ps.recover_torn_slots);
    st->StopSession();
    drop_store(st, "crashrec");
    fprintf(stderr, "PASS crashrecover\n");
}

// ── T17: disk-full (EFBIG/ENOSPC-class) fail-soft ────────────────────────────
static void test_enospc() {
    std::string dir = g_parent + "/enospc";
    { std::string cmd = "rm -rf " + dir; (void)system(cmd.c_str()); }
    mkdir(dir.c_str(), 0755);
    int stat = spawn_child({"--enospcchild", dir});
    CHECK(stat >= 0, "spawn failed");
    CHECK(WIFEXITED(stat), "enospc child crashed (sig %d) — engine aborted",
          WIFSIGNALED(stat) ? WTERMSIG(stat) : 0);
    CHECK(WIFEXITED(stat) && WEXITSTATUS(stat) == 0,
          "enospc child failed (exit %d)", WEXITSTATUS(stat));
    { std::string cmd = "rm -rf " + dir; (void)system(cmd.c_str()); }
    fprintf(stderr, "PASS enospc\n");
}

// ── T18: external log truncation — reads fail soft, never abort ─────────────
static void test_readfault() {
    const uint64_t N = 400000 / (uint64_t)g_scale;
    IKVStore* st = open_store("rdfault", 32ull << 20, true);   // tiny cache
    st->StartSession();
    GenValue v, out;
    for (uint64_t i = 0; i < N; ++i) { make_value(i, 1, 100, v); st->Upsert(i, v); }
    st->Checkpoint();
    std::string f = g_parent + "/rdfault/hydra_slots.dat";
    struct stat sb;
    CHECK(stat(f.c_str(), &sb) == 0, "slot log missing");
    // Cut the file mid-way at a NON-page-aligned offset: forces short reads
    // below the landed high-water mark on the miss path.
    CHECK(truncate(f.c_str(), sb.st_size / 2 + 100) == 0, "truncate failed");
    uint64_t found = 0, lost = 0;
    for (uint64_t i = 0; i < N; i += 3) {
        if (st->Read(i, out)) {
            found++;
            CHECK(check_value(i, out, 100) == 1,
                  "readfault: corrupt value served i=%" PRIu64, i);
        } else {
            lost++;
        }
    }
    // Async path across the truncated region too.
    for (uint64_t i = 1; i < N; i += 101) {
        if (read_async1(st, i, out))
            CHECK(check_value(i, out, 100) == 1, "readfault async corrupt");
    }
    HydraProdStats ps{};
    CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    CHECK(ps.read_errors > 0, "read_errors did not increment on truncation");
    CHECK(lost > 0, "truncation had no effect (test setup broken?)");
    CHECK(found > 0, "intact half fully lost");
    // Engine must remain writable and readable after read faults.
    make_value(N + 7, 3, 100, v);
    st->Upsert(N + 7, v);
    CHECK(st->Read(N + 7, out) && check_value(N + 7, out, 100) == 3,
          "readfault: store unusable after fault");
    st->StopSession();
    drop_store(st, "rdfault");
    fprintf(stderr, "PASS readfault\n");
}

// ── T19: bounded oversize map with honest rejection ──────────────────────────
static void test_oversizebound() {
    // Oversized values are DISK-resident (x-records in the slot log), so a
    // dataset far past the old RAM cap — and past the whole memory budget —
    // must be fully accepted with ZERO rejections and ZERO memory-cap use.
    // (The July-21 workload sweep proved the old capped-in-RAM design
    // silently dropped every oversized upsert past ~budget/8: 100% stale
    // reads at KVSTORE_VALUE_SIZE=1024, millions of false NotFounds on the
    // bimodal mix. This test pins the new contract.)
    const size_t budget = 64ull << 20;
    const uint64_t N = 30000;                    // 30K x 4 KB = 117 MB > budget
    IKVStore* st = open_store("osbound", budget, true);
    st->StartSession();
    GenValue v, out;
    HydraProdStats ps{};
    for (uint64_t i = 0; i < N; ++i) {
        make_value(i, 1, 4096, v);
        st->Upsert(i, v);
    }
    CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    CHECK(ps.rejected_oversize == 0,
          "healthy-disk oversized upserts rejected: %" PRIu64,
          ps.rejected_oversize);
    CHECK(ps.oversize_bytes <= budget / 8 + 8192,
          "x-records charged against the RAM cap: %" PRIu64, ps.oversize_bytes);
    uint64_t bad = 0;
    for (uint64_t k = 0; k < N; ++k)
        if (!st->Read(k, out) || check_value(k, out, 4096) != 1) bad++;
    CHECK(bad == 0, "oversizebound: %" PRIu64 "/%" PRIu64 " values damaged",
          bad, N);
    // Cross-thread visibility of oversized OVERWRITES (the exact audit
    // failure: writer-then-reader in different sessions must see the new
    // counter, never the stale one), including via the async path.
    for (uint64_t k = 0; k < 512; ++k) {
        make_value(k, 2, 1024, v);
        st->Upsert(k, v);
    }
    std::thread other([&] {
        st->StartSession();
        GenValue o;
        for (uint64_t k = 0; k < 512; ++k) {
            CHECK(st->Read(k, o) && check_value(k, o, 1024) == 2,
                  "cross-session oversized overwrite invisible k=%" PRIu64, k);
            CHECK(read_async1(st, k, o) && check_value(k, o, 1024) == 2,
                  "cross-session async oversized stale k=%" PRIu64, k);
        }
        st->StopSession();
    });
    other.join();
    // Delete an oversized key: gone now...
    CHECK(st->Delete(7), "oversized delete");
    CHECK(!st->Read(7, out), "deleted oversized key readable");
    // RMW on an oversized key produces an oversized result via the slow path
    uint8_t mod[2048];
    make_value(11, 0, 2048, v);
    memcpy(mod, v.data, 2048);
    st->RMW(11, mod, 2048);
    CHECK(st->Read(11, out) && out.size == 2048, "oversized RMW lost");
    // ...and across a clean restart (x-records + x-tombstones must recover:
    // length-aware scan, newest-LSN-wins, delete registry re-mark).
    st->Checkpoint();
    st->StopSession();
    delete st;
    st = open_store("osbound", budget, false);
    st->StartSession();
    CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    CHECK(ps.recover_ok == 1, "oversized recovery degraded");
    bad = 0;
    for (uint64_t k = 0; k < N; ++k) {
        if (k == 7 || k == 11) continue;
        uint32_t sz = k < 512 ? 1024 : 4096;
        uint64_t want = k < 512 ? 2 : 1;
        if (!st->Read(k, out) || check_value(k, out, sz) != want) bad++;
    }
    CHECK(bad == 0, "oversizebound: %" PRIu64 " values lost across restart",
          bad);
    CHECK(!st->Read(7, out), "deleted oversized key resurrected by recovery");
    CHECK(st->Read(11, out) && out.size == 2048, "RMW value lost by recovery");
    // Inline<->oversized flapping after recovery keeps LWW.
    for (uint64_t k = 0; k < 256; ++k) {
        make_value(k, 3, 100, v);  st->Upsert(k, v);
        make_value(k, 4, 4096, v); st->Upsert(k, v);
        make_value(k, 5, 60, v);   st->Upsert(k, v);
        CHECK(st->Read(k, out) && check_value(k, out, 60) == 5,
              "flap after recovery k=%" PRIu64, k);
    }
    st->StopSession();
    drop_store(st, "osbound");
    fprintf(stderr, "PASS oversizebound\n");
}

// ── T20: log compaction under concurrent readers ─────────────────────────────
static void test_compact() {
    setenv("HYDRA_COMPACT_FLOOR_MB", "4", 1);
    setenv("HYDRA_COMPACT_FACTOR", "2", 1);
    IKVStore* st = open_store("compact", 64ull << 20, true);
    unsetenv("HYDRA_COMPACT_FLOOR_MB");
    unsetenv("HYDRA_COMPACT_FACTOR");
    st->StartSession();
    const uint64_t K = 4096;
    const uint64_t ROUNDS = (uint64_t)(g_scale > 1 ? 120 : 400);
    std::vector<std::atomic<uint64_t>> comp(K), issue(K);
    for (uint64_t k = 0; k < K; ++k) { comp[k].store(0); issue[k].store(0); }
    std::atomic<bool> stop{false};
    std::vector<std::thread> rd;
    for (int r = 0; r < 2; ++r) {
        rd.emplace_back([&, r] {
            st->StartSession();
            std::mt19937_64 rng(9000 + r);
            GenValue o;
            while (!stop.load(std::memory_order_relaxed)) {
                uint64_t k = rng() % K;
                uint64_t lo = comp[k].load(std::memory_order_acquire);
                if (lo == 0) continue;
                bool ok = st->Read(k, o);
                uint64_t hi = issue[k].load(std::memory_order_acquire);
                CHECK(ok, "compact: key %" PRIu64 " lost mid-compaction", k);
                if (ok) {
                    uint64_t c = check_value(k, o, 100);
                    CHECK(c >= lo && c <= hi,
                          "compact STALE k=%" PRIu64 " c=%" PRIu64
                          " lo=%" PRIu64 " hi=%" PRIu64, k, c, lo, hi);
                }
            }
            st->StopSession();
        });
    }
    GenValue v;
    std::string f = g_parent + "/compact/hydra_slots.dat";
    struct stat sb;
    uint64_t peak_blocks = 0, peak_log = 0;
    HydraProdStats ps{};
    for (uint64_t rnd = 1; rnd <= ROUNDS; ++rnd) {
        for (uint64_t k = 0; k < K; ++k) {
            issue[k].store(rnd, std::memory_order_release);
            make_value(k, rnd, 100, v);
            st->Upsert(k, v);
            comp[k].store(rnd, std::memory_order_release);
        }
        st->Checkpoint();   // force every version to land (log grows)
        if (stat(f.c_str(), &sb) == 0 && (uint64_t)sb.st_blocks > peak_blocks)
            peak_blocks = (uint64_t)sb.st_blocks;
        CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
        if (ps.log_bytes > peak_log) peak_log = ps.log_bytes;
    }
    // give the compactor time to catch up, then settle
    for (int i = 0; i < 600 && ps.compactions_run == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    }
    stop.store(true);
    for (auto& t : rd) t.join();
    CHECK(ps.compactions_run > 0, "compaction never triggered (log %.1f MB, "
          "live %.1f MB)", ps.log_bytes / 1e6, ps.live_bytes / 1e6);
    CHECK(ps.reclaimed_bytes > 0, "compaction reclaimed nothing");
    // On-disk footprint must shrink from its peak (hole punch), or — where
    // punching is unsupported — allocated blocks must stop tracking the
    // logical write volume (extent reuse). Poll: the compactor is async.
    uint64_t now_blocks = peak_blocks;
    for (int i = 0; i < 600; ++i) {
        CHECK(stat(f.c_str(), &sb) == 0, "slot log missing");
        now_blocks = (uint64_t)sb.st_blocks;
        CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
        if (ps.punch_unsupported == 0 && now_blocks < (peak_blocks * 3) / 4) break;
        if (ps.punch_unsupported == 1 && ps.reclaimed_bytes > peak_log / 4) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (ps.punch_unsupported == 0)
        CHECK(now_blocks < (peak_blocks * 3) / 4,
              "st_blocks did not shrink: peak=%" PRIu64 " now=%" PRIu64,
              peak_blocks, now_blocks);
    // Position space is being recycled (the 512 GiB / 32-bit ceiling fix):
    CHECK(ps.log_bytes < peak_log + (16ull << 20),
          "log_bytes still growing without bound");
    // full correctness after compaction settles
    GenValue out;
    uint64_t bad = 0;
    for (uint64_t k = 0; k < K; ++k)
        if (!st->Read(k, out) || check_value(k, out, 100) != ROUNDS) bad++;
    CHECK(bad == 0, "compact: %" PRIu64 " keys wrong after compaction", bad);
    // ...and after a restart on the compacted (holey) log
    st->Checkpoint();
    st->StopSession();
    delete st;
    IKVStore* st2 = open_store("compact", 64ull << 20, false);
    st2->StartSession();
    bad = 0;
    for (uint64_t k = 0; k < K; ++k)
        if (!st2->Read(k, out) || check_value(k, out, 100) != ROUNDS) bad++;
    CHECK(bad == 0, "compact: %" PRIu64 " keys lost across restart", bad);
    st2->StopSession();
    drop_store(st2, "compact");
    fprintf(stderr, "PASS compact\n");
}

// ── T21: compaction of UNCACHED keys + deletes + oversize sidecar lifecycle ─
// A tiny budget (8 MiB -> ~32k cache entries) with 60k keys forces most keys
// out of cache, so the compactor must take its uncached-relocation path
// (verified_lookup + CLEAN admission) instead of the cache-resident one that
// test_compact exercises. Also covers: dead-slot skips for deleted keys and
// keys that moved to the overflow map, HYDRA_OVERSIZE_CAP_MB, recovery on a
// compacted log with a sidecar, and sidecar unlink once the last oversized
// key is deleted.
static void test_compactcold() {
    setenv("HYDRA_COMPACT_FLOOR_MB", "4", 1);
    setenv("HYDRA_COMPACT_FACTOR", "2", 1);
    setenv("HYDRA_OVERSIZE_CAP_MB", "1", 1);
    IKVStore* st = open_store("compactcold", 8ull << 20, true);
    unsetenv("HYDRA_COMPACT_FLOOR_MB");
    unsetenv("HYDRA_COMPACT_FACTOR");
    unsetenv("HYDRA_OVERSIZE_CAP_MB");
    st->StartSession();
    const uint64_t K = 60000;
    const uint64_t ROUNDS = 3;
    GenValue v, out;
    for (uint64_t rnd = 1; rnd <= ROUNDS; ++rnd) {
        for (uint64_t k = 0; k < K; ++k) {
            make_value(k, rnd, 100, v);
            st->Upsert(k, v);
        }
        st->Checkpoint();   // land every version: dead versions pile up
    }
    // Move keys 0..31 to oversized x-records (their landed inline slots
    // become dead versions superseded by the x-records' higher LSNs; the
    // compactor sees them lose the verified_lookup race). Disk-resident
    // x-records write NO overflow sidecar — that is the fixed contract
    // (the old capped-RAM overflow map silently dropped oversized data).
    for (uint64_t k = 0; k < 32; ++k) {
        make_value(k, ROUNDS + 1, 2048, v);
        st->Upsert(k, v);
    }
    st->Checkpoint();
    std::string side = g_parent + "/compactcold/hydra_overflow.dat";
    struct stat sb;
    CHECK(stat(side.c_str(), &sb) != 0,
          "x-resident oversized values must not create an overflow sidecar");
    // Tombstone a slice (compactor must skip deleted keys' old slots and
    // relocate their tombstones instead of dropping them).
    HydraProdStats ps{};
    CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    uint64_t comp0 = ps.compactions_run;
    for (uint64_t k = 32; k < K; k += 16)
        CHECK(st->Delete(k), "compactcold delete failed");
    st->Checkpoint();
    // Wait for at least one compaction pass that RAN AFTER the deletes, so
    // the deleted keys' slots/tombstones are actually processed.
    for (int i = 0; i < 600 && ps.compactions_run <= comp0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(hydra_get_prod_stats(st, &ps), "prod stats unavailable");
    }
    CHECK(ps.compactions_run > comp0, "cold compaction never ran post-delete "
          "(log %.1f MB, live %.1f MB)", ps.log_bytes / 1e6, ps.live_bytes / 1e6);
    CHECK(ps.reclaimed_bytes > 0, "cold compaction reclaimed nothing");
    // Full correctness while/after compaction: overflow keys, deleted keys,
    // and plain keys all read their exact last state.
    uint64_t bad = 0;
    for (uint64_t k = 0; k < K; ++k) {
        bool deleted = (k >= 32 && (k - 32) % 16 == 0);
        bool oversized = k < 32;
        bool ok = st->Read(k, out);
        if (deleted) { if (ok) bad++; continue; }
        if (!ok) { bad++; continue; }
        uint64_t c = check_value(k, out, oversized ? 2048 : 100);
        if (c != (oversized ? ROUNDS + 1 : ROUNDS)) bad++;
    }
    CHECK(bad == 0, "compactcold: %" PRIu64 " keys wrong after compaction", bad);
    st->Checkpoint();
    st->StopSession();
    delete st;
    // Restart on the compacted log + sidecar: everything must survive.
    IKVStore* st2 = open_store("compactcold", 8ull << 20, false);
    st2->StartSession();
    bad = 0;
    for (uint64_t k = 0; k < K; ++k) {
        bool deleted = (k >= 32 && (k - 32) % 16 == 0);
        bool oversized = k < 32;
        bool ok = st2->Read(k, out);
        uint64_t c = ok ? check_value(k, out, oversized ? 2048 : 100)
                        : UINT64_MAX;
        bool wrong = deleted ? ok
                             : (!ok || c != (oversized ? ROUNDS + 1 : ROUNDS));
        if (wrong && bad < 8)
            fprintf(stderr, "compactcold BAD k=%" PRIu64 " del=%d ovsz=%d "
                    "ok=%d size=%u ctr=%" PRIu64 "\n", k, (int)deleted,
                    (int)oversized, (int)ok, ok ? out.size : 0, c);
        if (wrong) bad++;
    }
    CHECK(bad == 0, "compactcold: %" PRIu64 " keys wrong after restart", bad);
    // Deleting EVERY remaining key empties the overflow map (the engine may
    // have absorbed inline keys into it under pin pressure — a documented
    // last-resort path — so deleting only the oversized keys is not enough).
    // An empty map must unlink the sidecar at the next Checkpoint.
    for (uint64_t k = 0; k < 32; ++k)
        CHECK(st2->Delete(k), "compactcold oversized delete failed");
    for (uint64_t k = 32; k < K; ++k) {
        bool deleted = (k >= 32 && (k - 32) % 16 == 0);
        if (!deleted) st2->Delete(k);
    }
    st2->Checkpoint();
    CHECK(stat(side.c_str(), &sb) != 0, "empty overflow sidecar not unlinked");
    st2->StopSession();
    delete st2;
    // Generation 3: every key was deleted in generation 2 — after ANOTHER
    // restart they must all stay deleted (recovery re-marks tombstone
    // winners, and compaction relocates rather than drops the tombstones
    // of still-deleted keys, so no stale copy can win the LSN race).
    IKVStore* st3 = open_store("compactcold", 8ull << 20, false);
    st3->StartSession();
    bad = 0;
    for (uint64_t k = 0; k < K; ++k)
        if (st3->Read(k, out)) bad++;
    CHECK(bad == 0,
          "compactcold: %" PRIu64 " deleted keys resurrected in gen 3", bad);
    st3->StopSession();
    drop_store(st3, "compactcold");
    fprintf(stderr, "PASS compactcold\n");
}

// ── bimodal: mixed inline/x-record sizes under concurrent write+read ────────
// Regression for the reference-node bimodal integrity failure: 20 B (inline)
// and 200 B (x-record) values by key class, tiny budget => heavy spill, so
// readers constantly walk disk chains INCLUDING the log's unaligned tail
// page while x-appends keep moving i_size. Before the x-extent read routing
// fix, O_DIRECT preads of that tail page failed with EINVAL on ext4 (i_size
// not block-aligned) and every fault surfaced as a false NotFound (~2K per
// 30 s scored-node run) or a stale read. Keys always exist here: NotFound is
// a hard failure; counters must be fresh (>= published). Also verifies the
// full state across a restart.
static void test_bimodal() {
    const uint64_t N = std::max<uint64_t>(300000 / g_scale, 1000);
    const int W = 4, R = 4;
    const uint64_t rounds = 4;
    // Force the compactor to run concurrently with x-extent reservations
    // (regression stress for the reservation-vs-compactor atomicity: a
    // punched just-reserved extent shows up here as corrupt/NotFound).
    setenv("HYDRA_COMPACT_FLOOR_MB", "8", 1);
    setenv("HYDRA_COMPACT_FACTOR", "1.1", 1);
    IKVStore* st = make_store("bimodal", 48ull << 20);   // tiny: heavy spill
    unsetenv("HYDRA_COMPACT_FLOOR_MB");
    unsetenv("HYDRA_COMPACT_FACTOR");
    st->StartSession();
    auto vsz = [](uint64_t k) -> uint32_t { return k % 10 == 0 ? 200 : 20; };
    {
        GenValue v;
        for (uint64_t i = 0; i < N; ++i) {
            make_value(i, 1, vsz(i), v);
            st->Upsert(i, v);
        }
        st->StopSession();
    }
    // pub[i] = highest counter whose Upsert has RETURNED for key i.
    std::vector<std::atomic<uint32_t>> pub(N);
    for (uint64_t i = 0; i < N; ++i) pub[i].store(1, std::memory_order_relaxed);
    std::atomic<uint64_t> nf{0}, stale{0}, bad{0};
    // Path-coverage census: the concurrent phase must exercise sync AND
    // async reads over inline AND x-record keys (vacuous-pass guard).
    std::atomic<uint64_t> rd_sync{0}, rd_async{0}, rd_x{0}, rd_in{0};
    std::atomic<bool> wdone{false};
    std::vector<std::thread> ths;
    for (int w = 0; w < W; ++w) {
        ths.emplace_back([&, w] {
            st->StartSession();
            GenValue v;
            for (uint64_t r = 2; r <= rounds; ++r)
                for (uint64_t i = (uint64_t)w; i < N; i += W) {
                    make_value(i, r, vsz(i), v);
                    st->Upsert(i, v);
                    pub[i].store((uint32_t)r, std::memory_order_release);
                }
            st->StopSession();
        });
    }
    for (int t = 0; t < R; ++t) {
        ths.emplace_back([&, t] {
            st->StartSession();
            GenValue out;
            uint64_t x = 0x9E3779B97F4A7C15ULL * (uint64_t)(t + 1);
            uint64_t myreads = 0;
            // Keep reading past writer completion until a minimum op count:
            // the checks must never pass vacuously on a scheduler that runs
            // the writers to completion before any reader starts.
            while (!wdone.load(std::memory_order_acquire) || myreads < 2000) {
                ++myreads;
                x = mix64(x + t + 1);
                uint64_t i = x % N;
                uint32_t c0 = pub[i].load(std::memory_order_acquire);
                bool sync = (x >> 32) & 1;
                bool ok = sync ? st->Read(i, out) : read_async1(st, i, out);
                if (!ok) { nf.fetch_add(1); continue; }
                (sync ? rd_sync : rd_async).fetch_add(1);
                (i % 10 == 0 ? rd_x : rd_in).fetch_add(1);
                uint64_t c = check_value(i, out, vsz(i));
                if (c == UINT64_MAX) bad.fetch_add(1);
                else if (c < c0) stale.fetch_add(1);
                else if (c > rounds) bad.fetch_add(1);
            }
            st->StopSession();
        });
    }
    for (int w = 0; w < W; ++w) ths[(size_t)w].join();
    wdone.store(true, std::memory_order_release);
    for (size_t t = W; t < ths.size(); ++t) ths[t].join();
    CHECK(nf.load() == 0, "bimodal: %" PRIu64 " false NotFounds", nf.load());
    CHECK(stale.load() == 0, "bimodal: %" PRIu64 " stale reads", stale.load());
    CHECK(bad.load() == 0, "bimodal: %" PRIu64 " corrupt reads", bad.load());
    CHECK(rd_sync.load() > 0 && rd_async.load() > 0 && rd_x.load() > 0 &&
          rd_in.load() > 0,
          "bimodal: vacuous run sync=%" PRIu64 " async=%" PRIu64 " x=%" PRIu64
          " inline=%" PRIu64, rd_sync.load(), rd_async.load(), rd_x.load(),
          rd_in.load());
    // Full sequential verify, then across a restart.
    st->StartSession();
    GenValue out;
    uint64_t miss = 0;
    for (uint64_t i = 0; i < N; ++i)
        if (!st->Read(i, out) || check_value(i, out, vsz(i)) != rounds) miss++;
    CHECK(miss == 0, "bimodal: %" PRIu64 " wrong before restart", miss);
    st->StopSession();
    delete st;
    IKVStore* st2 = open_store("bimodal", 48ull << 20, false);
    st2->StartSession();
    for (uint64_t i = 0; i < N; ++i) {
        bool ok = st2->Read(i, out);
        uint64_t c = ok ? check_value(i, out, vsz(i)) : 0;
        if (!ok || c != rounds) {
            if (miss < 5)
                fprintf(stderr, "bimodal DBG k=%" PRIu64 " sz=%u ok=%d ctr=%"
                        PRIu64 " outsz=%u\n", i, vsz(i), (int)ok, c,
                        ok ? out.size : 0);
            miss++;
        }
    }
    CHECK(miss == 0, "bimodal: %" PRIu64 " wrong after restart", miss);
    st2->StopSession();
    drop_store(st2, "bimodal");
    fprintf(stderr, "PASS bimodal\n");
}

// ── capwarn: init-time RAM-index capacity warning ────────────────────────────
// A hash_table_size hint that cannot fit the budget-sized fingerprint index
// must produce a LOUD init-time warning (a capacity-planning mistake would
// otherwise surface only as under-load rejections); a fitting hint must
// stay silent. Captures the store's real stderr via dup2 — no mocks.
static void test_capwarn() {
    std::string dir = g_parent + "/capwarn";
    std::string cmd = "rm -rf " + dir + " && mkdir -p " + dir;
    CHECK(system(cmd.c_str()) == 0, "capwarn mkdir failed");
    std::string log = g_parent + "/capwarn_err.txt";
    for (int big = 1; big >= 0; --big) {
        fflush(stderr);
        int saved = dup(2);
        CHECK(saved >= 0, "capwarn dup failed");
        FILE* lf = fopen(log.c_str(), "w");
        CHECK(lf != nullptr, "capwarn log open failed");
        dup2(fileno(lf), 2);
        IKVStore* st = create_kvstore();
        // 16 MiB budget => index capacity ~508K keys (2^19 words at ~97%
        // fill): a 4M-key hint must warn, a 1K-key hint must not.
        st->InitExtended(big ? (1ull << 22) : (1ull << 10), 16ull << 30,
                         16ull << 20, dir.c_str());
        delete st;
        fflush(stderr);
        dup2(saved, 2);
        close(saved);
        fclose(lf);
        std::string all;
        FILE* rf = fopen(log.c_str(), "r");
        CHECK(rf != nullptr, "capwarn log reopen failed");
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, rf)) > 0) all.append(buf, n);
        fclose(rf);
        bool warned =
            all.find("WARNING: hash_table_size") != std::string::npos;
        CHECK(warned == (big == 1), "capwarn: warning %s (big=%d)",
              warned ? "unexpected" : "missing", big);
    }
    cmd = "rm -rf " + dir + " " + log;
    (void)system(cmd.c_str());
    fprintf(stderr, "PASS capwarn\n");
}

// ── T28: delete crash-durability (July-21 audit Bug 5) ───────────────────────
// A Delete() that returned must survive a SIGKILL BEFORE any Checkpoint: the
// old async-reaper design left the on-disk tombstones to background work, so
// recovery resurrected durably-checkpointed values (audit: 60K of 97K keys).
// The delete-intent log (dlog) must suppress that for inline AND oversized
// (x-record) values, keep checkpointed reinserts alive, stay correct across
// a SECOND crash (the dlog survives until a quiescent Checkpoint), and be
// truncated once tombstones are durable.
static constexpr uint64_t kDcN  = 30000;         // inline keys
static constexpr uint64_t kDcOB = 1ull << 41;    // oversized key base
static constexpr uint64_t kDcON = 400;           // oversized keys (300 B)

// --delcrashchild <dir> <phase> <wfd>
static int delcrash_child(const char* dir, int phase, int wfd) {
    // Pause the background/cleaner reaper drains: only forced drains
    // (Checkpoint, destructor) may write tombstones. Post-checkpoint
    // deletes therefore depend PROVABLY on the dlog for crash durability —
    // the test can no longer pass by reaper luck (review item m14).
    setenv("HYDRA_REAP_PAUSE", "1", 1);
    IKVStore* st = create_kvstore();
    st->InitExtended(1ull << 20, 16ull << 30, 256ull << 20, dir);
    st->StartSession();
    GenValue v;
    if (phase == 1) {
        for (uint64_t k = 0; k < kDcN; ++k) {
            make_value(k, 1, 100, v);
            st->Upsert(k, v);
        }
        for (uint64_t i = 0; i < kDcON; ++i) {
            make_value(kDcOB + i, 1, 300, v);
            st->Upsert(kDcOB + i, v);
        }
        st->Checkpoint();                        // everything durable
        for (uint64_t k = 0; k < kDcN; k += 2) st->Delete(k);
        for (uint64_t i = 0; i < kDcON; i += 2) st->Delete(kDcOB + i);
        for (uint64_t k = 0; k < kDcN; k += 4) {         // reinsert subset
            make_value(k, 2, 100, v);
            st->Upsert(k, v);
        }
        st->Checkpoint();       // tombstones + reinserts durable, dlog reset
        // Post-checkpoint deletes: crash-durability rests on the dlog only.
        for (uint64_t k = 0; k < kDcN; k += 8) st->Delete(k);
        for (uint64_t i = 1; i < kDcON; i += 4) st->Delete(kDcOB + i);
        // Oversized reinsert AFTER a post-ckpt delete: the x-record writes
        // through with an LSN above the intent's, so recovery must keep it
        // (fresher-upsert-beats-intent, oversized facet).
        for (uint64_t i = 1; i < kDcON; i += 8) {
            make_value(kDcOB + i, 2, 300, v);
            st->Upsert(kDcOB + i, v);
        }
    } else {                                     // phase 2: second crash
        // Reinsert gen-1-DELETED keys RACING a Checkpoint (reviewer
        // R2-C2): each reinsert unmarks its key and may publish an
        // S_DIRTY overwrite into a set the checkpoint stripe already
        // scanned, no-op'ing the queued poison while the new value is
        // not itself durable. Legal post-crash states for these keys:
        // the reinsert (ctr=9) or NotFound (intent applied) — NEVER the
        // original pre-delete value (ctr=1/2).
        std::thread ck([&] { st->Checkpoint(); });
        for (uint64_t k = 0; k < kDcN; k += 16) {
            make_value(k, 9, 100, v);
            st->Upsert(k, v);
        }
        ck.join();
        for (uint64_t k = 1; k < kDcN; k += 10) st->Delete(k);
    }
    if (write(wfd, "R", 1) != 1) _Exit(3);
    for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
}

static int delcrash_spawn(const std::string& dir, int phase) {
    int p[2];
    if (pipe(p) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        close(p[0]);
        char ph[8], fdbuf[16];
        snprintf(ph, sizeof ph, "%d", phase);
        snprintf(fdbuf, sizeof fdbuf, "%d", p[1]);
        execl(g_argv0, g_argv0, "--delcrashchild", dir.c_str(), ph, fdbuf,
              (char*)nullptr);
        _Exit(127);
    }
    close(p[1]);
    char c = 0;
    bool ready = read(p[0], &c, 1) == 1 && c == 'R';
    close(p[0]);
    if (!ready) { kill(pid, SIGKILL); waitpid(pid, nullptr, 0); return -1; }
    kill(pid, SIGKILL);
    int stat = 0;
    waitpid(pid, &stat, 0);
    return WIFSIGNALED(stat) && WTERMSIG(stat) == SIGKILL ? 0 : -1;
}

static void test_delcrash() {
    std::string dir = g_parent + "/delcrash";
    std::string cmd = "rm -rf " + dir + " && mkdir -p " + dir;
    CHECK(system(cmd.c_str()) == 0, "mkdir failed");
    CHECK(delcrash_spawn(dir, 1) == 0, "phase-1 child failed");

    IKVStore* st = open_store("delcrash", 256ull << 20, false);
    st->StartSession();
    GenValue out;
    uint64_t resur = 0, lost = 0, badctr = 0;
    for (uint64_t k = 0; k < kDcN; ++k) {
        bool found = st->Read(k, out);
        if (k % 8 == 0 || (k % 2 == 0 && k % 4 != 0)) {
            if (found) resur++;                  // deleted: must stay dead
        } else if (k % 4 == 0) {                 // reinserted + checkpointed
            if (!found) lost++;
            else if (check_value(k, out, 100) != 2) badctr++;
        } else {                                 // odd: untouched
            if (!found) lost++;
            else if (check_value(k, out, 100) != 1) badctr++;
        }
    }
    uint64_t xresur = 0, xlost = 0;
    for (uint64_t i = 0; i < kDcON; ++i) {
        bool found = st->Read(kDcOB + i, out);
        bool dead = i % 2 == 0 || (i % 4 == 1 && i % 8 != 1);
        if (dead) {
            if (found) xresur++;
        } else {
            int expect = (i % 8 == 1) ? 2 : 1;  // reinserted-after-delete
            if (!found || check_value(kDcOB + i, out, 300) != expect)
                xlost++;
        }
    }
    CHECK(resur == 0, "crash resurrected %" PRIu64 " deleted inline keys "
          "(Bug 5)", resur);
    CHECK(xresur == 0, "crash resurrected %" PRIu64 " deleted oversized "
          "keys (Bug 5 x-record facet)", xresur);
    CHECK(lost == 0 && badctr == 0, "lost=%" PRIu64 " badctr=%" PRIu64
          " live inline keys after crash", lost, badctr);
    CHECK(xlost == 0, "lost %" PRIu64 " live oversized keys", xlost);
    st->StopSession();
    delete st;   // clean close (destructor Checkpoint may truncate the dlog)

    // Second crash generation: deletes on the RECOVERED store, no Checkpoint.
    CHECK(delcrash_spawn(dir, 2) == 0, "phase-2 child failed");
    st = open_store("delcrash", 256ull << 20, false);
    st->StartSession();
    resur = 0; lost = 0;
    for (uint64_t k = 0; k < kDcN; ++k) {
        bool found = st->Read(k, out);
        bool gen1_dead = k % 8 == 0 || (k % 2 == 0 && k % 4 != 0);
        bool gen2_dead = k % 2 == 1 && k % 10 == 1;
        if (k % 16 == 0) {
            // Reinserted racing the phase-2 Checkpoint: ctr=9 or NotFound
            // are legal; the pre-delete value resurfacing is not (R2-C2).
            if (found && check_value(k, out, 100) != 9) resur++;
        } else if (gen1_dead || gen2_dead) { if (found) resur++; }
        else if (!found) lost++;
    }
    CHECK(resur == 0, "%" PRIu64 " deleted keys resurrected after SECOND "
          "crash", resur);
    CHECK(lost == 0, "%" PRIu64 " live keys lost after second crash", lost);

    // Quiescent Checkpoint must truncate the dlog (tombstones now durable).
    st->Checkpoint();
    struct stat sb;
    std::string dlog = dir + "/hydra_dlog.dat";
    CHECK(stat(dlog.c_str(), &sb) == 0 && sb.st_size == 0,
          "dlog not truncated by quiescent Checkpoint (size=%lld)",
          (long long)(stat(dlog.c_str(), &sb) == 0 ? sb.st_size : -1));

    // Delete + reinsert + clean restart: the intent must NOT outlive the
    // fresher reinsert (LSN ordering).
    GenValue v;
    st->Delete(3);
    make_value(3, 7, 100, v);
    st->Upsert(3, v);
    st->StopSession();
    delete st;
    st = open_store("delcrash", 256ull << 20, false);
    st->StartSession();
    CHECK(st->Read(3, out) && check_value(3, out, 100) == 7,
          "reinsert after delete lost across restart");
    CHECK(!st->Read(8, out), "key 8 resurrected after clean restart");
    st->StopSession();
    drop_store(st, "delcrash");
    fprintf(stderr, "PASS delcrash\n");
}

int main(int argc, char** argv) {
    g_argv0 = argv[0];
    if (argc >= 5 && strcmp(argv[1], "--delcrashchild") == 0)
        return delcrash_child(argv[2], atoi(argv[3]), atoi(argv[4]));
    if (argc >= 3 && strcmp(argv[1], "--crashchild") == 0)
        return crash_child(argv[2], strtoull(argv[3], nullptr, 10),
                           atoi(argv[4]));
    if (argc >= 3 && strcmp(argv[1], "--enospcchild") == 0) {
        g_parent = "/";   // unused in child
        return enospc_child(argv[2]);
    }
    if (argc < 2) { fprintf(stderr, "usage: %s <spill_parent> [filter] [scale]\n", argv[0]); return 2; }
    g_parent = argv[1];
    const char* filter = argc > 2 ? argv[2] : "";
    if (argc > 3) g_scale = atoi(argv[3]) > 0 ? atoi(argv[3]) : 1;
    mkdir(g_parent.c_str(), 0755);
    struct { const char* name; void (*fn)(); } tests[] = {
        {"basic", test_basic},   {"lww", test_lww},       {"async", test_async},
        {"audit", test_audit},   {"oversized", test_oversized},
        {"delete", test_delete}, {"updel", test_updel},   {"alias", test_alias},
        {"updel2", test_updel2}, {"delcost", test_delcost},
        {"memfull", test_memfull}, {"recoverdrop", test_recoverdrop},
        {"tiny", test_tiny},
        {"nodisk", test_nodisk}, {"rmw", test_rmw},       {"twostores", test_twostores},
        {"ckpt", test_ckpt},     {"rywrite", test_rywrite},
        {"recover", test_recover}, {"crashrecover", test_crashrecover},
        {"enospc", test_enospc},   {"readfault", test_readfault},
        {"oversizebound", test_oversizebound}, {"compact", test_compact},
        {"compactcold", test_compactcold}, {"bimodal", test_bimodal},
        {"capwarn", test_capwarn}, {"delcrash", test_delcrash},
    };
    for (auto& t : tests)
        if (strstr(t.name, filter) || !*filter) t.fn();
    if (g_failures.load()) { fprintf(stderr, "FAILURES: %d\n", g_failures.load()); return 1; }
    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
