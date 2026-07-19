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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

extern IKVStore* create_kvstore();

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
    IKVStore* st = make_store("tiny", 64ull << 20);
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

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <spill_parent> [filter] [scale]\n", argv[0]); return 2; }
    g_parent = argv[1];
    const char* filter = argc > 2 ? argv[2] : "";
    if (argc > 3) g_scale = atoi(argv[3]) > 0 ? atoi(argv[3]) : 1;
    mkdir(g_parent.c_str(), 0755);
    struct { const char* name; void (*fn)(); } tests[] = {
        {"basic", test_basic},   {"lww", test_lww},       {"async", test_async},
        {"audit", test_audit},   {"oversized", test_oversized},
        {"delete", test_delete}, {"updel", test_updel},   {"alias", test_alias},
        {"tiny", test_tiny},
        {"nodisk", test_nodisk}, {"rmw", test_rmw},       {"twostores", test_twostores},
        {"ckpt", test_ckpt},     {"rywrite", test_rywrite},
    };
    for (auto& t : tests)
        if (strstr(t.name, filter) || !*filter) t.fn();
    if (g_failures.load()) { fprintf(stderr, "FAILURES: %d\n", g_failures.load()); return 1; }
    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
