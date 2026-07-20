// hydra.cc — HydraKV: a larger-than-memory key-value store for the 50:50 YCSB-A
// benchmark (250M keys x 100B values, 8 GiB memory budget, NVMe spill).
//
// Written from first principles (no code taken from any existing KV store).
//
// Architecture
// ────────────
//  * Disk tier: a single O_DIRECT "slot file" in the spill directory,
//    written as an APPEND-ONLY log of 128-byte slots: first inserts and all
//    write-backs stage into per-session 1 MiB chunk buffers landed
//    sequentially. Landed pages are immutable (except Delete's slot
//    tombstoning). The log has no GC — like FASTER, compaction is an
//    offline concern (documented limitation).
//  * Position index: an open-addressing fingerprint hash index (like
//    FASTER's hash table) mapping ARBITRARY uint64 keys to slot positions.
//    Each 8-byte word packs [fp32 | position+1]; buckets of 8 words (one
//    cache line), linear probing across buckets, lock-free CAS inserts.
//    Fingerprint matches are verified against the full key stored in the
//    on-disk slot (on the read path inline; on the write path lazily at
//    flush time, with a relocation slow path for the ~2^-60 false-match
//    case), so correctness never depends on fingerprints being unique.
//    Values > 101 B take a sharded in-memory overflow map (never exercised
//    by this workload, kept for interface completeness).
//  * Memory tier: a set-associative write-back cache (8-way, 120-byte
//    entries) sized from mem_budget_bytes. Reads that miss fetch the 4 KiB
//    page from disk and admit the value; blind upserts of uncached keys are
//    absorbed as dirty cache entries and flushed lazily.
//  * Background cleaner threads flush cold dirty entries so evictions almost
//    always find a clean victim and never block a worker on write I/O.
//  * Async reads: ReadAsync queues cache misses to a pool of I/O threads
//    (each doing synchronous O_DIRECT preads); the harness pipelines up to
//    256 in-flight reads per worker. Cache hits complete inline.
//  * Checkpoint(): flushes every dirty entry to the slot file and fdatasyncs.
//
// Correctness invariants
// ──────────────────────
//  * The slot file is an append-only log: sessions own disjoint page-aligned
//    extents and a landed page is never appended to again (partial flushes
//    continue at the next page boundary), so ordinary flushes never rewrite
//    landed bytes. The only in-place page writer is Delete's slot
//    tombstoning, serialized by a page lock stripe.
//  * Dirty entries created by the sequential-load path ("BUFFERED") are
//    pinned in cache until their chunk buffer has landed on disk, so a
//    stale chunk write can never clobber a newer flushed value and reads
//    always see the freshest value.
//  * Flush/evict uses a per-entry version counter: copy value+version under
//    the set lock, do the I/O unlocked, then re-check the version before
//    marking the entry clean. Concurrent upserts simply leave it dirty.

#include "kvstore_interface.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/syscall.h>
#define HYDRA_HAVE_URING 1
#endif

#if defined(__x86_64__)
#include <immintrin.h>
static inline void cpu_relax() { _mm_pause(); }
#else
static inline void cpu_relax() {}
#endif

namespace hydra {

// ── Geometry ─────────────────────────────────────────────────────────────────
// On-disk slot layout (128 B), crash-safe:
//   [0..7]    key            [8..11]  size+1 (0 = never landed)
//   [12..15]  prev pos+1     [16..116] value bytes (zero-padded to 101)
//   [117..123] 56-bit LSN    [124..127] CRC-32C over bytes [0..123]
// The LSN is a global monotonic staging counter: positions are NOT temporal
// (extents are pre-reserved per session), so version ordering — for reads,
// recovery and compaction — is decided by LSN, never by position. The CRC
// detects torn/partial slot writes after a crash (RocksDB/LevelDB-style
// tolerate-corrupted-tail recovery: bad-CRC slots are skipped, never served).
static constexpr size_t   kSlotBytes     = 128;   // fixed on-disk slot
static constexpr size_t   kSlotDataMax   = 101;   // was 112; 7 B LSN + 4 B CRC carved
static constexpr size_t   kSlotLsnOff    = 16 + kSlotDataMax;    // 117
static constexpr size_t   kSlotCrcOff    = kSlotLsnOff + 7;      // 124
static_assert(kSlotCrcOff + 4 == kSlotBytes, "slot layout must fill 128 B");
static constexpr size_t   kPageBytes     = 4096;  // O_DIRECT I/O unit
static constexpr size_t   kSlotsPerPage  = kPageBytes / kSlotBytes;   // 32
static constexpr size_t   kChunkSlots    = 8192;  // load-phase write batch
static constexpr size_t   kChunkBytes    = kChunkSlots * kSlotBytes;  // 1 MiB
static constexpr size_t   kAssoc         = 8;     // cache ways per set
// SLRU segmentation: new entries (read admissions and upsert-miss inserts)
// may only evict within the first kProbationWays ways of a set; the
// remaining "protected" ways are filled exclusively by promoting entries
// re-accessed while their recent bit was already set. One-pass scans and
// cold-write streams therefore churn 25% of the cache and can never
// displace the protected hot set (segmented-LRU / W-TinyLFU structure).
static constexpr size_t   kProbationWays = 4;
static constexpr size_t   kInlineMax     = 101;   // max value bytes in cache
static constexpr size_t   kEntryBytes    = 120;
static constexpr size_t   kBucketWords   = 8;     // index words per bucket (64 B)
static constexpr size_t   kPageLockCount = 1 << 16;
static constexpr size_t   kOverflowShards = 64;
static constexpr int      kIoThreads     = 8;     // async read-miss shards (io_uring rings)
static constexpr int      kRingDepth     = 128;   // in-flight reads per ring
static constexpr int      kSyncIoThreads = 48;    // fallback if io_uring unavailable
static constexpr int      kCleanerThreads = 4;    // background dirty flushers
static constexpr int      kCheckpointThreads = 32;
static constexpr uint64_t kCompactRegionSlots = 8192;  // 1 MiB compaction unit
static constexpr int      kDelShards = 64;        // deleted-key registry shards

// Entry states.
enum : uint8_t { S_EMPTY = 0, S_CLEAN = 1, S_DIRTY = 2, S_BUFFERED = 3,
                 S_FLUSHING = 4 /* dirty, staged in an in-flight flush chunk (pinned) */ };

// Entry.recent flag bits.
enum : uint8_t {
    R_RECENT = 1,   // CLOCK second-chance bit (cleared by aging sweeps)
};

// Relaxed-atomic byte with plain-byte syntax. Cleaner threads peek entry
// state/recency WITHOUT the set lock (a hint only; the locked rescan
// decides); mixing plain and atomic accesses to one byte is formally a data
// race, so every access to these two fields goes through this type. All ops
// are relaxed — on x86 this compiles to exactly the same MOVs as a plain
// uint8_t (no LOCK prefixes), so the hot path is unchanged.
struct RelaxedU8 {
    std::atomic<uint8_t> v;
    RelaxedU8() = default;
    RelaxedU8(const RelaxedU8& o) {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    RelaxedU8& operator=(const RelaxedU8& o) {
        v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
    RelaxedU8& operator=(uint8_t x) {
        v.store(x, std::memory_order_relaxed);
        return *this;
    }
    operator uint8_t() const { return v.load(std::memory_order_relaxed); }
    RelaxedU8& operator|=(uint8_t x) { return *this = (uint8_t)((uint8_t)*this | x); }
    RelaxedU8& operator&=(uint8_t x) { return *this = (uint8_t)((uint8_t)*this & x); }
};
static_assert(sizeof(RelaxedU8) == 1, "RelaxedU8 must be one byte");

struct Entry {
    uint64_t key;
    uint32_t pos1;    // slot position + 1; 0 = unknown (pure in-memory mode)
    uint32_t ver;     // bumped on every value write; survives eviction/reuse
                      // (32-bit so in-flight-flush ABA needs 2^32 same-key
                      // updates during one I/O — days, not seconds)
    uint8_t  size;    // value size in bytes (<= kInlineMax)
    RelaxedU8 state;  // peeked unlocked by cleaners (see RelaxedU8)
    RelaxedU8 recent; // flag bits (R_RECENT)
    uint8_t  data[kInlineMax];
};
static_assert(sizeof(Entry) == kEntryBytes, "Entry must be 120 bytes");

struct SpinLock {
    std::atomic<uint8_t> v{0};
    void lock() {
        for (;;) {
            uint8_t e = 0;
            if (v.compare_exchange_weak(e, 1, std::memory_order_acquire)) return;
            while (v.load(std::memory_order_relaxed)) cpu_relax();
        }
    }
    void unlock() { v.store(0, std::memory_order_release); }
};

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// Single-writer counter increment: relaxed load + relaxed store (plain MOVs);
// concurrent readers use relaxed loads. Not fetch_add — no lock prefix needed
// because each counter has exactly one writer thread.
static inline void bump(std::atomic<uint64_t>& c) {
    c.store(c.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

// ── Fingerprint position index ───────────────────────────────────────────────
// Open-addressing table of 8-byte words, each packing [fp32 | position+1].
// Word 0 = empty (positions are stored +1 so a live word is never 0).
// Buckets of kBucketWords words (one cache line); probing scans a bucket then
// moves to the next, wrapping around. Bucket choice uses the LOW 32 bits of
// mix64(key), the fingerprint uses the HIGH 32 bits, so a false candidate
// requires a full 64-bit hash collision (~2^-64 per pair) — and every
// candidate is still verified against the exact key stored in the on-disk
// slot before being trusted, so correctness never rests on the hash.
struct HashIndex {
    std::atomic<uint64_t>* words = nullptr;
    uint64_t nwords = 0;       // power of two
    uint64_t cap = 0;          // max live entries before callers overflow
    std::atomic<uint64_t> count{0};

    void init(uint64_t words_pow2) {
        nwords = words_pow2;
        cap = nwords - nwords / 32;   // ~97% fill hard limit
        void* p = mmap(nullptr, nwords * 8, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) { fprintf(stderr, "hydra: index mmap failed\n"); abort(); }
        words = static_cast<std::atomic<uint64_t>*>(p);
    }
    void destroy() { if (words) munmap(words, nwords * 8); words = nullptr; }

    static uint64_t pack(uint64_t h, uint32_t pos1) { return (h & 0xFFFFFFFF00000000ULL) | pos1; }
    uint64_t start_of(uint64_t h) const {
        return ((h & 0xFFFFFFFFULL) * kBucketWords) & (nwords - 1);
    }

    // Iterate fingerprint-match candidates. `it` starts at start_of(h) and
    // advances; returns candidate pos1, or 0 when an empty word is reached
    // (key definitely absent) or the whole table has been scanned.
    uint32_t next_candidate(uint64_t h, uint64_t& it, uint64_t& scanned) const {
        const uint64_t fp = h & 0xFFFFFFFF00000000ULL;
        while (scanned < nwords) {
            uint64_t w = words[it].load(std::memory_order_acquire);
            it = (it + 1) & (nwords - 1);
            scanned++;
            if (w == 0) return 0;                       // end of probe chain
            if ((w & 0xFFFFFFFF00000000ULL) == fp) {
                uint32_t pos1 = (uint32_t)w;
                if (pos1 != 0) return pos1;             // pos1==0 => tombstone
            }
        }
        return 0;
    }

    // First candidate convenience (unverified!).
    uint32_t first_candidate(uint64_t h) const {
        uint64_t it = start_of(h), scanned = 0;
        return next_candidate(h, it, scanned);
    }

    // Insert a new mapping. Returns false if the table is at capacity.
    bool insert(uint64_t h, uint32_t pos1) {
        if (count.load(std::memory_order_relaxed) >= cap) return false;
        uint64_t it = start_of(h);
        const uint64_t v = pack(h, pos1);
        uint64_t scanned = 0;
        while (scanned < nwords) {
            uint64_t w = words[it].load(std::memory_order_acquire);
            if (w == 0) {
                uint64_t expect = 0;
                if (words[it].compare_exchange_strong(expect, v,
                                                      std::memory_order_acq_rel)) {
                    count.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                continue;   // word got taken; re-examine it (no advance/count)
            }
            it = (it + 1) & (nwords - 1);
            scanned++;
        }
        return false;
    }

    // Move a key's mapping old_p1 -> new_p1 (log-append flush landed).
    // Single owner per key (S_FLUSHING) guarantees the old word is present.
    bool update(uint64_t h, uint32_t old_p1, uint32_t new_p1) {
        const uint64_t ov = pack(h, old_p1), nv = pack(h, new_p1);
        uint64_t it = start_of(h);
        for (uint64_t scanned = 0; scanned < nwords; ++scanned) {
            uint64_t w = words[it].load(std::memory_order_acquire);
            if (w == 0) return false;
            if (w == ov) {
                if (words[it].compare_exchange_strong(w, nv,
                                                      std::memory_order_acq_rel))
                    return true;
                continue;   // re-examine (word changed under us)
            }
            it = (it + 1) & (nwords - 1);
        }
        return false;
    }

    // (No erase method: Delete keeps index words — they route same-fp
    // chains — and poisons the slots instead; see HydraStore::Delete.)
};

// Aligned buffer helper (O_DIRECT requires 4 KiB-aligned buffers).
static uint8_t* alloc_aligned(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, kPageBytes, bytes) != 0) {
        fprintf(stderr, "hydra: posix_memalign(%zu) failed\n", bytes);
        abort();
    }
    return static_cast<uint8_t*>(p);
}

// ── CRC-32C (Castagnoli) — hardware crc32 instruction when available ─────────
#if defined(__SSE4_2__)
static inline uint32_t crc32c(const uint8_t* p, size_t n) {
    uint64_t c = 0xFFFFFFFFu;
    while (n >= 8) {
        uint64_t x;
        memcpy(&x, p, 8);
        c = _mm_crc32_u64(c, x);
        p += 8; n -= 8;
    }
    uint32_t c32 = (uint32_t)c;
    while (n) { c32 = _mm_crc32_u8(c32, *p++); n--; }
    return c32 ^ 0xFFFFFFFFu;
}
#else
// Table-driven software CRC-32C (identical values to the SSE4.2 path).
static uint32_t g_crc32c_tab[256];
static const bool g_crc32c_init = [] {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1)));
        g_crc32c_tab[i] = c;
    }
    return true;
}();
static inline uint32_t crc32c(const uint8_t* p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i)
        c = (c >> 8) ^ g_crc32c_tab[(c ^ p[i]) & 0xFF];
    return c ^ 0xFFFFFFFFu;
}
#endif

// ── Slot LSN/CRC accessors ───────────────────────────────────────────────────
static inline uint64_t slot_lsn(const uint8_t* slot) {
    uint64_t v = 0;
    memcpy(&v, slot + kSlotLsnOff, 7);   // little-endian 56-bit
    return v;
}
static inline void slot_seal(uint8_t* slot, uint64_t lsn) {
    memcpy(slot + kSlotLsnOff, &lsn, 7);
    uint32_t c = crc32c(slot, kSlotCrcOff);
    memcpy(slot + kSlotCrcOff, &c, 4);
}
static inline bool slot_crc_ok(const uint8_t* slot) {
    uint32_t c;
    memcpy(&c, slot + kSlotCrcOff, 4);
    return c == crc32c(slot, kSlotCrcOff);
}

// EINTR/EAGAIN-safe pread. Returns bytes read (< n only at EOF), or -1 on a
// real I/O error (FAIL-SOFT: callers count the error and treat the data as
// unavailable — the process never dies for a disk fault).
static ssize_t xpread(int fd, void* buf, size_t n, off_t off) {
    size_t done = 0;
    int spins = 0;
    while (done < n) {
        ssize_t r = pread(fd, (char*)buf + done, n - done, off + (off_t)done);
        if (r < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN) {   // transient: back off, retry
                if (++spins > 64) std::this_thread::yield();
                continue;
            }
            return -1;
        }
        if (r == 0) break;   // EOF: caller sees zero-filled tail
        done += (size_t)r;
    }
    return (ssize_t)done;
}

// EINTR/EAGAIN-safe full pwrite. Returns false on error/zero progress
// (FAIL-SOFT: the caller keeps its state dirty/staged and retries later —
// after a failed write nothing is ever marked clean or durable).
static bool xpwrite(int fd, const void* buf, size_t n, off_t off) {
    size_t done = 0;
    int spins = 0;
    while (done < n) {
        ssize_t r = pwrite(fd, (const char*)buf + done, n - done,
                           off + (off_t)done);
        if (r < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN) {
                if (++spins > 64) std::this_thread::yield();
                continue;
            }
            return false;
        }
        if (r == 0) return false;
        done += (size_t)r;
    }
    return true;
}

// ── Minimal raw io_uring wrapper (no liburing on the box) ────────────────────
// Single-threaded use: one ring per io thread. Only IORING_OP_READ is used.
#ifdef HYDRA_HAVE_URING
static int sys_io_uring_setup(unsigned entries, struct io_uring_params* p) {
    return (int)syscall(__NR_io_uring_setup, entries, p);
}
static int sys_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete,
                              unsigned flags) {
    return (int)syscall(__NR_io_uring_enter, fd, to_submit, min_complete,
                        flags, nullptr, 0);
}

struct Uring {
    int fd = -1;
    unsigned sq_entries = 0;
    unsigned *sq_head = nullptr, *sq_tail = nullptr, *sq_mask = nullptr,
             *sq_array = nullptr;
    io_uring_sqe* sqes = nullptr;
    unsigned *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
    io_uring_cqe* cqes = nullptr;
    void *sq_mm = nullptr, *cq_mm = nullptr, *sqe_mm = nullptr;
    size_t sq_mm_len = 0, cq_mm_len = 0, sqe_mm_len = 0;
    unsigned pending = 0;   // prepped but not yet submitted

    ~Uring() { destroy(); }   // RAII: rings leaked fds/mappings otherwise

    bool init(unsigned entries) {
        io_uring_params p{};
        fd = sys_io_uring_setup(entries, &p);
        if (fd < 0) return false;
        sq_mm_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_mm_len = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single) sq_mm_len = cq_mm_len = std::max(sq_mm_len, cq_mm_len);
        sq_mm = mmap(nullptr, sq_mm_len, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq_mm == MAP_FAILED) { destroy(); return false; }
        cq_mm = single ? sq_mm
                       : mmap(nullptr, cq_mm_len, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (cq_mm == MAP_FAILED) { cq_mm = nullptr; destroy(); return false; }
        sqe_mm_len = p.sq_entries * sizeof(io_uring_sqe);
        sqe_mm = mmap(nullptr, sqe_mm_len, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (sqe_mm == MAP_FAILED) { sqe_mm = nullptr; destroy(); return false; }
        uint8_t* sb = (uint8_t*)sq_mm;
        uint8_t* cb = (uint8_t*)cq_mm;
        sq_head  = (unsigned*)(sb + p.sq_off.head);
        sq_tail  = (unsigned*)(sb + p.sq_off.tail);
        sq_mask  = (unsigned*)(sb + p.sq_off.ring_mask);
        sq_array = (unsigned*)(sb + p.sq_off.array);
        cq_head  = (unsigned*)(cb + p.cq_off.head);
        cq_tail  = (unsigned*)(cb + p.cq_off.tail);
        cq_mask  = (unsigned*)(cb + p.cq_off.ring_mask);
        cqes     = (io_uring_cqe*)(cb + p.cq_off.cqes);
        sqes     = (io_uring_sqe*)sqe_mm;
        sq_entries = p.sq_entries;
        return true;
    }
    void destroy() {
        if (sqe_mm) munmap(sqe_mm, sqe_mm_len);
        if (cq_mm && cq_mm != sq_mm) munmap(cq_mm, cq_mm_len);
        if (sq_mm) munmap(sq_mm, sq_mm_len);
        sq_mm = cq_mm = sqe_mm = nullptr;
        if (fd >= 0) close(fd);
        fd = -1;
    }
    // Queue a pread. Caller bounds in-flight ops by sq_entries, so a free
    // sqe always exists. user_data identifies the op.
    void prep_read(int file_fd, void* buf, unsigned len, off_t off, uint64_t ud) {
        unsigned tail = *sq_tail;               // single producer
        unsigned idx = tail & *sq_mask;
        io_uring_sqe* e = &sqes[idx];
        memset(e, 0, sizeof(*e));
        e->opcode = IORING_OP_READ;
        e->fd = file_fd;
        e->addr = (uint64_t)(uintptr_t)buf;
        e->len = len;
        e->off = (uint64_t)off;
        e->user_data = ud;
        sq_array[idx] = idx;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        pending++;
    }
    // Submit everything prepped; optionally block for >= 1 completion.
    // Accounts for partial submission: io_uring_enter may consume fewer than
    // to_submit SQEs, and returns the consumed count (EINTR during the
    // GETEVENTS wait after consuming SQEs also reports the count).
    void enter(bool wait) {
        unsigned subs = pending;
        pending = 0;
        int spins = 0;
        for (;;) {
            int r = sys_io_uring_enter(fd, subs, wait ? 1 : 0,
                                       wait ? IORING_ENTER_GETEVENTS : 0);
            if (r < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EBUSY) {
                    // Nothing consumed: retry with the same count, backing
                    // off so a persistent EAGAIN/EBUSY (kernel memory
                    // pressure) does not become a hard busy-loop.
                    if (++spins > 64) std::this_thread::yield();
                    else cpu_relax();
                    continue;
                }
                fprintf(stderr, "hydra: io_uring_enter failed (errno=%d)\n",
                        errno);
                abort();
            }
            if ((unsigned)r < subs) { subs -= (unsigned)r; continue; }
            return;
        }
    }
    // Reap one completion; false if none available.
    bool reap(uint64_t& ud, int& res) {
        unsigned head = *cq_head;
        unsigned tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
        if (head == tail) return false;
        io_uring_cqe* c = &cqes[head & *cq_mask];
        ud = c->user_data;
        res = c->res;
        __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
        return true;
    }
};
#endif  // HYDRA_HAVE_URING

// ── Per-thread session ───────────────────────────────────────────────────────
struct Session {
    // async reads in flight for this session
    std::atomic<uint64_t> outstanding{0};

    // sequential-load chunk buffer (lazy)
    uint8_t* chunk_buf = nullptr;
    uint64_t chunk_base = 0;     // next slot position of the chunk
    uint32_t chunk_fill = 0;     // slots used since base
    uint32_t chunk_cap = 0;      // slots remaining in the reserved extent
    bool     chunk_active = false;  // extent reserved (survives failed insert)
    uint64_t extent_base = 0;    // reserved extent origin (ownership registry)
    uint32_t extent_slots = 0;   // reserved extent length in slots
    struct ChunkRec {          // one staged slot in the chunk
        uint64_t key;
        uint32_t ver;          // entry version at stage time
        uint32_t old_p1;       // 0 = first insert; else previous position+1
        uint32_t new_p1;       // staged position+1
    };
    std::vector<ChunkRec> chunk_meta;
    uint32_t io_rr = 0;        // round-robin io shard selector

    // last-page read buffer (helps the stride-1 validation scan)
    uint8_t* page_buf = nullptr;
    uint64_t page_no = ~0ULL;
    uint64_t page_epoch = 0;

    // scratch page for read-modify-write flushes
    uint8_t* rmw_buf = nullptr;

    // stats — written only by the owning thread via bump() (a relaxed atomic
    // store compiling to a plain MOV), read concurrently by GetCacheStats
    // with relaxed loads: race-free, zero hot-path cost.
    std::atomic<uint64_t> read_hits{0}, read_misses{0};
    std::atomic<uint64_t> rmw_hits{0}, rmw_misses{0};
    std::atomic<uint64_t> evictions{0};

    ~Session() {
        free(chunk_buf);
        free(page_buf);
        free(rmw_buf);
    }
};

class HydraStore final : public IKVStore {
public:
    HydraStore() : gen_(0) {}   // gen_ assigned in InitExtended (g_store_gen)
    ~HydraStore() override;

    void Init(size_t hash_table_size, size_t log_size_bytes) override {
        InitExtended(hash_table_size, log_size_bytes, 0, nullptr);
    }
    void InitExtended(size_t hash_table_size, size_t log_size_bytes,
                      size_t mem_budget_bytes, const char* storage_path) override;

    void StartSession() override;
    void StopSession() override;
    void Refresh() override {}

    bool Read(uint64_t key, GenValue& out) override;
    OpStatus ReadAsync(ReadSlot* slot) override;
    void CompletePending(bool wait) override;
    void Upsert(uint64_t key, const GenValue& value) override;
    void RMW(uint64_t key, const uint8_t* mod_data, size_t mod_size) override;
    bool Delete(uint64_t key) override;
    void Checkpoint() override;
    CacheStats GetCacheStats() const override;

private:
    // ---- helpers -----------------------------------------------------------
    Session* session();
    uint64_t set_of(uint64_t key) const {
        // fastrange: uniform map of the mixed key onto [0, nsets_)
        return (uint64_t)(((__uint128_t)mix64(key) * nsets_) >> 64);
    }
    Entry* set_base(uint64_t s) { return entries_ + s * kAssoc; }

public:
    // Production observability (extern "C" seam; not part of IKVStore).
    struct ProdStats {
        uint64_t durable_ok, recover_ok, recovered_keys, recover_torn_slots;
        uint64_t write_errors, read_errors, rejected_oversize, oversize_bytes;
        uint64_t compactions_run, log_bytes, live_bytes, reclaimed_bytes;
        uint64_t buffered_fallback, punch_unsupported;
    };
    void FillProdStats(ProdStats& out) const;

private:
    bool cache_probe(uint64_t key, GenValue& out);          // hit path
    // Promote a twice-hit probation entry into a protected way (lock held).
    void promote_way(uint64_t s, int w);
    OpStatus miss_read(uint64_t key, GenValue& out, Session* se);
    void upsert_inline(uint64_t key, const uint8_t* data, uint16_t size);
    int  find_victim(uint64_t s, Session* se);              // set locked; may drop+retake
    // Log-append flush: stage a DIRTY entry's bytes into the session's
    // sequential chunk (entry -> S_FLUSHING, pinned until the chunk lands).
    // Returns true if the chunk is now full (caller must flush_chunk after
    // releasing the set lock). Requires set lock s held; e = &set_base(s)[w].
    bool stage_flush(uint64_t s, int w, Session* se);
    // Disk-verified index lookup (reads candidate slots); 0 = absent.
    uint32_t verified_lookup(uint64_t key, Session* se);
    // Re-stage a durable tombstone for a still-deleted key whose original
    // tombstone slot is about to be reclaimed by compaction (no index word,
    // no cache entry — the slot exists purely so recovery keeps the key
    // deleted). Returns false if the disk/position space refuses it.
    bool restage_tombstone(uint64_t key, Session* se);
    // Overwrite a landed slot's size field with an unmatchable value so no
    // read can ever match it again (prev chain stays intact). Delete-only.
    void tombstone_slot(uint64_t position, Session* se);
    void flush_chunk(Session* se);
    void flush_partials(Session* se);  // land a partially-filled chunk
    // Doorkeeper admission filter (two-generation bitmaps).
    bool dk_seen_and_mark(uint64_t h);
    void dk_maybe_rotate();
    void ensure_chunk(Session* se);
    bool read_page(uint64_t page, uint8_t* buf);             // O_DIRECT pread
    void cleaner_main(int id);
    void io_main(int shard);
    void stop_background();
    // Crash recovery: rebuild the index from the slot log (per-slot CRC
    // verified, newest-LSN-wins, tombstone-aware). Runs once from Init.
    void recover_log(uint64_t file_bytes);
    void load_overflow_file(const uint64_t* rk, const uint64_t* rl);
    void write_overflow_file();          // Checkpoint: persist oversize map
    void sync_dir();                     // fsync dir_ (sidecar rename/unlink)
    // Background log compaction: relocate live slots out of a region, then
    // punch the region and recycle its position space.
    void compactor_main();
    bool compact_region(Session* se, uint64_t base);
    void note_extent_owned(uint64_t base, uint32_t slots);
    void note_extent_done(uint64_t base);
    // Doorkeeper-gated clean admission of a disk-read value (non-blocking).
    void admit_read(uint64_t key, uint32_t p1, const uint8_t* data,
                    uint16_t size, Session* se);
#ifdef HYDRA_HAVE_URING
    // Async read-miss state machine (one MissOp per in-flight request).
    struct MissOp {
        ReadSlot* slot = nullptr;
        Session*  sess = nullptr;    // requester's session (outstanding ctr)
        uint64_t  h = 0;             // mix64(key)
        uint64_t  it = 0;            // index candidate iterator
        uint64_t  scanned = 0;
        uint32_t  cur = 0;           // chain slot position+1 being read
        uint32_t  hops = 0;          // chain-walk bound (defensive)
        uint64_t  page = 0;          // page currently in buf (in flight)
        uint8_t*  buf = nullptr;     // fixed 4 KiB aligned buffer
        uint32_t  best_p1 = 0;       // newest (highest-LSN) verified match
        uint64_t  best_lsn = 0;
        uint32_t  filled = 0;        // bytes of the current page already read
        uint16_t  best_sz = 0;
        uint8_t   best_data[kSlotDataMax];
    };
    void io_main_uring(int shard);
    // Start servicing a request; returns true iff an IO was submitted.
    bool uring_start(MissOp& op, uint32_t oi, Uring& ring, Session* io_se);
    // Handle a completed page read; returns true iff another IO was submitted.
    bool uring_advance(MissOp& op, uint32_t oi, Uring& ring, int res,
                       Session* io_se);
    std::unique_ptr<Uring> rings_[kIoThreads];
    bool use_uring_ = false;
#endif

    // overflow (oversized values / index-capacity fallback); in-memory,
    // oversized entries charged against oversize_cap_, persisted by
    // Checkpoint into an atomically-renamed sidecar file (with LSNs so
    // recovery can order them against slot-log versions).
    struct OvfVal {
        std::string bytes;
        uint64_t lsn;
    };
    struct OverflowShard {
        std::mutex mu;
        std::unordered_map<uint64_t, OvfVal> map;
    };
    OverflowShard& shard(uint64_t key) { return overflow_[mix64(key) % kOverflowShards]; }
    static uint64_t ovf_charge(size_t size) { return size + 64; }
    bool overflow_contains(uint64_t key);
    bool overflow_get(uint64_t key, GenValue& out);
    void overflow_put(uint64_t key, const uint8_t* data, size_t size);
    bool overflow_erase(uint64_t key);
    bool oversize_would_fit(uint64_t key, size_t size);

    // ---- state -------------------------------------------------------------
    size_t   mem_budget_ = 0;
    bool     disk_mode_ = false;
    int      fd_ = -1;
    std::string dir_;
    uint64_t gen_ = 0;             // unique per-instance id (TLS validation)

    uint64_t nsets_ = 0;
    Entry*   entries_ = nullptr;         // nsets_ * kAssoc
    SpinLock* set_locks_ = nullptr;      // per set
    SpinLock  page_locks_[kPageLockCount];

    HashIndex index_;                        // key -> position+1
    // Cache-line isolation (perf, no semantic change): several atomics below
    // are READ on every op by workers while a NEIGHBORING atomic is written
    // at high rate by other threads; without alignas(64) they share a 64-B
    // line and every write invalidates the readers' cached copy (false
    // sharing). Write-hot fields get their own line; read-mostly fields are
    // grouped away from them.
    alignas(64) std::atomic<uint64_t> next_slot_{0};     // slot allocator
    // Staging tokens: every staged chunk record (BUFFERED insert or FLUSHING
    // write-back) stamps the entry's ver with a fresh globally unique token,
    // so a flush record's pin-ownership check (key+state+pos1+ver) can never
    // be satisfied by a DIFFERENT staging — e.g. a post-delete reinsert of
    // the same key in another cache way (per-way counters could collide
    // there). False ownership now needs a full 2^32 token wrap while a
    // record is parked AND matching key/pos1 (documented residual).
    // stage_ver_ + lsn_ share one write-hot line (both bumped per staged
    // slot, usually by the same thread in the same operation).
    alignas(64) std::atomic<uint32_t> stage_ver_{1};

    // Global staging LSN (56 bits stored per slot): total-orders every slot
    // version across sessions. Bumped per staged slot / tombstone / oversize
    // put. 2^56 writes at 10 M/s = 228 years before wrap.
    std::atomic<uint64_t> lsn_{1};

    // ── production state / observability ─────────────────────────────────
    // landed_hw_ is loaded on every read_page/EOF classification; keep it
    // off the stagers' lsn_ line.
    alignas(64) std::atomic<uint64_t> landed_hw_{0};  // highest durably-landed byte+1
    std::atomic<uint64_t> read_errors_{0};    // failed/short page reads
    std::atomic<uint64_t> write_errors_{0};   // failed slot-log writes
    std::atomic<bool>     durable_ok_{true};  // sticky: any write/sync fail
    std::atomic<uint64_t> recover_ok_{0};
    std::atomic<uint64_t> recovered_keys_{0};
    std::atomic<uint64_t> recover_torn_{0};
    std::atomic<uint64_t> rejected_oversize_{0};
    std::atomic<uint64_t> oversize_bytes_{0};  // charged bytes in overflow map
    uint64_t              oversize_cap_ = 0;   // budget/8 (HYDRA_OVERSIZE_CAP_MB)
    std::atomic<uint64_t> compactions_run_{0};
    std::atomic<uint64_t> reclaimed_bytes_{0};   // net outstanding (reuse subtracts)
    std::atomic<uint64_t> reclaimed_total_{0};   // cumulative (stats)
    std::atomic<bool>     punch_unsupported_{false};
    bool                  buffered_fallback_ = false;
    bool                  overflow_file_seen_ = false;  // sidecar file exists
    std::mutex            ckpt_mu_;            // serializes overflow-file writes

    // Free/owned extent registry: compaction recycles punched regions
    // through here; sessions must never have an in-flight chunk inside a
    // region being compacted. Cold path only (one lock per 8192 slots).
    std::mutex free_mu_;
    std::vector<std::pair<uint64_t, uint32_t>> free_extents_;  // base, slots
    // Loaded on every fresh-chunk grab; isolate from the mutex/vectors above.
    alignas(64) std::atomic<uint64_t> free_count_{0};
    std::vector<std::pair<uint64_t, uint32_t>> owned_extents_; // base, slots
    double   compact_factor_ = 2.0;    // HYDRA_COMPACT_FACTOR
    uint64_t compact_floor_ = 256ull << 20;   // HYDRA_COMPACT_FLOOR_MB
    uint64_t compact_cursor_ = 0;
    std::thread compactor_;

    // Doorkeeper: one-hit-wonder read admissions are filtered through two
    // rotating bitmap generations (frequency >= 2 required to cache a read
    // miss). Sized from the budget; rotated by cleaner 0.
    // Read-mostly doorkeeper lookup state on its own line; dk_marks_ (bumped
    // on every doorkeeper mark, i.e. every read miss) on a separate line.
    alignas(64) std::atomic<uint64_t>* dk_bits_[2] = {nullptr, nullptr};
    uint64_t dk_mask_ = 0;                   // bit-index mask (bits count - 1)
    std::atomic<int> dk_cur_{0};
    alignas(64) std::atomic<uint64_t> dk_marks_{0};

    OverflowShard overflow_[kOverflowShards];
    // Loaded on read paths (zero-overflow fast path); written rarely.
    alignas(64) std::atomic<uint64_t> overflow_count_{0};

    // flush_epoch_ is loaded per verified-lookup page-buffer check; keep it
    // away from the admission-rate occupancy_ counter.
    alignas(64) std::atomic<uint64_t> flush_epoch_{0};  // invalidates session page bufs
    alignas(64) std::atomic<uint64_t> occupancy_{0};
    std::atomic<uint64_t> key_mismatch_{0};

    // Deleted-key registry. Delete() marks the key BEFORE its cache sweep;
    // admissions and flush landings re-check it under the SAME set lock the
    // sweep uses, so every staged/in-flight copy of a deleted key is either
    // (a) staged before the sweep's lock acquisition — the sweep sees the
    // pin and erases/waits — or (b) staged after — the stager observes the
    // mark (release/acquire through set_locks_) and skips. Zero cost while
    // no key was ever deleted: one relaxed-MOV load. A mark persists until
    // the key is re-upserted (Upsert/RMW unmark first), which also keeps
    // Delete() idempotent: a second Delete of the same key returns false.
    // Sharded (kDelShards-way) to remove the audit's "global mutex cliff":
    // once any key was ever deleted, readers pay a shard mutex, not one
    // process-global mutex. del_active_ keeps the zero-deletes fast path.
    struct DelShard {
        std::mutex mu;
        std::unordered_set<uint64_t> set;
    };
    DelShard del_shards_[kDelShards];
    // Loaded (relaxed/acquire) on every admission/landing; written only by
    // Delete/reinsert — isolate so neighbors can't invalidate it.
    alignas(64) std::atomic<uint64_t> del_active_{0};    // total marked keys

    DelShard& del_shard(uint64_t key) {
        return del_shards_[(mix64(key) >> 32) % kDelShards];
    }
    bool is_deleted(uint64_t key) {
        if (del_active_.load(std::memory_order_acquire) == 0) return false;
        DelShard& sh = del_shard(key);
        std::lock_guard<std::mutex> g(sh.mu);
        return sh.set.count(key) != 0;
    }
    bool del_mark(uint64_t key) {    // true iff key was not already marked
        DelShard& sh = del_shard(key);
        std::lock_guard<std::mutex> g(sh.mu);
        bool fresh = sh.set.insert(key).second;
        if (fresh) del_active_.fetch_add(1, std::memory_order_release);
        return fresh;
    }
    void del_unmark(uint64_t key) {
        if (del_active_.load(std::memory_order_acquire) == 0) return;
        DelShard& sh = del_shard(key);
        std::lock_guard<std::mutex> g(sh.mu);
        if (sh.set.erase(key))
            del_active_.fetch_sub(1, std::memory_order_release);
    }

    // sessions
    mutable std::mutex sess_mu_;
    std::vector<std::unique_ptr<Session>> sessions_;
    std::vector<Session*> free_sess_;        // drained, reusable sessions

    // async read I/O pool
    struct IoReq { ReadSlot* slot; Session* sess; };
    struct IoShard {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<IoReq> q;
    };
    IoShard io_shards_[kIoThreads];          // one queue per io thread
    std::atomic<bool> stopping_{false};
    std::vector<std::thread> io_threads_;
    std::vector<std::thread> cleaners_;
};

// ── thread-local session registry ────────────────────────────────────────────
// A raw `tls_owner == this` check would break if a store is destroyed and a
// new one is allocated at the same address (stale tls_sess -> use-after-free).
// Each store instance gets a globally unique generation id; the cached session
// is valid only if both the pointer AND the generation match.
static std::atomic<uint64_t> g_store_gen{1};
static thread_local HydraStore* tls_owner = nullptr;
static thread_local uint64_t    tls_gen = 0;
static thread_local Session*    tls_sess = nullptr;

Session* HydraStore::session() {
    if (tls_owner == this && tls_gen == gen_ && tls_sess) return tls_sess;
    auto se = std::make_unique<Session>();   // may be discarded if reusing
    Session* raw = nullptr;
    {
        std::lock_guard<std::mutex> lk(sess_mu_);
        if (!free_sess_.empty()) {           // reuse a drained session
            raw = free_sess_.back();
            free_sess_.pop_back();
        } else {
            raw = se.get();
            sessions_.push_back(std::move(se));
        }
    }
    tls_owner = this;
    tls_gen = gen_;
    tls_sess = raw;
    return raw;
}

void HydraStore::StartSession() { (void)session(); }

void HydraStore::StopSession() {
    Session* se = session();
    // Drain async reads FIRST (their completions never touch our chunk —
    // io threads use their own sessions — but draining first keeps the
    // "session is quiescent when recycled" invariant simple), then land any
    // partial chunk so staged (pinned) entries unpin.
    while (se->outstanding.load(std::memory_order_acquire) != 0)
        std::this_thread::yield();
    flush_partials(se);
    {   // recycle: unbounded phases/Checkpoints must not leak sessions
        std::lock_guard<std::mutex> lk(sess_mu_);
        free_sess_.push_back(se);
    }
    tls_sess = nullptr;   // thread may be reused for a different phase
    tls_owner = nullptr;
}

// ── Init ─────────────────────────────────────────────────────────────────────
void HydraStore::InitExtended(size_t, size_t, size_t mem_budget_bytes,
                              const char* storage_path) {
    gen_ = g_store_gen.fetch_add(1, std::memory_order_relaxed);
    mem_budget_ = mem_budget_bytes ? mem_budget_bytes : (8ULL << 30);
    disk_mode_ = storage_path && storage_path[0];

    // Oversize budget cap: the overflow map is charged against the memory
    // budget and hard-capped; over-cap upserts are rejected and counted.
    oversize_cap_ = mem_budget_ / 8;
    if (const char* e = getenv("HYDRA_OVERSIZE_CAP_MB"))
        oversize_cap_ = strtoull(e, nullptr, 10) << 20;
    if (const char* e = getenv("HYDRA_COMPACT_FACTOR")) {
        double f = atof(e);
        if (f >= 1.1) compact_factor_ = f;
    }
    if (const char* e = getenv("HYDRA_COMPACT_FLOOR_MB"))
        compact_floor_ = strtoull(e, nullptr, 10) << 20;

    if (disk_mode_) {
        dir_ = storage_path;
        std::string file = dir_ + "/hydra_slots.dat";
        // NO O_TRUNC: an existing slot log is RECOVERED, not destroyed.
        fd_ = open(file.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
        if (fd_ < 0) {
            // Fall back to buffered I/O if the filesystem rejects O_DIRECT.
            // Surfaced (not silent): logged here and via prod stats.
            fd_ = open(file.c_str(), O_RDWR | O_CREAT, 0644);
            if (fd_ >= 0) {
                buffered_fallback_ = true;
                fprintf(stderr, "hydra: O_DIRECT unavailable on %s — using "
                        "buffered I/O (surfaced in prod stats)\n", dir_.c_str());
            }
        }
        if (fd_ < 0) {
            // Init-time invariant: no storage at all — cannot construct.
            fprintf(stderr, "hydra: cannot open %s\n", file.c_str());
            abort();
        }

        // Fingerprint hash index: largest power-of-two word count whose
        // footprint is <= budget/4 (2 GiB / 268M entries at an 8 GiB budget).
        uint64_t words = 64;
        while (words * 2 * 8 <= mem_budget_ / 4) words *= 2;
        index_.init(words);

        // Crash / restart recovery: a non-empty log is scanned (CRC-checked,
        // newest-LSN-wins) to rebuild the index before anything else runs.
        // An overflow sidecar beside an EMPTY log (oversized-only store)
        // must also be recovered: recover_log(0) skips the scan and loads it.
        struct stat sb, ov;
        bool have_sidecar =
            stat((dir_ + "/hydra_overflow.dat").c_str(), &ov) == 0;
        if (fstat(fd_, &sb) == 0 && (sb.st_size > 0 || have_sidecar))
            recover_log((uint64_t)sb.st_size);
    }

    if (disk_mode_) {
        // Doorkeeper bitmaps: two generations, each budget/64 bytes (128 MiB
        // at 8 GiB), so a read miss must be the key's 2nd+ recent access to
        // be admitted. Sized from the budget only — no workload assumptions.
        uint64_t dk_bytes = 64;
        while (dk_bytes * 2 * 64 <= mem_budget_) dk_bytes *= 2;
        dk_mask_ = dk_bytes * 8 - 1;
        for (int g = 0; g < 2; ++g) {
            void* p = mmap(nullptr, dk_bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) { fprintf(stderr, "hydra: dk mmap failed\n"); abort(); }
            dk_bits_[g] = static_cast<std::atomic<uint64_t>*>(p);
        }
    }

    // Cache sizing: leave room for the hash index and doorkeeper plus
    // allocator/thread slack, spend the rest on the value cache.
    size_t reserve = (disk_mode_ ? index_.nwords * 8 + (dk_mask_ + 1) / 4 : 0)
                     + (512ULL << 20);
    size_t cache_bytes = mem_budget_ > reserve ? mem_budget_ - reserve
                                               : mem_budget_ / 2;
    nsets_ = cache_bytes / (kEntryBytes * kAssoc);
    if (nsets_ < 64) nsets_ = 64;

    entries_ = static_cast<Entry*>(
        mmap(nullptr, nsets_ * kAssoc * sizeof(Entry), PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (entries_ == MAP_FAILED) {
        fprintf(stderr, "hydra: cache alloc failed\n");
        abort();
    }
    set_locks_ = new SpinLock[nsets_];

    if (disk_mode_) {
#ifdef HYDRA_HAVE_URING
        // HYDRA_NO_URING: test-only knob to exercise the synchronous pread
        // fallback pool on kernels that do have io_uring. Default: off.
        use_uring_ = (getenv("HYDRA_NO_URING") == nullptr);
        for (int i = 0; use_uring_ && i < kIoThreads; ++i) {
            rings_[i] = std::make_unique<Uring>();
            if (!rings_[i]->init(kRingDepth)) { use_uring_ = false; break; }
        }
        if (use_uring_) {
            for (int i = 0; i < kIoThreads; ++i)
                io_threads_.emplace_back(&HydraStore::io_main_uring, this, i);
        } else {
            for (int i = 0; i < kIoThreads; ++i)
                if (rings_[i]) { rings_[i]->destroy(); rings_[i].reset(); }
        }
        if (!use_uring_)
#endif
        {
            // Fallback: blocking preads spread over many threads; multiple
            // threads may share one shard queue.
            for (int i = 0; i < kSyncIoThreads; ++i)
                io_threads_.emplace_back(&HydraStore::io_main, this,
                                         i % kIoThreads);
        }
        for (int i = 0; i < kCleanerThreads; ++i)
            cleaners_.emplace_back(&HydraStore::cleaner_main, this, i);
        compactor_ = std::thread(&HydraStore::compactor_main, this);
    }

    fprintf(stderr,
            "hydra: init budget=%.2f GB cache=%.2f GB sets=%llu assoc=%zu "
            "disk=%s\n",
            mem_budget_ / 1e9, nsets_ * kAssoc * (double)kEntryBytes / 1e9,
            (unsigned long long)nsets_, kAssoc,
            disk_mode_ ? dir_.c_str() : "(none)");
}

void HydraStore::stop_background() {
    if (stopping_.exchange(true)) return;
    for (auto& sh : io_shards_) {
        std::lock_guard<std::mutex> lk(sh.mu);   // fence queued producers
        sh.cv.notify_all();
    }
    for (auto& t : io_threads_) t.join();
    for (auto& t : cleaners_) t.join();
    if (compactor_.joinable()) compactor_.join();
    io_threads_.clear();
    cleaners_.clear();
}

HydraStore::~HydraStore() {
    stop_background();
    if (disk_mode_ && fd_ >= 0) {
        // Clean-shutdown durability (best effort): land every session's
        // parked partial chunk, persist the oversize map, fdatasync. All
        // threads are joined — single-threaded from here.
        {
            std::lock_guard<std::mutex> lk(sess_mu_);
            for (auto& se : sessions_)
                if (se->chunk_buf && se->chunk_fill > 0) flush_chunk(se.get());
        }
        write_overflow_file();
        if (fdatasync(fd_) < 0) durable_ok_.store(false);
    }
    if (entries_) munmap(entries_, nsets_ * kAssoc * sizeof(Entry));
    delete[] set_locks_;
    index_.destroy();
    for (int g = 0; g < 2; ++g)
        if (dk_bits_[g]) munmap(dk_bits_[g], (dk_mask_ + 1) / 8);
    if (fd_ >= 0) close(fd_);
    uint64_t mism = key_mismatch_.load();
    if (mism) fprintf(stderr, "hydra: WARNING %llu slot key mismatches\n",
                      (unsigned long long)mism);
}

// ── Disk primitives ──────────────────────────────────────────────────────────
// FAIL-SOFT: a failed page read zero-fills (looks like "no landed slot"),
// counts a read error, and returns false; the caller's lookup degrades to
// NotFound instead of killing the process.
bool HydraStore::read_page(uint64_t page, uint8_t* buf) {
    ssize_t n = xpread(fd_, buf, kPageBytes, (off_t)(page * kPageBytes));
    if (n < 0) {
        read_errors_.fetch_add(1, std::memory_order_relaxed);
        memset(buf, 0, kPageBytes);
        return false;
    }
    if ((size_t)n < kPageBytes) {
        memset(buf + n, 0, kPageBytes - n);
        // A short read BELOW the landed high-water mark is a real fault
        // (external truncation, dying device) — beyond it, it is just a
        // pre-reserved extent that has not landed yet (normal EOF).
        if ((page + 1) * kPageBytes <= landed_hw_.load(std::memory_order_relaxed))
            read_errors_.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

// Disk-verified index lookup: probes the fingerprint index and reads each
// candidate slot (following the prev-version chain on key mismatch) until
// the stored key matches exactly. Returns position+1 or 0 if the key has no
// landed slot. (Rare path; used by Delete.)
uint32_t HydraStore::verified_lookup(uint64_t key, Session* se) {
    if (!se->page_buf) { se->page_buf = alloc_aligned(kPageBytes); se->page_no = ~0ULL; }
    uint64_t h = mix64(key);
    uint64_t it = index_.start_of(h), scanned = 0;
    uint32_t p1, best = 0;
    uint64_t best_lsn = 0;
    while ((p1 = index_.next_candidate(h, it, scanned)) != 0) {
        uint32_t cur = p1, hops = 0;
        while (cur != 0) {
            uint64_t page = (uint64_t)(cur - 1) / kSlotsPerPage;
            if (!read_page(page, se->page_buf)) break;   // fail-soft
            se->page_no = ~0ULL;   // scratch use; don't trust as page cache
            const uint8_t* slot =
                se->page_buf + ((cur - 1) % kSlotsPerPage) * kSlotBytes;
            uint64_t skey; uint32_t sz1, prev;
            memcpy(&skey, slot, 8);
            memcpy(&sz1, slot + 8, 4);
            memcpy(&prev, slot + 12, 4);
            if (sz1 == 0) break;
            // Size must be plausible for a match: Delete tombstones a slot
            // by setting an out-of-range size, which must never match again.
            // CRC guards against trusting torn slots after a crash. Newest
            // is decided by LSN (positions are not temporal).
            if (skey == key && (uint64_t)(sz1 - 1) <= kSlotDataMax &&
                slot_crc_ok(slot)) {
                uint64_t l = slot_lsn(slot);
                if (l > best_lsn) { best_lsn = l; best = cur; }
                break;
            }
            // Chains are TEMPORAL (prepend-only), not position-ordered:
            // pre-reserved extents mean a newer slot can have a lower
            // position. Bound the walk defensively against corrupt loops.
            if (prev == cur || ++hops > (1u << 20)) break;
            cur = prev;               // fp collision: walk the version chain
        }
    }
    return best;
}

// Tombstone a landed slot in place: rewrite its 4 KiB page with the slot's
// size field set out of range, so no read/lookup can match the slot again,
// while its prev pointer keeps older chain links traversable for OTHER keys.
// Landed pages are never rewritten by the log-append flush path, so the only
// writers of a landed page are Delete calls, serialized by the page lock.
void HydraStore::tombstone_slot(uint64_t position, Session* se) {
    if (!se->rmw_buf) se->rmw_buf = alloc_aligned(kPageBytes);
    uint64_t page = position / kSlotsPerPage;
    SpinLock& pl = page_locks_[page % kPageLockCount];
    pl.lock();
    if (!read_page(page, se->rmw_buf)) { pl.unlock(); return; }  // fail-soft
    uint8_t* slot = se->rmw_buf + (position % kSlotsPerPage) * kSlotBytes;
    uint32_t sz1 = 0xFFFFFFFFu;   // size 0xFFFFFFFE: unmatchable, non-zero
    memcpy(slot + 8, &sz1, 4);
    // Fresh LSN + reseal: the tombstone must be the key's NEWEST version so
    // recovery keeps the key deleted (recovery is newest-LSN-wins).
    slot_seal(slot, lsn_.fetch_add(1, std::memory_order_relaxed));
    if (!xpwrite(fd_, se->rmw_buf, kPageBytes, (off_t)(page * kPageBytes))) {
        // FAIL-SOFT: in-process deletion stays correct (the del-registry
        // mark gates every reader); only the delete's crash-durability is
        // degraded, which the sticky flag surfaces.
        write_errors_.fetch_add(1, std::memory_order_relaxed);
        durable_ok_.store(false, std::memory_order_relaxed);
    }
    pl.unlock();
    flush_epoch_.fetch_add(1, std::memory_order_relaxed);
}

// Re-stage a fresh tombstone slot for `key` into the session's chunk (no
// chunk_meta record: the landing loop never links it — the slot exists only
// for recovery's newest-LSN-wins scan). Staged and sealed UNDER the key's
// set lock: del_unmark + value staging on a reinsert run under this same
// lock, so a tombstone sealed here can never carry a higher LSN than a
// concurrent reinsert's value (which would erase the key at recovery).
bool HydraStore::restage_tombstone(uint64_t key, Session* se) {
    uint64_t s = set_of(key);
    for (int tries = 0; tries < 4; ++tries) {
        set_locks_[s].lock();
        if (!is_deleted(key)) {         // reinserted meanwhile: no tombstone
            set_locks_[s].unlock();
            return true;
        }
        ensure_chunk(se);
        if (se->chunk_active && se->chunk_cap > 0 &&
            se->chunk_fill >= se->chunk_cap) {
            // Full-but-unlanded chunk (a previous landing failed): retry the
            // landing outside the lock; if the disk stays broken, give up —
            // the caller leaves the region unreclaimed.
            set_locks_[s].unlock();
            flush_chunk(se);
            if (se->chunk_fill > 0) return false;
            continue;
        }
        uint64_t position = se->chunk_base + se->chunk_fill;
        if (position + 1 > 0xFFFFFFFFULL) {
            set_locks_[s].unlock();
            return false;
        }
        uint8_t* slot = se->chunk_buf + (size_t)se->chunk_fill * kSlotBytes;
        memset(slot, 0, kSlotBytes);
        memcpy(slot, &key, 8);
        uint32_t sz1 = 0xFFFFFFFFu;     // size 0xFFFFFFFE: unmatchable
        memcpy(slot + 8, &sz1, 4);      // prev = 0: no chain
        slot_seal(slot, lsn_.fetch_add(1, std::memory_order_relaxed));
        se->chunk_fill++;
        bool full = (se->chunk_fill == se->chunk_cap);
        set_locks_[s].unlock();
        if (full) flush_chunk(se);
        return true;
    }
    return false;
}

// Log-append flush staging: copy the DIRTY entry's bytes into the session's
// sequential chunk at a fresh position; the entry becomes S_FLUSHING and
// stays pinned (unevictable, value authoritative in cache) until the chunk
// lands, at which point flush_chunk moves the index word old->new.
// Called with set lock s held. Returns true if the chunk is now full and the
// caller must flush_chunk() after releasing the set lock.
bool HydraStore::stage_flush(uint64_t s, int w, Session* se) {
    (void)s;
    Entry& e = set_base(s)[w];
    ensure_chunk(se);
    // A full-but-unlanded chunk (previous flush_chunk failed, e.g. ENOSPC)
    // cannot accept more slots: tell the caller to (re)try flush_chunk.
    if (se->chunk_active && se->chunk_cap > 0 && se->chunk_fill >= se->chunk_cap)
        return true;
    uint64_t position = se->chunk_base + se->chunk_fill;
    if (position + 1 > 0xFFFFFFFFULL) {
        // 32-bit position space exhausted (>512 GB file): keep the entry
        // dirty. The background compactor recycles position space, so this
        // is only reachable when compaction cannot keep up.
        return false;
    }
    uint8_t* slot = se->chunk_buf + (size_t)se->chunk_fill * kSlotBytes;
    uint32_t sz1 = (uint32_t)e.size + 1;
    uint32_t prev = e.pos1;             // back-pointer: previous version slot
    memcpy(slot, &e.key, 8);
    memcpy(slot + 8, &sz1, 4);
    memcpy(slot + 12, &prev, 4);
    memcpy(slot + 16, e.data, e.size);
    memset(slot + 16 + e.size, 0, kSlotDataMax - e.size);
    slot_seal(slot, lsn_.fetch_add(1, std::memory_order_relaxed));
    se->chunk_fill++;
    e.ver = stage_ver_.fetch_add(1, std::memory_order_relaxed);  // unique pin
    se->chunk_meta.push_back(
        Session::ChunkRec{e.key, e.ver, e.pos1, (uint32_t)position + 1});
    e.state = S_FLUSHING;
    return se->chunk_fill == se->chunk_cap;
}
// ── Load-phase sequential chunk writer ───────────────────────────────────────
void HydraStore::ensure_chunk(Session* se) {
    if (!se->chunk_buf) {
        se->chunk_buf = alloc_aligned(kChunkBytes);
        se->chunk_meta.reserve(kChunkSlots);
    }
    if (se->chunk_fill == 0 && !se->chunk_active) {
        uint64_t base = 0;
        uint32_t slots = 0;
        // Recycle punched extents first (compaction feeds this list); the
        // relaxed counter keeps the scored hot path lock-free while empty.
        if (free_count_.load(std::memory_order_relaxed) > 0) {
            std::lock_guard<std::mutex> lk(free_mu_);
            if (!free_extents_.empty()) {
                auto ext = free_extents_.back();
                free_extents_.pop_back();
                free_count_.fetch_sub(1, std::memory_order_relaxed);
                base = ext.first;
                slots = std::min<uint32_t>(ext.second, (uint32_t)kChunkSlots);
                reclaimed_bytes_.fetch_sub((uint64_t)slots * kSlotBytes,
                                           std::memory_order_relaxed);
                if (ext.second > slots) {
                    free_extents_.push_back({ext.first + slots,
                                             ext.second - slots});
                    free_count_.fetch_add(1, std::memory_order_relaxed);
                }
                owned_extents_.push_back({base, slots});
            }
        }
        if (slots == 0) {
            base = next_slot_.fetch_add(kChunkSlots, std::memory_order_relaxed);
            slots = (uint32_t)kChunkSlots;
            note_extent_owned(base, slots);
        }
        se->chunk_base = base;
        se->chunk_cap = slots;
        se->extent_base = base;
        se->extent_slots = slots;
        se->chunk_meta.clear();
        se->chunk_active = true;   // extent stays reserved until fully consumed
    }
}

void HydraStore::note_extent_owned(uint64_t base, uint32_t slots) {
    std::lock_guard<std::mutex> lk(free_mu_);
    owned_extents_.push_back({base, slots});
}

void HydraStore::note_extent_done(uint64_t base) {
    std::lock_guard<std::mutex> lk(free_mu_);
    for (size_t i = 0; i < owned_extents_.size(); ++i) {
        if (owned_extents_[i].first == base) {
            owned_extents_[i] = owned_extents_.back();
            owned_extents_.pop_back();
            return;
        }
    }
}

void HydraStore::flush_partials(Session* se) {
    if (se->chunk_buf && se->chunk_fill > 0) flush_chunk(se);
}

void HydraStore::flush_chunk(Session* se) {
    if (se->chunk_fill == 0) return;
    size_t used = (size_t)se->chunk_fill * kSlotBytes;
    size_t len = (used + kPageBytes - 1) & ~(kPageBytes - 1);
    memset(se->chunk_buf + used, 0, len - used);
    off_t off = (off_t)(se->chunk_base * kSlotBytes);
    if (!xpwrite(fd_, se->chunk_buf, len, off)) {
        // FAIL-SOFT (ENOSPC/EIO): nothing landed. Keep the chunk staged —
        // every staged entry stays pinned in cache with its value still
        // authoritative — and retry on the next flush_chunk call (cleaner,
        // Checkpoint, StopSession, destructor, or the next full-chunk
        // trigger). Sticky counters surface the condition.
        write_errors_.fetch_add(1, std::memory_order_relaxed);
        durable_ok_.store(false, std::memory_order_relaxed);
        return;
    }
    // Advance the landed high-water mark (used to tell real short reads from
    // pre-reserved not-yet-landed extents).
    uint64_t hw = (uint64_t)off + len;
    uint64_t cur_hw = landed_hw_.load(std::memory_order_relaxed);
    while (cur_hw < hw &&
           !landed_hw_.compare_exchange_weak(cur_hw, hw,
                                             std::memory_order_relaxed)) {}

    // The chunk is durable: publish flushed positions and unpin entries.
    //  - first inserts (old_p1 == 0): index word was inserted at stage time;
    //    BUFFERED -> CLEAN (or DIRTY if re-upserted since staging).
    //  - flush records: land under the PIN-OWNERSHIP protocol below.
    for (auto& rec : se->chunk_meta) {
        uint64_t s = set_of(rec.key);
        // Warm the index bucket and the set's ways before taking the lock:
        // the landing does its index CAS under the set lock, and an
        // in-lock DRAM miss would lengthen every contender's spin.
        __builtin_prefetch(&index_.words[index_.start_of(mix64(rec.key))]);
        __builtin_prefetch(set_base(s));
        set_locks_[s].lock();
        Entry* base = set_base(s);
        Entry* own = nullptr;
        for (size_t w = 0; w < kAssoc; ++w) {
            Entry& e = base[w];
            if (e.state != S_EMPTY && e.key == rec.key) { own = &e; break; }
        }
        if (rec.old_p1 == 0) {
            // First insert: its index word was inserted at stage time, and
            // Delete waits for BUFFERED pins, so the pin is still present.
            if (own && own->state == S_BUFFERED) {
                if (own->ver == rec.ver) {
                    own->state = S_CLEAN;
                } else {
                    own->state = S_DIRTY;   // newer value needs another flush
                }
            }
            set_locks_[s].unlock();
            continue;
        }
        // Flush record: land it ONLY while we still own the pin — the exact
        // staging we made must still be in the cache (same key, S_FLUSHING,
        // same version, same source position). Otherwise the record is
        // ABANDONED and its slot stays unreachable:
        //  - Delete erased the pin: linking would resurrect the old bytes
        //    (even after a later reinsert unmarked the key);
        //  - re-upserted mid-flight (ver moved): the newer bytes go back to
        //    DIRTY below and reflush from old_p1.
        // The index CAS runs UNDER the set lock so it serializes with
        // Delete's sweep (mark -> sweep -> tombstone all under/after this
        // same lock). ver is a per-way monotonic 32-bit counter, so a false
        // ownership match needs a 2^32 wrap while this record is parked
        // (documented residual risk, same class as FASTER's ABA guards).
        if (!own || own->state != S_FLUSHING || own->pos1 != rec.old_p1) {
            // The slot at rec.new_p1 has already landed with live-looking
            // bytes but will never be linked. If the pin was broken by a
            // Delete, reseal that orphan as a tombstone NOW (still under
            // the set lock, like Delete's own poison pass, so a concurrent
            // reinsert cannot land a newer value with a lower LSN than
            // this tombstone). Otherwise recovery could resurrect the
            // deleted key from the orphan once the regions holding the
            // real tombstones are compacted away.
            if (is_deleted(rec.key))
                tombstone_slot((uint64_t)rec.new_p1 - 1, se);
            set_locks_[s].unlock();
            continue;
        }
        if (own->ver != rec.ver) {          // re-upserted while in flight
            own->state = S_DIRTY;           // reflush newer bytes from old_p1
            set_locks_[s].unlock();
            continue;
        }
        if (!index_.update(mix64(rec.key), rec.old_p1, rec.new_p1)) {
            // A same-fp key's flush replaced our head word (the shared-chain
            // protocol: fp-group members share one word, whoever lands last
            // owns the head, older versions hang off the prev chain in
            // TEMPORAL prepend order). Do NOT insert a second root: slot
            // positions are not version-ordered across sessions (extents
            // are pre-reserved), so a reader could not tell which root is
            // newer. Re-point the entry at the current head and retry: the
            // next flush prepends our value in front of the whole chain.
            uint32_t holder = index_.first_candidate(mix64(rec.key));
            if (holder != 0) {
                own->pos1 = holder;         // chain through the new head
                own->state = S_DIRTY;       // retry via cleaner/victim path
            } else {
                // No head word at all (defensive: Delete keeps words, so
                // this should not happen). Drop the staged copy.
                own->state = S_EMPTY;
                occupancy_.fetch_sub(1, std::memory_order_relaxed);
            }
            set_locks_[s].unlock();
            continue;
        }
        own->pos1 = rec.new_p1;
        own->state = S_CLEAN;       // ver==rec.ver checked above
        set_locks_[s].unlock();
    }
    flush_epoch_.fetch_add(1, std::memory_order_relaxed);
    // Continue inside the reserved extent at the next PAGE boundary (landed
    // pages are immutable): a partial flush no longer abandons up to 8191
    // reserved slots, which would burn the 32-bit position space ~8000x
    // faster than actual writes.
    uint32_t consumed = (uint32_t)(len / kSlotBytes);
    if (consumed < se->chunk_cap) {
        se->chunk_base += consumed;
        se->chunk_cap -= consumed;
    } else {
        se->chunk_cap = 0;
        se->chunk_active = false;
        note_extent_done(se->extent_base);   // compactor may touch it now
    }
    se->chunk_fill = 0;
    se->chunk_meta.clear();
}

// ── Victim selection (called with set s locked) ──────────────────────────────
// Returns a way index whose entry may be overwritten. May temporarily drop the
// set lock to flush a dirty victim; always returns with the lock held.
// Returns -1 if the caller should rescan the set (it changed while unlocked).
int HydraStore::find_victim(uint64_t s, Session* se) {
    // SLRU: inserts may claim an EMPTY way anywhere (so the cache fills at
    // load/cold-start), but may EVICT only within the probation ways;
    // resident protected entries are displaced exclusively through
    // promote_way, so bulk-cold traffic can never flush the hot set.
    Entry* base = set_base(s);
    for (int w = (int)kProbationWays; w < (int)kAssoc; ++w)
        if (base[w].state == S_EMPTY) return w;
    int clean_old = -1, clean_any = -1, dirty_old = -1, dirty_any = -1;
    for (int w = 0; w < (int)kProbationWays; ++w) {
        Entry& e = base[w];
        switch (e.state) {
        case S_EMPTY:
            return w;
        case S_CLEAN:
            if (!(e.recent & R_RECENT) && clean_old < 0) clean_old = w;
            if (clean_any < 0) clean_any = w;
            break;
        case S_DIRTY:
            if (!(e.recent & R_RECENT) && dirty_old < 0) dirty_old = w;
            if (dirty_any < 0) dirty_any = w;
            break;
        default:  // S_BUFFERED: pinned, skip
            break;
        }
    }
    // CLOCK aging: when every probation way is "recent", strip their bits so
    // the segment stays evictable and the hot set can drift over time.
    if (clean_old < 0 && dirty_old < 0) {
        for (int w = 0; w < (int)kProbationWays; ++w)
            base[w].recent &= (uint8_t)~R_RECENT;
    }
    if (clean_old >= 0 || clean_any >= 0) {
        int v = clean_old >= 0 ? clean_old : clean_any;
        bump(se->evictions);
        occupancy_.fetch_sub(1, std::memory_order_relaxed);
        return v;
    }
    int d = dirty_old >= 0 ? dirty_old : dirty_any;
    if (d >= 0) {
        // Sync-assist: stage the dirty victim into our flush chunk (it
        // becomes S_FLUSHING/pinned; evictable once the chunk lands).
        // Cheap: a memcpy, no inline disk RMW. (stage_flush can refuse on
        // position-space exhaustion; the entry then just stays DIRTY.)
        bool full = stage_flush(s, d, se);
        if (full) {
            set_locks_[s].unlock();
            flush_chunk(se);
            set_locks_[s].lock();
            return -1;   // set may have changed while unlocked
        }
    }
    // Forward progress: probation has no immediate victim (dirty just
    // staged, or everything pinned). Fall back to a clean protected way so
    // writers never spin waiting for chunk landings. This is rare in steady
    // state, so the protected segment still repels bulk-cold churn.
    int po = -1, pa = -1;
    for (int w = (int)kProbationWays; w < (int)kAssoc; ++w) {
        Entry& e = base[w];
        if (e.state != S_CLEAN) continue;
        if (!(e.recent & R_RECENT) && po < 0) po = w;
        if (pa < 0) pa = w;
    }
    int v = po >= 0 ? po : pa;
    if (v >= 0) {
        bump(se->evictions);
        occupancy_.fetch_sub(1, std::memory_order_relaxed);
        return v;
    }
    if (d >= 0) return -1;   // staged; caller rescans while it lands
    // Every probation way is pinned (BUFFERED/FLUSHING) and no clean
    // protected way exists. Flush our own chunk to unpin ours, then have
    // the caller retry; other threads' flushes will unpin the rest.
    set_locks_[s].unlock();
    flush_partials(se);
    std::this_thread::yield();
    set_locks_[s].lock();
    return -1;
}

// ── Overflow (rare path) ─────────────────────────────────────────────────────
bool HydraStore::overflow_contains(uint64_t key) {
    OverflowShard& sh = shard(key);
    std::lock_guard<std::mutex> lk(sh.mu);
    return sh.map.find(key) != sh.map.end();
}

bool HydraStore::overflow_get(uint64_t key, GenValue& out) {
    if (overflow_count_.load(std::memory_order_relaxed) == 0) return false;
    OverflowShard& sh = shard(key);
    std::lock_guard<std::mutex> lk(sh.mu);
    auto it = sh.map.find(key);
    if (it == sh.map.end()) return false;
    out.size = (uint32_t)it->second.bytes.size();
    memcpy(out.data, it->second.bytes.data(), it->second.bytes.size());
    return true;
}

void HydraStore::overflow_put(uint64_t key, const uint8_t* data, size_t size) {
    OverflowShard& sh = shard(key);
    std::lock_guard<std::mutex> lk(sh.mu);
    int64_t delta = (size > kInlineMax) ? (int64_t)ovf_charge(size) : 0;
    uint64_t lsn = lsn_.fetch_add(1, std::memory_order_relaxed);
    auto it = sh.map.find(key);
    if (it != sh.map.end()) {
        if (it->second.bytes.size() > kInlineMax)
            delta -= (int64_t)ovf_charge(it->second.bytes.size());
        it->second.bytes.assign(reinterpret_cast<const char*>(data), size);
        it->second.lsn = lsn;
    } else {
        sh.map.emplace(key,
            OvfVal{std::string(reinterpret_cast<const char*>(data), size), lsn});
        overflow_count_.fetch_add(1, std::memory_order_relaxed);
    }
    if (delta > 0) oversize_bytes_.fetch_add((uint64_t)delta, std::memory_order_relaxed);
    else if (delta < 0) oversize_bytes_.fetch_sub((uint64_t)(-delta), std::memory_order_relaxed);
}

bool HydraStore::overflow_erase(uint64_t key) {
    if (overflow_count_.load(std::memory_order_relaxed) == 0) return false;
    OverflowShard& sh = shard(key);
    std::lock_guard<std::mutex> lk(sh.mu);
    auto it = sh.map.find(key);
    if (it == sh.map.end()) return false;
    if (it->second.bytes.size() > kInlineMax)
        oversize_bytes_.fetch_sub(ovf_charge(it->second.bytes.size()),
                                  std::memory_order_relaxed);
    sh.map.erase(it);
    overflow_count_.fetch_sub(1, std::memory_order_relaxed);
    return true;
}

// Advisory cap check for oversized upserts (honest rejection: over the cap,
// nothing is inserted, truncated or evicted — the caller counts the reject).
// Concurrent same-instant oversized upserts can overshoot the cap by at most
// one value each (bounded by thread count; documented).
bool HydraStore::oversize_would_fit(uint64_t key, size_t size) {
    uint64_t credit = 0;
    {
        OverflowShard& sh = shard(key);
        std::lock_guard<std::mutex> lk(sh.mu);
        auto it = sh.map.find(key);
        if (it != sh.map.end() && it->second.bytes.size() > kInlineMax)
            credit = ovf_charge(it->second.bytes.size());
    }
    return oversize_bytes_.load(std::memory_order_relaxed) + ovf_charge(size)
           <= oversize_cap_ + credit;
}

// ── Doorkeeper (read-miss admission filter) ─────────────────────────────────
// Two rotating bitmap generations approximate "seen within the recent
// window". A read miss is admitted to the cache only if its key was already
// seen (frequency >= 2), so streams of one-hit wonders cannot evict warmer
// entries. Rotation clears the older generation once enough marks accumulate
// (window ~= bits/8 accesses), which also lets the hot set drift.
bool HydraStore::dk_seen_and_mark(uint64_t h) {
    if (!dk_bits_[0]) return true;
    uint64_t bit = (h * 0x9E3779B97F4A7C15ULL) & dk_mask_;
    uint64_t word = bit >> 6, mask = 1ULL << (bit & 63);
    int cur = dk_cur_.load(std::memory_order_relaxed);
    bool seen =
        (dk_bits_[cur][word].load(std::memory_order_relaxed) & mask) ||
        (dk_bits_[cur ^ 1][word].load(std::memory_order_relaxed) & mask);
    if (!seen) {
        dk_bits_[cur][word].fetch_or(mask, std::memory_order_relaxed);
        dk_marks_.fetch_add(1, std::memory_order_relaxed);
    }
    return seen;
}

void HydraStore::dk_maybe_rotate() {
    if (!dk_bits_[0]) return;
    if (dk_marks_.load(std::memory_order_relaxed) < (dk_mask_ + 1) / 8) return;
    int cur = dk_cur_.load(std::memory_order_relaxed);
    // Atomic word clears: workers concurrently load/fetch_or these words,
    // so a plain memset would be a data race (could tear/lose marks).
    std::atomic<uint64_t>* g = dk_bits_[cur ^ 1];
    uint64_t words = (dk_mask_ + 1) / 64;
    for (uint64_t i = 0; i < words; ++i)
        g[i].store(0, std::memory_order_relaxed);
    dk_cur_.store(cur ^ 1, std::memory_order_relaxed);
    dk_marks_.store(0, std::memory_order_relaxed);
}

// ── Read paths ───────────────────────────────────────────────────────────────
bool HydraStore::cache_probe(uint64_t key, GenValue& out) {
    uint64_t s = set_of(key);
    set_locks_[s].lock();
    Entry* base = set_base(s);
    for (size_t w = 0; w < kAssoc; ++w) {
        Entry& e = base[w];
        if (e.state != S_EMPTY && e.key == key) {
            out.size = e.size;
            memcpy(out.data, e.data, e.size);
            bool again = (e.recent & R_RECENT) != 0;
            e.recent |= R_RECENT;
            if (again && w < kProbationWays) promote_way(s, (int)w);
            set_locks_[s].unlock();
            return true;
        }
    }
    set_locks_[s].unlock();
    return false;
}

// A probation entry was re-accessed while still marked recent: swap it with
// the coldest protected way so scans/cold writes can no longer evict it.
// Called with set lock s held; safe for any entry states (pins travel with
// the Entry struct and flush bookkeeping locates entries by key scan).
void HydraStore::promote_way(uint64_t s, int w) {
    Entry* base = set_base(s);
    int target = -1;
    for (int t = (int)kProbationWays; t < (int)kAssoc; ++t) {
        Entry& p = base[t];
        if (p.state == S_EMPTY) { target = t; break; }
        if (p.state == S_CLEAN && !(p.recent & R_RECENT) && target < 0) target = t;
    }
    if (target < 0) {
        // Every protected way is recent (or pinned/dirty): CLOCK-age the
        // clean ones so the protected segment adapts to drifting hot sets.
        for (int t = (int)kProbationWays; t < (int)kAssoc; ++t) {
            Entry& p = base[t];
            if (p.state == S_CLEAN) {
                p.recent &= (uint8_t)~R_RECENT;
                if (target < 0) target = t;
            }
        }
    }
    if (target < 0) return;   // all protected ways dirty/pinned: skip
    Entry tmp = base[target];
    base[target] = base[w];
    base[w] = tmp;
    base[w].recent &= (uint8_t)~R_RECENT;   // demoted: evictable next
}

OpStatus HydraStore::miss_read(uint64_t key, GenValue& out, Session* se) {
    // Re-check the cache: the key may have been upserted/admitted since the
    // caller's probe. The cached value is always the freshest.
    if (cache_probe(key, out)) return OpStatus::Ok;
    if (overflow_get(key, out)) return OpStatus::Ok;
    if (!disk_mode_) return OpStatus::NotFound;

    if (!se->page_buf) { se->page_buf = alloc_aligned(kPageBytes); se->page_no = ~0ULL; }

    // Probe the fingerprint index; verify every candidate against the exact
    // key stored in its on-disk slot, following the prev-version chain on a
    // mismatch (fingerprint collisions park older keys behind newer ones).
    // A key may be reachable through MULTIPLE candidate words (two same-fp
    // same-bucket keys can each have inserted their own root), so take the
    // match with the HIGHEST position across candidates. Within one chain
    // the FIRST match is the newest (versions are prepended in temporal
    // order).
    uint64_t h = mix64(key);
    uint64_t itc = index_.start_of(h), scanned = 0;
    uint32_t p1;
    uint32_t best_p1 = 0;
    uint64_t best_lsn = 0;
    uint16_t best_sz = 0;
    uint8_t best_data[kSlotDataMax];
    while ((p1 = index_.next_candidate(h, itc, scanned)) != 0) {
        uint32_t cur = p1, hops = 0;
        while (cur != 0) {
            uint32_t position = cur - 1;
            uint64_t page = (uint64_t)position / kSlotsPerPage;
            uint64_t epoch = flush_epoch_.load(std::memory_order_relaxed);
            if (se->page_no != page || se->page_epoch != epoch) {
                if (!read_page(page, se->page_buf)) {   // fail-soft
                    se->page_no = ~0ULL;
                    break;
                }
                se->page_no = page;
                se->page_epoch = epoch;
            }
            const uint8_t* slot =
                se->page_buf + (position % kSlotsPerPage) * kSlotBytes;
            uint64_t skey; uint32_t sz1, prev;
            memcpy(&skey, slot, 8);
            memcpy(&sz1, slot + 8, 4);
            memcpy(&prev, slot + 12, 4);
            if (sz1 == 0) {
                // Candidate slot has not landed yet. If it is really ours,
                // our value is pinned in cache (BUFFERED/FLUSHING) —
                // re-probe. Otherwise keep probing.
                se->page_no = ~0ULL;
                if (cache_probe(key, out)) return OpStatus::Ok;
                break;
            }
            uint16_t sz = (uint16_t)(sz1 - 1);
            if (skey == key && sz <= kSlotDataMax) {
                // CRC gate: torn crash remnants must never be served. LSN
                // decides "newest" across chains (positions not temporal).
                if (slot_crc_ok(slot)) {
                    uint64_t l = slot_lsn(slot);
                    if (l > best_lsn) {
                        best_lsn = l;
                        best_p1 = cur;
                        best_sz = sz;
                        memcpy(best_data, slot + 16, sz);
                    }
                }
                break;   // newest match within this chain
            }
            key_mismatch_.fetch_add(1, std::memory_order_relaxed);
            // Chains are TEMPORAL (prepend-only), not position-ordered.
            // Bound the walk defensively against corrupt loops.
            if (prev == cur || ++hops > (1u << 20)) break;
            cur = prev;               // walk the version chain
        }
    }
    if (best_p1 == 0) return OpStatus::NotFound;

    // Freshness: if the key entered the cache while we were doing I/O, the
    // cached copy supersedes the disk copy.
    if (cache_probe(key, out)) return OpStatus::Ok;

    out.size = best_sz;
    memcpy(out.data, best_data, best_sz);

    if (best_sz <= kInlineMax) admit_read(key, best_p1, best_data, best_sz, se);
    return OpStatus::Ok;
}

// Admit (clean) a disk-read value if the key has been seen before (the
// doorkeeper filters one-hit wonders so cold reads cannot evict warmer
// entries) and a non-blocking victim exists.
// [Measured: gating admissions on the doorkeeper and protecting admitted
//  entries (R_RECENT) beat both always-admit and cold-admit variants; the
//  speculative page-neighbor co-admission path below is kept for reference
//  but is disabled — it burned io-thread CPU and churned probation for a
//  net hit-rate LOSS on the 30 s cold-window benchmark.]
void HydraStore::admit_read(uint64_t key, uint32_t p1, const uint8_t* data,
                            uint16_t size, Session* se) {
    if (!dk_seen_and_mark(mix64(key))) return;
    uint64_t s = set_of(key);
    // Warm the two cache lines the locked section will touch (the index
    // bucket of the linearizability gate below and the set's first way)
    // BEFORE acquiring the lock, so the DRAM misses happen unlocked.
    __builtin_prefetch(&index_.words[index_.start_of(mix64(key))]);
    __builtin_prefetch(set_base(s));
    set_locks_[s].lock();
    Entry* base = set_base(s);
    int victim = -1;
    bool present = false;
    for (int w = 0; w < (int)kAssoc; ++w) {
        Entry& e = base[w];
        if (e.state != S_EMPTY && e.key == key) { present = true; break; }
        if (e.state == S_EMPTY) { if (victim < 0) victim = w; }   // empty: anywhere
        else if (w < (int)kProbationWays &&                       // evict: probation only
                 e.state == S_CLEAN && !(e.recent & R_RECENT) && victim < 0) victim = w;
    }
    if (!present && victim < 0) {
        // CLOCK aging on the read path too: strip R_RECENT from clean
        // probation ways so read-heavy / drifting workloads keep admitting
        // instead of freezing the segment.
        for (int w = 0; w < (int)kProbationWays; ++w) {
            Entry& e = base[w];
            if (e.state == S_CLEAN) {
                e.recent &= (uint8_t)~R_RECENT;
                if (victim < 0) victim = w;
            }
        }
    }
    if (!present && victim >= 0) {
        // Linearizability gate: verify UNDER THE SET LOCK that p1 is still
        // the key's newest indexed position. A concurrent upsert can land a
        // newer slot (index word rewritten old->new) and drop its cache
        // entry while our disk read was in flight; admitting the older
        // bytes as CLEAN would then serve stale data until eviction. The
        // stale case is exactly "p1 no longer among the candidates, or a
        // higher-position candidate exists". (A same-fp alias of another
        // key with a higher position only causes a skipped admission —
        // harmless and astronomically rare.)
        uint64_t h = mix64(key);
        uint64_t it = index_.start_of(h), scanned = 0;
        uint32_t c, maxc = 0;
        bool indexed = false;
        while ((c = index_.next_candidate(h, it, scanned)) != 0) {
            if (c == p1) indexed = true;
            if (c > maxc) maxc = c;
        }
        if (!indexed || maxc != p1) { set_locks_[s].unlock(); return; }
        // Deleted-key + overflow gates, both under the set lock: a Delete
        // or oversized Upsert concurrent with our disk read makes these
        // bytes non-authoritative (TOCTOU-safe here, racy anywhere else:
        // Delete's sweep and this admission serialize on set_locks_[s], so
        // whichever runs second sees the other's effect).
        if (is_deleted(key) ||
            (overflow_count_.load(std::memory_order_relaxed) != 0 &&
             overflow_contains(key))) {
            set_locks_[s].unlock();
            return;
        }
        Entry& e = base[victim];
        if (e.state == S_CLEAN) { bump(se->evictions); occupancy_.fetch_sub(1, std::memory_order_relaxed); }
        e.key = key;
        e.pos1 = p1;
        e.ver = e.ver + 1;
        e.size = (uint8_t)size;
        e.state = S_CLEAN;
        e.recent = R_RECENT;   // doorkeeper-known: protect from next victim scan
        memcpy(e.data, data, size);
        occupancy_.fetch_add(1, std::memory_order_relaxed);
    }
    set_locks_[s].unlock();
}

bool HydraStore::Read(uint64_t key, GenValue& out) {
    Session* se = session();
    // Delete monotonicity: a read that starts after Delete(key) returned
    // must never find the key (until re-upserted), even if a stale copy is
    // still landing. Free when no key was ever deleted (one relaxed load).
    if (is_deleted(key)) { bump(se->read_misses); return false; }
    if (cache_probe(key, out)) { bump(se->read_hits); return true; }
    bump(se->read_misses);
    if (!disk_mode_) return overflow_get(key, out);
    return miss_read(key, out, se) == OpStatus::Ok;
}

OpStatus HydraStore::ReadAsync(ReadSlot* slot) {
    Session* se = session();
    if (is_deleted(slot->key)) {   // see Read(): delete monotonicity
        bump(se->read_misses);
        slot->status = OpStatus::NotFound;
        slot->done.store(1, std::memory_order_release);
        return OpStatus::NotFound;
    }
    if (cache_probe(slot->key, slot->out)) {
        bump(se->read_hits);
        slot->status = OpStatus::Ok;
        slot->done.store(1, std::memory_order_release);
        return OpStatus::Ok;
    }
    bump(se->read_misses);
    if (!disk_mode_) {
        bool ok = overflow_get(slot->key, slot->out);
        slot->status = ok ? OpStatus::Ok : OpStatus::NotFound;
        slot->done.store(1, std::memory_order_release);
        return slot->status;
    }
    se->outstanding.fetch_add(1, std::memory_order_relaxed);
    IoShard& sh = io_shards_[se->io_rr++ % kIoThreads];
    {
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.q.push_back(IoReq{slot, se});
    }
    sh.cv.notify_one();
    return OpStatus::Pending;
}

void HydraStore::CompletePending(bool wait) {
    Session* se = session();
    if (wait) {
        flush_partials(se);
        while (se->outstanding.load(std::memory_order_acquire) != 0)
            std::this_thread::yield();
    }
}

void HydraStore::io_main(int shard) {
    Session* se = session();   // private session for page buffers
    IoShard& sh = io_shards_[shard];
    std::unique_lock<std::mutex> lk(sh.mu);
    for (;;) {
        sh.cv.wait(lk, [&] {
            return stopping_.load(std::memory_order_relaxed) || !sh.q.empty();
        });
        if (stopping_.load(std::memory_order_relaxed) && sh.q.empty()) return;
        IoReq r = sh.q.front();
        sh.q.pop_front();
        lk.unlock();

        OpStatus st = miss_read(r.slot->key, r.slot->out, se);
        r.slot->status = st;
        r.slot->done.store(1, std::memory_order_release);
        r.sess->outstanding.fetch_sub(1, std::memory_order_release);

        lk.lock();
    }
}

#ifdef HYDRA_HAVE_URING
// Publish a completed async read and release the requester's pipeline slot.
static inline void finish_slot(ReadSlot* slot, Session* sess, OpStatus st) {
    slot->status = st;
    slot->done.store(1, std::memory_order_release);
    sess->outstanding.fetch_sub(1, std::memory_order_release);
}

// Start servicing a queued read miss: re-check cache/overflow, then submit
// the first candidate page read. Returns true iff an IO was submitted.
bool HydraStore::uring_start(MissOp& op, uint32_t oi, Uring& ring,
                             Session* io_se) {
    (void)io_se;
    ReadSlot* slot = op.slot;
    if (cache_probe(slot->key, slot->out) ||
        overflow_get(slot->key, slot->out)) {
        finish_slot(slot, op.sess, OpStatus::Ok);
        return false;
    }
    op.h = mix64(slot->key);
    op.it = index_.start_of(op.h);
    op.scanned = 0;
    op.hops = 0;
    op.best_p1 = 0;
    op.best_lsn = 0;
    op.best_sz = 0;
    op.filled = 0;
    uint32_t p1 = index_.next_candidate(op.h, op.it, op.scanned);
    if (p1 == 0) {
        finish_slot(slot, op.sess, OpStatus::NotFound);
        return false;
    }
    op.cur = p1;
    op.page = (uint64_t)(op.cur - 1) / kSlotsPerPage;
    ring.prep_read(fd_, op.buf, kPageBytes, (off_t)(op.page * kPageBytes), oi);
    return true;
}

// Continue a read miss after its page read completed. Mirrors miss_read:
// verify the slot key, walk the prev version chain on mismatch, advance to
// the next fingerprint candidate when a chain ends. Same-page hops parse
// without further IO. Returns true iff another IO was submitted.
bool HydraStore::uring_advance(MissOp& op, uint32_t oi, Uring& ring, int res,
                               Session* io_se) {
    ReadSlot* slot = op.slot;
    const uint64_t key = slot->key;
    if (res < 0) {
        if (res == -EINTR || res == -EAGAIN) {   // transient: retry remainder
            ring.prep_read(fd_, op.buf + op.filled, kPageBytes - op.filled,
                           (off_t)(op.page * kPageBytes) + op.filled, oi);
            return true;
        }
        // FAIL-SOFT: count the fault, abandon this chain, keep evaluating
        // the remaining candidates (the process never dies for a bad read).
        read_errors_.fetch_add(1, std::memory_order_relaxed);
        memset(op.buf, 0, kPageBytes);
        op.cur = 0;
        op.filled = 0;
    } else if (op.filled + (uint32_t)res < kPageBytes) {
        if (res > 0) {
            // Mid-page short read: resubmit the remainder instead of
            // zero-filling bytes the device still owes us.
            op.filled += (uint32_t)res;
            ring.prep_read(fd_, op.buf + op.filled, kPageBytes - op.filled,
                           (off_t)(op.page * kPageBytes) + op.filled, oi);
            return true;
        }
        // EOF: normal beyond the landed high-water mark (pre-reserved
        // extent); below it, a real fault (external truncation).
        memset(op.buf + op.filled, 0, kPageBytes - op.filled);
        if ((op.page + 1) * kPageBytes <=
            landed_hw_.load(std::memory_order_relaxed))
            read_errors_.fetch_add(1, std::memory_order_relaxed);
        op.filled = 0;
    } else {
        op.filled = 0;
    }
    for (;;) {
        if (op.cur == 0) {
            uint32_t p1 = index_.next_candidate(op.h, op.it, op.scanned);
            if (p1 == 0) {
                // All candidate chains examined: serve the newest verified
                // match (highest position; the log only appends), if any.
                if (op.best_p1 == 0) {
                    finish_slot(slot, op.sess, OpStatus::NotFound);
                    return false;
                }
                // Freshness: a value cached while we did IO supersedes disk.
                if (!cache_probe(key, slot->out)) {
                    slot->out.size = op.best_sz;
                    memcpy(slot->out.data, op.best_data, op.best_sz);
                    if (op.best_sz <= kInlineMax)
                        admit_read(key, op.best_p1, op.best_data, op.best_sz,
                                   io_se);
                }
                finish_slot(slot, op.sess, OpStatus::Ok);
                return false;
            }
            op.cur = p1;
            op.hops = 0;   // hop bound is per candidate chain
        }
        uint32_t position = op.cur - 1;
        uint64_t page = (uint64_t)position / kSlotsPerPage;
        if (page != op.page) {
            op.page = page;
            op.filled = 0;
            ring.prep_read(fd_, op.buf, kPageBytes,
                           (off_t)(page * kPageBytes), oi);
            return true;
        }
        const uint8_t* sp = op.buf + (position % kSlotsPerPage) * kSlotBytes;
        uint64_t skey; uint32_t sz1, prev;
        memcpy(&skey, sp, 8);
        memcpy(&sz1, sp + 8, 4);
        memcpy(&prev, sp + 12, 4);
        if (sz1 == 0) {
            // Candidate slot has not landed yet. If it is really ours, our
            // value is pinned in cache (BUFFERED/FLUSHING) — re-probe.
            if (cache_probe(key, slot->out)) {
                finish_slot(slot, op.sess, OpStatus::Ok);
                return false;
            }
            op.cur = 0;   // chain dead-ends: try the next candidate
            continue;
        }
        uint16_t sz = (uint16_t)(sz1 - 1);
        if (skey == key && sz <= kSlotDataMax) {
            // CRC-gated; newest across chains decided by LSN (positions
            // are not temporal; compaction recycles them).
            if (slot_crc_ok(sp)) {
                uint64_t l = slot_lsn(sp);
                if (l > op.best_lsn) {
                    op.best_lsn = l;
                    op.best_p1 = op.cur;
                    op.best_sz = sz;
                    memcpy(op.best_data, sp + 16, sz);
                }
            }
            op.cur = 0;   // done with this chain: try the next candidate
            continue;
        }
        key_mismatch_.fetch_add(1, std::memory_order_relaxed);
        // Chains are TEMPORAL (prepend-only), not position-ordered.
        // Bound the walk defensively against corrupt loops.
        if (prev == op.cur || ++op.hops > (1u << 20)) { op.cur = 0; continue; }
        op.cur = prev;   // walk the version chain
    }
}

void HydraStore::io_main_uring(int shard) {
    Session* io_se = session();
    IoShard& sh = io_shards_[shard];
    Uring& ring = *rings_[shard];
    std::vector<MissOp> ops((size_t)kRingDepth);
    uint8_t* bufs = alloc_aligned((size_t)kRingDepth * kPageBytes);
    for (int i = 0; i < kRingDepth; ++i)
        ops[(size_t)i].buf = bufs + (size_t)i * kPageBytes;
    std::vector<uint32_t> free_ops;
    for (int i = kRingDepth - 1; i >= 0; --i) free_ops.push_back((uint32_t)i);
    unsigned inflight = 0;
    std::deque<IoReq> local;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(sh.mu);
            if (local.empty() && inflight == 0) {
                sh.cv.wait(lk, [&] {
                    return stopping_.load(std::memory_order_relaxed) ||
                           !sh.q.empty();
                });
                if (stopping_.load(std::memory_order_relaxed) && sh.q.empty()) {
                    free(bufs);
                    return;
                }
            }
            while (!sh.q.empty() && local.size() < (size_t)kRingDepth) {
                local.push_back(sh.q.front());
                sh.q.pop_front();
            }
        }
        while (!local.empty() && !free_ops.empty()) {
            IoReq r = local.front();
            local.pop_front();
            uint32_t oi = free_ops.back();
            MissOp& op = ops[oi];
            op.slot = r.slot;
            op.sess = r.sess;
            if (uring_start(op, oi, ring, io_se)) {
                free_ops.pop_back();
                inflight++;
            }
        }
        if (inflight == 0) continue;
        ring.enter(true);   // submit prepped sqes; block for >= 1 completion
        uint64_t ud; int res;
        while (ring.reap(ud, res)) {
            if (!uring_advance(ops[ud], (uint32_t)ud, ring, res, io_se)) {
                free_ops.push_back((uint32_t)ud);
                inflight--;
            }
        }
    }
}
#endif  // HYDRA_HAVE_URING

// ── Write paths ──────────────────────────────────────────────────────────────
void HydraStore::upsert_inline(uint64_t key, const uint8_t* data, uint16_t size) {
    Session* se = session();
    uint64_t s = set_of(key);
    int attempts = 0;   // bounds retry loops when the disk is failing
    for (;;) {
        set_locks_[s].lock();
        Entry* base = set_base(s);
        // In-place update?
        for (size_t w = 0; w < kAssoc; ++w) {
            Entry& e = base[w];
            if (e.state != S_EMPTY && e.key == key) {
                e.ver = e.ver + 1;
                e.size = (uint8_t)size;
                memcpy(e.data, data, size);
                bool again = (e.recent & R_RECENT) != 0;
                e.recent |= R_RECENT;
                if (e.state == S_CLEAN) e.state = S_DIRTY;
                if (again && w < kProbationWays) promote_way(s, (int)w);
                // Under the set lock: an unlocked erase could delete a NEWER
                // oversized value published in between (lock order set->
                // overflow is shared with the oversized Upsert path).
                if (overflow_count_.load(std::memory_order_relaxed)) overflow_erase(key);
                del_unmark(key);   // publish-then-unmark under the set lock
                set_locks_[s].unlock();
                return;
            }
        }
        uint64_t h = mix64(key);
        uint32_t p1 = index_.first_candidate(h);
        // Writes also feed the frequency filter. A first-seen (one-hit-
        // wonder) written key is admitted — the value must live somewhere
        // until it is flushed — but WITHOUT the recent bit, so once the
        // cleaner lands it, it is the next eviction victim instead of
        // polluting the hot set.
        bool seen = dk_seen_and_mark(h);
        int v = find_victim(s, se);
        if (v < 0) {   // set changed; retry (bounded when the disk is dead)
            set_locks_[s].unlock();
            if (++attempts > 64) {
                // Pathological pin-up (every way pinned behind a failing
                // disk): absorb the write in the overflow map so Upsert
                // never livelocks and never loses the bytes.
                set_locks_[s].lock();
                overflow_put(key, data, size);
                del_unmark(key);
                set_locks_[s].unlock();
                return;
            }
            continue;
        }
        Entry& e = base[v];
        uint8_t new_state;
        uint8_t flags = seen ? R_RECENT : 0;
        if (p1 == 0) {
            // First insert of this key: reserve a slot position and stage the
            // bytes in the sequential chunk buffer (pinned until it lands).
            ensure_chunk(se);
            if (se->chunk_active && se->chunk_fill >= se->chunk_cap) {
                // Full-but-unlanded chunk (an earlier landing failed, e.g.
                // ENOSPC). Retry the landing OUTSIDE the lock a few times;
                // if the disk stays broken, absorb the write in the
                // overflow map — Upsert must never hang or crash.
                if (e.state != S_EMPTY)
                    occupancy_.fetch_add(1, std::memory_order_relaxed);
                if (++attempts <= 16) {
                    set_locks_[s].unlock();
                    flush_chunk(se);
                    continue;
                }
                overflow_put(key, data, size);
                del_unmark(key);
                set_locks_[s].unlock();
                return;
            }
            uint64_t position = se->chunk_base + se->chunk_fill;
            if (position + 1 > 0xFFFFFFFFULL ||
                !index_.insert(h, (uint32_t)position + 1)) {
                // Slot space or index capacity exhausted — overflow map.
                // Roll back the eviction accounting: the victim entry was not
                // actually replaced (find_victim already decremented
                // occupancy for a non-empty clean victim).
                if (e.state != S_EMPTY)
                    occupancy_.fetch_add(1, std::memory_order_relaxed);
                overflow_put(key, data, size);
                del_unmark(key);   // publish-then-unmark under the set lock
                set_locks_[s].unlock();
                return;
            }
            uint8_t* slot = se->chunk_buf + (size_t)se->chunk_fill * kSlotBytes;
            uint32_t sz1 = (uint32_t)size + 1;
            uint32_t prev = 0;
            memcpy(slot, &key, 8);
            memcpy(slot + 8, &sz1, 4);
            memcpy(slot + 12, &prev, 4);
            memcpy(slot + 16, data, size);
            memset(slot + 16 + size, 0, kSlotDataMax - size);
            slot_seal(slot, lsn_.fetch_add(1, std::memory_order_relaxed));
            se->chunk_fill++;
            p1 = (uint32_t)position + 1;
            new_state = S_BUFFERED;
        } else {
            // Key exists (or an astronomically rare fingerprint collision —
            // harmless: flushes append with a prev back-pointer, so a stolen
            // head is still reachable through the version chain).
            new_state = S_DIRTY;
        }
        e.key = key;
        e.pos1 = p1;
        // BUFFERED entries are pins referenced by a chunk record: stamp a
        // globally unique staging token (see stage_ver_). Plain dirty
        // entries just bump their way-local version.
        e.ver = (new_state == S_BUFFERED)
                    ? stage_ver_.fetch_add(1, std::memory_order_relaxed)
                    : e.ver + 1;
        e.size = (uint8_t)size;
        e.state = new_state;
        e.recent = flags;
        memcpy(e.data, data, size);
        occupancy_.fetch_add(1, std::memory_order_relaxed);
        if (new_state == S_BUFFERED)
            se->chunk_meta.push_back(Session::ChunkRec{key, e.ver, 0, p1});
        bool full = (se->chunk_active && se->chunk_fill == se->chunk_cap);
        if (overflow_count_.load(std::memory_order_relaxed)) overflow_erase(key);
        del_unmark(key);   // publish-then-unmark, all under the set lock
        set_locks_[s].unlock();
        if (full) flush_chunk(se);
        return;
    }
}

void HydraStore::Upsert(uint64_t key, const GenValue& value) {
    // Re-inserting a deleted key: the mark is cleared AT THE PUBLICATION
    // POINT, under the key's set lock (upsert_inline / the oversized path
    // below), never earlier — an early unmark would let an in-flight
    // admission of pre-delete bytes slip through the is_deleted gate while
    // the new value is not yet visible. (del_unmark is a no-op fast path
    // while the registry is empty.)
    if (!disk_mode_) {
        overflow_put(key, value.data, value.size);
        del_unmark(key);
        return;
    }
    if (value.size > kInlineMax) {
        // Bounded oversize: reject (and count) before touching ANY state, so
        // a rejected upsert can never damage the key's current value.
        if (!oversize_would_fit(key, value.size)) {
            rejected_oversize_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // Drop any stale inline cache entry so reads see the overflow value.
        // A BUFFERED entry stays pinned until its chunk lands (removing it
        // early would let the stale chunk write escape its cancellation), so
        // wait for the landing first — rare path, bounded by chunk flush.
        uint64_t s = set_of(key);
        int spins = 0;
        for (;;) {
            bool buffered = false;
            set_locks_[s].lock();
            Entry* base = set_base(s);
            for (size_t w = 0; w < kAssoc; ++w) {
                Entry& e = base[w];
                if (e.state != S_EMPTY && e.key == key) {
                    if (e.state == S_BUFFERED || e.state == S_FLUSHING) { buffered = true; break; }
                    e.state = S_EMPTY;
                    occupancy_.fetch_sub(1, std::memory_order_relaxed);
                    break;
                }
            }
            if (!buffered) break;
            set_locks_[s].unlock();
            Session* se = session();
            flush_partials(se);
            // The pin may belong to ANOTHER session's in-flight chunk; wait
            // politely (bounded by that chunk's landing) instead of burning
            // a core. Rare path — never taken by inline-sized workloads.
            if (++spins > 100)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            else
                std::this_thread::yield();
        }
        overflow_put(key, value.data, value.size);
        del_unmark(key);   // publish-then-unmark under the set lock
        set_locks_[s].unlock();
        return;
    }
    upsert_inline(key, value.data, (uint16_t)value.size);
}

void HydraStore::RMW(uint64_t key, const uint8_t* mod_data, size_t mod_size) {
    // Atomic add of the leading 8 bytes; remaining bytes take the mod bytes.
    static SpinLock rmw_locks[1024];
    SpinLock& l = rmw_locks[mix64(key) % 1024];
    l.lock();
    Session* se = session();
    // Fast path: cache-resident key — read+modify+write entirely UNDER the
    // key's set lock, which closes the RMW-vs-Upsert lost-update window for
    // resident keys (plain Upserts of the same key take the same lock).
    // Non-resident keys fall back to the read+upsert path below, which is
    // atomic against other RMWs (rmw_locks) but not against a concurrent
    // blind Upsert (documented residual; first RMW residency closes it).
    if (disk_mode_ && !is_deleted(key) && mod_size >= 1) {
        uint64_t s = set_of(key);
        set_locks_[s].lock();
        Entry* base = set_base(s);
        for (size_t w = 0; w < kAssoc; ++w) {
            Entry& e = base[w];
            if (e.state == S_EMPTY || e.key != key) continue;
            GenValue nv2;
            nv2.size = (uint32_t)std::min<size_t>(mod_size, GenValue::kMaxSize);
            memcpy(nv2.data, mod_data, nv2.size);
            if (nv2.size > kInlineMax) break;   // oversized result: slow path
            uint64_t a = 0, b = 0;
            if (e.size >= 8) memcpy(&a, e.data, 8);
            if (nv2.size >= 8) {
                memcpy(&b, nv2.data, 8);
                a += b;
                memcpy(nv2.data, &a, 8);
            }
            e.ver = e.ver + 1;
            e.size = (uint8_t)nv2.size;
            memcpy(e.data, nv2.data, nv2.size);
            e.recent |= R_RECENT;
            if (e.state == S_CLEAN) e.state = S_DIRTY;
            if (overflow_count_.load(std::memory_order_relaxed)) overflow_erase(key);
            set_locks_[s].unlock();
            bump(se->rmw_hits);
            l.unlock();
            return;
        }
        set_locks_[s].unlock();
    }
    GenValue cur;
    bool found = false;
    if (!is_deleted(key)) {   // deleted key: RMW sees "absent" (insert)
        found = cache_probe(key, cur);
        if (!found && disk_mode_) found = miss_read(key, cur, se) == OpStatus::Ok;
        if (!found) found = overflow_get(key, cur);
    }
    GenValue nv;
    nv.size = (uint32_t)std::min<size_t>(mod_size, GenValue::kMaxSize);
    memcpy(nv.data, mod_data, nv.size);
    if (found) {
        bump(se->rmw_hits);
        uint64_t a = 0, b = 0;
        if (cur.size >= 8) memcpy(&a, cur.data, 8);
        if (nv.size >= 8) { memcpy(&b, nv.data, 8); a += b; memcpy(nv.data, &a, 8); }
    } else {
        bump(se->rmw_misses);
    }
    Upsert(key, nv);
    l.unlock();
}

bool HydraStore::Delete(uint64_t key) {
    if (!disk_mode_) return overflow_erase(key);
    // Mark FIRST. Every admission path and every flush landing re-checks
    // the registry / pin ownership under the key's set lock, so any copy
    // staged after our sweep below sees the mark and is skipped/dropped,
    // and any copy staged before is visible to the sweep. Reads that start
    // after Delete returns observe the mark and report NotFound even while
    // a stale copy is still somewhere in flight.
    bool newly = del_mark(key);
    bool existed = overflow_erase(key);
    uint64_t s = set_of(key);
    // Sweep the cache. CLEAN/DIRTY entries are erased outright. FLUSHING
    // entries (staged in some session's in-flight chunk) are erased too:
    // their landing re-checks pin ownership under this same lock and never
    // links the staged slot — waiting for the landing instead could stall
    // on another session's parked partial chunk. BUFFERED entries must land
    // first: their index word already points at the reserved slot, so the
    // slot has to become readable before verified_lookup below can poison
    // it on disk.
    int spins = 0;
    for (;;) {
        bool buffered = false;
        set_locks_[s].lock();
        Entry* base = set_base(s);
        for (size_t w = 0; w < kAssoc; ++w) {
            Entry& e = base[w];
            if (e.state != S_EMPTY && e.key == key) {
                if (e.state == S_BUFFERED) { buffered = true; break; }
                e.state = S_EMPTY;
                occupancy_.fetch_sub(1, std::memory_order_relaxed);
                existed = true;
                break;
            }
        }
        set_locks_[s].unlock();
        if (!buffered) break;
        flush_partials(session());
        // Pin may belong to another session's chunk; back off politely.
        if (++spins > 100)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        else
            std::this_thread::yield();
    }
    // Unlink EVERY on-disk version of the key. Alias repair (historic) and
    // prev chains hanging off OTHER keys' words can reach our slots —
    // erasing a single index word would let a later read resurrect an older
    // version. Loop: verified_lookup returns the newest still-matching
    // slot; tombstone it on disk (unmatchable size, chain intact). The
    // index word is KEPT: it routes same-fp chains for other keys.
    // Terminates because each pass permanently unmatches one slot. Slot
    // bytes are leaked until a compactor runs (log GC is out of scope, as
    // in FASTER's manual Log.Compact).
    Session* se = session();
    uint32_t p1;
    int poison_passes = 0;
    while ((p1 = verified_lookup(key, se)) != 0) {
        // Liveness bound: each pass normally unmatches one slot, but on a
        // dead disk tombstone_slot's page write fails soft and the same
        // slot keeps matching forever. In-process deletion stays correct
        // (the registry mark gates every reader); only crash-durability is
        // degraded, which the sticky flags already surface.
        if (++poison_passes > 4096) break;
        // Re-check the mark UNDER THE SET LOCK before each tombstone: a
        // concurrent Upsert republishes the key and unmarks it under this
        // same lock, and its relanded slot would otherwise be poisoned here
        // ("present until eviction, then absent" — non-linearizable). Once
        // the mark is gone, every remaining older on-disk version is
        // shadowed by the reinsert's chain prepend, so stopping is safe.
        // Holding the set lock across the (rare) tombstone I/O also blocks
        // a landing from publishing a new slot mid-poison (landings take
        // this lock). Lock order set -> page is unique to this path.
        set_locks_[s].lock();
        if (!is_deleted(key)) { set_locks_[s].unlock(); break; }
        tombstone_slot((uint64_t)p1 - 1, se);
        set_locks_[s].unlock();
        existed = true;
    }
    // Nothing existed anywhere (absent-key delete): drop the mark again so
    // the registry cannot grow without bound under absent-delete churn.
    // Safe: we erased no value, so there is no stale copy to hide; any
    // concurrent Upsert's value is legitimately visible (Upsert-after-
    // Delete unmarks anyway).
    if (newly && !existed) del_unmark(key);
    // Idempotence: a second Delete with no re-insert in between returns
    // false even if it swept up residue of the first one.
    return newly && existed;
}

// ── Cleaner: keep sets stocked with clean victims ────────────────────────────
void HydraStore::cleaner_main(int id) {
    Session* se = session();
    uint64_t s = (nsets_ / kCleanerThreads) * (uint64_t)id;
    for (;;) {
        if (stopping_.load(std::memory_order_relaxed)) return;
        uint64_t cleaned = 0, scanned = 0;
        while (scanned < 65536) {
            if (stopping_.load(std::memory_order_relaxed)) return;
            s++;
            if (s >= nsets_) s = 0;
            scanned++;
            Entry* base = set_base(s);
            // Cheap unlocked peek: any COLD dirty? (relaxed atomic byte
            // loads so the unlocked scan is race-free; staleness is fine —
            // it is only a hint, the locked rescan below decides.)
            // Hot (recent) dirty entries are deliberately left alone: they
            // coalesce many upserts into one eventual write-back. Flushing
            // them here would hammer the hottest set locks and re-stage the
            // same keys forever (each flush lands already-stale bytes).
            bool maybe = false;
            for (size_t w = 0; w < kProbationWays; ++w) {
                uint8_t st = base[w].state;    // RelaxedU8: race-free peek
                uint8_t rc = base[w].recent;
                if (st == S_DIRTY && !(rc & R_RECENT)) { maybe = true; break; }
            }
            if (!maybe) continue;
            set_locks_[s].lock();
            bool full = false;
            // Probation ways only: that is where eviction needs clean
            // victims. Protected dirty entries are hot by construction and
            // keep coalescing writes until demoted (or Checkpoint).
            for (int w = 0; w < (int)kProbationWays; ++w) {
                if (base[w].state != S_DIRTY) continue;
                if (base[w].recent & R_RECENT) continue;   // hot: coalesce
                full = stage_flush(s, w, se);
                cleaned++;
                if (full) break;
            }
            set_locks_[s].unlock();
            if (full) flush_chunk(se);
        }
        // Land partial chunks so staged (pinned) entries unpin promptly.
        flush_partials(se);
        if (id == 0) dk_maybe_rotate();
        // Pacing: cleaners are an optimization only — find_victim's
        // sync-assist staging bounds eviction latency — so cap background
        // CPU instead of spinning over 5.9M sets at full speed.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(cleaned ? 1 : 10));
    }
}

// ── Checkpoint: flush all dirty entries, then fdatasync ─────────────────────
void HydraStore::Checkpoint() {
    if (!disk_mode_) return;
    // Land the CALLER's partial chunk first: its own completed Upserts may
    // still sit BUFFERED in RAM. (Other sessions' partials land at their own
    // flush/StopSession — single-owner chunks cannot be stolen.)
    flush_partials(session());
    std::vector<std::thread> ts;
    uint64_t stripe = (nsets_ + kCheckpointThreads - 1) / kCheckpointThreads;
    for (int t = 0; t < kCheckpointThreads; ++t) {
        ts.emplace_back([this, t, stripe] {
            Session* se = session();
            uint64_t lo = stripe * (uint64_t)t;
            uint64_t hi = std::min<uint64_t>(lo + stripe, nsets_);
            for (uint64_t s = lo; s < hi; ++s) {
                Entry* base = set_base(s);
                for (int w = 0; w < (int)kAssoc; ++w) {
                    set_locks_[s].lock();
                    if (base[w].state != S_DIRTY) { set_locks_[s].unlock(); continue; }
                    bool full = stage_flush(s, w, se);
                    set_locks_[s].unlock();
                    if (full) flush_chunk(se);
                }
            }
            StopSession();   // flush partials + recycle the session
        });
    }
    for (auto& t : ts) t.join();
    while (fdatasync(fd_) < 0) {
        if (errno == EINTR) continue;
        // FAIL-SOFT: durability is degraded, not the process. The sticky
        // flag can never flip back to true (fsyncgate lesson: a later
        // "successful" fsync proves nothing about earlier lost writes).
        fprintf(stderr, "hydra: fdatasync failed (errno=%d)\n", errno);
        write_errors_.fetch_add(1, std::memory_order_relaxed);
        durable_ok_.store(false, std::memory_order_relaxed);
        break;
    }
    // Persist the oversize/overflow map (atomic tmp+rename sidecar).
    write_overflow_file();
}

// ── Crash recovery ───────────────────────────────────────────────────────────
// Rebuild the fingerprint index from the slot log: scan every slot, skip
// never-landed (zero) and CRC-invalid (torn) slots, keep the newest version
// per key by LSN. Tombstones participate: a newest-by-LSN tombstone keeps its
// key deleted (its index word then points at an unmatchable slot — the same
// state a live Delete leaves behind). Temporary per-word key/LSN arrays are
// freed before the cache is allocated, so peak RSS stays inside the budget.
void HydraStore::recover_log(uint64_t file_bytes) {
    const uint64_t nwords = index_.nwords;
    auto* rk = static_cast<uint64_t*>(
        mmap(nullptr, nwords * 8, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    auto* rl = static_cast<uint64_t*>(
        mmap(nullptr, nwords * 8, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    auto* rt = static_cast<uint8_t*>(     // 1 = word's winner is a tombstone
        mmap(nullptr, nwords, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    if (rk == MAP_FAILED || rl == MAP_FAILED || rt == MAP_FAILED) {
        fprintf(stderr, "hydra: recovery scratch mmap failed\n");
        abort();   // init-time invariant: cannot construct without RAM
    }
    const size_t kScan = 1 << 20;
    uint8_t* buf = alloc_aligned(kScan);
    // Slots past the 32-bit position space (a >512 GiB file) cannot be
    // indexed: stop the scan there rather than truncating positions.
    uint64_t scan_bytes = (file_bytes / kSlotBytes) * kSlotBytes;
    const uint64_t kMaxBytes = 0xFFFFFFFEULL * kSlotBytes;
    if (scan_bytes > kMaxBytes) scan_bytes = kMaxBytes;
    const uint64_t data_bytes = scan_bytes;
    uint64_t max_lsn = 0, dropped = 0, scan_errors = 0;
    for (uint64_t base = 0; base < data_bytes; base += kScan) {
        size_t want = (size_t)std::min<uint64_t>(kScan, data_bytes - base);
        size_t aligned = (want + kPageBytes - 1) & ~(kPageBytes - 1);
        ssize_t got = xpread(fd_, buf, aligned, (off_t)base);
        if (got < 0) {   // unreadable region: skip (fail-soft, degraded)
            read_errors_.fetch_add(1, std::memory_order_relaxed);
            scan_errors++;
            continue;
        }
        if ((size_t)got < want) memset(buf + got, 0, want - (size_t)got);
        for (size_t o = 0; o + kSlotBytes <= want; o += kSlotBytes) {
            const uint8_t* slot = buf + o;
            uint32_t sz1;
            memcpy(&sz1, slot + 8, 4);
            if (sz1 == 0) continue;              // hole / never landed
            if (!slot_crc_ok(slot)) {            // torn write: never served
                recover_torn_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            uint64_t key, lsn = slot_lsn(slot);
            memcpy(&key, slot, 8);
            uint32_t p1 = (uint32_t)((base + o) / kSlotBytes) + 1;
            if (lsn > max_lsn) max_lsn = lsn;
            uint8_t tomb = (uint64_t)(sz1 - 1) > kSlotDataMax ? 1 : 0;
            uint64_t h = mix64(key);
            const uint64_t fp = h & 0xFFFFFFFF00000000ULL;
            uint64_t it = index_.start_of(h);
            bool placed = false;
            for (uint64_t scanned = 0; scanned < nwords; ++scanned) {
                uint64_t w = index_.words[it].load(std::memory_order_relaxed);
                if (w == 0) {
                    if (index_.count.load(std::memory_order_relaxed) >=
                        index_.cap)
                        break;                    // over capacity: drop
                    index_.words[it].store(HashIndex::pack(h, p1),
                                           std::memory_order_relaxed);
                    index_.count.fetch_add(1, std::memory_order_relaxed);
                    rk[it] = key;
                    rl[it] = lsn;
                    rt[it] = tomb;
                    placed = true;
                    break;
                }
                if ((w & 0xFFFFFFFF00000000ULL) == fp && rk[it] == key) {
                    if (lsn > rl[it]) {           // newest-LSN-wins
                        index_.words[it].store(HashIndex::pack(h, p1),
                                               std::memory_order_relaxed);
                        rl[it] = lsn;
                        rt[it] = tomb;
                    }
                    placed = true;
                    break;
                }
                it = (it + 1) & (nwords - 1);
            }
            if (!placed) dropped++;
        }
    }
    free(buf);
    // Resume appends at the next page boundary past the recovered log; the
    // recovered log itself is the landed high-water mark.
    uint64_t nslots = (file_bytes + kSlotBytes - 1) / kSlotBytes;
    uint64_t next = (nslots + kSlotsPerPage - 1) & ~(uint64_t)(kSlotsPerPage - 1);
    next_slot_.store(next, std::memory_order_relaxed);
    landed_hw_.store(file_bytes & ~(uint64_t)(kPageBytes - 1),
                     std::memory_order_relaxed);
    load_overflow_file(rk, rl);   // needs rk/rl for LSN ordering
    // Re-mark tombstone-winning keys in the delete registry, so the
    // "compaction relocates a still-deleted key's tombstone" invariant
    // (restage_tombstone) holds across restart generations too — otherwise
    // a second-generation compactor would reclaim the tombstone while a
    // stale unlinked copy of the key still survives elsewhere in the log.
    // Keys whose overflow-sidecar entry outlived the tombstone are live.
    for (uint64_t it = 0; it < nwords; ++it)
        if (rt[it] && index_.words[it].load(std::memory_order_relaxed) != 0 &&
            !overflow_contains(rk[it]))
            del_mark(rk[it]);
    munmap(rk, nwords * 8);
    munmap(rl, nwords * 8);
    munmap(rt, nwords);
    lsn_.store(std::max(lsn_.load(std::memory_order_relaxed), max_lsn + 1),
               std::memory_order_relaxed);
    recovered_keys_.store(index_.count.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
    // recover_ok: 1 = clean, 0 = DEGRADED (unreadable regions were skipped
    // or the file exceeded the position space — keys may be missing).
    recover_ok_.store((scan_errors == 0 &&
                       (file_bytes / kSlotBytes) * kSlotBytes == data_bytes)
                          ? 1 : 0,
                      std::memory_order_relaxed);
    if (dropped)
        fprintf(stderr, "hydra: WARNING recovery dropped %llu keys "
                "(index capacity)\n", (unsigned long long)dropped);
    fprintf(stderr, "hydra: recovered %llu keys, %llu torn slots skipped\n",
            (unsigned long long)recovered_keys_.load(),
            (unsigned long long)recover_torn_.load());
}

// ── Oversize/overflow sidecar persistence ────────────────────────────────────
// Format: [magic u64][count u64] then per entry [key u64][lsn u64][size u32]
// [crc32c u32][bytes]. Written to a temp file, fdatasync'd, then atomically
// rename()d — a crash mid-write can never tear the previous sidecar.
static constexpr uint64_t kOvfMagic = 0x31564F4152445948ULL;   // "HYDRAOV1"

void HydraStore::write_overflow_file() {
    if (!disk_mode_) return;
    std::lock_guard<std::mutex> lk(ckpt_mu_);
    std::string path = dir_ + "/hydra_overflow.dat";
    if (overflow_count_.load(std::memory_order_relaxed) == 0) {
        if (overflow_file_seen_) {
            if (unlink(path.c_str()) == 0 || errno == ENOENT) {
                overflow_file_seen_ = false;
                sync_dir();          // make the unlink itself crash-durable
            } else {                 // sidecar may resurrect stale entries
                write_errors_.fetch_add(1, std::memory_order_relaxed);
                durable_ok_.store(false, std::memory_order_relaxed);
            }
        }
        return;
    }
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) {
        durable_ok_.store(false, std::memory_order_relaxed);
        write_errors_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    uint64_t hdr[2] = {kOvfMagic, 0};
    bool ok = fwrite(hdr, 8, 2, f) == 2;
    uint64_t count = 0;
    std::vector<uint8_t> rec;
    for (size_t i = 0; ok && i < kOverflowShards; ++i) {
        OverflowShard& sh = overflow_[i];
        std::lock_guard<std::mutex> g(sh.mu);
        for (auto& kv : sh.map) {
            uint32_t size = (uint32_t)kv.second.bytes.size();
            rec.resize(24 + size);
            memcpy(rec.data(), &kv.first, 8);
            memcpy(rec.data() + 8, &kv.second.lsn, 8);
            memcpy(rec.data() + 16, &size, 4);
            memcpy(rec.data() + 24, kv.second.bytes.data(), size);
            uint32_t crc = crc32c(rec.data(), 20);   // key|lsn|size
            crc ^= crc32c(rec.data() + 24, size);    // bytes (xor-combined)
            memcpy(rec.data() + 20, &crc, 4);
            ok = fwrite(rec.data(), 1, rec.size(), f) == rec.size();
            if (!ok) break;
            count++;
        }
    }
    if (ok) {   // patch the entry count into the header
        ok = fseek(f, 8, SEEK_SET) == 0 && fwrite(&count, 8, 1, f) == 1;
    }
    ok = ok && fflush(f) == 0 && fdatasync(fileno(f)) == 0;
    fclose(f);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        durable_ok_.store(false, std::memory_order_relaxed);
        write_errors_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    sync_dir();   // rename is only crash-durable once the directory is
    overflow_file_seen_ = true;
}

// fsync the store directory so a rename()/unlink() of the sidecar survives
// power loss (fail-soft: sticky durability flag on failure).
void HydraStore::sync_dir() {
    int dfd = open(dir_.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd < 0 || fsync(dfd) < 0)
        durable_ok_.store(false, std::memory_order_relaxed);
    if (dfd >= 0) close(dfd);
}

void HydraStore::load_overflow_file(const uint64_t* rk, const uint64_t* rl) {
    std::string path = dir_ + "/hydra_overflow.dat";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;
    overflow_file_seen_ = true;
    uint64_t hdr[2];
    if (fread(hdr, 8, 2, f) != 2 || hdr[0] != kOvfMagic) {
        fclose(f);
        return;
    }
    const uint64_t nwords = index_.nwords;
    std::vector<uint8_t> rec;
    for (uint64_t i = 0; i < hdr[1]; ++i) {
        uint8_t head[24];
        if (fread(head, 1, 24, f) != 24) break;
        uint64_t key, lsn;
        uint32_t size, crc;
        memcpy(&key, head, 8);
        memcpy(&lsn, head + 8, 8);
        memcpy(&size, head + 16, 4);
        memcpy(&crc, head + 20, 4);
        if (size > (64u << 20)) break;           // corrupt: stop
        if (size > GenValue::kMaxSize) {         // larger than the interface
            // can ever serve: skip it — never let a persisted record
            // overflow a GenValue buffer at read time.
            recover_torn_.fetch_add(1, std::memory_order_relaxed);
            if (fseek(f, (long)size, SEEK_CUR) != 0) break;
            continue;
        }
        rec.resize(size);
        if (size && fread(rec.data(), 1, size, f) != size) break;
        uint32_t want = crc32c(head, 20) ^ crc32c(rec.data(), size);
        if (want != crc) {                       // torn entry: skip
            recover_torn_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        // A newer slot-log version supersedes this sidecar entry.
        uint64_t h = mix64(key);
        const uint64_t fp = h & 0xFFFFFFFF00000000ULL;
        uint64_t it = index_.start_of(h);
        uint64_t slot_newest = 0;
        for (uint64_t scanned = 0; scanned < nwords; ++scanned) {
            uint64_t w = index_.words[it].load(std::memory_order_relaxed);
            if (w == 0) break;
            if ((w & 0xFFFFFFFF00000000ULL) == fp && rk[it] == key) {
                slot_newest = rl[it];
                break;
            }
            it = (it + 1) & (nwords - 1);
        }
        if (lsn <= slot_newest) continue;        // slot log wins
        OverflowShard& sh = shard(key);
        std::lock_guard<std::mutex> g(sh.mu);
        if (sh.map.emplace(key, OvfVal{std::string((const char*)rec.data(),
                                                   size), lsn}).second) {
            overflow_count_.fetch_add(1, std::memory_order_relaxed);
            if (size > kInlineMax)
                oversize_bytes_.fetch_add(ovf_charge(size),
                                          std::memory_order_relaxed);
        }
        if (lsn >= lsn_.load(std::memory_order_relaxed))
            lsn_.store(lsn + 1, std::memory_order_relaxed);
    }
    fclose(f);
}

// ── Background log compaction ────────────────────────────────────────────────
// Cyclic scavenger: when the log carries more than compact_factor_ x live
// bytes (and is above the floor), pick a region, relocate its live slots
// through the ordinary cache-flush protocol (so every existing correctness
// interlock applies), punch the region, and recycle its position space.
// Correctness never depends on the punch: LSN ordering makes any stale bytes
// that survive a failed punch lose every read/recovery comparison.
void HydraStore::compactor_main() {
    Session* se = session();
    while (!stopping_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (stopping_.load(std::memory_order_relaxed)) break;
        uint64_t alloc = next_slot_.load(std::memory_order_relaxed) * kSlotBytes;
        uint64_t rec = reclaimed_bytes_.load(std::memory_order_relaxed);
        uint64_t log_bytes = alloc > rec ? alloc - rec : 0;
        uint64_t live = index_.count.load(std::memory_order_relaxed) * kSlotBytes;
        if (log_bytes < compact_floor_) continue;
        if ((double)log_bytes < compact_factor_ * (double)live) continue;
        uint64_t total = next_slot_.load(std::memory_order_relaxed);
        if (total < kCompactRegionSlots) continue;
        if (compact_cursor_ + kCompactRegionSlots > total) compact_cursor_ = 0;
        uint64_t base = compact_cursor_;
        compact_cursor_ += kCompactRegionSlots;
        if (compact_region(se, base))
            compactions_run_.fetch_add(1, std::memory_order_relaxed);
    }
    flush_partials(se);
}

bool HydraStore::compact_region(Session* se, uint64_t base) {
    const uint64_t nslots = kCompactRegionSlots;
    {   // never touch extents owned by active sessions or already free
        std::lock_guard<std::mutex> lk(free_mu_);
        for (auto& e : owned_extents_)
            if (e.first < base + nslots && base < e.first + e.second)
                return false;
        for (auto& e : free_extents_)
            if (e.first < base + nslots && base < e.first + e.second)
                return false;
    }
    if (!se->page_buf) {
        se->page_buf = alloc_aligned(kPageBytes);
        se->page_no = ~0ULL;
    }
    uint8_t* buf = alloc_aligned(nslots * kSlotBytes);
    ssize_t got = xpread(fd_, buf, nslots * kSlotBytes,
                         (off_t)(base * kSlotBytes));
    if (got < 0) {
        read_errors_.fetch_add(1, std::memory_order_relaxed);
        free(buf);
        return false;
    }
    if ((size_t)got < nslots * kSlotBytes)
        memset(buf + got, 0, nslots * kSlotBytes - (size_t)got);
    bool all_clear = true;
    for (uint64_t i = 0; i < nslots; ++i) {
        if (stopping_.load(std::memory_order_relaxed)) {
            free(buf);
            return false;
        }
        const uint8_t* slot = buf + i * kSlotBytes;
        uint32_t sz1;
        memcpy(&sz1, slot + 8, 4);
        if (sz1 == 0) continue;                   // hole / never landed
        if (!slot_crc_ok(slot)) continue;         // torn garbage: reclaimable
        uint64_t sz = (uint64_t)(sz1 - 1);
        uint64_t key;
        memcpy(&key, slot, 8);
        if (sz > kSlotDataMax) {                  // tombstone
            // A still-deleted key's tombstone must SURVIVE reclamation:
            // stale unlinked copies of the key may exist elsewhere in the
            // log (broken-pin landings, superseded chain slots), and
            // punching the key's last tombstone would let recovery
            // resurrect them. Relocate it (fresh LSN, no index word);
            // reinserted keys (mark gone) are plain garbage. On failure,
            // keep the region unreclaimed and retry later.
            if (is_deleted(key) && !restage_tombstone(key, se))
                all_clear = false;
            continue;
        }
        uint32_t p1 = (uint32_t)(base + i) + 1;
        uint64_t h = mix64(key);
        uint64_t s = set_of(key);
        set_locks_[s].lock();
        Entry* bse = set_base(s);
        int own = -1;
        for (int w = 0; w < (int)kAssoc; ++w)
            if (bse[w].state != S_EMPTY && bse[w].key == key) { own = w; break; }
        if (own >= 0) {
            if (bse[own].pos1 == p1 &&
                (bse[own].state == S_CLEAN || bse[own].state == S_DIRTY)) {
                // Live, cache-resident: relocate via the ordinary flush
                // protocol (S_FLUSHING pin, landing CASes the index word).
                bool full = stage_flush(s, own, se);
                set_locks_[s].unlock();
                if (full) flush_chunk(se);
            } else if (bse[own].pos1 == p1) {
                set_locks_[s].unlock();
                all_clear = false;                // pinned mid-flight: later
            } else {
                set_locks_[s].unlock();           // superseded: dead slot
            }
            continue;
        }
        set_locks_[s].unlock();
        // Uncached: is this slot the key's newest version anywhere?
        if (is_deleted(key) || overflow_contains(key)) continue;   // dead
        if (verified_lookup(key, se) != p1) continue;              // dead
        // Admit CLEAN, then relocate through the normal flush protocol.
        set_locks_[s].lock();
        Entry* b2 = set_base(s);
        int v = find_victim(s, se);
        if (v < 0) {
            set_locks_[s].unlock();
            all_clear = false;
            continue;
        }
        bool present = false, still = false;
        for (int w = 0; w < (int)kAssoc; ++w)
            if (w != v && b2[w].state != S_EMPTY && b2[w].key == key) {
                present = true;
                break;
            }
        {   // re-validate reachability under the lock (find_victim may have
            // dropped it): the word must still route to this exact slot
            uint64_t it = index_.start_of(h), scanned = 0;
            uint32_t c;
            while ((c = index_.next_candidate(h, it, scanned)) != 0)
                if (c == p1) { still = true; break; }
        }
        if (present || !still || is_deleted(key)) {
            if (b2[v].state != S_EMPTY)   // roll back find_victim's eviction
                occupancy_.fetch_add(1, std::memory_order_relaxed);
            set_locks_[s].unlock();
            if (present || is_deleted(key)) continue;   // superseded: dead
            all_clear = false;                          // moved: retry later
            continue;
        }
        Entry& e = b2[v];
        e.key = key;
        e.pos1 = p1;
        e.ver = e.ver + 1;
        e.size = (uint8_t)sz;
        e.state = S_CLEAN;
        e.recent = 0;                        // do not pollute recency
        memcpy(e.data, slot + 16, sz);
        occupancy_.fetch_add(1, std::memory_order_relaxed);
        bool full = stage_flush(s, v, se);
        set_locks_[s].unlock();
        if (full) flush_chunk(se);
    }
    flush_partials(se);                      // land all relocations
    // If the landing failed (ENOSPC/EIO), relocations AND restaged
    // tombstones are still parked in the chunk buffer: the region must not
    // be reclaimed on the strength of writes that never reached the disk.
    if (se->chunk_fill > 0) all_clear = false;
    if (all_clear) {
        // Verification pass: no index word may still route to the region
        // and no cache entry may still reference it. Otherwise retry later.
        for (uint64_t i = 0; i < nslots && all_clear; ++i) {
            const uint8_t* slot = buf + i * kSlotBytes;
            uint32_t sz1;
            memcpy(&sz1, slot + 8, 4);
            if (sz1 == 0 || !slot_crc_ok(slot)) continue;
            if ((uint64_t)(sz1 - 1) > kSlotDataMax) continue;   // tombstone
            uint64_t key;
            memcpy(&key, slot, 8);
            uint32_t p1 = (uint32_t)(base + i) + 1;
            uint64_t h = mix64(key);
            uint64_t it = index_.start_of(h), scanned = 0;
            uint32_t c;
            while ((c = index_.next_candidate(h, it, scanned)) != 0)
                if (c == p1) { all_clear = false; break; }
            if (!all_clear) break;
            uint64_t s = set_of(key);
            set_locks_[s].lock();
            Entry* bse = set_base(s);
            for (int w = 0; w < (int)kAssoc; ++w)
                if (bse[w].state != S_EMPTY && bse[w].key == key &&
                    bse[w].pos1 == p1) {
                    all_clear = false;
                    break;
                }
            set_locks_[s].unlock();
        }
    }
    free(buf);
    if (!all_clear) return false;
    // Durability ordering: relocated copies (and restaged tombstones) must
    // be DURABLE before their checkpoint-durable predecessors are punched,
    // or a crash between the two could lose the only surviving copy.
    if (fdatasync(fd_) < 0) {
        write_errors_.fetch_add(1, std::memory_order_relaxed);
        durable_ok_.store(false, std::memory_order_relaxed);
        return false;                        // retry the region later
    }
    // Punch the region (space) and recycle it (position space). A failed
    // punch is nonfatal: extent reuse alone stops file growth, and stale
    // bytes lose every LSN comparison anyway.
    off_t roff = (off_t)(base * kSlotBytes);
    off_t rlen = (off_t)(nslots * kSlotBytes);
#if defined(__linux__) && defined(FALLOC_FL_PUNCH_HOLE)
    while (fallocate(fd_, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                     roff, rlen) < 0) {
        if (errno == EINTR) continue;
        punch_unsupported_.store(true, std::memory_order_relaxed);
        break;
    }
#else
    (void)roff; (void)rlen;
    punch_unsupported_.store(true, std::memory_order_relaxed);
#endif
    {
        std::lock_guard<std::mutex> lk(free_mu_);
        free_extents_.push_back({base, (uint32_t)nslots});
        free_count_.fetch_add(1, std::memory_order_relaxed);
    }
    reclaimed_bytes_.fetch_add(nslots * kSlotBytes, std::memory_order_relaxed);
    reclaimed_total_.fetch_add(nslots * kSlotBytes, std::memory_order_relaxed);
    return true;
}

// ── Production stats seam ────────────────────────────────────────────────────
void HydraStore::FillProdStats(ProdStats& out) const {
    out.durable_ok = durable_ok_.load(std::memory_order_relaxed) ? 1 : 0;
    out.recover_ok = recover_ok_.load(std::memory_order_relaxed);
    out.recovered_keys = recovered_keys_.load(std::memory_order_relaxed);
    out.recover_torn_slots = recover_torn_.load(std::memory_order_relaxed);
    out.write_errors = write_errors_.load(std::memory_order_relaxed);
    out.read_errors = read_errors_.load(std::memory_order_relaxed);
    out.rejected_oversize = rejected_oversize_.load(std::memory_order_relaxed);
    out.oversize_bytes = oversize_bytes_.load(std::memory_order_relaxed);
    out.compactions_run = compactions_run_.load(std::memory_order_relaxed);
    uint64_t alloc = next_slot_.load(std::memory_order_relaxed) * kSlotBytes;
    uint64_t rec = reclaimed_bytes_.load(std::memory_order_relaxed);
    out.log_bytes = alloc > rec ? alloc - rec : 0;
    out.live_bytes = index_.count.load(std::memory_order_relaxed) * kSlotBytes;
    out.reclaimed_bytes = reclaimed_total_.load(std::memory_order_relaxed);
    out.buffered_fallback = buffered_fallback_ ? 1 : 0;
    out.punch_unsupported =
        punch_unsupported_.load(std::memory_order_relaxed) ? 1 : 0;
}

CacheStats HydraStore::GetCacheStats() const {
    CacheStats cs;
    {
        std::lock_guard<std::mutex> lk(sess_mu_);
        for (auto& se : sessions_) {
            cs.read_hits += se->read_hits.load(std::memory_order_relaxed);
            cs.read_misses += se->read_misses.load(std::memory_order_relaxed);
            cs.rmw_hits += se->rmw_hits.load(std::memory_order_relaxed);
            cs.rmw_misses += se->rmw_misses.load(std::memory_order_relaxed);
            cs.evictions += se->evictions.load(std::memory_order_relaxed);
        }
    }
    cs.hot_bytes = occupancy_.load() * kEntryBytes +
                   oversize_bytes_.load(std::memory_order_relaxed);
    cs.total_bytes = next_slot_.load() * kSlotBytes + cs.hot_bytes;
    cs.budget_bytes = mem_budget_;
    return cs;
}

}  // namespace hydra

IKVStore* create_kvstore() { return new hydra::HydraStore(); }

// Production observability seam (see kvstore_interface.h note in tests): the
// IKVStore interface is frozen by the harness, so prod stats are exported via
// a plain C-linkage function; callers declare a byte-compatible struct.
extern "C" bool hydra_get_prod_stats(IKVStore* st,
                                     hydra::HydraStore::ProdStats* out) {
    auto* h = dynamic_cast<hydra::HydraStore*>(st);
    if (!h || !out) return false;
    h->FillProdStats(*out);
    return true;
}
