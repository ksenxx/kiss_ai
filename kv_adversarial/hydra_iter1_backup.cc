// hydra.cc — HydraKV: a larger-than-memory key-value store for the 50:50 YCSB-A
// benchmark (250M keys x 100B values, 8 GiB memory budget, NVMe spill).
//
// Written from first principles (no code taken from any existing KV store).
//
// Architecture
// ────────────
//  * Disk tier: a single O_DIRECT "slot file" in the spill directory. Every
//    key is assigned a fixed 128-byte slot the first time it is inserted
//    (positions handed out by an atomic counter, so the load phase writes the
//    file strictly sequentially through per-thread 1 MiB chunk buffers).
//    Updates are written back in place (4 KiB page read-modify-write), so the
//    store never grows the file during steady state — no GC needed, no leaks.
//  * Position index: an open-addressing fingerprint hash index (like
//    FASTER's hash table) mapping ARBITRARY uint64 keys to slot positions.
//    Each 8-byte word packs [fp32 | position+1]; buckets of 8 words (one
//    cache line), linear probing across buckets, lock-free CAS inserts.
//    Fingerprint matches are verified against the full key stored in the
//    on-disk slot (on the read path inline; on the write path lazily at
//    flush time, with a relocation slow path for the ~2^-60 false-match
//    case), so correctness never depends on fingerprints being unique.
//    Values > 102 B take a sharded in-memory overflow map (never exercised
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
//  * A key's bytes on disk are only ever changed by flushing that key's own
//    cache entry; flushes of *other* keys in the same 4 KiB page rewrite the
//    page with freshly-read content and are serialized by a page lock stripe,
//    so concurrent page RMWs never lose updates, and un-locked reads of a
//    page can never observe wrong bytes for a key that is absent from cache.
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
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <immintrin.h>
static inline void cpu_relax() { _mm_pause(); }
#else
static inline void cpu_relax() {}
#endif

namespace hydra {

// ── Geometry ─────────────────────────────────────────────────────────────────
static constexpr size_t   kSlotBytes     = 128;   // fixed on-disk slot
static constexpr size_t   kSlotDataMax   = 116;   // 128 - key(8) - size(4)
static constexpr size_t   kPageBytes     = 4096;  // O_DIRECT I/O unit
static constexpr size_t   kSlotsPerPage  = kPageBytes / kSlotBytes;   // 32
static constexpr size_t   kChunkSlots    = 8192;  // load-phase write batch
static constexpr size_t   kChunkBytes    = kChunkSlots * kSlotBytes;  // 1 MiB
static constexpr size_t   kAssoc         = 8;     // cache ways per set
static constexpr size_t   kInlineMax     = 101;   // max value bytes in cache
static constexpr size_t   kEntryBytes    = 120;
static constexpr size_t   kBucketWords   = 8;     // index words per bucket (64 B)
static constexpr size_t   kPageLockCount = 1 << 16;
static constexpr size_t   kOverflowShards = 64;
static constexpr int      kIoThreads     = 48;    // async read-miss service
static constexpr int      kCleanerThreads = 4;    // background dirty flushers
static constexpr int      kCheckpointThreads = 32;

// Entry states.
enum : uint8_t { S_EMPTY = 0, S_CLEAN = 1, S_DIRTY = 2, S_BUFFERED = 3 };

// Entry.recent flag bits.
enum : uint8_t {
    R_RECENT = 1,   // CLOCK second-chance bit (cleared by aging sweeps)
    R_UNVER  = 2,   // pos1 came from an unverified fingerprint match; the
                    // flush path verifies the slot key before overwriting
};

struct Entry {
    uint64_t key;
    uint32_t pos1;    // slot position + 1; 0 = unknown (pure in-memory mode)
    uint32_t ver;     // bumped on every value write; survives eviction/reuse
                      // (32-bit so in-flight-flush ABA needs 2^32 same-key
                      // updates during one I/O — days, not seconds)
    uint8_t  size;    // value size in bytes (<= kInlineMax)
    uint8_t  state;
    uint8_t  recent;  // flag bits (R_RECENT | R_UNVER)
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
            uint64_t cur = it;
            it = (it + 1) & (nwords - 1);
            scanned++;
            (void)cur;
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

    // Tombstone the word holding (h -> pos1). Used only by Delete.
    bool erase(uint64_t h, uint32_t pos1) {
        const uint64_t v = pack(h, pos1);
        uint64_t it = start_of(h);
        for (uint64_t scanned = 0; scanned < nwords; ++scanned) {
            uint64_t w = words[it].load(std::memory_order_acquire);
            if (w == 0) return false;
            if (w == v) {
                uint64_t tomb = w & 0xFFFFFFFF00000000ULL;  // pos1 -> 0
                if (tomb == 0) tomb = 1ULL << 32;           // never store word 0
                if (words[it].compare_exchange_strong(w, tomb,
                                                      std::memory_order_acq_rel)) {
                    count.fetch_sub(1, std::memory_order_relaxed);
                    return true;
                }
            }
            it = (it + 1) & (nwords - 1);
        }
        return false;
    }
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

// EINTR-safe pread. Returns bytes read (< n only at EOF). Fail-stop on error:
// silently continuing after an I/O error would risk serving corrupt data.
static size_t xpread(int fd, void* buf, size_t n, off_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = pread(fd, (char*)buf + done, n - done, off + (off_t)done);
        if (r < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "hydra: pread failed (errno=%d)\n", errno);
            abort();
        }
        if (r == 0) break;   // EOF: caller sees zero-filled tail
        done += (size_t)r;
    }
    return done;
}

// EINTR-safe full pwrite. Fail-stop on error/short write: state must never
// be marked clean/durable after a failed write.
static void xpwrite(int fd, const void* buf, size_t n, off_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t r = pwrite(fd, (const char*)buf + done, n - done,
                           off + (off_t)done);
        if (r < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "hydra: pwrite failed (errno=%d)\n", errno);
            abort();
        }
        if (r == 0) {
            fprintf(stderr, "hydra: pwrite wrote 0 bytes\n");
            abort();
        }
        done += (size_t)r;
    }
}

// ── Per-thread session ───────────────────────────────────────────────────────
struct Session {
    // async reads in flight for this session
    std::atomic<uint64_t> outstanding{0};

    // sequential-load chunk buffer (lazy)
    uint8_t* chunk_buf = nullptr;
    uint64_t chunk_base = 0;     // first slot position of the chunk
    uint32_t chunk_fill = 0;     // slots used
    bool     chunk_active = false;  // extent reserved (survives failed insert)
    std::vector<std::pair<uint64_t, uint32_t>> chunk_meta;  // (key, ver)

    // last-page read buffer (helps the stride-1 validation scan)
    uint8_t* page_buf = nullptr;
    uint64_t page_no = ~0ULL;
    uint64_t page_epoch = 0;

    // scratch page for read-modify-write flushes
    uint8_t* rmw_buf = nullptr;

    // stats
    uint64_t read_hits = 0, read_misses = 0;
    uint64_t rmw_hits = 0, rmw_misses = 0;
    uint64_t evictions = 0;

    ~Session() {
        free(chunk_buf);
        free(page_buf);
        free(rmw_buf);
    }
};

class HydraStore final : public IKVStore {
public:
    HydraStore() = default;
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

    bool cache_probe(uint64_t key, GenValue& out);          // hit path
    OpStatus miss_read(uint64_t key, GenValue& out, Session* se);
    void upsert_inline(uint64_t key, const uint8_t* data, uint16_t size);
    int  find_victim(uint64_t s, Session* se);              // set locked; may drop+retake
    // In-place slot write (4 KiB RMW). Verifies the slot's stored key first;
    // on a fingerprint false-match it relocates the key (rare slow path) and
    // returns the final position+1 (callers refresh Entry::pos1 with it).
    uint32_t flush_slot_rmw(uint32_t position, uint64_t key,
                            const uint8_t* data, uint16_t size, Session* se);
    // After a flushed entry lands, refresh its pos1 / clear R_UNVER.
    void finish_flush(uint64_t s, int w, const Entry& snap, uint32_t final_p1);
    // Disk-verified index lookup (reads candidate slots); 0 = absent.
    uint32_t verified_lookup(uint64_t key, Session* se);
    void flush_chunk(Session* se);
    void ensure_chunk(Session* se);
    bool read_page(uint64_t page, uint8_t* buf);             // O_DIRECT pread
    void cleaner_main(int id);
    void io_main();
    void stop_background();

    // overflow (out-of-range keys / oversized values); in-memory
    struct OverflowShard {
        std::mutex mu;
        std::unordered_map<uint64_t, std::string> map;
    };
    OverflowShard& shard(uint64_t key) { return overflow_[mix64(key) % kOverflowShards]; }
    bool overflow_get(uint64_t key, GenValue& out);
    void overflow_put(uint64_t key, const uint8_t* data, size_t size);
    bool overflow_erase(uint64_t key);

    // ---- state -------------------------------------------------------------
    size_t   mem_budget_ = 0;
    bool     disk_mode_ = false;
    int      fd_ = -1;
    std::string dir_;

    uint64_t nsets_ = 0;
    Entry*   entries_ = nullptr;         // nsets_ * kAssoc
    SpinLock* set_locks_ = nullptr;      // per set
    SpinLock  page_locks_[kPageLockCount];

    HashIndex index_;                        // key -> position+1
    std::mutex relocate_mu_;                 // serializes rare relocation path
    std::atomic<uint64_t> next_slot_{0};     // slot allocator

    OverflowShard overflow_[kOverflowShards];
    std::atomic<uint64_t> overflow_count_{0};

    std::atomic<uint64_t> flush_epoch_{0};   // invalidates session page bufs
    std::atomic<uint64_t> occupancy_{0};
    std::atomic<uint64_t> dirty_count_{0};
    std::atomic<uint64_t> key_mismatch_{0};

    // sessions
    mutable std::mutex sess_mu_;
    std::vector<std::unique_ptr<Session>> sessions_;

    // async read I/O pool
    struct IoReq { ReadSlot* slot; Session* sess; };
    std::mutex io_mu_;
    std::condition_variable io_cv_;
    std::deque<IoReq> io_q_;
    std::atomic<bool> stopping_{false};
    std::vector<std::thread> io_threads_;
    std::vector<std::thread> cleaners_;
};

// ── thread-local session registry ────────────────────────────────────────────
static thread_local HydraStore* tls_owner = nullptr;
static thread_local Session*    tls_sess = nullptr;

Session* HydraStore::session() {
    if (tls_owner == this && tls_sess) return tls_sess;
    auto se = std::make_unique<Session>();
    Session* raw = se.get();
    {
        std::lock_guard<std::mutex> lk(sess_mu_);
        sessions_.push_back(std::move(se));
    }
    tls_owner = this;
    tls_sess = raw;
    return raw;
}

void HydraStore::StartSession() { (void)session(); }

void HydraStore::StopSession() {
    Session* se = session();
    if (se->chunk_buf && se->chunk_fill > 0) flush_chunk(se);
    while (se->outstanding.load(std::memory_order_acquire) != 0)
        std::this_thread::yield();
    tls_sess = nullptr;   // thread may be reused for a different phase
    tls_owner = nullptr;
}

// ── Init ─────────────────────────────────────────────────────────────────────
void HydraStore::InitExtended(size_t, size_t, size_t mem_budget_bytes,
                              const char* storage_path) {
    mem_budget_ = mem_budget_bytes ? mem_budget_bytes : (8ULL << 30);
    disk_mode_ = storage_path && storage_path[0];

    if (disk_mode_) {
        dir_ = storage_path;
        std::string file = dir_ + "/hydra_slots.dat";
        fd_ = open(file.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
        if (fd_ < 0) {
            // Fall back to buffered I/O if the filesystem rejects O_DIRECT.
            fd_ = open(file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        }
        if (fd_ < 0) {
            fprintf(stderr, "hydra: cannot open %s\n", file.c_str());
            abort();
        }

        // Fingerprint hash index: largest power-of-two word count whose
        // footprint is <= budget/4 (2 GiB / 268M entries at an 8 GiB budget).
        uint64_t words = 64;
        while (words * 2 * 8 <= mem_budget_ / 4) words *= 2;
        index_.init(words);
    }

    // Cache sizing: leave room for the hash index plus allocator/thread
    // slack, spend the rest on the value cache.
    size_t reserve = (disk_mode_ ? index_.nwords * 8 : 0) + (512ULL << 20);
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
        for (int i = 0; i < kIoThreads; ++i)
            io_threads_.emplace_back(&HydraStore::io_main, this);
        for (int i = 0; i < kCleanerThreads; ++i)
            cleaners_.emplace_back(&HydraStore::cleaner_main, this, i);
    }

    fprintf(stderr,
            "hydra: init budget=%.2f GB cache=%.2f GB sets=%llu assoc=%zu "
            "disk=%s\n",
            mem_budget_ / 1e9, nsets_ * kAssoc * (double)kEntryBytes / 1e9,
            (unsigned long long)nsets_, kAssoc,
            disk_mode_ ? dir_.c_str() : "(none)");
}

void HydraStore::stop_background() {
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        if (stopping_) return;
        stopping_ = true;
    }
    io_cv_.notify_all();
    for (auto& t : io_threads_) t.join();
    for (auto& t : cleaners_) t.join();
    io_threads_.clear();
    cleaners_.clear();
}

HydraStore::~HydraStore() {
    stop_background();
    if (entries_) munmap(entries_, nsets_ * kAssoc * sizeof(Entry));
    delete[] set_locks_;
    index_.destroy();
    if (fd_ >= 0) close(fd_);
    uint64_t mism = key_mismatch_.load();
    if (mism) fprintf(stderr, "hydra: WARNING %llu slot key mismatches\n",
                      (unsigned long long)mism);
}

// ── Disk primitives ──────────────────────────────────────────────────────────
bool HydraStore::read_page(uint64_t page, uint8_t* buf) {
    size_t n = xpread(fd_, buf, kPageBytes, (off_t)(page * kPageBytes));
    if (n < kPageBytes) memset(buf + n, 0, kPageBytes - n);   // EOF tail
    return true;
}

// Disk-verified index lookup: probes the fingerprint index and reads each
// candidate slot until the stored key matches exactly. Returns position+1 or
// 0 if the key has no verified slot. (Rare path; used by relocation/Delete.)
uint32_t HydraStore::verified_lookup(uint64_t key, Session* se) {
    if (!se->page_buf) { se->page_buf = alloc_aligned(kPageBytes); se->page_no = ~0ULL; }
    uint64_t h = mix64(key);
    uint64_t it = index_.start_of(h), scanned = 0;
    uint32_t p1;
    while ((p1 = index_.next_candidate(h, it, scanned)) != 0) {
        uint64_t page = (uint64_t)(p1 - 1) / kSlotsPerPage;
        read_page(page, se->page_buf);
        se->page_no = ~0ULL;   // scratch use; don't trust as page cache
        const uint8_t* slot =
            se->page_buf + ((p1 - 1) % kSlotsPerPage) * kSlotBytes;
        uint64_t skey; uint32_t sz1;
        memcpy(&skey, slot, 8);
        memcpy(&sz1, slot + 8, 4);
        if (sz1 != 0 && skey == key) return p1;
    }
    return 0;
}

// In-place slot flush: 4 KiB read-modify-write under the page lock stripe.
// Verifies that the slot really belongs to `key` before overwriting; on a
// fingerprint false-match (probability ~2^-60 per lookup) the key is
// relocated to its true slot, or a fresh slot if it never had one. Returns
// the final position+1 the value was written to.
uint32_t HydraStore::flush_slot_rmw(uint32_t position, uint64_t key,
                                    const uint8_t* data, uint16_t size,
                                    Session* se) {
    if (!se->rmw_buf) se->rmw_buf = alloc_aligned(kPageBytes);
    uint8_t* buf = se->rmw_buf;
    for (int attempt = 0; ; ++attempt) {
        uint64_t page = (uint64_t)position / kSlotsPerPage;
        SpinLock& pl = page_locks_[page & (kPageLockCount - 1)];
        pl.lock();
        read_page(page, buf);
        uint8_t* slot = buf + (position % kSlotsPerPage) * kSlotBytes;
        uint64_t skey; uint32_t osz1;
        memcpy(&skey, slot, 8);
        memcpy(&osz1, slot + 8, 4);
        if (osz1 == 0 && attempt == 0) {
            // Candidate slot has not landed yet, so it CANNOT be verified as
            // ours (if it were ours we would still be S_BUFFERED-pinned, not
            // RMW-flushed). Never overwrite an unverified blank slot — defer:
            // the entry stays dirty and is retried after the chunk lands.
            pl.unlock();
            return 0;
        }
        if (osz1 != 0 && skey != key && attempt == 0) {
            // Fingerprint false-match: this slot belongs to another key.
            pl.unlock();
            key_mismatch_.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(relocate_mu_);
            uint32_t true_p1 = verified_lookup(key, se);
            if (true_p1 == 0) {
                // Key never had a slot: allocate a WHOLE PRIVATE PAGE of
                // slots (keeps every chunk_base reservation 4 KiB-aligned for
                // O_DIRECT and keeps page ownership disjoint from chunk
                // writers), write it while still holding relocate_mu_ (so a
                // racing flusher's verified_lookup sees the landed key, never
                // allocating a duplicate), and index it.
                uint64_t np = next_slot_.fetch_add(kSlotsPerPage,
                                                   std::memory_order_relaxed);
                true_p1 = (uint32_t)np + 1;
                if (np + 1 > 0xFFFFFFFFULL || !index_.insert(mix64(key), true_p1)) {
                    overflow_put(key, data, size);
                    return position + 1;   // value safe in overflow map
                }
                uint64_t npage = (uint64_t)np / kSlotsPerPage;
                SpinLock& npl = page_locks_[npage & (kPageLockCount - 1)];
                npl.lock();
                read_page(npage, buf);
                uint8_t* nslot = buf + (np % kSlotsPerPage) * kSlotBytes;
                memcpy(nslot, &key, 8);
                uint32_t nsz1 = (uint32_t)size + 1;
                memcpy(nslot + 8, &nsz1, 4);
                memcpy(nslot + 12, data, size);
                memset(nslot + 12 + size, 0, kSlotDataMax - size);
                xpwrite(fd_, buf, kPageBytes, (off_t)(npage * kPageBytes));
                npl.unlock();
                flush_epoch_.fetch_add(1, std::memory_order_relaxed);
                return true_p1;
            }
            position = true_p1 - 1;
            continue;   // rewrite at the verified position
        }
        memcpy(slot, &key, 8);
        uint32_t sz1 = (uint32_t)size + 1;
        memcpy(slot + 8, &sz1, 4);
        memcpy(slot + 12, data, size);
        memset(slot + 12 + size, 0, kSlotDataMax - size);
        xpwrite(fd_, buf, kPageBytes, (off_t)(page * kPageBytes));
        pl.unlock();
        flush_epoch_.fetch_add(1, std::memory_order_relaxed);
        return position + 1;
    }
}

// Post-flush bookkeeping shared by cleaner/victim/checkpoint paths: mark the
// entry clean if unchanged, and refresh pos1 (relocation) / clear R_UNVER.
// Guarded by the full (key, ver, state) identity so a stale flusher whose
// entry was concurrently updated/evicted/replaced has NO side effects; the
// newer generation simply gets flushed (and re-verified) again later.
// final_p1 == 0 means the flush was deferred (unlanded candidate slot).
void HydraStore::finish_flush(uint64_t s, int w, const Entry& snap,
                              uint32_t final_p1) {
    if (final_p1 == 0) return;
    Entry& e = set_base(s)[w];
    if (e.state == S_DIRTY && e.key == snap.key && e.ver == snap.ver) {
        e.pos1 = final_p1;
        e.recent &= (uint8_t)~R_UNVER;
        e.state = S_CLEAN;
        dirty_count_.fetch_sub(1, std::memory_order_relaxed);
    }
}

// ── Load-phase sequential chunk writer ───────────────────────────────────────
void HydraStore::ensure_chunk(Session* se) {
    if (!se->chunk_buf) {
        se->chunk_buf = alloc_aligned(kChunkBytes);
        se->chunk_meta.reserve(kChunkSlots);
    }
    if (se->chunk_fill == 0 && !se->chunk_active) {
        se->chunk_base = next_slot_.fetch_add(kChunkSlots, std::memory_order_relaxed);
        se->chunk_meta.clear();
        se->chunk_active = true;   // extent stays reserved until flushed
    }
}

void HydraStore::flush_chunk(Session* se) {
    if (se->chunk_fill == 0) return;
    size_t used = (size_t)se->chunk_fill * kSlotBytes;
    size_t len = (used + kPageBytes - 1) & ~(kPageBytes - 1);
    memset(se->chunk_buf + used, 0, len - used);
    off_t off = (off_t)(se->chunk_base * kSlotBytes);
    xpwrite(fd_, se->chunk_buf, len, off);

    // Unpin: entries whose value is unchanged become CLEAN; re-upserted ones
    // fall back to DIRTY so the cleaner writes the newer bytes later.
    for (auto& km : se->chunk_meta) {
        uint64_t s = set_of(km.first);
        set_locks_[s].lock();
        Entry* base = set_base(s);
        for (size_t w = 0; w < kAssoc; ++w) {
            Entry& e = base[w];
            if (e.state == S_BUFFERED && e.key == km.first) {
                if (e.ver == km.second) {
                    e.state = S_CLEAN;
                    dirty_count_.fetch_sub(1, std::memory_order_relaxed);
                } else {
                    e.state = S_DIRTY;
                }
                break;
            }
        }
        set_locks_[s].unlock();
    }
    se->chunk_fill = 0;
    se->chunk_meta.clear();
    se->chunk_active = false;
}

// ── Victim selection (called with set s locked) ──────────────────────────────
// Returns a way index whose entry may be overwritten. May temporarily drop the
// set lock to flush a dirty victim; always returns with the lock held.
// Returns -1 if the caller should rescan the set (it changed while unlocked).
int HydraStore::find_victim(uint64_t s, Session* se) {
    Entry* base = set_base(s);
    int clean_old = -1, clean_any = -1, dirty_old = -1, dirty_any = -1;
    for (int w = 0; w < (int)kAssoc; ++w) {
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
    // CLOCK aging: when every resident way is "recent", strip the bits so the
    // set stays evictable and the hot set can drift over time.
    if (clean_old < 0 && dirty_old < 0) {
        for (int w = 0; w < (int)kAssoc; ++w)
            base[w].recent &= (uint8_t)~R_RECENT;
    }
    if (clean_old >= 0 || clean_any >= 0) {
        int v = clean_old >= 0 ? clean_old : clean_any;
        se->evictions++;
        occupancy_.fetch_sub(1, std::memory_order_relaxed);
        return v;
    }
    int d = dirty_old >= 0 ? dirty_old : dirty_any;
    if (d >= 0) {
        // Sync-assist flush: copy out, write back unlocked, re-verify, then
        // have the caller rescan (the set may have changed while unlocked).
        Entry snap = base[d];
        set_locks_[s].unlock();
        uint32_t fp1 = flush_slot_rmw(snap.pos1 - 1, snap.key, snap.data,
                                      snap.size, se);
        set_locks_[s].lock();
        finish_flush(s, d, snap, fp1);
        if (fp1 == 0) {
            // Deferred (candidate slot unlanded): let the owning chunk land
            // instead of spinning on this victim.
            set_locks_[s].unlock();
            std::this_thread::yield();
            set_locks_[s].lock();
        }
        return -1;
    }
    // Every way is pinned (BUFFERED). Flush our own chunk to unpin ours, then
    // have the caller retry; other threads' flushes will unpin the rest.
    set_locks_[s].unlock();
    if (se->chunk_fill > 0) flush_chunk(se);
    std::this_thread::yield();
    set_locks_[s].lock();
    return -1;
}

// ── Overflow (rare path) ─────────────────────────────────────────────────────
bool HydraStore::overflow_get(uint64_t key, GenValue& out) {
    if (overflow_count_.load(std::memory_order_relaxed) == 0) return false;
    OverflowShard& sh = shard(key);
    std::lock_guard<std::mutex> lk(sh.mu);
    auto it = sh.map.find(key);
    if (it == sh.map.end()) return false;
    out.size = (uint32_t)it->second.size();
    memcpy(out.data, it->second.data(), it->second.size());
    return true;
}

void HydraStore::overflow_put(uint64_t key, const uint8_t* data, size_t size) {
    OverflowShard& sh = shard(key);
    std::lock_guard<std::mutex> lk(sh.mu);
    auto r = sh.map.insert_or_assign(key,
        std::string(reinterpret_cast<const char*>(data), size));
    if (r.second) overflow_count_.fetch_add(1, std::memory_order_relaxed);
}

bool HydraStore::overflow_erase(uint64_t key) {
    if (overflow_count_.load(std::memory_order_relaxed) == 0) return false;
    OverflowShard& sh = shard(key);
    std::lock_guard<std::mutex> lk(sh.mu);
    if (sh.map.erase(key)) {
        overflow_count_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
    return false;
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
            e.recent |= R_RECENT;
            set_locks_[s].unlock();
            return true;
        }
    }
    set_locks_[s].unlock();
    return false;
}

OpStatus HydraStore::miss_read(uint64_t key, GenValue& out, Session* se) {
    // Re-check the cache: the key may have been upserted/admitted since the
    // caller's probe. The cached value is always the freshest.
    if (cache_probe(key, out)) return OpStatus::Ok;
    if (overflow_get(key, out)) return OpStatus::Ok;
    if (!disk_mode_) return OpStatus::NotFound;

    if (!se->page_buf) { se->page_buf = alloc_aligned(kPageBytes); se->page_no = ~0ULL; }

    // Probe the fingerprint index; verify every candidate against the exact
    // key stored in its on-disk slot. Almost always the first candidate is
    // the right one (fp false matches need a 64-bit hash collision).
    uint64_t h = mix64(key);
    uint64_t itc = index_.start_of(h), scanned = 0;
    uint32_t p1;
    const uint8_t* slot = nullptr;
    uint16_t size = 0;
    for (;;) {
        p1 = index_.next_candidate(h, itc, scanned);
        if (p1 == 0) return OpStatus::NotFound;
        uint32_t position = p1 - 1;
        uint64_t page = (uint64_t)position / kSlotsPerPage;
        uint64_t epoch = flush_epoch_.load(std::memory_order_relaxed);
        if (se->page_no != page || se->page_epoch != epoch) {
            read_page(page, se->page_buf);
            se->page_no = page;
            se->page_epoch = epoch;
        }
        slot = se->page_buf + (position % kSlotsPerPage) * kSlotBytes;
        uint64_t skey; uint32_t sz1;
        memcpy(&skey, slot, 8);
        memcpy(&sz1, slot + 8, 4);
        if (sz1 == 0) {
            // Candidate slot has not landed yet. If it is really ours, our
            // value is pinned in cache (BUFFERED) — re-probe. Otherwise it
            // is a false match: keep probing.
            se->page_no = ~0ULL;
            if (cache_probe(key, out)) return OpStatus::Ok;
            continue;
        }
        uint16_t sz = (uint16_t)(sz1 - 1);
        if (skey != key || sz > kSlotDataMax) {
            key_mismatch_.fetch_add(1, std::memory_order_relaxed);
            continue;   // false match: keep probing
        }
        size = sz;
        break;
    }

    // Freshness: if the key entered the cache while we were doing I/O, the
    // cached copy supersedes the disk copy.
    if (cache_probe(key, out)) return OpStatus::Ok;

    out.size = size;
    memcpy(out.data, slot + 12, size);

    // Admit (clean) if a non-blocking victim exists.
    if (size <= kInlineMax) {
        uint64_t s = set_of(key);
        set_locks_[s].lock();
        Entry* base = set_base(s);
        int victim = -1;
        bool present = false;
        for (int w = 0; w < (int)kAssoc; ++w) {
            Entry& e = base[w];
            if (e.state != S_EMPTY && e.key == key) { present = true; break; }
            if (e.state == S_EMPTY) { if (victim < 0) victim = w; }
            else if (e.state == S_CLEAN && !(e.recent & R_RECENT) && victim < 0) victim = w;
        }
        if (!present && victim < 0) {
            // CLOCK aging on the read path too: strip R_RECENT from clean
            // ways so read-heavy / drifting workloads keep admitting instead
            // of freezing the cache (writes are not required to age a set).
            for (int w = 0; w < (int)kAssoc; ++w) {
                Entry& e = base[w];
                if (e.state == S_CLEAN) {
                    e.recent &= (uint8_t)~R_RECENT;
                    if (victim < 0) victim = w;
                }
            }
        }
        if (!present && victim >= 0) {
            Entry& e = base[victim];
            if (e.state == S_CLEAN) { se->evictions++; occupancy_.fetch_sub(1, std::memory_order_relaxed); }
            e.key = key;
            e.pos1 = p1;
            e.ver = e.ver + 1;
            e.size = (uint8_t)size;
            e.state = S_CLEAN;
            e.recent = 0;
            memcpy(e.data, slot + 12, size);
            occupancy_.fetch_add(1, std::memory_order_relaxed);
        }
        set_locks_[s].unlock();
    }
    return OpStatus::Ok;
}

bool HydraStore::Read(uint64_t key, GenValue& out) {
    Session* se = session();
    if (cache_probe(key, out)) { se->read_hits++; return true; }
    se->read_misses++;
    if (!disk_mode_) return overflow_get(key, out);
    return miss_read(key, out, se) == OpStatus::Ok;
}

OpStatus HydraStore::ReadAsync(ReadSlot* slot) {
    Session* se = session();
    if (cache_probe(slot->key, slot->out)) {
        se->read_hits++;
        slot->status = OpStatus::Ok;
        slot->done.store(1, std::memory_order_release);
        return OpStatus::Ok;
    }
    se->read_misses++;
    if (!disk_mode_) {
        bool ok = overflow_get(slot->key, slot->out);
        slot->status = ok ? OpStatus::Ok : OpStatus::NotFound;
        slot->done.store(1, std::memory_order_release);
        return slot->status;
    }
    se->outstanding.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(io_mu_);
        io_q_.push_back(IoReq{slot, se});
    }
    io_cv_.notify_one();
    return OpStatus::Pending;
}

void HydraStore::CompletePending(bool wait) {
    Session* se = session();
    if (wait) {
        if (se->chunk_buf && se->chunk_fill > 0) flush_chunk(se);
        while (se->outstanding.load(std::memory_order_acquire) != 0)
            std::this_thread::yield();
    }
}

void HydraStore::io_main() {
    Session* se = session();   // private session for page buffers
    std::unique_lock<std::mutex> lk(io_mu_);
    for (;;) {
        io_cv_.wait(lk, [&] { return stopping_ || !io_q_.empty(); });
        if (stopping_ && io_q_.empty()) return;
        IoReq r = io_q_.front();
        io_q_.pop_front();
        lk.unlock();

        OpStatus st = miss_read(r.slot->key, r.slot->out, se);
        r.slot->status = st;
        r.slot->done.store(1, std::memory_order_release);
        r.sess->outstanding.fetch_sub(1, std::memory_order_release);

        lk.lock();
    }
}

// ── Write paths ──────────────────────────────────────────────────────────────
void HydraStore::upsert_inline(uint64_t key, const uint8_t* data, uint16_t size) {
    Session* se = session();
    uint64_t s = set_of(key);
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
                e.recent |= R_RECENT;
                if (e.state == S_CLEAN) {
                    e.state = S_DIRTY;
                    dirty_count_.fetch_add(1, std::memory_order_relaxed);
                }
                set_locks_[s].unlock();
                if (overflow_count_.load(std::memory_order_relaxed)) overflow_erase(key);
                return;
            }
        }
        uint64_t h = mix64(key);
        uint32_t p1 = index_.first_candidate(h);
        int v = find_victim(s, se);
        if (v < 0) { set_locks_[s].unlock(); continue; }   // set changed; retry
        Entry& e = base[v];
        uint8_t new_state;
        uint8_t flags = R_RECENT;
        if (p1 == 0) {
            // First insert of this key: reserve a slot position and stage the
            // bytes in the sequential chunk buffer (pinned until it lands).
            ensure_chunk(se);
            uint64_t position = se->chunk_base + se->chunk_fill;
            if (position + 1 > 0xFFFFFFFFULL ||
                !index_.insert(h, (uint32_t)position + 1)) {
                // Slot space or index capacity exhausted — overflow map.
                // Roll back the eviction accounting: the victim entry was not
                // actually replaced (find_victim already decremented
                // occupancy for a non-empty clean victim).
                if (e.state != S_EMPTY)
                    occupancy_.fetch_add(1, std::memory_order_relaxed);
                set_locks_[s].unlock();
                overflow_put(key, data, size);
                return;
            }
            uint8_t* slot = se->chunk_buf + (size_t)se->chunk_fill * kSlotBytes;
            memcpy(slot, &key, 8);
            uint32_t sz1 = (uint32_t)size + 1;
            memcpy(slot + 8, &sz1, 4);
            memcpy(slot + 12, data, size);
            memset(slot + 12 + size, 0, kSlotDataMax - size);
            se->chunk_fill++;
            p1 = (uint32_t)position + 1;
            new_state = S_BUFFERED;
        } else {
            // Key (almost certainly) exists: adopt the first candidate
            // position optimistically; the flush path verifies the slot's
            // stored key before overwriting (R_UNVER).
            new_state = S_DIRTY;
            flags |= R_UNVER;
        }
        e.key = key;
        e.pos1 = p1;
        e.ver = e.ver + 1;
        e.size = (uint8_t)size;
        e.state = new_state;
        e.recent = flags;
        memcpy(e.data, data, size);
        occupancy_.fetch_add(1, std::memory_order_relaxed);
        dirty_count_.fetch_add(1, std::memory_order_relaxed);
        if (new_state == S_BUFFERED)
            se->chunk_meta.emplace_back(key, e.ver);
        bool full = (se->chunk_buf && se->chunk_fill == kChunkSlots);
        set_locks_[s].unlock();
        if (overflow_count_.load(std::memory_order_relaxed)) overflow_erase(key);
        if (full) flush_chunk(se);
        return;
    }
}

void HydraStore::Upsert(uint64_t key, const GenValue& value) {
    if (!disk_mode_) { overflow_put(key, value.data, value.size); return; }
    if (value.size > kInlineMax) {
        // Drop any stale inline cache entry so reads see the overflow value.
        // A BUFFERED entry stays pinned until its chunk lands (removing it
        // early would let the stale chunk write escape its cancellation), so
        // wait for the landing first — rare path, bounded by chunk flush.
        uint64_t s = set_of(key);
        for (;;) {
            bool buffered = false;
            set_locks_[s].lock();
            Entry* base = set_base(s);
            for (size_t w = 0; w < kAssoc; ++w) {
                Entry& e = base[w];
                if (e.state != S_EMPTY && e.key == key) {
                    if (e.state == S_BUFFERED) { buffered = true; break; }
                    if (e.state == S_DIRTY)
                        dirty_count_.fetch_sub(1, std::memory_order_relaxed);
                    e.state = S_EMPTY;
                    occupancy_.fetch_sub(1, std::memory_order_relaxed);
                    break;
                }
            }
            if (!buffered) break;
            set_locks_[s].unlock();
            Session* se = session();
            if (se->chunk_buf && se->chunk_fill > 0) flush_chunk(se);
            std::this_thread::yield();
        }
        overflow_put(key, value.data, value.size);
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
    GenValue cur;
    bool found = cache_probe(key, cur);
    if (!found && disk_mode_) found = miss_read(key, cur, se) == OpStatus::Ok;
    if (!found) found = overflow_get(key, cur);
    GenValue nv;
    nv.size = (uint32_t)std::min<size_t>(mod_size, GenValue::kMaxSize);
    memcpy(nv.data, mod_data, nv.size);
    if (found) {
        se->rmw_hits++;
        uint64_t a = 0, b = 0;
        if (cur.size >= 8) memcpy(&a, cur.data, 8);
        if (nv.size >= 8) { memcpy(&b, nv.data, 8); a += b; memcpy(nv.data, &a, 8); }
    } else {
        se->rmw_misses++;
    }
    Upsert(key, nv);
    l.unlock();
}

bool HydraStore::Delete(uint64_t key) {
    bool existed = overflow_erase(key);
    if (!disk_mode_) return existed;
    uint64_t s = set_of(key);
    // Never remove a BUFFERED (pinned) entry: wait for its chunk to land so
    // the index unlink below can verify the slot and no stale chunk write
    // can resurrect the key. Rare path; Delete is unused by workload id 0.
    for (;;) {
        bool buffered = false;
        set_locks_[s].lock();
        Entry* base = set_base(s);
        for (size_t w = 0; w < kAssoc; ++w) {
            Entry& e = base[w];
            if (e.state != S_EMPTY && e.key == key) {
                if (e.state == S_BUFFERED) { buffered = true; break; }
                if (e.state == S_DIRTY)
                    dirty_count_.fetch_sub(1, std::memory_order_relaxed);
                e.state = S_EMPTY;
                occupancy_.fetch_sub(1, std::memory_order_relaxed);
                existed = true;
                break;
            }
        }
        set_locks_[s].unlock();
        if (!buffered) break;
        Session* se = session();
        if (se->chunk_buf && se->chunk_fill > 0) flush_chunk(se);
        std::this_thread::yield();
    }
    // Unlink from the index (verified: reads candidate slots from disk).
    // The old slot itself is leaked (Delete is unused by this workload);
    // a production compactor would reclaim it.
    uint32_t p1 = verified_lookup(key, session());
    if (p1 != 0 && index_.erase(mix64(key), p1)) existed = true;
    return existed;
}

// ── Cleaner: keep sets stocked with clean victims ────────────────────────────
void HydraStore::cleaner_main(int id) {
    Session* se = session();
    uint64_t s = (nsets_ / kCleanerThreads) * (uint64_t)id;
    for (;;) {
        if (stopping_) return;
        uint64_t cleaned = 0, scanned = 0;
        while (scanned < 65536) {
            if (stopping_) return;
            s++;
            if (s >= nsets_) s = 0;
            scanned++;
            Entry* base = set_base(s);
            // Under heavy dirty pressure, age recent dirty entries too so the
            // backlog cannot grow without bound (hot keys that keep being
            // re-touched simply re-set their bit before the next sweep).
            bool pressure = dirty_count_.load(std::memory_order_relaxed) >
                            (nsets_ * kAssoc) / 4;
            // cheap unlocked peek: any dirty? (relaxed atomic byte loads so
            // the unlocked scan is race-free; staleness is fine — it is only
            // a hint, the locked rescan below decides)
            bool maybe = false;
            for (size_t w = 0; w < kAssoc; ++w) {
                uint8_t st = __atomic_load_n(&base[w].state, __ATOMIC_RELAXED);
                uint8_t rc = __atomic_load_n(&base[w].recent, __ATOMIC_RELAXED);
                if (st == S_DIRTY &&
                    (pressure || !(rc & R_RECENT))) { maybe = true; break; }
            }
            if (!maybe) continue;
            set_locks_[s].lock();
            int d = -1;
            for (int w = 0; w < (int)kAssoc; ++w) {
                Entry& ce = base[w];
                if (ce.state != S_DIRTY) continue;
                if (!(ce.recent & R_RECENT)) { d = w; break; }
                if (pressure) ce.recent &= (uint8_t)~R_RECENT;  // age; flush next sweep
            }
            if (d < 0) { set_locks_[s].unlock(); continue; }
            Entry snap = base[d];
            set_locks_[s].unlock();
            uint32_t fp1 = flush_slot_rmw(snap.pos1 - 1, snap.key, snap.data,
                                          snap.size, se);
            set_locks_[s].lock();
            Entry& e = base[d];
            bool was_dirty = (e.state == S_DIRTY && e.key == snap.key &&
                              e.ver == snap.ver);
            finish_flush(s, d, snap, fp1);
            if (was_dirty && fp1 != 0) cleaned++;
            set_locks_[s].unlock();
        }
        if (cleaned == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

// ── Checkpoint: flush all dirty entries, then fdatasync ─────────────────────
void HydraStore::Checkpoint() {
    if (!disk_mode_) return;
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
                    Entry snap = base[w];
                    set_locks_[s].unlock();
                    uint32_t fp1 = flush_slot_rmw(snap.pos1 - 1, snap.key,
                                                  snap.data, snap.size, se);
                    set_locks_[s].lock();
                    finish_flush(s, w, snap, fp1);
                    set_locks_[s].unlock();
                }
            }
        });
    }
    for (auto& t : ts) t.join();
    fdatasync(fd_);
}

CacheStats HydraStore::GetCacheStats() const {
    CacheStats cs;
    {
        std::lock_guard<std::mutex> lk(sess_mu_);
        for (auto& se : sessions_) {
            cs.read_hits += se->read_hits;
            cs.read_misses += se->read_misses;
            cs.rmw_hits += se->rmw_hits;
            cs.rmw_misses += se->rmw_misses;
            cs.evictions += se->evictions;
        }
    }
    cs.hot_bytes = occupancy_.load() * kEntryBytes;
    cs.total_bytes = next_slot_.load() * kSlotBytes + cs.hot_bytes;
    cs.budget_bytes = mem_budget_;
    return cs;
}

}  // namespace hydra

IKVStore* create_kvstore() { return new hydra::HydraStore(); }
