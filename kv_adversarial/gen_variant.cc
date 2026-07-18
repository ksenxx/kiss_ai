// gen_variant.cc — adversarial-but-realistic workload variants for the 50:50
// KV benchmark. Same file format as traces/gen_traces.cc (flat little-endian
// uint64 arrays; harness derives counts from filesize/8).
//
// All variants use SPARSE random 64-bit keys (production keys are hashes/IDs,
// never dense 0..N-1 — cf. Twitter cache-trace, FB FAST'20): the loaded key
// set is { sm64(i ^ seed) : i in 0..N-1 } (sm64 = SplitMix64 finalizer, a
// bijection, so keys are unique). Run traces pick an insertion index idx in
// [0,N) from a realistic popularity model, then emit keyof(idx).
//
// Popularity models (mode):
//   scrambled  — Zipf(theta) rank -> FNV-scrambled index (hotness spread
//                across the keyspace; YCSB ScrambledZipfian, like the
//                original workload but over sparse keys).
//   clustered  — Zipf(theta) rank used directly as insertion index (hot keys
//                are the earliest-inserted, adjacent in insertion order —
//                key-range locality as seen in FB UDB/ZippyDB, FAST'20).
//   drift      — ScrambledZipfian whose scramble seed crossfades to a new
//                one every EPOCH ops (gradual hot-set churn, as in Twitter
//                production diurnal/drift patterns; OSDI'20).
//   mix        — 60% scrambled + 25% clustered + 15% drifting (holdout-style
//                blend of all three real-world patterns).
//
// build: g++ -O2 -std=c++17 gen_variant.cc -o gen_variant
// usage: ./gen_variant N M theta mode key_seed run_seed load_out run_out
//   e.g. ./gen_variant 250000000 250000000 0.95 scrambled 0xA5A5 1 load.dat run.dat
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>

static inline uint64_t sm64(uint64_t x) {           // SplitMix64 finalizer
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
static inline uint64_t fnv1a(uint64_t x) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < 8; ++i) { h ^= (x & 0xff); h *= 1099511628211ULL; x >>= 8; }
  return h;
}

int main(int argc, char** argv) {
  if (argc < 9) {
    fprintf(stderr, "usage: %s N M theta mode key_seed run_seed load_out run_out\n", argv[0]);
    return 1;
  }
  const uint64_t N = strtoull(argv[1], nullptr, 0);
  const uint64_t M = strtoull(argv[2], nullptr, 0);
  const double THETA = strtod(argv[3], nullptr);
  const std::string mode = argv[4];
  const uint64_t key_seed = strtoull(argv[5], nullptr, 0);
  const uint64_t run_seed = strtoull(argv[6], nullptr, 0);
  const char* lf = argv[7];
  const char* rf = argv[8];

  auto keyof = [&](uint64_t idx) { return sm64(idx ^ key_seed); };

  // --- load: sparse unique 64-bit keys, insertion order i = 0..N-1 ---------
  {
    FILE* f = fopen(lf, "wb");
    if (!f) { perror("open load"); return 1; }
    std::vector<uint64_t> buf; buf.reserve(1u << 20);
    for (uint64_t i = 0; i < N; ++i) {
      buf.push_back(keyof(i));
      if (buf.size() == (1u << 20)) { fwrite(buf.data(), 8, buf.size(), f); buf.clear(); }
    }
    if (!buf.empty()) fwrite(buf.data(), 8, buf.size(), f);
    fclose(f);
    fprintf(stderr, "wrote %s: %llu sparse keys (seed=%#llx)\n", lf,
            (unsigned long long)N, (unsigned long long)key_seed);
  }

  // --- run: Zipf(theta) over ranks (Gray et al. SIGMOD'94) ------------------
  double zn = 0.0;
  for (uint64_t i = 1; i <= N; ++i) zn += 1.0 / std::pow((double)i, THETA);
  const double z2 = 1.0 + std::pow(0.5, THETA);
  const double alpha = 1.0 / (1.0 - THETA);
  const double eta = (1.0 - std::pow(2.0 / (double)N, 1.0 - THETA)) / (1.0 - z2 / zn);
  std::mt19937_64 g(run_seed * 0x9E3779B97F4A7C15ULL + 12345);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  const uint64_t EPOCH = 24u * 1000 * 1000;   // drift crossfade period (ops)
  auto zipf_rank = [&]() -> uint64_t {
    double u = U(g), uz = u * zn;
    if (uz < 1.0) return 0;
    if (uz < z2)  return 1;
    uint64_t z = (uint64_t)((double)N * std::pow(eta * u - eta + 1.0, alpha));
    return z >= N ? N - 1 : z;
  };
  auto scrambled_idx = [&](uint64_t z, uint64_t sseed) {
    return fnv1a(z ^ (sseed * 0x9E3779B97F4A7C15ULL)) % N;
  };

  FILE* f = fopen(rf, "wb");
  if (!f) { perror("open run"); return 1; }
  std::vector<uint64_t> buf; buf.reserve(1u << 20);
  for (uint64_t t = 0; t < M; ++t) {
    uint64_t z = zipf_rank();
    uint64_t idx;
    std::string m = mode;
    if (mode == "mix") {
      double r = U(g);
      m = r < 0.60 ? "scrambled" : (r < 0.85 ? "clustered" : "drift");
    }
    if (m == "scrambled") {
      idx = scrambled_idx(z, run_seed);
    } else if (m == "clustered") {
      idx = z;                                  // hot = earliest inserted
    } else {                                    // drift: gradual re-scramble
      uint64_t e = t / EPOCH;
      double frac = (double)(t % EPOCH) / (double)EPOCH;
      uint64_t se = (U(g) < frac) ? e + 1 : e;
      idx = scrambled_idx(z, run_seed + 1000003ULL * se);
    }
    buf.push_back(keyof(idx));
    if (buf.size() == (1u << 20)) { fwrite(buf.data(), 8, buf.size(), f); buf.clear(); }
  }
  if (!buf.empty()) fwrite(buf.data(), 8, buf.size(), f);
  fclose(f);
  fprintf(stderr, "wrote %s: %llu ops (mode=%s theta=%.2f run_seed=%llu)\n", rf,
          (unsigned long long)M, mode.c_str(), THETA, (unsigned long long)run_seed);
  return 0;
}
