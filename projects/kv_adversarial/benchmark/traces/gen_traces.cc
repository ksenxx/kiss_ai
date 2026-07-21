// gen_traces.cc — generate the two trace files for the 50:50 KV benchmark, so the
// task is fully self-contained (no 2 GB download needed).
//
//   build:  g++ -O2 -std=c++17 gen_traces.cc -o gen_traces
//   run:    ./gen_traces 250000000 250000000 load_250M.dat run_250M.dat
//           (args: N_load_keys  M_run_ops  load_out  run_out)
//
// Output format (what the harness reads): a flat little-endian array of uint64
// keys; the harness derives the key count from filesize/8.
//   load_250M.dat : a random permutation of the dense keyspace 0..N-1 (each key once).
//   run_250M.dat  : M access keys drawn Zipfian (theta=0.95), then SCRAMBLED via a
//                   hash so "hot" keys are spread across the keyspace (hotness is
//                   uncorrelated with key value) — matching the scored workload.
//
// NOTE: this produces a fresh *equivalent* workload (same distribution/skew), so
// throughput is comparable but the exact bytes differ from any reference .dat.
// It needs ~N*8 bytes of RAM for the load permutation (~2 GB at N=250M).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static constexpr double THETA = 0.95;

static inline uint64_t fnv1a(uint64_t x) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < 8; ++i) { h ^= (x & 0xff); h *= 1099511628211ULL; x >>= 8; }
  return h;
}
static double zeta(uint64_t n, double t) {
  double s = 0.0;
  for (uint64_t i = 1; i <= n; ++i) s += 1.0 / std::pow((double)i, t);
  return s;
}

int main(int argc, char** argv) {
  uint64_t N = argc > 1 ? strtoull(argv[1], nullptr, 10) : 250000000ULL;
  uint64_t M = argc > 2 ? strtoull(argv[2], nullptr, 10) : 250000000ULL;
  const char* lf = argc > 3 ? argv[3] : "load_250M.dat";
  const char* rf = argc > 4 ? argv[4] : "run_250M.dat";

  // --- load: a random permutation of 0..N-1 (dense, each key exactly once) ---
  {
    std::vector<uint64_t> a(N);
    for (uint64_t i = 0; i < N; ++i) a[i] = i;
    std::mt19937_64 g(12345ULL);
    std::shuffle(a.begin(), a.end(), g);
    FILE* f = std::fopen(lf, "wb");
    if (!f) { perror("open load"); return 1; }
    std::fwrite(a.data(), sizeof(uint64_t), N, f);
    std::fclose(f);
    std::fprintf(stderr, "wrote %s: %llu keys\n", lf, (unsigned long long)N);
  }

  // --- run: scrambled Zipfian (theta=0.95) over 0..N-1 (Gray et al.) ---
  {
    const double zn = zeta(N, THETA);
    const double z2 = 1.0 + std::pow(0.5, THETA);           // zeta(2, theta)
    const double alpha = 1.0 / (1.0 - THETA);
    const double eta = (1.0 - std::pow(2.0 / (double)N, 1.0 - THETA)) / (1.0 - z2 / zn);
    std::mt19937_64 g(67890ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    FILE* f = std::fopen(rf, "wb");
    if (!f) { perror("open run"); return 1; }
    std::vector<uint64_t> buf;
    buf.reserve(1u << 20);
    for (uint64_t i = 0; i < M; ++i) {
      double u = U(g), uz = u * zn;
      uint64_t z;
      if (uz < 1.0)       z = 0;
      else if (uz < z2)   z = 1;
      else                z = (uint64_t)((double)N * std::pow(eta * u - eta + 1.0, alpha));
      if (z >= N) z = N - 1;
      uint64_t key = fnv1a(z) % N;   // scramble: spread hot keys across the keyspace
      buf.push_back(key);
      if (buf.size() == (1u << 20)) { std::fwrite(buf.data(), sizeof(uint64_t), buf.size(), f); buf.clear(); }
    }
    if (!buf.empty()) std::fwrite(buf.data(), sizeof(uint64_t), buf.size(), f);
    std::fclose(f);
    std::fprintf(stderr, "wrote %s: %llu keys (Zipf theta=%.2f, scrambled)\n",
                 rf, (unsigned long long)M, THETA);
  }
  return 0;
}
