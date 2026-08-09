"""Security-focused adversarial end-to-end tests.

Each test here actively tries to BREAK the benchmark system the way a hostile
or buggy candidate method (or caller) would: cheating via frame inspection,
mutating shared arrays, returning degenerate output, hanging forever, passing
malicious module names, or exploiting seed-derivation weaknesses. All tests are
real end-to-end tests -- no mocks -- exercising the actual generators, harness,
and methods.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
import sys
import time

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datagen import Dataset, make_benchmark  # noqa: E402
from harness import (  # noqa: E402
    _load_method,
    score_test,
    score_test_module,
    score_validation,
)


def _checksum(arr: np.ndarray) -> str:
    """Stable content hash of an array."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Cross-process determinism (regression test for the salted-hash() seed bug). #
# --------------------------------------------------------------------------- #
def test_cross_process_determinism():
    """Datasets must be bit-identical across processes with different hash salts.

    The original implementation derived RNG seeds from Python's salted
    ``hash()``, so every process built *different* benchmarks. This test runs
    two fresh interpreters with different PYTHONHASHSEED values and asserts the
    generated data is identical.
    """
    snippet = (
        "from datagen import make_benchmark\n"
        "import hashlib\n"
        "h = hashlib.sha256()\n"
        "for name in ('perturb_replogle', 'enhancer_gene_linking'):\n"
        "    ds = make_benchmark(name, master_seed=0)\n"
        "    for a in (ds.x_train, ds.y_train, ds.x_test, ds.y_test):\n"
        "        h.update(a.tobytes())\n"
        "print(h.hexdigest())\n"
    )
    digests = []
    for salt in ("1", "31337"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
        assert out.returncode == 0, out.stderr
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1], "dataset generation depends on PYTHONHASHSEED"


# --------------------------------------------------------------------------- #
# Cheating method: frame-walking toward the sealed test labels.               #
# --------------------------------------------------------------------------- #
def test_frame_walking_cheat_cannot_reach_test_labels():
    """A method that walks the ``f_back`` frame chain must find no Dataset.

    Before hardening, ``fit_predict`` could reach the harness frame holding
    ``ds`` and return ``ds.y_test`` for a perfect sealed-test score. The
    harness now runs methods on a fresh thread whose stack contains no harness
    frames, so the cheat finds nothing and must fall back to a constant guess.
    """
    found: dict[str, object] = {"dataset": None}

    def cheat(x_train, y_train, x_eval):
        frame = inspect.currentframe()
        while frame is not None:
            for value in list(frame.f_locals.values()):
                if isinstance(value, Dataset):
                    found["dataset"] = value
                    return np.asarray(value.y_test, dtype=np.float64)
            frame = frame.f_back
        return np.zeros(x_eval.shape[0])

    ds = make_benchmark("perturb_replogle", master_seed=0)
    score = score_test(cheat, ds)
    assert found["dataset"] is None, "cheat reached a Dataset through the frame chain"
    assert score < 10.0, f"cheat scored {score:.3f}; test labels leaked"


# --------------------------------------------------------------------------- #
# Cheating method: mutating its inputs.                                       #
# --------------------------------------------------------------------------- #
def test_mutating_method_is_blocked_and_dataset_survives():
    """In-place writes to the provided arrays must fail, leaving data intact."""
    ds = make_benchmark("perturb_adamson", master_seed=0)
    before = [_checksum(a) for a in (ds.x_train, ds.y_train, ds.x_val, ds.x_test, ds.y_test)]

    def vandal(x_train, y_train, x_eval):
        x_eval[:, 0] = 0.0  # must raise: arrays handed to methods are read-only
        return np.zeros(x_eval.shape[0])

    with pytest.raises(ValueError):
        score_test(vandal, ds)

    after = [_checksum(a) for a in (ds.x_train, ds.y_train, ds.x_val, ds.x_test, ds.y_test)]
    assert before == after, "benchmark arrays were corrupted by a hostile method"

    # And the untouched dataset still scores identically to a fresh build.
    import methods.tuso_evolved as evolved
    fresh = make_benchmark("perturb_adamson", master_seed=0)
    assert score_test(evolved.fit_predict, ds) == score_test(evolved.fit_predict, fresh)


def test_dataset_arrays_are_read_only():
    """Direct writes to any split array must raise."""
    ds = make_benchmark("enhancer_gene_linking", master_seed=0)
    for arr in (ds.x_train, ds.y_train, ds.x_val, ds.y_val, ds.x_test, ds.y_test):
        with pytest.raises(ValueError):
            arr[0] = 123.0


# --------------------------------------------------------------------------- #
# Degenerate method output.                                                   #
# --------------------------------------------------------------------------- #
def test_constant_output_scores_at_chance():
    """A constant predictor must be scored (no crash) far below the 99 gate."""
    for name, ceiling in (("perturb_replogle", 10.0), ("enhancer_gene_linking", 55.0)):
        ds = make_benchmark(name, master_seed=0)
        score = score_validation(lambda xt, yt, xe: np.full(xe.shape[0], 0.5), ds)
        assert score < ceiling, f"{name}: constant output scored {score:.3f}"


def test_non_numeric_output_rejected():
    """A method returning non-numeric junk must raise ValueError."""
    ds = make_benchmark("perturb_replogle", master_seed=0)
    with pytest.raises(ValueError):
        score_validation(lambda xt, yt, xe: ["junk"] * xe.shape[0], ds)


# --------------------------------------------------------------------------- #
# Resource limits.                                                            #
# --------------------------------------------------------------------------- #
def test_hanging_method_times_out():
    """A method that stalls must raise TimeoutError, not block the evaluator."""
    def sleeper(x_train, y_train, x_eval):
        time.sleep(30.0)
        return np.zeros(x_eval.shape[0])

    ds = make_benchmark("perturb_replogle", master_seed=0)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        score_validation(sleeper, ds, timeout_s=1.0)
    assert time.monotonic() - start < 10.0


# --------------------------------------------------------------------------- #
# Extreme / malformed seeds.                                                  #
# --------------------------------------------------------------------------- #
def test_extreme_seeds_are_deterministic():
    """A huge master seed must work and reproduce bit-identically."""
    a = make_benchmark("perturb_norman", master_seed=2**62)
    b = make_benchmark("perturb_norman", master_seed=2**62)
    assert _checksum(a.x_train) == _checksum(b.x_train)
    assert _checksum(a.y_test) == _checksum(b.y_test)


def test_invalid_seeds_rejected():
    """Negative, boolean, and non-integer seeds must raise ValueError."""
    for bad in (-1, -(2**40), True, 1.5, "7", None):
        for name in ("perturb_replogle", "enhancer_gene_linking"):
            with pytest.raises(ValueError):
                make_benchmark(name, master_seed=bad)


# --------------------------------------------------------------------------- #
# Malicious module names.                                                     #
# --------------------------------------------------------------------------- #
def test_load_method_rejects_malicious_names():
    """Only dotted identifier paths may reach importlib."""
    for bad in (
        "methods/../../etc",
        "methods..baseline",
        "methods.baseline; import os",
        "os.system('true')",
        ".hidden",
        "",
        "methods.base-line",
        123,
    ):
        with pytest.raises(ValueError):
            _load_method(bad)
    # A legitimate name still loads.
    assert callable(_load_method("methods.baseline"))


# --------------------------------------------------------------------------- #
# CLI argument hardening.                                                     #
# --------------------------------------------------------------------------- #
def test_run_benchmarks_rejects_bad_threshold():
    """An unparseable or non-finite threshold must exit 2 before evaluating."""
    for bad in ("notafloat", "nan"):
        out = subprocess.run(
            [sys.executable, "run_benchmarks.py", "methods.tuso_evolved", bad],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 2, out.stderr


def test_eval_runner_rejects_unknown_benchmark():
    """An unknown benchmark name must exit 2 with a clear message."""
    out = subprocess.run(
        [sys.executable, "eval_runner.py", "no_such_benchmark", "methods.baseline"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 2
    assert "unknown benchmark" in out.stderr


# --------------------------------------------------------------------------- #
# Cross-thread frame walking / gc scanning, defeated by process isolation.    #
# --------------------------------------------------------------------------- #
_CROSS_THREAD_CHEAT = '''
"""Hostile method: scans every live frame in every thread, then the gc heap."""
import gc
import sys

import numpy as np


def fit_predict(x_train, y_train, x_eval):
    # Channel 1: walk ALL threads' frame chains (defeats thread-only isolation).
    for frame in list(sys._current_frames().values()):
        f = frame
        while f is not None:
            for value in list(f.f_locals.values()):
                if hasattr(value, "y_test"):
                    return np.asarray(value.y_test, dtype=np.float64)
            f = f.f_back
    # Channel 2: scan the whole heap for a label-like 1-D array of eval length.
    ours = {id(x_train), id(y_train), id(x_eval)}
    n = x_eval.shape[0]
    for obj in gc.get_objects():
        if (
            isinstance(obj, np.ndarray)
            and obj.ndim == 1
            and obj.shape[0] == n
            and obj.dtype == np.float64
            and id(obj) not in ours
        ):
            return np.asarray(obj, dtype=np.float64)
    return np.zeros(n)
'''


def test_cross_thread_and_gc_cheat_blocked_by_process_isolation():
    """Even sys._current_frames() + gc.get_objects() must not reach labels.

    Thread-level isolation alone is defeated by ``sys._current_frames()``
    (verified during review: such a cheat scored 100.0 in-process). The
    official evaluation path therefore runs candidate MODULES in a separate
    OS process where the sealed labels simply do not exist in memory. This
    test runs a maximally hostile module through that path and asserts it
    scores at chance.
    """
    cheat_name = f"cheat_mod_{os.getpid()}"
    cheat_path = os.path.join(PROJECT_ROOT, cheat_name + ".py")
    with open(cheat_path, "w") as fh:
        fh.write(_CROSS_THREAD_CHEAT)
    try:
        ds = make_benchmark("perturb_replogle", master_seed=0)
        score = score_test_module(cheat_name, ds)
        assert score < 10.0, f"hostile module scored {score:.3f}; sealed labels leaked"
    finally:
        os.unlink(cheat_path)


# --------------------------------------------------------------------------- #
# SystemExit laundering.                                                      #
# --------------------------------------------------------------------------- #
def test_system_exit_method_cannot_abort_evaluation():
    """A method raising SystemExit(0) must surface as ValueError, not exit 0.

    Before hardening, the harness re-raised the candidate's exception object
    verbatim, so ``raise SystemExit(0)`` inside a method terminated the whole
    evaluator with a *success* status before the gate could score anything.
    """
    def escape_artist(x_train, y_train, x_eval):
        raise SystemExit(0)

    ds = make_benchmark("perturb_replogle", master_seed=0)
    with pytest.raises(ValueError, match="SystemExit"):
        score_validation(escape_artist, ds)


def test_method_exceptions_convert_to_value_error():
    """Arbitrary method exceptions must surface as ValueError with context."""
    def crasher(x_train, y_train, x_eval):
        raise RuntimeError("boom")

    ds = make_benchmark("perturb_replogle", master_seed=0)
    with pytest.raises(ValueError, match="RuntimeError: boom"):
        score_validation(crasher, ds)


# --------------------------------------------------------------------------- #
# NumPy integer seed overflow.                                                #
# --------------------------------------------------------------------------- #
def test_numpy_integer_seed_does_not_overflow_or_alias():
    """np.int64 seeds must behave exactly like Python int seeds (no wraparound).

    ``np.int64(2**62) * 1000`` silently overflows to 0 in fixed-width
    arithmetic, which previously made the huge seed alias master_seed=0.
    """
    big_np = make_benchmark("perturb_norman", master_seed=np.int64(2**62))
    big_py = make_benchmark("perturb_norman", master_seed=2**62)
    zero = make_benchmark("perturb_norman", master_seed=0)
    assert _checksum(big_np.x_train) == _checksum(big_py.x_train)
    assert _checksum(big_np.x_train) != _checksum(zero.x_train)


# --------------------------------------------------------------------------- #
# Strict 1-D prediction contract.                                             #
# --------------------------------------------------------------------------- #
def test_matrix_output_with_matching_size_rejected():
    """A 2-D (r, c) output whose r*c equals the row count must be rejected.

    The old validator ravel()ed first, so any matrix with the right *element*
    count slipped through the shape check.
    """
    ds = make_benchmark("perturb_replogle", master_seed=0)
    n = ds.x_val.shape[0]
    assert n % 2 == 0
    with pytest.raises(ValueError, match="shape"):
        score_validation(lambda xt, yt, xe: np.zeros((2, n // 2)), ds)
    # A trailing singleton dimension (sklearn-style column vector) is fine.
    score = score_validation(lambda xt, yt, xe: np.zeros((xe.shape[0], 1)), ds)
    assert score <= 0.0  # constant regression output scores at/below chance


# --------------------------------------------------------------------------- #
# Dataset ownership and irreversibility of the write protection.              #
# --------------------------------------------------------------------------- #
def test_dataset_copies_and_freezes_without_touching_caller_arrays():
    """Dataset must not freeze/alias caller arrays, and its own arrays must
    stay read-only even against setflags(write=True)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 3))
    y = rng.normal(size=8)
    ds = Dataset("toy", "regression", x, y, x.copy(), y.copy(), x.copy(), y.copy(), "r2")

    # Caller arrays are untouched (still writable, not aliased).
    assert x.flags.writeable and y.flags.writeable
    x[0, 0] = 123.0
    assert ds.x_train[0, 0] != 123.0

    # The exposed arrays are read-only views whose protection cannot be
    # re-enabled directly.
    with pytest.raises(ValueError):
        ds.x_train.setflags(write=True)
    with pytest.raises(ValueError):
        ds.y_test.setflags(write=True)
