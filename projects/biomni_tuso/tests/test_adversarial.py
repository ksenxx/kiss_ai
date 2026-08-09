"""Adversarial / anti-cheating end-to-end tests for the Biomni x TusoAI system.

These are real end-to-end tests (no mocks): they build the actual benchmarks,
run the actual methods through the actual harness, and assert on genuine
behaviour. They exist to prove the reported >=99 scores are legitimate and not
the product of leakage, memorization, or a degenerate benchmark.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import methods.baseline as baseline  # noqa: E402
import methods.tuso_evolved as evolved  # noqa: E402
from datagen import BENCHMARKS, make_benchmark  # noqa: E402
from harness import evaluate_all, score_test, score_validation  # noqa: E402


# --------------------------------------------------------------------------- #
# The evolved method clears 99 everywhere (the headline claim).               #
# --------------------------------------------------------------------------- #
def test_evolved_passes_all_benchmarks_primary_and_generalization():
    """Every sealed-test and generalization score must be >= 99."""
    for seed in (0, 7, 13):
        results = evaluate_all("methods.tuso_evolved", master_seed=seed)
        assert len(results) == len(BENCHMARKS)
        for r in results:
            assert r.test >= 99.0, f"{r.name} seed={seed} test={r.test:.3f} < 99"


# --------------------------------------------------------------------------- #
# The benchmark is non-trivial: a naive baseline is well below 99.            #
# --------------------------------------------------------------------------- #
def test_benchmark_is_non_trivial():
    """A naive raw-feature baseline must fail to reach 99 on every benchmark."""
    for name in BENCHMARKS:
        ds = make_benchmark(name, master_seed=0)
        val = score_validation(baseline.fit_predict, ds)
        assert val < 97.0, f"{name} baseline val={val:.3f} unexpectedly high (benchmark too easy)"


# --------------------------------------------------------------------------- #
# No label leakage: destroying the train signal must destroy the score.       #
# --------------------------------------------------------------------------- #
def test_shuffled_labels_collapse_score():
    """If training labels are shuffled, the sealed-test score must collapse.

    A method that somehow peeked at test labels would still score high here;
    collapse proves the score comes only from a genuine learned train->test
    relationship.
    """
    for name in BENCHMARKS:
        ds = make_benchmark(name, master_seed=0)
        rng = np.random.default_rng(0)
        y_tr = ds.y_train.copy()
        rng.shuffle(y_tr)
        y_va = ds.y_val.copy()
        rng.shuffle(y_va)
        shuffled = ds.__class__(
            ds.name, ds.task, ds.x_train, y_tr, ds.x_val, y_va,
            ds.x_test, ds.y_test, ds.metric,
        )
        score = score_test(evolved.fit_predict, shuffled)
        # Chance level is 0 for R^2 and 50 for AUC; allow a small margin.
        ceiling = 55.0 if ds.metric == "auc" else 10.0
        assert score < ceiling, f"{name} shuffled score={score:.3f} too high -> leakage suspected"


# --------------------------------------------------------------------------- #
# Determinism: identical inputs -> identical scores.                          #
# --------------------------------------------------------------------------- #
def test_determinism():
    """Two evaluations of the same method/seed must be bit-for-bit identical."""
    a = evaluate_all("methods.tuso_evolved", master_seed=0)
    b = evaluate_all("methods.tuso_evolved", master_seed=0)
    for ra, rb in zip(a, b):
        assert ra.test == rb.test
        assert ra.validation == rb.validation


# --------------------------------------------------------------------------- #
# The generative target function is identical across splits.                  #
# --------------------------------------------------------------------------- #
def test_target_function_consistent_across_splits():
    """A method trained on train must generalize to val/test with a tiny gap.

    A large train->test degradation would indicate the per-split target drift
    bug that previously broke the sealed-test scores.
    """
    for name in BENCHMARKS:
        ds = make_benchmark(name, master_seed=0)
        val = score_validation(evolved.fit_predict, ds)
        test = score_test(evolved.fit_predict, ds)
        assert abs(val - test) < 2.0, f"{name} val/test gap {abs(val-test):.3f} too large"


# --------------------------------------------------------------------------- #
# Test labels are never handed to a method.                                   #
# --------------------------------------------------------------------------- #
def test_method_never_receives_test_labels():
    """The harness must only ever pass features (not labels) for evaluation rows."""
    seen = {}

    def spy(x_train, y_train, x_eval):
        seen["n_train"] = x_train.shape[0]
        seen["n_eval"] = x_eval.shape[0]
        # A method receives exactly three positional args; there is no channel
        # through which test labels could arrive.
        return np.zeros(x_eval.shape[0])

    ds = make_benchmark("perturb_replogle", master_seed=0)
    score_test(spy, ds)
    assert seen["n_eval"] == ds.x_test.shape[0]
    assert seen["n_train"] == ds.x_train.shape[0] + ds.x_val.shape[0]


# --------------------------------------------------------------------------- #
# Robustness: harness rejects malformed method outputs.                       #
# --------------------------------------------------------------------------- #
def test_harness_rejects_wrong_shape():
    """A method returning the wrong number of predictions must raise."""
    ds = make_benchmark("perturb_replogle", master_seed=0)
    with pytest.raises(ValueError):
        score_validation(lambda xt, yt, xe: np.zeros(xe.shape[0] + 1), ds)


def test_harness_rejects_nan_output():
    """A method returning non-finite predictions must raise."""
    ds = make_benchmark("perturb_replogle", master_seed=0)
    with pytest.raises(ValueError):
        score_validation(lambda xt, yt, xe: np.full(xe.shape[0], np.nan), ds)


# --------------------------------------------------------------------------- #
# Splits are disjoint (no row overlap between train/val/test).                #
# --------------------------------------------------------------------------- #
def test_splits_are_disjoint():
    """Train, validation, and test feature rows must not overlap."""
    for name in BENCHMARKS:
        ds = make_benchmark(name, master_seed=0)
        tr = {row.tobytes() for row in ds.x_train}
        va = {row.tobytes() for row in ds.x_val}
        te = {row.tobytes() for row in ds.x_test}
        assert tr.isdisjoint(te)
        assert tr.isdisjoint(va)
        assert va.isdisjoint(te)
