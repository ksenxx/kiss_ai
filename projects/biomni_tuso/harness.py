"""TusoAI-style evaluation harness.

A *method* is any object exposing::

    fit_predict(x_train, y_train, x_eval) -> np.ndarray

returning predictions for ``x_eval`` (continuous for regression, class-1
probabilities for classification). This mirrors TusoAI's contract where a
candidate is evaluated by a reference script that trains and prints
``tuso_evaluate: <score>`` (see the TusoAI README / ``optimization.py``).

Scoring is a standard 0-100 percentage:
    * regression  -> R^2 * 100
    * classification -> ROC-AUC * 100

Split discipline (anti-cheating):
    * ``score_validation`` trains on TRAIN and scores on VALIDATION. This is the
      only signal a discovery loop is allowed to optimize against.
    * ``score_test`` trains on TRAIN+VAL and scores on the SEALED TEST split.
      Methods never receive test labels.
    * ``score_generalization`` rebuilds every benchmark from a *different master
      seed* and reports sealed-test scores, proving the method is not overfit to
      one sampling of the generative process.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass

import numpy as np
from datagen import BENCHMARKS, Dataset, make_benchmark
from sklearn.metrics import r2_score, roc_auc_score

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHILD_RUNNER = os.path.join(_PROJECT_DIR, "_method_child.py")

#: Dotted path of plain Python identifiers -- the only module names the harness
#: will import. Blocks path traversal, shell metacharacters, and anything that
#: is not a legitimate importable module path.
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")

#: Wall-clock budget (seconds) for a single fit_predict call. A hung or
#: runaway candidate method raises TimeoutError instead of blocking the
#: evaluator forever.
DEFAULT_TIMEOUT_S = 900.0


def _score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the 0-100 benchmark score for one split."""
    if metric == "r2":
        return float(r2_score(y_true, y_pred) * 100.0)
    if metric == "auc":
        return float(roc_auc_score(y_true, y_pred) * 100.0)
    raise ValueError(f"Unknown metric: {metric!r}")


def _load_method(module_name: str):
    """Import a candidate method module and return its ``fit_predict`` callable.

    The module name is validated against a strict dotted-identifier pattern
    before being handed to ``importlib``, so CLI callers cannot smuggle in path
    traversal or other malformed import targets.

    Args:
        module_name: Importable dotted module path, e.g. ``methods.baseline``.

    Returns:
        The module's ``fit_predict`` callable.

    Raises:
        ValueError: If the module name is not a plain dotted identifier path.
        AttributeError: If the module lacks a ``fit_predict`` function.
    """
    if not isinstance(module_name, str) or not _MODULE_NAME_RE.fullmatch(module_name):
        raise ValueError(f"invalid method module name: {module_name!r}")
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
    if not hasattr(mod, "fit_predict"):
        raise AttributeError(f"{module_name} has no fit_predict(x_train, y_train, x_eval)")
    return mod.fit_predict


def _readonly_copy(arr: np.ndarray) -> np.ndarray:
    """Return a private, write-protected copy of an array.

    Candidate methods only ever see these copies, so they can neither mutate
    the benchmark's arrays nor observe later harness state through shared
    memory.
    """
    out = np.array(arr, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _validate_predictions(pred_obj, n_rows: int) -> np.ndarray:
    """Coerce and validate a candidate method's raw output.

    Args:
        pred_obj: Whatever the method returned.
        n_rows: Number of evaluation rows the method was asked to predict.

    Returns:
        A validated 1-D float64 array of length ``n_rows``.

    Raises:
        ValueError: If the output is non-numeric, not 1-D (a trailing
            singleton dimension is tolerated), the wrong length, or contains
            non-finite values.
    """
    try:
        pred = np.asarray(pred_obj, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"method returned non-numeric predictions: {exc}") from exc
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim != 1:
        raise ValueError(f"method returned predictions of shape {pred.shape}; expected 1-D")
    if pred.shape[0] != n_rows:
        raise ValueError(f"method returned {pred.shape[0]} predictions for {n_rows} rows")
    if not np.all(np.isfinite(pred)):
        raise ValueError("method returned non-finite predictions")
    return pred


def _predict(
    fit_predict, x_train, y_train, x_eval, timeout_s: float = DEFAULT_TIMEOUT_S
) -> np.ndarray:
    """Run an in-process candidate callable under best-effort isolation.

    Defenses applied to every call:
      * The method receives read-only *copies* of the data (mutation attempts
        raise, and the harness's own arrays can never be corrupted).
      * The method runs on a fresh worker thread whose own ``f_back`` chain
        contains no harness frames. NOTE: this is *best-effort* only -- a
        hostile method could still reach other threads' frames through
        ``sys._current_frames()`` or scan ``gc.get_objects()``. Genuine
        isolation is provided by :func:`_predict_module`, which runs the
        method in a separate OS process; that is what ``evaluate_all`` and
        the CLI entry points use.
      * A wall-clock timeout bounds runaway methods.
      * Any exception raised by the method (including control-flow exceptions
        such as ``SystemExit``, which a hostile method could otherwise use to
        make the acceptance gate exit successfully) is converted into a
        ``ValueError``.
      * The returned predictions are coerced and validated (numeric, 1-D,
        correct length, all finite).

    Args:
        fit_predict: Callable ``(x_train, y_train, x_eval) -> predictions``.
        x_train: Training features.
        y_train: Training labels.
        x_eval: Evaluation features (labels are never passed).
        timeout_s: Wall-clock budget for the call, in seconds.

    Returns:
        Validated 1-D float64 prediction array.

    Raises:
        TimeoutError: If the method exceeds ``timeout_s``.
        ValueError: If the method raises, or returns malformed output.
    """
    xt = _readonly_copy(x_train)
    yt = _readonly_copy(y_train)
    xe = _readonly_copy(x_eval)
    box: dict[str, object] = {}

    def _worker() -> None:
        try:
            box["pred"] = fit_predict(xt, yt, xe)
        except BaseException as exc:  # noqa: BLE001 - report, don't swallow silently
            box["exc"] = exc

    thread = threading.Thread(target=_worker, name="tuso-candidate", daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"method exceeded the {timeout_s:.0f}s evaluation budget")
    if "exc" in box:
        exc = box["exc"]
        raise ValueError(f"method raised {type(exc).__name__}: {exc}") from exc  # type: ignore[arg-type]

    return _validate_predictions(box.get("pred"), xe.shape[0])


def _predict_module(module_name: str, x_train, y_train, x_eval,
                    timeout_s: float = DEFAULT_TIMEOUT_S) -> np.ndarray:
    """Run a candidate method *module* in an isolated child OS process.

    This is the strong isolation path used for official evaluation: the child
    process receives ONLY the serialized ``(x_train, y_train, x_eval)`` arrays
    and returns only a prediction array. Sealed test labels live exclusively
    in the parent process's memory, so no amount of frame walking,
    ``sys._current_frames()``, or ``gc.get_objects()`` scanning inside the
    candidate can reach them.

    Args:
        module_name: Importable dotted module path exposing ``fit_predict``.
        x_train: Training features.
        y_train: Training labels.
        x_eval: Evaluation features (labels are never passed).
        timeout_s: Wall-clock budget; the child is killed on expiry.

    Returns:
        Validated 1-D float64 prediction array.

    Raises:
        ValueError: If the module name is invalid, the child fails, or the
            output is malformed.
        TimeoutError: If the child exceeds ``timeout_s``.
    """
    if not isinstance(module_name, str) or not _MODULE_NAME_RE.fullmatch(module_name):
        raise ValueError(f"invalid method module name: {module_name!r}")
    with tempfile.TemporaryDirectory(prefix="tuso_eval_") as tmp_dir:
        in_path = os.path.join(tmp_dir, "in.npz")
        out_path = os.path.join(tmp_dir, "out.npy")
        np.savez(
            in_path,
            x_train=np.asarray(x_train, dtype=np.float64),
            y_train=np.asarray(y_train, dtype=np.float64),
            x_eval=np.asarray(x_eval, dtype=np.float64),
        )
        try:
            proc = subprocess.run(
                [sys.executable, _CHILD_RUNNER, module_name, in_path, out_path],
                cwd=_PROJECT_DIR, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"method exceeded the {timeout_s:.0f}s evaluation budget") from exc
        if proc.returncode != 0 or not os.path.exists(out_path):
            detail = (proc.stderr or "").strip()[-500:] or "no output produced"
            raise ValueError(
                f"method subprocess failed (exit {proc.returncode}): {detail}"
            )
        pred = np.load(out_path, allow_pickle=False)
    return _validate_predictions(pred, np.asarray(x_eval).shape[0])


def score_validation(fit_predict, ds: Dataset, timeout_s: float = DEFAULT_TIMEOUT_S) -> float:
    """Train on TRAIN, score on VALIDATION (the only optimizable signal)."""
    pred = _predict(fit_predict, ds.x_train, ds.y_train, ds.x_val, timeout_s=timeout_s)
    return _score(ds.metric, ds.y_val, pred)


def score_test(fit_predict, ds: Dataset, timeout_s: float = DEFAULT_TIMEOUT_S) -> float:
    """Train on TRAIN+VAL, score on the SEALED TEST split."""
    x = np.concatenate([ds.x_train, ds.x_val], axis=0)
    y = np.concatenate([ds.y_train, ds.y_val], axis=0)
    pred = _predict(fit_predict, x, y, ds.x_test, timeout_s=timeout_s)
    return _score(ds.metric, ds.y_test, pred)


def score_validation_module(module_name: str, ds: Dataset,
                            timeout_s: float = DEFAULT_TIMEOUT_S) -> float:
    """Like :func:`score_validation`, but with child-process isolation."""
    pred = _predict_module(module_name, ds.x_train, ds.y_train, ds.x_val, timeout_s=timeout_s)
    return _score(ds.metric, ds.y_val, pred)


def score_test_module(module_name: str, ds: Dataset,
                      timeout_s: float = DEFAULT_TIMEOUT_S) -> float:
    """Like :func:`score_test`, but with child-process isolation."""
    x = np.concatenate([ds.x_train, ds.x_val], axis=0)
    y = np.concatenate([ds.y_train, ds.y_val], axis=0)
    pred = _predict_module(module_name, x, y, ds.x_test, timeout_s=timeout_s)
    return _score(ds.metric, ds.y_test, pred)


@dataclass
class BenchResult:
    """Per-benchmark scores."""

    name: str
    validation: float
    test: float


def evaluate_all(module_name: str, master_seed: int = 0) -> list[BenchResult]:
    """Evaluate a method module across every benchmark with process isolation.

    Every ``fit_predict`` call runs in a separate child OS process that only
    ever receives train features/labels and evaluation features; the sealed
    test labels stay in this (parent) process.

    Args:
        module_name: Importable module exposing ``fit_predict``.
        master_seed: Seed for benchmark construction (use a fresh one for the
            generalization check).

    Returns:
        A list of :class:`BenchResult`, one per benchmark.
    """
    if not isinstance(module_name, str) or not _MODULE_NAME_RE.fullmatch(module_name):
        raise ValueError(f"invalid method module name: {module_name!r}")
    results = []
    for name in BENCHMARKS:
        ds = make_benchmark(name, master_seed=master_seed)
        val = score_validation_module(module_name, ds)
        test = score_test_module(module_name, ds)
        results.append(BenchResult(name, val, test))
    return results
