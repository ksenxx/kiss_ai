"""Isolated candidate-method runner (child process side).

Executed by ``harness._predict_module`` in a separate OS process. The child
receives ONLY the training features/labels and the evaluation features via an
``.npz`` file, runs the candidate module's ``fit_predict``, and writes the raw
predictions to an ``.npy`` file. Sealed test labels never enter this process,
so a hostile candidate has no in-memory channel to them.

Usage:
    python _method_child.py <method_module> <in.npz> <out.npy>
"""

from __future__ import annotations

import importlib
import re
import sys

import numpy as np

_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def main() -> int:
    """Load inputs, run the candidate method, and persist its predictions.

    Returns:
        Process exit status: 0 on success, 2 on usage/validation errors.
    """
    if len(sys.argv) != 4:
        print("usage: python _method_child.py <method_module> <in.npz> <out.npy>", file=sys.stderr)
        return 2
    module_name, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if not _MODULE_NAME_RE.fullmatch(module_name):
        print(f"invalid method module name: {module_name!r}", file=sys.stderr)
        return 2
    with np.load(in_path, allow_pickle=False) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_eval = data["x_eval"]
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "fit_predict"):
        print(f"{module_name} has no fit_predict(x_train, y_train, x_eval)", file=sys.stderr)
        return 2
    pred = np.asarray(mod.fit_predict(x_train, y_train, x_eval), dtype=np.float64)
    np.save(out_path, pred, allow_pickle=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
