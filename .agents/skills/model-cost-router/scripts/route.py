#!/usr/bin/env python3
# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Pick the cheapest runnable model for a routing tier and log the decision.

Standard library only.  Model prices come from the local Sorcar catalog
(``$KISS_HOME/MODEL_INFO.json``, default ``~/.kiss/MODEL_INFO.json``) when
it exists and from ``assets/tiers.json`` otherwise.  When the ``kiss``
package is importable the candidate list is filtered to models whose
provider credential is configured; otherwise every candidate is reported
with ``"runnable": null`` (unknown).

Sub-commands::

    route.py menu  [--in N --out N] [--catalog F]   priced menu of every tier
    route.py pick  --tier small|medium|frontier [--in N --out N] [--exclude M ...] [--catalog F]
    route.py estimate --model M --in N --out N [--catalog F]
    route.py log   --unit U --tier T --model M --reason R [--outcome O] [--file F]

``pick`` prints one JSON object with the chosen model; every command exits
1 with a message on stderr when it cannot produce an answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parent.parent
TIERS_FILE = SKILL_DIR / "assets" / "tiers.json"
TIER_NAMES = ("small", "medium", "frontier")
DEFAULT_LOG = Path("tmp") / "MODEL_DECISIONS.md"


def load_tiers() -> dict[str, list[dict[str, Any]]]:
    """Return the ordered candidate lists from ``assets/tiers.json``."""
    with TIERS_FILE.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {tier: raw[tier] for tier in TIER_NAMES}


def default_catalog_path() -> Path:
    """Return ``$KISS_HOME/MODEL_INFO.json`` (``~/.kiss`` when unset)."""
    return Path(os.environ.get("KISS_HOME", str(Path.home() / ".kiss"))) / "MODEL_INFO.json"


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Return the model catalog at *path*, or an empty dict when absent or malformed."""
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def runnable_models() -> set[str] | None:
    """Return the models with a configured credential, or None when unknown."""
    try:
        from kiss.core.models.model_info import get_available_models
    except ImportError:
        return None
    return set(get_available_models())


def catalog_prices(model: str, catalog: dict[str, dict[str, Any]]) -> tuple[float, float] | None:
    """Return (input, output) USD per 1M tokens from *catalog*, or None when not priced there."""
    entry = catalog.get(model) or {}
    prices = entry.get("input_price_per_1M"), entry.get("output_price_per_1M")
    if prices[0] is None or prices[1] is None:
        return None
    return float(prices[0]), float(prices[1])


def price_of(candidate: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> tuple[float, float]:
    """Return (input, output) USD per 1M tokens, preferring the local catalog."""
    return catalog_prices(candidate["model"], catalog) or (
        float(candidate["input"]),
        float(candidate["output"]),
    )


def estimate_usd(input_price: float, output_price: float, tokens_in: int, tokens_out: int) -> float:
    """Return the USD cost of *tokens_in* prompt and *tokens_out* completion tokens."""
    return (input_price * tokens_in + output_price * tokens_out) / 1_000_000


def describe(
    candidate: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    runnable: set[str] | None,
    tokens_in: int,
    tokens_out: int,
) -> dict[str, Any]:
    """Return one priced, availability-annotated candidate record."""
    input_price, output_price = price_of(candidate, catalog)
    return {
        "model": candidate["model"],
        "input_per_1M": input_price,
        "output_per_1M": output_price,
        "estimated_usd": round(estimate_usd(input_price, output_price, tokens_in, tokens_out), 4),
        "runnable": None if runnable is None else candidate["model"] in runnable,
        "note": candidate.get("note", ""),
    }


def tier_candidates(
    tier: str, catalog_path: Path, tokens_in: int, tokens_out: int, exclude: set[str]
) -> list[dict[str, Any]]:
    """Return the described candidates of *tier* in preference order, minus *exclude*."""
    catalog = load_catalog(catalog_path)
    runnable = runnable_models()
    return [
        describe(candidate, catalog, runnable, tokens_in, tokens_out)
        for candidate in load_tiers()[tier]
        if candidate["model"] not in exclude
    ]


def cmd_menu(args: argparse.Namespace) -> int:
    """Print every tier's candidates as JSON."""
    menu = {
        tier: tier_candidates(tier, Path(args.catalog), args.tokens_in, args.tokens_out, set())
        for tier in TIER_NAMES
    }
    print(json.dumps(menu, indent=2))
    return 0


def cmd_pick(args: argparse.Namespace) -> int:
    """Print the first runnable candidate of the requested tier."""
    candidates = tier_candidates(
        args.tier, Path(args.catalog), args.tokens_in, args.tokens_out, set(args.exclude)
    )
    usable = [c for c in candidates if c["runnable"] is not False]
    if not usable:
        print(
            f"no runnable model in tier {args.tier!r}; "
            "configure an API key or edit assets/tiers.json",
            file=sys.stderr,
        )
        return 1
    chosen = dict(usable[0], tier=args.tier)
    print(json.dumps(chosen, indent=2))
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    """Print the estimated USD cost of a call to one model."""
    listed = [c for tier in load_tiers().values() for c in tier if c["model"] == args.model]
    if listed:
        prices = price_of(listed[0], load_catalog(Path(args.catalog)))
    else:
        found = catalog_prices(args.model, load_catalog(Path(args.catalog)))
        if found is None:
            print(
                f"unknown model {args.model!r}: not in the local catalog or assets/tiers.json",
                file=sys.stderr,
            )
            return 1
        prices = found
    cost = estimate_usd(prices[0], prices[1], args.tokens_in, args.tokens_out)
    print(json.dumps({"model": args.model, "estimated_usd": round(cost, 4)}))
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    """Append one routing decision to the Markdown ledger."""
    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        columns = "| time (UTC) | unit | tier | model | reason | outcome |"
        rule = "|---|---|---|---|---|---|"
        path.write_text(f"# Model routing decisions\n\n{columns}\n{rule}\n", encoding="utf-8")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    cells = [stamp, args.unit, args.tier, args.model, args.reason, args.outcome]
    with path.open("a", encoding="utf-8") as handle:
        cleaned = [" ".join(cell.replace("|", "/").split()) for cell in cells]
        handle.write("| " + " | ".join(cleaned) + " |\n")
    print(f"logged to {path}")
    return 0


def add_pricing_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--in`` / ``--out`` token counts and ``--catalog`` path."""
    parser.add_argument(
        "--in", dest="tokens_in", type=int, default=200_000, help="prompt tokens (default 200k)"
    )
    parser.add_argument(
        "--out", dest="tokens_out", type=int, default=20_000, help="completion tokens (default 20k)"
    )
    parser.add_argument(
        "--catalog",
        default=str(default_catalog_path()),
        help="MODEL_INFO.json to read prices from (default $KISS_HOME)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    menu = sub.add_parser("menu", help="priced menu of every tier")
    add_pricing_args(menu)
    menu.set_defaults(func=cmd_menu)

    pick = sub.add_parser("pick", help="cheapest runnable model of one tier")
    pick.add_argument("--tier", choices=TIER_NAMES, required=True)
    pick.add_argument(
        "--exclude", nargs="*", default=[], help="models that already failed this unit"
    )
    add_pricing_args(pick)
    pick.set_defaults(func=cmd_pick)

    estimate = sub.add_parser("estimate", help="USD cost of one call")
    estimate.add_argument("--model", required=True)
    add_pricing_args(estimate)
    estimate.set_defaults(func=cmd_estimate)

    log = sub.add_parser("log", help="append a decision to the ledger")
    log.add_argument("--unit", required=True)
    log.add_argument("--tier", choices=TIER_NAMES, required=True)
    log.add_argument("--model", required=True)
    log.add_argument("--reason", required=True)
    log.add_argument("--outcome", default="pending")
    log.add_argument("--file", default=str(DEFAULT_LOG))
    log.set_defaults(func=cmd_log)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
