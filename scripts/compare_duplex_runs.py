#!/usr/bin/env python3
"""Compare two duplex run bundles while enforcing one changed variable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "moshi"))

from moshi.qualification import (
    QualificationError,
    compare_runs,
    load_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--experimental-variable",
        required=True,
        help="One dotted canonical identity field allowed to differ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON here instead of stdout",
    )
    return parser


def _write_report(report: dict, output: Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    try:
        baseline_run, baseline_metrics = load_bundle(args.baseline)
        candidate_run, candidate_metrics = load_bundle(args.candidate)
        report = compare_runs(
            baseline_run,
            baseline_metrics,
            candidate_run,
            candidate_metrics,
            experimental_variable=args.experimental_variable,
        )
    except QualificationError:
        _write_report(
            {
                "schema_version": 1,
                "verdict": "Inconclusive",
                "reason_code": 10,
                "error_code": 1,
            },
            args.output,
        )
        print("error: run bundles are not comparable", file=sys.stderr)
        return 2
    _write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
