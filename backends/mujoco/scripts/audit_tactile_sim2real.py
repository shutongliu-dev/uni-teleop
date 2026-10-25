"""Audit real-robot tactile packets without copying raw data into the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonic_mujoco.tactile_audit import audit_tactile_sources, write_audit_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="LeRobot dataset directories or individual parquet files",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="public-safe source label; repeat once per source",
    )
    parser.add_argument("--fps", type=float, help="override dataset record rate")
    parser.add_argument("--deadband-counts", type=int, default=2)
    parser.add_argument("--quiet-quantile", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_tactile_sources(
        args.sources,
        labels=args.labels,
        fps=args.fps,
        deadband_counts=args.deadband_counts,
        quiet_quantile=args.quiet_quantile,
    )
    if args.output is None:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    write_audit_report(args.output, report)
    print(f"Tactile audit saved to {args.output}")


if __name__ == "__main__":
    main()
