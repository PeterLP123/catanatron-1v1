#!/usr/bin/env python3
"""Publish a validated promotion/final report as compact tracked evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from catanatron.gym.result_artifacts import publish_compact_result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Tracked destination, normally docs/results/<experiment>.json.",
    )
    args = parser.parse_args(argv)
    output = publish_compact_result(args.report, args.output)
    print(f"Published validated evidence: {output}")


if __name__ == "__main__":
    main()
