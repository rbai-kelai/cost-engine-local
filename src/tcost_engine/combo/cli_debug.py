"""CLI for debug_perturb (mosek-style pre vs post t-cost tables)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcost_engine.commission import DEFAULT_COMMISH_MILS
from tcost_engine.combo.cli import DEFAULT_SOD
from tcost_engine.combo.debug_perturb import (
    DEFAULT_OUTDIR,
    format_debug_perturb,
    result_summary,
    run_debug_perturb,
    write_perturb_artifacts,
)
from tcost_engine.lseg.ds2 import parse_iso_date
from tcost_engine.lseg.pull import write_frame
from tcost_engine.types import TransactionCostError


def add_debug_perturb_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "debug-perturb",
        help=(
            "Debugger perturb: Stage-C-style pre-tcost vs post-tcost "
            "(MOC fill + VWAP fill)"
        ),
    )
    parser.add_argument(
        "--sod",
        default=DEFAULT_SOD,
        help=f"Local or s3:// SOD parquet (default: {DEFAULT_SOD})",
    )
    parser.add_argument("--h5", default=None, help="Datastream2 H5 path or s3:// URL")
    parser.add_argument("--start", default=None, help="First SOD date inclusive")
    parser.add_argument("--end", default=None, help="Last SOD date inclusive")
    parser.add_argument(
        "--mils",
        default=str(DEFAULT_COMMISH_MILS),
        help=f"Commission mils per share (default {DEFAULT_COMMISH_MILS})",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Write yearly tables + daily series here (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional extra daily series path (.parquet or .csv)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache root (default ~/.cache/tcost-engine): SOD, prices, combo-cost",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass price / combo-cost caches and rebuild",
    )
    parser.add_argument(
        "--fill",
        choices=("vwap", "moc"),
        default=None,
        help=argparse.SUPPRESS,  # deprecated: both scenarios always reported
    )
    parser.add_argument(
        "--no-spread",
        action="store_true",
        help="Omit LSEG EOD half-spread from VWAP fill (MOC never charges spread)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    parser.add_argument(
        "--no-outdir",
        action="store_true",
        help="Skip writing the perturb outdir artifacts",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Do not SSH-hop (used on kelai-team-robert; Mac entrypoint strips this)",
    )


def cli_debug_perturb(args: argparse.Namespace) -> int:
    start = parse_iso_date(args.start) if args.start else None
    end = parse_iso_date(args.end) if args.end else None
    try:
        result = run_debug_perturb(
            args.sod,
            h5_path=args.h5,
            start=start,
            end=end,
            mils=args.mils,
            cache_dir=args.cache_dir,
            refresh=bool(args.refresh),
            include_spread=not bool(getattr(args, "no_spread", False)),
            fill=getattr(args, "fill", None),
        )
    except Exception as exc:
        try:
            from botocore.exceptions import NoCredentialsError as _NoCreds
        except ImportError:
            _NoCreds = ()  # type: ignore[assignment,misc]
        if _NoCreds and isinstance(exc, _NoCreds):
            print(
                "error: AWS credentials not configured — needed for "
                "s3://kelaidata/.../ds2_data.h5 (LSEG prices). "
                "Configure AWS credentials, or place the H5 at "
                "/data/robert/lseg/Datastream2/ds2_data.h5.",
                file=sys.stderr,
            )
            return 2
        if isinstance(
            exc,
            (TransactionCostError, FileNotFoundError, ImportError, OSError, ValueError, KeyError),
        ):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise

    if not args.no_outdir:
        out = write_perturb_artifacts(result, Path(args.outdir))
        print(f"wrote {out}/", file=sys.stderr)
        print("  README.md", file=sys.stderr)
        print("  pre_tcost_yearly.csv", file=sys.stderr)
        print("  moc_post_tcost_yearly.csv", file=sys.stderr)
        print("  vwap_post_tcost_yearly.csv", file=sys.stderr)
        print("  daily_pre_post.parquet (or .csv)", file=sys.stderr)

    if args.output:
        out = write_frame(result.daily, Path(args.output))
        print(f"wrote {out}", file=sys.stderr)

    if args.json:
        json.dump(result_summary(result), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_debug_perturb(result))
    return 0
