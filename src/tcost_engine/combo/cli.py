"""CLI for costing a combo SOD dollar panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcost_engine.commission import DEFAULT_COMMISH_MILS
from tcost_engine.combo.cost import cost_combo_sod, result_summary
from tcost_engine.lseg.ds2 import parse_iso_date
from tcost_engine.lseg.pull import write_frame
from tcost_engine.types import TransactionCostError, dec_str


DEFAULT_SOD = (
    "/data/robert/stage_c_pinnet_mktbeta_pos_20261005_gto3e-4_to28/"
    "stage_c_pinnet_pos_2021_2026_gto3e-4_to28.parquet"
)
DEFAULT_SOD_S3 = (
    "s3://kelai-team-robert/stage_c_pinnet_mktbeta_pos_20261005_gto3e-4_to28/"
    "stage_c_pinnet_pos_2021_2026_gto3e-4_to28.parquet"
)


def add_cost_combo_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "cost-combo",
        help="Cost DoD trades from a combo SOD dollar parquet against LSEG prices",
    )
    parser.add_argument(
        "--sod",
        default=DEFAULT_SOD,
        help=f"Local or s3:// SOD parquet (default: {DEFAULT_SOD})",
    )
    parser.add_argument(
        "--h5",
        default=None,
        help="Datastream2 H5 path or s3:// URL (default: box path / prod S3)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="First SOD date inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Last SOD date inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--mils",
        default=str(DEFAULT_COMMISH_MILS),
        help=f"Commission mils per share (default {DEFAULT_COMMISH_MILS})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write daily t-cost summary to .parquet or .csv",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for S3 downloads",
    )
    parser.add_argument(
        "--fill",
        choices=("vwap", "moc"),
        default="vwap",
        help="Execution assumption: vwap (mils+VWAP−close) or moc (mils only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON totals on stdout",
    )


def run_cost_combo(args: argparse.Namespace) -> int:
    start = parse_iso_date(args.start) if args.start else None
    end = parse_iso_date(args.end) if args.end else None
    try:
        result = cost_combo_sod(
            args.sod,
            h5_path=args.h5,
            start=start,
            end=end,
            mils=args.mils,
            cache_dir=args.cache_dir,
            fill=str(args.fill),
        )
    except (TransactionCostError, FileNotFoundError, ImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        out = write_frame(result.daily, Path(args.output))
        print(f"wrote {out}", file=sys.stderr)

    summary = result_summary(result)
    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(
            "combo SOD t-costs\n"
            f"  trades              {summary['n_trades']:,}\n"
            f"  fills               {summary['n_fills']:,}\n"
            f"  dropped             {summary['n_dropped']:,}\n"
            f"  days                {summary['n_days']:,}\n"
            f"  trade notional      {dec_str(result.trade_notional)}\n"
            f"  commish             {dec_str(result.total_commish)}\n"
            f"  spread              {dec_str(result.total_spread)}\n"
            f"  intraday slippage   {dec_str(result.total_intraday_slippage)}\n"
            f"  total               {dec_str(result.total_cost)}",
            file=sys.stdout,
        )
    return 0
