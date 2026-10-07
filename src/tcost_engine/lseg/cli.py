"""CLI for LSEG Datastream2 pulls (OHLCV + TOP500)."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from tcost_engine.lseg.ds2 import parse_iso_date
from tcost_engine.lseg.paths import DEFAULT_DS2_H5_LOCAL, DEFAULT_DS2_H5_S3
from tcost_engine.lseg.pull import pull_ohlcv, pull_top500, summarize_pull, write_frame


def add_lseg_parser(sub: argparse._SubParsersAction) -> None:
    lseg = sub.add_parser(
        "lseg",
        help="Pull LSEG Datastream2 OHLCV / TOP500 (same H5 as the AWS box)",
    )
    lseg_sub = lseg.add_subparsers(dest="lseg_command", required=True)

    ohlcv = lseg_sub.add_parser(
        "ohlcv",
        help="Pull daily OHLCV (unadjusted and/or adjusted)",
    )
    _add_common(ohlcv)
    ohlcv.add_argument(
        "--adjustment",
        choices=("unadjusted", "adjusted", "both"),
        default="both",
        help="Which OHLCV scale to export (default: both)",
    )
    ohlcv.add_argument(
        "--universe",
        choices=("top500", "all"),
        default="top500",
        help="Restrict to TOP500 members that day (default) or all names",
    )
    ohlcv.set_defaults(_lseg_handler=_run_ohlcv)

    top500 = lseg_sub.add_parser(
        "top500",
        help="Pull point-in-time TOP500 universe constituents",
    )
    _add_common(top500)
    top500.set_defaults(_lseg_handler=_run_top500)


def run_lseg(args: argparse.Namespace) -> int:
    return int(args._lseg_handler(args))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--h5",
        default=None,
        help=(
            f"Path or s3:// URL to ds2_data.h5. Default: {DEFAULT_DS2_H5_LOCAL} "
            f"if present, else {DEFAULT_DS2_H5_S3}"
        ),
    )
    parser.add_argument(
        "--start",
        default="2016-01-01",
        help="First market date inclusive (default: 2016-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Last market date inclusive (default: latest in H5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output .parquet or .csv path",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for S3 downloads",
    )
    parser.add_argument(
        "--no-ticker",
        action="store_true",
        help="Omit ticker column (INFOCODE only)",
    )


def _run_ohlcv(args: argparse.Namespace) -> int:
    start, end = _dates(args)
    df = pull_ohlcv(
        args.h5,
        start=start,
        end=end,
        adjustment=args.adjustment,
        universe=args.universe,
        cache_dir=args.cache_dir,
        include_ticker=not args.no_ticker,
    )
    out = write_frame(df, Path(args.output))
    print(summarize_pull(df, label="ohlcv"))
    print(f"wrote {out}")
    return 0


def _run_top500(args: argparse.Namespace) -> int:
    start, end = _dates(args)
    df = pull_top500(
        args.h5,
        start=start,
        end=end,
        cache_dir=args.cache_dir,
        include_ticker=not args.no_ticker,
    )
    out = write_frame(df, Path(args.output))
    print(summarize_pull(df, label="top500"))
    print(f"wrote {out}")
    return 0


def _dates(args: argparse.Namespace) -> tuple[date, date | None]:
    start = parse_iso_date(args.start)
    end = parse_iso_date(args.end) if args.end else None
    return start, end
