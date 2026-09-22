"""Command line for a single fill or a blotter CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tcm.commission import (
    BpsOfNotional,
    CommissionSchedule,
    Composite,
    FlatFee,
    NoCommission,
    PerShare,
)
from tcm.model import TransactionCostModel
from tcm.report import blotter_to_dict, format_report
from tcm.types import (
    BidAsk,
    Fill,
    FullSpreadBps,
    Liquidity,
    OneWaySpreadBps,
    Side,
    Spread,
    TransactionCostError,
    to_decimal,
)

_COLUMNS = {
    "symbol",
    "side",
    "quantity",
    "price",
    "bid",
    "ask",
    "full_spread_bps",
    "one_way_spread_bps",
    "liquidity",
    "order_id",
}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        commission = _commission(args)
        model = TransactionCostModel(commission=commission, maker_capture=args.maker_capture)
        if args.command == "cost":
            blotter = model.cost_many([_fill_from_args(args)])
        else:
            blotter = model.cost_many(_fills_from_csv(Path(args.csv)))
    except (TransactionCostError, OSError, UnicodeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(blotter_to_dict(blotter), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_report(blotter))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcm",
        description=(
            "Cost trades as commission plus spread. "
            "Market impact is not modeled and is reported as zero."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cost = sub.add_parser("cost", help="Cost one fill")
    _add_fill_args(cost)
    _add_schedule_args(cost)

    blotter = sub.add_parser("blotter", help="Cost fills from a CSV file")
    blotter.add_argument("csv", help="CSV with side, quantity, price, and a spread")
    _add_schedule_args(blotter)
    return parser


def _add_fill_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="")
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    parser.add_argument("--qty", required=True, help="Unsigned quantity")
    parser.add_argument("--price", required=True, help="Execution price")
    parser.add_argument("--bid")
    parser.add_argument("--ask")
    parser.add_argument("--full-spread-bps", help="Quoted width (ask-bid) in bps. Taker pays half.")
    parser.add_argument(
        "--one-way-spread-bps",
        help="One-way spread cost in bps. Not halved.",
    )
    parser.add_argument(
        "--liquidity",
        default="taker",
        choices=["taker", "maker", "midpoint"],
    )
    parser.add_argument("--order-id")


def _add_schedule_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--per-share", help="Commission currency amount per share")
    parser.add_argument(
        "--min-commission",
        default="0",
        help="Order minimum applied to the per-share schedule. Requires --per-share.",
    )
    parser.add_argument("--flat", help="Flat commission once per order")
    parser.add_argument("--commission-bps", help="Commission in bps of execution notional")
    parser.add_argument(
        "--maker-capture",
        default="0",
        help="Fraction of the half-spread a maker earns, from 0 to 1. Default 0.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")


def _commission(args: argparse.Namespace) -> CommissionSchedule:
    minimum = to_decimal(args.min_commission, name="min commission")
    parts: list[CommissionSchedule] = []
    if args.per_share is not None:
        parts.append(PerShare(args.per_share, minimum=args.min_commission))
    elif minimum != 0:
        raise TransactionCostError("--min-commission requires --per-share")
    if args.flat is not None:
        parts.append(FlatFee(args.flat))
    if args.commission_bps is not None:
        parts.append(BpsOfNotional(args.commission_bps))
    if len(parts) == 0:
        return NoCommission()
    if len(parts) == 1:
        return parts[0]
    return Composite(*parts)


def _fill_from_args(args: argparse.Namespace) -> Fill:
    return Fill(
        symbol=args.symbol,
        side=Side.parse(args.side),
        quantity=args.qty,
        price=args.price,
        spread=_spread_from_values(args.bid, args.ask, args.full_spread_bps, args.one_way_spread_bps),
        liquidity=Liquidity.parse(args.liquidity),
        order_id=args.order_id,
    )


def _fills_from_csv(path: Path) -> list[Fill]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TransactionCostError("CSV has no header")
        fields = [(name or "").strip().lower() for name in reader.fieldnames]
        unknown = [name for name in fields if name not in _COLUMNS]
        if unknown:
            raise TransactionCostError(
                "unknown CSV column(s): "
                + ", ".join(unknown)
                + ". Expected: "
                + ", ".join(sorted(_COLUMNS))
            )
        missing = {"side", "quantity", "price"} - set(fields)
        if missing:
            raise TransactionCostError("CSV is missing column(s): " + ", ".join(sorted(missing)))
        fills: list[Fill] = []
        for row_number, raw in enumerate(reader, start=2):
            try:
                fills.append(_fill_from_row(_normalize_row(raw)))
            except TransactionCostError as exc:
                raise TransactionCostError(f"row {row_number}: {exc}") from exc
        return fills


def _normalize_row(raw: dict[str | None, str | None]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key, value in raw.items():
        if key is None:
            raise TransactionCostError("row has extra values with no header")
        text = value.strip() if isinstance(value, str) else ""
        row[key.strip().lower()] = text
    return row


def _fill_from_row(row: dict[str, str]) -> Fill:
    liquidity = row.get("liquidity") or "taker"
    order_id = row.get("order_id") or None
    return Fill(
        symbol=row.get("symbol", ""),
        side=row.get("side", ""),
        quantity=row.get("quantity", ""),
        price=row.get("price", ""),
        spread=_spread_from_values(
            row.get("bid") or None,
            row.get("ask") or None,
            row.get("full_spread_bps") or None,
            row.get("one_way_spread_bps") or None,
        ),
        liquidity=liquidity,
        order_id=order_id,
    )


def _spread_from_values(
    bid: str | None,
    ask: str | None,
    full_spread_bps: str | None,
    one_way_spread_bps: str | None,
) -> Spread:
    has_quote = bid is not None or ask is not None
    kinds = [
        has_quote,
        full_spread_bps is not None,
        one_way_spread_bps is not None,
    ]
    if sum(bool(kind) for kind in kinds) != 1:
        raise TransactionCostError(
            "provide exactly one spread: bid and ask, full-spread-bps, or one-way-spread-bps"
        )
    if has_quote:
        if bid is None or ask is None:
            raise TransactionCostError("bid and ask must be provided together")
        return BidAsk(bid, ask)
    if full_spread_bps is not None:
        return FullSpreadBps(full_spread_bps)
    return OneWaySpreadBps(one_way_spread_bps or "0")
