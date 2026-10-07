"""Spread cost from a quote or an explicit spread assumption.

This is the cost of where the fill sits relative to the mid, taken from the
quoted width. With LSEG Datastream2, bid/ask are closing prints used as a
proxy for that day's intraday spread. It is not the gap between the execution
price and the touch, and not VWAP vs close (that is residual / slippage).
"""

from __future__ import annotations

from decimal import Decimal

from tcost_engine.types import BidAsk, Fill, FullSpreadBps, Liquidity, OneWaySpreadBps, dec_str


def half_spread_per_unit(fill: Fill) -> tuple[Decimal, str]:
    """One-way quote cost per unit of quantity, before liquidity adjustment."""
    spread = fill.spread
    if isinstance(spread, BidAsk):
        half: Decimal = spread.half_spread
        detail = (
            f"({dec_str(spread.ask)} - {dec_str(spread.bid)}) / 2 = {dec_str(half)}"
        )
        return half, detail
    if isinstance(spread, FullSpreadBps):
        reference = _reference_price(spread.reference_price, fill)
        half = reference * spread.bps / Decimal(10000) / Decimal(2)
        detail = (
            f"half of {dec_str(spread.bps)} bps quoted spread"
            f" on {dec_str(reference)} = {dec_str(half)}"
        )
        return half, detail
    if isinstance(spread, OneWaySpreadBps):
        reference = _reference_price(spread.reference_price, fill)
        half = reference * spread.bps / Decimal(10000)
        detail = (
            f"{dec_str(spread.bps)} bps one-way spread"
            f" on {dec_str(reference)} = {dec_str(half)}"
        )
        return half, detail
    raise TypeError(f"unsupported spread: {type(spread).__name__}")


def spread_cost(fill: Fill, *, maker_capture: Decimal) -> tuple[Decimal, str]:
    """Currency spread cost for one fill. Positive is a cost to the trader."""
    half, half_detail = half_spread_per_unit(fill)
    quantity: Decimal = fill.quantity  # type: ignore[assignment]
    gross = half * quantity
    if fill.liquidity is Liquidity.TAKER:
        amount = gross
        detail = f"taker half-spread {half_detail} × {dec_str(quantity)} = {dec_str(amount)}"
        return amount, detail
    if fill.liquidity is Liquidity.MIDPOINT:
        return Decimal(0), f"midpoint pays no spread; half-spread {half_detail} ignored"
    if fill.liquidity is Liquidity.MAKER:
        amount = -maker_capture * gross
        detail = (
            f"maker earns {dec_str(maker_capture)} of half-spread {half_detail}; "
            f"{dec_str(-maker_capture)} × {dec_str(half)} × {dec_str(quantity)} = {dec_str(amount)}"
        )
        return amount, detail
    raise TypeError(f"unsupported liquidity: {fill.liquidity!r}")


def _reference_price(reference: Decimal | None, fill: Fill) -> Decimal:
    if reference is None:
        return fill.price  # type: ignore[return-value]
    return reference
