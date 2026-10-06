"""Market impact and residual (VWAP vs close slippage) trading costs.

Literature decomposition (Northfield / diBartolomeo; Deutsche Bank; Bocconi):

    total = commish + spread + market_impact + residual

Commish and spread are computed elsewhere. Residual here is the VWAP-against-
close benchmark slippage. Market impact (size-dependent price move from *this*
trade) stays deferred at zero.
"""

from __future__ import annotations

from decimal import Decimal

from tcost_engine.types import Fill, Side, dec_str


def market_impact() -> tuple[Decimal, str]:
    """Currency impact cost. Always zero until an impact model is added."""
    return Decimal(0), "not modeled (size-dependent price move deferred)"


def residual_cost(fill: Fill) -> tuple[Decimal, str]:
    """VWAP vs close benchmark slippage for one fill.

    ``fill.price`` is the VWAP (execution). ``fill.close`` is the close
    benchmark. Positive is a cost to the trader:

        slippage = side × (VWAP − close) × quantity

    with buy = +1 and sell = −1. That equals −intraday mark-to-close PnL from
    a VWAP fill. Omit ``close`` to leave this term at zero.
    """
    if fill.close is None:
        return (
            Decimal(0),
            "no close benchmark; VWAP vs close slippage omitted",
        )
    vwap: Decimal = fill.price  # type: ignore[assignment]
    close: Decimal = fill.close  # type: ignore[assignment]
    quantity: Decimal = fill.quantity  # type: ignore[assignment]
    sign = Decimal(1) if fill.side is Side.BUY else Decimal(-1)
    amount = sign * (vwap - close) * quantity
    detail = (
        f"{fill.side.value} side × (VWAP {dec_str(vwap)} − close {dec_str(close)})"
        f" × {dec_str(quantity)} = {dec_str(amount)}"
    )
    return amount, detail


def vwap_close_slippage(fill: Fill) -> tuple[Decimal, str]:
    """Alias for :func:`residual_cost` — VWAP vs close benchmark slippage."""
    return residual_cost(fill)
