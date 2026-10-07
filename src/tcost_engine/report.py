"""Text and JSON reports. Amounts are exact decimals, not rounded currency."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

from tcost_engine.model import BlotterCost, OrderCost
from tcost_engine.types import Charge, cost_bps, dec_str


def format_report(blotter: BlotterCost) -> str:
    lines = [
        "tcost-engine report",
        "Decomposition: commish + spread + market impact + residual",
        "Market impact: not modeled (0)",
        "Residual: VWAP vs close benchmark slippage",
        "",
    ]
    if len(blotter.orders) == 0:
        lines.append("No fills.")
        lines.append("")
    for order in blotter.orders:
        lines.extend(_format_order(order))
        lines.append("")
    lines.extend(_format_totals("Total", blotter))
    return "\n".join(lines) + "\n"


def blotter_to_dict(blotter: BlotterCost) -> dict[str, Any]:
    return {
        "decomposition": "commish + spread + market_impact + residual",
        "market_impact": "not_modeled",
        "residual": "vwap_vs_close",
        "execution_notional": dec_str(blotter.execution_notional),
        "commish": dec_str(blotter.commish),
        "commission": dec_str(blotter.commission),
        "spread": dec_str(blotter.spread),
        "market_impact_cost": dec_str(blotter.market_impact),
        "residual_cost": dec_str(blotter.residual),
        "slippage": dec_str(blotter.slippage),
        "total": dec_str(blotter.total),
        "total_bps": dec_str(blotter.total_bps),
        "orders": [_order_to_dict(order) for order in blotter.orders],
    }


def _format_order(order: OrderCost) -> list[str]:
    label = order.symbol or "(no symbol)"
    order_label = order.order_id if order.order_id is not None else "(single fill)"
    count = len(order.fills)
    noun = "fill" if count == 1 else "fills"
    lines = [
        f"Order {order_label}  {label} {order.side.value}  {count} {noun}",
        f"  execution notional  {dec_str(order.execution_notional)}",
        f"  commish             {dec_str(order.commish_amount)}",
    ]
    lines.extend(_charge_details(order.commish, indent=4))
    lines.append(f"  spread              {dec_str(order.spread)}")
    for index, line in enumerate(order.fills, start=1):
        fill = line.fill
        prefix = f"    fill {index}  " if len(order.fills) > 1 else "    "
        close_txt = (
            f"  close {dec_str(fill.close)}" if fill.close is not None else ""
        )
        lines.append(
            f"{prefix}qty {dec_str(fill.quantity)} @ VWAP {dec_str(fill.price)}"
            f"{close_txt}"
            f"  {fill.liquidity.value}  {dec_str(line.spread)}"
        )
        lines.append(f"      {line.spread_detail}")
    lines.append(f"  market impact       {dec_str(order.market_impact)}")
    lines.append(f"      {order.fills[0].market_impact_detail}")
    lines.append(f"  residual (slippage) {dec_str(order.residual)}")
    for index, line in enumerate(order.fills, start=1):
        prefix = f"    fill {index}  " if len(order.fills) > 1 else "    "
        lines.append(f"{prefix}{line.residual_detail}")
    lines.append(
        f"  total               {dec_str(order.total)}"
        f"  ({_bps(order.total, order.execution_notional)} bps of execution notional)"
    )
    return lines


def _charge_details(charge: Charge, indent: int) -> list[str]:
    pad = " " * indent
    lines = [f"{pad}{charge.detail}"]
    for part in charge.parts:
        lines.extend(_charge_details(part, indent + 2))
    return lines


def _format_totals(title: str, blotter: BlotterCost) -> list[str]:
    notional = blotter.execution_notional
    return [
        title,
        f"  orders              {len(blotter.orders)}",
        f"  execution notional  {dec_str(notional)}",
        f"  commish             {dec_str(blotter.commish)}  ({_bps(blotter.commish, notional)} bps)",
        f"  spread              {dec_str(blotter.spread)}  ({_bps(blotter.spread, notional)} bps)",
        f"  market impact       {dec_str(blotter.market_impact)}  (not modeled)",
        f"  residual (slippage) {dec_str(blotter.residual)}"
        f"  ({_bps(blotter.residual, notional)} bps)",
        f"  total               {dec_str(blotter.total)}  ({_bps(blotter.total, notional)} bps)",
    ]


def _bps(cost: Decimal, notional: Decimal) -> str:
    """Basis points for display, rounded to 0.0001. The model values stay exact."""
    displayed = cost_bps(cost, notional).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
    return dec_str(displayed)


def _order_to_dict(order: OrderCost) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side.value,
        "quantity": dec_str(order.quantity),
        "execution_notional": dec_str(order.execution_notional),
        "commish": dec_str(order.commish_amount),
        "commission": dec_str(order.commission_amount),
        "commish_detail": order.commish.detail,
        "commission_detail": order.commish.detail,
        "spread": dec_str(order.spread),
        "market_impact": dec_str(order.market_impact),
        "market_impact_detail": "not modeled",
        "residual": dec_str(order.residual),
        "slippage": dec_str(order.slippage),
        "residual_detail": order.fills[0].residual_detail if len(order.fills) == 1 else "see fills",
        "total": dec_str(order.total),
        "total_bps": dec_str(order.total_bps),
        "fills": [
            {
                "quantity": dec_str(line.fill.quantity),
                "price": dec_str(line.fill.price),
                "vwap": dec_str(line.fill.price),
                "close": dec_str(line.fill.close) if line.fill.close is not None else None,
                "liquidity": line.fill.liquidity.value,
                "execution_notional": dec_str(line.execution_notional),
                "commish_allocated": dec_str(order.commish_allocated[index]),
                "commission_allocated": dec_str(order.commish_allocated[index]),
                "spread": dec_str(line.spread),
                "spread_detail": line.spread_detail,
                "market_impact": dec_str(line.market_impact),
                "residual": dec_str(line.residual),
                "slippage": dec_str(line.slippage),
                "residual_detail": line.residual_detail,
                "total": dec_str(order.fill_total(index)),
            }
            for index, line in enumerate(order.fills)
        ],
    }
