"""Broker commission schedules (the broker piece of agency cost).

Compose these with exchange and tax fees from ``tcm.fees`` via ``Composite``
to build the full agency term in:

    total = agency + spread + market_impact + residual

A schedule is applied once per order. Fills that share an order id are one
order; a fill with no order id is its own order. Per-share and bps amounts
scale with size, so splitting them across fills does not change the total.
A flat fee or a minimum does change with the grouping, and those apply once
per order unless wrapped in PerFill.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from tcm.types import Charge, Number, OrderView, TransactionCostError, dec_str, to_decimal


class CommissionSchedule(Protocol):
    def charge(self, order: OrderView) -> Charge:
        """Currency commission for the order. Negative means a rebate."""


def _require_non_negative_minimum(minimum: Decimal) -> None:
    if minimum < 0:
        raise TransactionCostError("minimum commission cannot be negative")


def _apply_floor(raw: Decimal, minimum: Decimal) -> tuple[Decimal, bool]:
    if minimum > 0 and raw < minimum:
        return minimum, True
    return raw, False


@dataclass(frozen=True)
class NoCommission:
    """Charge nothing. Use this when a run is spread-only on purpose."""

    def charge(self, order: OrderView) -> Charge:
        del order
        return Charge("none", Decimal(0), "no commission")


@dataclass(frozen=True)
class PerShare:
    """max(rate × order quantity, minimum).

    The minimum is a floor on what the trader pays for the order. Leave it at
    0 for a rebate (a negative rate). A positive minimum replaces a smaller
    raw amount, including a negative one.
    """

    rate: Number
    minimum: Number = Decimal(0)

    def __post_init__(self) -> None:
        rate = to_decimal(self.rate, name="per-share rate")
        minimum = to_decimal(self.minimum, name="minimum commission")
        _require_non_negative_minimum(minimum)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "minimum", minimum)

    def charge(self, order: OrderView) -> Charge:
        rate: Decimal = self.rate  # type: ignore[assignment]
        minimum: Decimal = self.minimum  # type: ignore[assignment]
        raw = rate * order.quantity
        amount, bound = _apply_floor(raw, minimum)
        detail = f"{dec_str(rate)} per share × {dec_str(order.quantity)} = {dec_str(raw)}"
        if bound:
            detail += f"; order minimum {dec_str(minimum)} binds"
        return Charge("per_share", amount, detail)


@dataclass(frozen=True)
class FlatFee:
    """A fixed currency amount once per order. Negative is a per-order rebate."""

    amount: Number

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", to_decimal(self.amount, name="flat commission"))

    def charge(self, order: OrderView) -> Charge:
        del order
        amount: Decimal = self.amount  # type: ignore[assignment]
        return Charge("flat", amount, f"flat {dec_str(amount)} per order")


@dataclass(frozen=True)
class BpsOfNotional:
    """max(execution notional × bps / 10,000, minimum)."""

    bps: Number
    minimum: Number = Decimal(0)

    def __post_init__(self) -> None:
        bps = to_decimal(self.bps, name="commission bps")
        minimum = to_decimal(self.minimum, name="minimum commission")
        _require_non_negative_minimum(minimum)
        object.__setattr__(self, "bps", bps)
        object.__setattr__(self, "minimum", minimum)

    def charge(self, order: OrderView) -> Charge:
        bps: Decimal = self.bps  # type: ignore[assignment]
        minimum: Decimal = self.minimum  # type: ignore[assignment]
        raw = order.notional * bps / Decimal(10000)
        amount, bound = _apply_floor(raw, minimum)
        detail = (
            f"{dec_str(bps)} bps × notional {dec_str(order.notional)} / 10000"
            f" = {dec_str(raw)}"
        )
        if bound:
            detail += f"; order minimum {dec_str(minimum)} binds"
        return Charge("bps", amount, detail)


@dataclass(frozen=True)
class PercentOfNotional:
    """max(execution notional × percent / 100, minimum). One percent is 100 bps."""

    percent: Number
    minimum: Number = Decimal(0)

    def __post_init__(self) -> None:
        percent = to_decimal(self.percent, name="commission percent")
        minimum = to_decimal(self.minimum, name="minimum commission")
        _require_non_negative_minimum(minimum)
        object.__setattr__(self, "percent", percent)
        object.__setattr__(self, "minimum", minimum)

    def charge(self, order: OrderView) -> Charge:
        percent: Decimal = self.percent  # type: ignore[assignment]
        minimum: Decimal = self.minimum  # type: ignore[assignment]
        raw = order.notional * percent / Decimal(100)
        amount, bound = _apply_floor(raw, minimum)
        detail = f"{dec_str(percent)}% × notional {dec_str(order.notional)} = {dec_str(raw)}"
        if bound:
            detail += f"; order minimum {dec_str(minimum)} binds"
        return Charge("percent", amount, detail)


@dataclass(frozen=True, init=False)
class Composite:
    """Sum of schedules. Each schedule is applied to the same order."""

    schedules: tuple[CommissionSchedule, ...]

    def __init__(self, *schedules: CommissionSchedule) -> None:
        if len(schedules) == 0:
            raise TransactionCostError("composite commission needs at least one schedule")
        object.__setattr__(self, "schedules", schedules)

    def charge(self, order: OrderView) -> Charge:
        parts = tuple(schedule.charge(order) for schedule in self.schedules)
        amount = sum((part.amount for part in parts), Decimal(0))
        detail = " + ".join(f"{part.name} {dec_str(part.amount)}" for part in parts)
        return Charge("composite", amount, detail, parts)


@dataclass(frozen=True)
class PerFill:
    """Apply a schedule to each fill on its own, then sum.

    Use this for a ticket fee charged per fill. A PerShare minimum wrapped
    here binds on each fill instead of once per order.
    """

    schedule: CommissionSchedule

    def charge(self, order: OrderView) -> Charge:
        parts = tuple(
            self.schedule.charge(OrderView(fills=(fill,), side=order.side))
            for fill in order.fills
        )
        amount = sum((part.amount for part in parts), Decimal(0))
        detail = " + ".join(part.detail for part in parts)
        return Charge("per_fill", amount, detail, parts)


@dataclass(frozen=True)
class AtLeast:
    """Floor on another schedule's order charge. The floor is not applied per fill."""

    schedule: CommissionSchedule
    minimum: Number

    def __post_init__(self) -> None:
        minimum = to_decimal(self.minimum, name="minimum commission")
        _require_non_negative_minimum(minimum)
        object.__setattr__(self, "minimum", minimum)

    def charge(self, order: OrderView) -> Charge:
        inner = self.schedule.charge(order)
        minimum: Decimal = self.minimum  # type: ignore[assignment]
        if inner.amount >= minimum:
            return inner
        detail = f"{inner.detail}; floor {dec_str(minimum)} binds"
        return Charge("at_least", minimum, detail, (inner,))
