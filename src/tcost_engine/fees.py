"""Exchange, regulatory, and transfer fees.

These are the non-broker pieces of commish cost in the Northfield / diBartolomeo
decomposition (commish = broker commission + exchange / custody / tax fees).
They are explicit and known in advance. Rates are configurable; defaults are
illustrative placeholders and must be set to the live published rate before
production use.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from tcost_engine.types import Charge, Number, OrderView, Side, TransactionCostError, dec_str, to_decimal


@dataclass(frozen=True)
class OnSide:
    """Apply a schedule only when the order is on the listed side(s)."""

    schedule: object
    sides: tuple[Side, ...]
    name: str = "on_side"

    def __post_init__(self) -> None:
        if len(self.sides) == 0:
            raise TransactionCostError("OnSide needs at least one side")
        if len(set(self.sides)) != len(self.sides):
            raise TransactionCostError("OnSide sides must be unique")
        for side in self.sides:
            if not isinstance(side, Side):
                raise TransactionCostError("OnSide sides must be Side values")

    def charge(self, order: OrderView) -> Charge:
        if order.side not in self.sides:
            side_names = ", ".join(side.value for side in self.sides)
            return Charge(
                self.name,
                Decimal(0),
                f"{self.name}: not charged on {order.side.value} (only {side_names})",
            )
        inner = self.schedule.charge(order)  # type: ignore[attr-defined]
        return Charge(self.name, inner.amount, inner.detail, (inner,))


def OnBuy(schedule: object, *, name: str = "on_buy") -> OnSide:
    return OnSide(schedule, (Side.BUY,), name=name)


def OnSell(schedule: object, *, name: str = "on_sell") -> OnSide:
    return OnSide(schedule, (Side.SELL,), name=name)


@dataclass(frozen=True)
class SecFee:
    """US SEC Section 31 transaction fee on covered equity sells.

    Charged as rate × sell notional. The published rate changes; pass the
    current rate as a decimal fraction of notional (for example "0.0000278"
    for $27.80 per million). Buy orders are zero.
    """

    rate: Number
    name: str = "sec_fee"

    def __post_init__(self) -> None:
        rate = to_decimal(self.rate, name="SEC fee rate")
        if rate < 0:
            raise TransactionCostError("SEC fee rate cannot be negative")
        object.__setattr__(self, "rate", rate)

    def charge(self, order: OrderView) -> Charge:
        if order.side is Side.BUY:
            return Charge(self.name, Decimal(0), f"{self.name}: not charged on buys")
        rate: Decimal = self.rate  # type: ignore[assignment]
        amount = order.notional * rate
        detail = (
            f"SEC Section 31 {dec_str(rate)} × sell notional {dec_str(order.notional)}"
            f" = {dec_str(amount)}"
        )
        return Charge(self.name, amount, detail)


@dataclass(frozen=True)
class FinraTaf:
    """FINRA Trading Activity Fee on sells, per share with an optional cap.

    Default rate and cap are illustrative. Confirm against the live FINRA TAF
    schedule before production use. Buy orders are zero.
    """

    rate_per_share: Number = "0.000166"
    cap: Number | None = "8.30"
    name: str = "finra_taf"

    def __post_init__(self) -> None:
        rate = to_decimal(self.rate_per_share, name="FINRA TAF rate")
        if rate < 0:
            raise TransactionCostError("FINRA TAF rate cannot be negative")
        cap = self.cap
        if cap is not None:
            cap = to_decimal(cap, name="FINRA TAF cap")
            if cap < 0:
                raise TransactionCostError("FINRA TAF cap cannot be negative")
        object.__setattr__(self, "rate_per_share", rate)
        object.__setattr__(self, "cap", cap)

    def charge(self, order: OrderView) -> Charge:
        if order.side is Side.BUY:
            return Charge(self.name, Decimal(0), f"{self.name}: not charged on buys")
        rate: Decimal = self.rate_per_share  # type: ignore[assignment]
        raw = rate * order.quantity
        cap = self.cap
        if cap is not None and raw > cap:
            amount = cap  # type: ignore[assignment]
            detail = (
                f"FINRA TAF {dec_str(rate)} × {dec_str(order.quantity)} = {dec_str(raw)}; "
                f"cap {dec_str(amount)} binds"
            )
        else:
            amount = raw
            detail = f"FINRA TAF {dec_str(rate)} × {dec_str(order.quantity)} = {dec_str(amount)}"
        return Charge(self.name, amount, detail)


@dataclass(frozen=True)
class StampDuty:
    """Transfer tax as a percent of buy notional (default UK SDRT-style 0.5%).

    Charged only on buys. Set percent to the local rate for the market you
    trade. Sells are zero.
    """

    percent: Number = "0.5"
    name: str = "stamp_duty"

    def __post_init__(self) -> None:
        percent = to_decimal(self.percent, name="stamp duty percent")
        if percent < 0:
            raise TransactionCostError("stamp duty percent cannot be negative")
        object.__setattr__(self, "percent", percent)

    def charge(self, order: OrderView) -> Charge:
        if order.side is Side.SELL:
            return Charge(self.name, Decimal(0), f"{self.name}: not charged on sells")
        percent: Decimal = self.percent  # type: ignore[assignment]
        amount = order.notional * percent / Decimal(100)
        detail = (
            f"stamp duty {dec_str(percent)}% × buy notional {dec_str(order.notional)}"
            f" = {dec_str(amount)}"
        )
        return Charge(self.name, amount, detail)
