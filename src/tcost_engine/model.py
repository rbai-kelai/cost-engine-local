"""Combine agency (commission + fees) and spread for fills and blotters.

Literature-aligned decomposition (Northfield / diBartolomeo):

    total = agency + spread + market_impact + residual

This release costs agency and spread. Market impact and residual are reported
as zero. The half-spread is the transparent bid-ask cost of taking liquidity;
it is not implementation shortfall and does not include size-dependent impact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from tcost_engine.commission import CommissionSchedule
from tcost_engine.impact import market_impact, residual_cost
from tcost_engine.spread import spread_cost
from tcost_engine.types import Charge, Fill, FillEconomics, OrderView, TransactionCostError, cost_bps, to_decimal


@dataclass(frozen=True)
class FillSpread:
    """Spread plus deferred impact and residual terms for a single fill."""

    fill: Fill
    spread: Decimal
    spread_detail: str
    market_impact: Decimal
    market_impact_detail: str
    residual: Decimal
    residual_detail: str

    @property
    def execution_notional(self) -> Decimal:
        return self.fill.execution_notional


@dataclass(frozen=True)
class OrderCost:
    """Cost of one order: one agency charge plus each fill's spread."""

    order_id: str | None
    fills: tuple[FillSpread, ...]
    agency: Charge
    agency_allocated: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if len(self.fills) != len(self.agency_allocated):
            raise TransactionCostError("agency allocation does not match fills")

    @property
    def symbol(self) -> str:
        return self.fills[0].fill.symbol

    @property
    def side(self):
        return self.fills[0].fill.side

    @property
    def quantity(self) -> Decimal:
        return sum((line.fill.quantity for line in self.fills), Decimal(0))

    @property
    def execution_notional(self) -> Decimal:
        return sum((line.execution_notional for line in self.fills), Decimal(0))

    @property
    def agency_amount(self) -> Decimal:
        return self.agency.amount

    @property
    def commission(self) -> Charge:
        """Alias for agency. Kept for callers that still say commission."""
        return self.agency

    @property
    def commission_amount(self) -> Decimal:
        return self.agency_amount

    @property
    def commission_allocated(self) -> tuple[Decimal, ...]:
        return self.agency_allocated

    @property
    def spread(self) -> Decimal:
        return sum((line.spread for line in self.fills), Decimal(0))

    @property
    def market_impact(self) -> Decimal:
        return sum((line.market_impact for line in self.fills), Decimal(0))

    @property
    def residual(self) -> Decimal:
        return sum((line.residual for line in self.fills), Decimal(0))

    @property
    def total(self) -> Decimal:
        return self.agency_amount + self.spread + self.market_impact + self.residual

    @property
    def total_bps(self) -> Decimal:
        return cost_bps(self.total, self.execution_notional)

    def fill_total(self, index: int) -> Decimal:
        line = self.fills[index]
        return (
            self.agency_allocated[index]
            + line.spread
            + line.market_impact
            + line.residual
        )


@dataclass(frozen=True)
class BlotterCost:
    """Costs for a list of fills, grouped into orders."""

    orders: tuple[OrderCost, ...]

    @property
    def execution_notional(self) -> Decimal:
        return sum((order.execution_notional for order in self.orders), Decimal(0))

    @property
    def agency(self) -> Decimal:
        return sum((order.agency_amount for order in self.orders), Decimal(0))

    @property
    def commission(self) -> Decimal:
        """Alias for agency currency total."""
        return self.agency

    @property
    def spread(self) -> Decimal:
        return sum((order.spread for order in self.orders), Decimal(0))

    @property
    def market_impact(self) -> Decimal:
        return sum((order.market_impact for order in self.orders), Decimal(0))

    @property
    def residual(self) -> Decimal:
        return sum((order.residual for order in self.orders), Decimal(0))

    @property
    def total(self) -> Decimal:
        return self.agency + self.spread + self.market_impact + self.residual

    @property
    def total_bps(self) -> Decimal:
        return cost_bps(self.total, self.execution_notional)

    def by_symbol(self) -> dict[str, BlotterCost]:
        grouped: dict[str, list[OrderCost]] = {}
        for order in self.orders:
            grouped.setdefault(order.symbol, []).append(order)
        return {symbol: BlotterCost(tuple(orders)) for symbol, orders in grouped.items()}


@dataclass(frozen=True)
class TransactionCostModel:
    """total = agency + spread + market_impact + residual.

    Agency is broker commission plus optional exchange / tax fees. Spread for
    a taker is the half-spread from the quote (or from an explicit spread in
    bps). Market impact and residual (trend / opportunity) are fixed at zero.

    The distance between the execution price and the touch is not a cost here;
    that is where size-dependent impact would go later.

    ``commission`` is accepted as a synonym for ``agency``.

    maker_capture is the fraction of the half-spread a maker is assumed to
    earn, from 0 (no spread cost and no capture) to 1 (earns the full
    half-spread). It does not affect taker or midpoint fills.
    """

    agency: CommissionSchedule | None = None
    maker_capture: Decimal | str | int = Decimal(0)
    commission: CommissionSchedule | None = None

    def __post_init__(self) -> None:
        if self.agency is None and self.commission is None:
            raise TransactionCostError("provide agency= (or commission=) schedule")
        if self.agency is not None and self.commission is not None and self.agency is not self.commission:
            raise TransactionCostError("pass agency= or commission=, not both")
        schedule = self.agency if self.agency is not None else self.commission
        capture = to_decimal(self.maker_capture, name="maker_capture")
        if capture < 0 or capture > 1:
            raise TransactionCostError("maker_capture must be between 0 and 1")
        object.__setattr__(self, "agency", schedule)
        object.__setattr__(self, "commission", schedule)
        object.__setattr__(self, "maker_capture", capture)

    def cost(self, fill: Fill) -> OrderCost:
        """Cost one fill as its own order.

        An order minimum or flat fee is applied to this fill alone. For
        several fills that share an order id, use cost_many so the order
        charge is applied once.
        """
        return self.cost_many([fill]).orders[0]

    def cost_many(self, fills: Sequence[Fill]) -> BlotterCost:
        """Cost fills, charging order-level agency once per order id.

        Fills with the same order id must share a symbol and a side. Fills
        with no order id are each charged as a separate order.
        """
        orders = tuple(self._cost_order(tuple(group)) for group in _group_orders(fills))
        return BlotterCost(orders)

    def _cost_order(self, fills: tuple[Fill, ...]) -> OrderCost:
        if len(fills) == 0:
            raise TransactionCostError("an order must contain at least one fill")
        economics = tuple(
            FillEconomics(quantity=fill.quantity, notional=fill.execution_notional)  # type: ignore[arg-type]
            for fill in fills
        )
        schedule: CommissionSchedule = self.agency  # type: ignore[assignment]
        agency = schedule.charge(OrderView(fills=economics, side=fills[0].side))
        weights = [fill.quantity for fill in fills]  # type: ignore[misc]
        allocated = _allocate(agency.amount, weights)
        capture: Decimal = self.maker_capture  # type: ignore[assignment]
        lines = []
        for fill in fills:
            spread, spread_detail = spread_cost(fill, maker_capture=capture)
            impact, impact_detail = market_impact()
            residual, residual_detail = residual_cost()
            lines.append(
                FillSpread(
                    fill=fill,
                    spread=spread,
                    spread_detail=spread_detail,
                    market_impact=impact,
                    market_impact_detail=impact_detail,
                    residual=residual,
                    residual_detail=residual_detail,
                )
            )
        return OrderCost(
            order_id=fills[0].order_id,
            fills=tuple(lines),
            agency=agency,
            agency_allocated=tuple(allocated),
        )


def _group_orders(fills: Sequence[Fill]) -> list[list[Fill]]:
    groups: dict[str, list[Fill]] = {}
    order_keys: list[str] = []
    for index, fill in enumerate(fills):
        key = fill.order_id if fill.order_id is not None else f"__fill_{index}"
        if key not in groups:
            groups[key] = []
            order_keys.append(key)
        else:
            previous = groups[key][0]
            if previous.symbol != fill.symbol or previous.side is not fill.side:
                raise TransactionCostError(
                    f"order {fill.order_id} mixes symbol or side "
                    f"({previous.symbol} {previous.side.value} vs {fill.symbol} {fill.side.value})"
                )
        groups[key].append(fill)
    return [groups[key] for key in order_keys]


def _allocate(total: Decimal, weights: list[Decimal]) -> list[Decimal]:
    """Split total across weights. The last slice absorbs the remainder."""
    if len(weights) == 0:
        return []
    weight_sum = sum(weights, Decimal(0))
    if weight_sum == 0:
        raise TransactionCostError("cannot allocate agency across zero quantity")
    allocated: list[Decimal] = []
    running = Decimal(0)
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            allocated.append(total - running)
        else:
            share = total * weight / weight_sum
            allocated.append(share)
            running += share
    return allocated
