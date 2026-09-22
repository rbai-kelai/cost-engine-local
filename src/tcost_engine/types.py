"""Trades, quotes, and shared errors for the transaction cost model."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Union


class TransactionCostError(ValueError):
    """A trade or schedule the model cannot cost."""


Number = Union[Decimal, str, int]


def to_decimal(value: Number, *, name: str) -> Decimal:
    """Parse a money or quantity input. Floats are rejected so binary fractions never enter."""
    if isinstance(value, bool) or isinstance(value, float):
        raise TransactionCostError(
            f"{name} must be Decimal, str, or int (got {type(value).__name__}). "
            "Pass a string so the fraction stays exact."
        )
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except InvalidOperation as exc:
            raise TransactionCostError(f"{name} is not a decimal: {value!r}") from exc
    else:
        raise TransactionCostError(
            f"{name} must be Decimal, str, or int (got {type(value).__name__})"
        )
    if not parsed.is_finite():
        raise TransactionCostError(f"{name} must be finite")
    return parsed


def dec_str(value: Decimal) -> str:
    """Exact decimal text without trailing zeros."""
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def cost_bps(cost: Decimal, notional: Decimal) -> Decimal:
    """Express a currency cost in basis points of notional."""
    if notional == 0:
        if cost == 0:
            return Decimal(0)
        raise TransactionCostError("cannot express a non-zero cost in bps of zero notional")
    if notional < 0:
        raise TransactionCostError("notional must be non-negative")
    return cost / notional * Decimal(10000)


class Side(Enum):
    BUY = "buy"
    SELL = "sell"

    @classmethod
    def parse(cls, value: str) -> Side:
        try:
            return cls(value.strip().lower())
        except (AttributeError, ValueError) as exc:
            raise TransactionCostError(f"side must be buy or sell, got {value!r}") from exc


class Liquidity(Enum):
    """How the fill interacted with the quote.

    Taker crosses and pays the half-spread. Maker rests; any spread earned is
    controlled by the model's maker capture, which defaults to none. Midpoint
    trades at the mid and pays no spread.
    """

    TAKER = "taker"
    MAKER = "maker"
    MIDPOINT = "midpoint"

    @classmethod
    def parse(cls, value: str) -> Liquidity:
        try:
            return cls(value.strip().lower())
        except (AttributeError, ValueError) as exc:
            raise TransactionCostError(
                f"liquidity must be taker, maker, or midpoint, got {value!r}"
            ) from exc


@dataclass(frozen=True)
class BidAsk:
    """Touch quote. The one-way taker cost is half of ask minus bid."""

    bid: Number
    ask: Number

    def __post_init__(self) -> None:
        bid = to_decimal(self.bid, name="bid")
        ask = to_decimal(self.ask, name="ask")
        if bid <= 0 or ask <= 0:
            raise TransactionCostError("bid and ask must be positive")
        if ask < bid:
            raise TransactionCostError(f"crossed quote: bid {dec_str(bid)} is above ask {dec_str(ask)}")
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)  # type: ignore[operator]

    @property
    def half_spread(self) -> Decimal:
        return (self.ask - self.bid) / Decimal(2)  # type: ignore[operator]

    @property
    def full_spread(self) -> Decimal:
        return self.ask - self.bid  # type: ignore[operator]


@dataclass(frozen=True)
class FullSpreadBps:
    """Quoted spread width in basis points: (ask - bid) / reference * 10,000.

    A taker pays half of this. If reference_price is omitted, the fill price is
    the reference, and the taker cost is exactly half these bps of execution notional.
    """

    bps: Number
    reference_price: Number | None = None

    def __post_init__(self) -> None:
        bps = to_decimal(self.bps, name="full spread bps")
        if bps < 0:
            raise TransactionCostError("spread bps cannot be negative")
        reference = self.reference_price
        if reference is not None:
            reference = to_decimal(reference, name="spread reference price")
            if reference <= 0:
                raise TransactionCostError("spread reference price must be positive")
        object.__setattr__(self, "bps", bps)
        object.__setattr__(self, "reference_price", reference)


@dataclass(frozen=True)
class OneWaySpreadBps:
    """One-way spread cost already expressed in basis points. Not halved.

    Use this when the input is the cost of crossing from the mid to the touch,
    rather than the full quoted width. That one-way cost is the effective
    half-spread term in an arrival-price / mid-to-touch decomposition.
    """

    bps: Number
    reference_price: Number | None = None

    def __post_init__(self) -> None:
        bps = to_decimal(self.bps, name="one-way spread bps")
        if bps < 0:
            raise TransactionCostError("spread bps cannot be negative")
        reference = self.reference_price
        if reference is not None:
            reference = to_decimal(reference, name="spread reference price")
            if reference <= 0:
                raise TransactionCostError("spread reference price must be positive")
        object.__setattr__(self, "bps", bps)
        object.__setattr__(self, "reference_price", reference)


Spread = Union[BidAsk, FullSpreadBps, OneWaySpreadBps]


@dataclass(frozen=True)
class Fill:
    """One execution. Quantity is unsigned; side carries the direction."""

    side: Side | str
    quantity: Number
    price: Number
    spread: Spread
    symbol: str = ""
    liquidity: Liquidity | str = Liquidity.TAKER
    order_id: str | None = None

    def __post_init__(self) -> None:
        side = self.side if isinstance(self.side, Side) else Side.parse(str(self.side))
        quantity = to_decimal(self.quantity, name="quantity")
        price = to_decimal(self.price, name="price")
        if quantity <= 0:
            raise TransactionCostError("quantity must be positive")
        if price <= 0:
            raise TransactionCostError("price must be positive")
        if not isinstance(self.spread, (BidAsk, FullSpreadBps, OneWaySpreadBps)):
            raise TransactionCostError(
                "spread must be BidAsk, FullSpreadBps, or OneWaySpreadBps"
            )
        liquidity = (
            self.liquidity
            if isinstance(self.liquidity, Liquidity)
            else Liquidity.parse(str(self.liquidity))
        )
        symbol = self.symbol.strip()
        order_id = self.order_id
        if order_id is not None:
            order_id = order_id.strip()
            if order_id == "":
                raise TransactionCostError(
                    "order_id cannot be blank; omit it to treat the fill as its own order"
                )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "liquidity", liquidity)
        object.__setattr__(self, "order_id", order_id)

    @property
    def execution_notional(self) -> Decimal:
        return self.quantity * self.price  # type: ignore[operator]


@dataclass(frozen=True, init=False)
class Charge:
    """One commish amount and the arithmetic that produced it."""

    name: str
    amount: Decimal
    detail: str
    parts: tuple[Charge, ...]

    def __init__(
        self,
        name: str,
        amount: Decimal,
        detail: str,
        parts: tuple[Charge, ...] = (),
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "amount", amount)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True)
class FillEconomics:
    quantity: Decimal
    notional: Decimal


@dataclass(frozen=True)
class OrderView:
    """What a commish schedule sees for one order.

    Side is required so sell-only fees (SEC Section 31, FINRA TAF) and buy-only
    fees (UK stamp duty) can fire correctly.
    """

    fills: tuple[FillEconomics, ...]
    side: Side

    def __post_init__(self) -> None:
        if len(self.fills) == 0:
            raise TransactionCostError("an order must contain at least one fill")
        if not isinstance(self.side, Side):
            raise TransactionCostError("order side must be Side.BUY or Side.SELL")

    @property
    def quantity(self) -> Decimal:
        return sum((fill.quantity for fill in self.fills), Decimal(0))

    @property
    def notional(self) -> Decimal:
        return sum((fill.notional for fill in self.fills), Decimal(0))

    @property
    def fill_count(self) -> int:
        return len(self.fills)
