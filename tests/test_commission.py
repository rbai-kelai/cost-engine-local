from decimal import Decimal

import pytest

from tcost_engine import (
    AtLeast,
    BpsOfNotional,
    Composite,
    FlatFee,
    NoCommission,
    PercentOfNotional,
    PerFill,
    PerShare,
    Side,
    TransactionCostError,
)
from tcost_engine.types import FillEconomics, OrderView


def order(qty: str, notional: str, fills: int = 1, side: Side = Side.BUY) -> OrderView:
    quantity = Decimal(qty)
    each_qty = quantity / fills
    each_notional = Decimal(notional) / fills
    return OrderView(
        fills=tuple(FillEconomics(each_qty, each_notional) for _ in range(fills)),
        side=side,
    )


def test_per_share_without_minimum() -> None:
    charge = PerShare("0.005").charge(order("1000", "50020"))
    assert charge.amount == Decimal("5")
    assert charge.detail == "0.005 per share × 1000 = 5"


def test_per_share_minimum_binds() -> None:
    charge = PerShare("0.005", minimum="1").charge(order("100", "1000"))
    assert charge.amount == Decimal("1")
    assert charge.detail == "0.005 per share × 100 = 0.5; order minimum 1 binds"


def test_per_share_minimum_does_not_bind_and_is_not_mentioned() -> None:
    charge = PerShare("0.005", minimum="1").charge(order("1000", "50020"))
    assert charge.amount == Decimal("5")
    assert "minimum" not in charge.detail


def test_rebate_is_kept_when_minimum_is_zero() -> None:
    charge = PerShare("-0.002").charge(order("1000", "50020"))
    assert charge.amount == Decimal("-2")


def test_positive_minimum_replaces_a_rebate() -> None:
    charge = PerShare("-0.002", minimum="1").charge(order("1000", "50020"))
    assert charge.amount == Decimal("1")


def test_negative_minimum_is_rejected() -> None:
    with pytest.raises(TransactionCostError, match="minimum"):
        PerShare("0.005", minimum="-1")


def test_bps_of_notional() -> None:
    charge = BpsOfNotional("1").charge(order("1000", "50020"))
    assert charge.amount == Decimal("5.002")
    assert charge.detail == "1 bps × notional 50020 / 10000 = 5.002"


def test_one_percent_matches_100_bps() -> None:
    view = order("1000", "50020")
    assert PercentOfNotional("1").charge(view).amount == BpsOfNotional("100").charge(view).amount
    assert PercentOfNotional("1").charge(view).amount == Decimal("500.20")


def test_flat_fee_ignores_size() -> None:
    assert FlatFee("2").charge(order("1", "10")).amount == Decimal("2")
    assert FlatFee("2").charge(order("1000", "10")).amount == Decimal("2")


def test_no_commission() -> None:
    assert NoCommission().charge(order("10", "100")).amount == Decimal("0")


def test_composite_sums_parts() -> None:
    charge = Composite(PerShare("0.005"), BpsOfNotional("1")).charge(order("1000", "50020"))
    assert charge.amount == Decimal("10.002")
    assert [part.amount for part in charge.parts] == [Decimal("5"), Decimal("5.002")]


def test_empty_composite_is_rejected() -> None:
    with pytest.raises(TransactionCostError, match="composite"):
        Composite()


def test_per_fill_applies_the_minimum_on_each_fill() -> None:
    view = order("100", "1000", fills=2)
    once = PerShare("0.005", minimum="1").charge(view)
    each = PerFill(PerShare("0.005", minimum="1")).charge(view)
    assert once.amount == Decimal("1")
    assert each.amount == Decimal("2")


def test_at_least_binds_on_the_combined_schedule() -> None:
    schedule = AtLeast(Composite(PerShare("0.001"), FlatFee("0.10")), minimum="1")
    charge = schedule.charge(order("100", "1000"))
    assert charge.amount == Decimal("1")
    assert "floor 1 binds" in charge.detail


def test_at_least_is_transparent_when_the_floor_does_not_bind() -> None:
    charge = AtLeast(PerShare("0.005"), minimum="1").charge(order("1000", "50020"))
    assert charge.name == "per_share"
    assert charge.amount == Decimal("5")


def test_float_rate_is_rejected() -> None:
    with pytest.raises(TransactionCostError, match="float"):
        PerShare(0.005)  # type: ignore[arg-type]
