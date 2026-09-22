from decimal import Decimal

import pytest

from tcm import (
    BidAsk,
    Fill,
    FullSpreadBps,
    NoCommission,
    OneWaySpreadBps,
    TransactionCostError,
    TransactionCostModel,
)


def _fill(**overrides: object) -> Fill:
    values: dict[str, object] = {
        "side": "buy",
        "quantity": "1000",
        "price": "50.02",
        "spread": BidAsk("50.00", "50.02"),
    }
    values.update(overrides)
    return Fill(**values)  # type: ignore[arg-type]


def test_buy_and_sell_pay_the_same_half_spread() -> None:
    model = TransactionCostModel(NoCommission())
    buy = model.cost(_fill(side="buy"))
    sell = model.cost(_fill(side="sell", price="50.00"))
    assert buy.spread == Decimal("10")
    assert sell.spread == buy.spread
    assert "taker half-spread" in buy.fills[0].spread_detail
    assert buy.fills[0].spread_detail == (
        "taker half-spread (50.02 - 50) / 2 = 0.01 × 1000 = 10"
    )


def test_full_spread_bps_is_halved_and_one_way_is_not() -> None:
    model = TransactionCostModel(NoCommission())
    common = {"side": "buy", "quantity": "10", "price": "100"}
    full = model.cost(Fill(**common, spread=FullSpreadBps("10")))  # type: ignore[arg-type]
    one_way = model.cost(Fill(**common, spread=OneWaySpreadBps("10")))  # type: ignore[arg-type]
    assert full.spread == Decimal("0.5")
    assert one_way.spread == Decimal("1")


def test_full_spread_bps_matches_the_quote_it_describes() -> None:
    model = TransactionCostModel(NoCommission())
    quoted = model.cost(
        Fill(side="buy", quantity="10", price="100.02", spread=BidAsk("99.98", "100.02"))
    )
    assumed = model.cost(
        Fill(
            side="buy",
            quantity="10",
            price="100.02",
            spread=FullSpreadBps("4", reference_price="100"),
        )
    )
    assert quoted.spread == Decimal("0.2")
    assert assumed.spread == quoted.spread


def test_one_way_bps_without_a_reference_is_exact_on_execution_notional() -> None:
    model = TransactionCostModel(NoCommission())
    result = model.cost(
        Fill(side="sell", quantity="5", price="80", spread=OneWaySpreadBps("8"))
    )
    assert result.spread == Decimal("0.32")
    assert result.total_bps == Decimal("8")


def test_full_spread_bps_without_a_reference_costs_half_in_bps() -> None:
    model = TransactionCostModel(NoCommission())
    result = model.cost(
        Fill(side="buy", quantity="5", price="80", spread=FullSpreadBps("8"))
    )
    assert result.total_bps == Decimal("4")


def test_maker_capture_zero_one_and_midpoint() -> None:
    fill = _fill(liquidity="maker")
    none = TransactionCostModel(NoCommission(), maker_capture="0").cost(fill)
    full = TransactionCostModel(NoCommission(), maker_capture="1").cost(fill)
    half = TransactionCostModel(NoCommission(), maker_capture="0.5").cost(fill)
    midpoint = TransactionCostModel(NoCommission()).cost(_fill(liquidity="midpoint"))
    assert none.spread == Decimal("0")
    assert full.spread == Decimal("-10")
    assert half.spread == Decimal("-5")
    assert midpoint.spread == Decimal("0")
    assert "midpoint" in midpoint.fills[0].spread_detail


def test_maker_capture_must_be_a_fraction() -> None:
    with pytest.raises(TransactionCostError, match="maker_capture"):
        TransactionCostModel(NoCommission(), maker_capture="1.1")
    with pytest.raises(TransactionCostError, match="maker_capture"):
        TransactionCostModel(NoCommission(), maker_capture="-0.1")


def test_locked_quote_has_no_spread_and_crossed_quote_is_rejected() -> None:
    model = TransactionCostModel(NoCommission())
    locked = model.cost(_fill(spread=BidAsk("10", "10"), price="10", quantity="4"))
    assert locked.spread == Decimal("0")
    with pytest.raises(TransactionCostError, match="crossed"):
        BidAsk("10", "9")


def test_negative_spread_bps_is_rejected() -> None:
    with pytest.raises(TransactionCostError, match="spread bps"):
        FullSpreadBps("-1")
    with pytest.raises(TransactionCostError, match="spread bps"):
        OneWaySpreadBps("-1")
