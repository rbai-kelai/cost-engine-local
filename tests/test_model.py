from decimal import Decimal

import pytest

from tcost_engine import (
    BidAsk,
    BpsOfNotional,
    Composite,
    Fill,
    FlatFee,
    NoCommission,
    PerShare,
    TransactionCostError,
    TransactionCostModel,
)
from tcost_engine.report import blotter_to_dict, format_report
from tcost_engine.types import cost_bps


def buy(**overrides: object) -> Fill:
    values: dict[str, object] = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": "1000",
        "price": "50.02",
        "spread": BidAsk("50.00", "50.02"),
    }
    values.update(overrides)
    return Fill(**values)  # type: ignore[arg-type]


def test_readme_example() -> None:
    model = TransactionCostModel(commission=PerShare("0.005", minimum="1"))
    result = model.cost(buy())
    assert result.execution_notional == Decimal("50020")
    assert result.commission_amount == Decimal("5")
    assert result.spread == Decimal("10")
    assert result.market_impact == Decimal("0")
    assert result.residual == Decimal("0")
    assert result.total == Decimal("15")
    assert result.total == (
        result.commish_amount + result.spread + result.market_impact + result.residual
    )
    assert result.total_bps == Decimal(15) / Decimal(50020) * Decimal(10000)
    assert "deferred" in result.fills[0].market_impact_detail
    assert "deferred" in result.fills[0].residual_detail


def test_trading_through_the_quote_does_not_add_impact() -> None:
    model = TransactionCostModel(NoCommission())
    at_ask = model.cost(buy(price="50.02"))
    through = model.cost(buy(price="50.10"))
    assert at_ask.spread == Decimal("10")
    assert through.spread == at_ask.spread
    assert through.market_impact == Decimal("0")
    assert through.total == at_ask.total
    assert through.execution_notional == Decimal("50100")
    assert through.execution_notional != at_ask.execution_notional


def test_order_minimum_is_charged_once_across_fills() -> None:
    model = TransactionCostModel(commission=PerShare("0.005", minimum="1"))
    fills = [
        buy(quantity="50", price="10", spread=BidAsk("10", "10.02"), order_id="ord-1"),
        buy(quantity="50", price="10", spread=BidAsk("10", "10.02"), order_id="ord-1"),
    ]
    together = model.cost_many(fills)
    assert len(together.orders) == 1
    assert together.commission == Decimal("1")
    assert sum(together.orders[0].commission_allocated, Decimal(0)) == Decimal("1")
    separate = model.cost(fills[0]).commission_amount + model.cost(fills[1]).commission_amount
    assert separate == Decimal("2")


def test_flat_fee_is_once_per_order_and_allocation_sums_exactly() -> None:
    model = TransactionCostModel(commission=FlatFee("1"))
    quote = BidAsk("10", "10.02")
    fills = [
        buy(quantity="1", price="10", spread=quote, order_id="ord-1"),
        buy(quantity="1", price="10", spread=quote, order_id="ord-1"),
        buy(quantity="1", price="10", spread=quote, order_id="ord-1"),
    ]
    result = model.cost_many(fills)
    allocated = result.orders[0].commission_allocated
    assert result.commission == Decimal("1")
    assert sum(allocated, Decimal(0)) == Decimal("1")
    assert allocated[0] == allocated[1]
    assert allocated[2] == Decimal("1") - allocated[0] - allocated[1]


def test_fills_without_an_order_id_are_separate_orders() -> None:
    model = TransactionCostModel(commission=FlatFee("1"))
    quote = BidAsk("10", "10")
    result = model.cost_many(
        [
            buy(quantity="1", price="10", spread=quote),
            buy(quantity="1", price="10", spread=quote),
        ]
    )
    assert len(result.orders) == 2
    assert result.commission == Decimal("2")


def test_order_cannot_mix_symbol_or_side() -> None:
    model = TransactionCostModel(NoCommission())
    quote = BidAsk("10", "10.02")
    with pytest.raises(TransactionCostError, match="mixes"):
        model.cost_many(
            [
                buy(symbol="AAPL", price="10", quantity="1", spread=quote, order_id="ord-1"),
                buy(symbol="MSFT", price="10", quantity="1", spread=quote, order_id="ord-1"),
            ]
        )


def test_blotter_groups_by_symbol_and_keeps_impact_at_zero() -> None:
    model = TransactionCostModel(commission=PerShare("0.005", minimum="1"))
    blotter = model.cost_many(
        [
            buy(order_id="ord-1", quantity="600", price="50.02", spread=BidAsk("50.00", "50.02")),
            buy(order_id="ord-1", quantity="400", price="50.03", spread=BidAsk("50.01", "50.03")),
            Fill(
                symbol="MSFT",
                side="sell",
                quantity="200",
                price="420.10",
                spread=BidAsk("420.00", "420.20"),
                order_id="ord-2",
            ),
            Fill(
                symbol="SPY",
                side="buy",
                quantity="100",
                price="500.05",
                spread=BidAsk("500.00", "500.10"),
                liquidity="maker",
                order_id="ord-3",
            ),
        ]
    )
    assert blotter.execution_notional == Decimal("184049")
    assert blotter.commission == Decimal("7")
    assert blotter.spread == Decimal("30")
    assert blotter.market_impact == Decimal("0")
    assert blotter.total == Decimal("37")
    assert blotter.by_symbol()["AAPL"].total == Decimal("15")
    assert blotter.by_symbol()["MSFT"].spread == Decimal("20")
    assert blotter.by_symbol()["SPY"].spread == Decimal("0")
    assert blotter.by_symbol()["SPY"].commission == Decimal("1")


def test_empty_blotter() -> None:
    blotter = TransactionCostModel(NoCommission()).cost_many([])
    assert blotter.total == Decimal("0")
    assert blotter.total_bps == Decimal("0")
    assert "No fills." in format_report(blotter)


def test_report_rounds_bps_and_json_keeps_exact_decimals() -> None:
    model = TransactionCostModel(commission=PerShare("0.005"))
    blotter = model.cost_many([buy()])
    text = format_report(blotter)
    assert "Market impact: not modeled (0)" in text
    assert "Residual (trend / opportunity): not modeled (0)" in text
    assert "2.9988 bps" in text
    payload = blotter_to_dict(blotter)
    assert payload["market_impact"] == "not_modeled"
    assert payload["residual"] == "not_modeled"
    assert payload["market_impact_cost"] == "0"
    assert payload["residual_cost"] == "0"
    assert payload["commish"] == "5"
    assert payload["total"] == "15"
    assert Decimal(payload["total_bps"]) == cost_bps(Decimal("15"), Decimal("50020"))


def test_composite_commission_on_a_fill() -> None:
    model = TransactionCostModel(commission=Composite(PerShare("0.005"), BpsOfNotional("1")))
    result = model.cost(buy())
    assert result.commission_amount == Decimal("10.002")
    assert result.total == Decimal("20.002")


def test_invalid_fill_inputs() -> None:
    with pytest.raises(TransactionCostError, match="float"):
        buy(price=50.02)
    with pytest.raises(TransactionCostError, match="quantity"):
        buy(quantity="0")
    with pytest.raises(TransactionCostError, match="price"):
        buy(price="-1")
    with pytest.raises(TransactionCostError, match="side"):
        buy(side="short")
    with pytest.raises(TransactionCostError, match="order_id"):
        buy(order_id="  ")
    with pytest.raises(TransactionCostError, match="liquidity"):
        buy(liquidity="hidden")
