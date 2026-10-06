from decimal import Decimal

import pytest

from tcost_engine import (
    BidAsk,
    Fill,
    NoCommission,
    TransactionCostError,
    TransactionCostModel,
    vwap_close_slippage,
)


def test_buy_pays_vwap_above_close() -> None:
    model = TransactionCostModel(NoCommission())
    result = model.cost(
        Fill(
            side="buy",
            quantity="1000",
            price="50.10",
            close="50.00",
            spread=BidAsk("50.00", "50.02"),
        )
    )
    assert result.spread == Decimal("10")
    assert result.residual == Decimal("100")
    assert result.slippage == result.residual
    assert result.market_impact == Decimal("0")
    assert result.total == Decimal("110")
    assert "VWAP 50.1 − close 50" in result.fills[0].residual_detail


def test_sell_pays_vwap_below_close() -> None:
    model = TransactionCostModel(NoCommission())
    result = model.cost(
        Fill(
            side="sell",
            quantity="200",
            price="420.10",
            close="420.20",
            spread=BidAsk("420.00", "420.20"),
        )
    )
    # sell × (420.10 − 420.20) × 200 = +20
    assert result.slippage == Decimal("20")
    assert result.spread == Decimal("20")
    assert result.total == Decimal("40")


def test_favorable_vwap_is_a_negative_cost() -> None:
    amount, detail = vwap_close_slippage(
        Fill(
            side="buy",
            quantity="100",
            price="99",
            close="100",
            spread=BidAsk("99", "100"),
        )
    )
    assert amount == Decimal("-100")
    assert "buy side" in detail


def test_omitted_close_leaves_slippage_at_zero() -> None:
    model = TransactionCostModel(NoCommission())
    result = model.cost(
        Fill(side="buy", quantity="10", price="50", spread=BidAsk("49.99", "50.01"))
    )
    assert result.residual == Decimal("0")
    assert "omitted" in result.fills[0].residual_detail


def test_close_must_be_positive() -> None:
    with pytest.raises(TransactionCostError, match="close"):
        Fill(side="buy", quantity="1", price="10", close="0", spread=BidAsk("10", "10.01"))
