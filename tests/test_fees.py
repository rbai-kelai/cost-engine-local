from decimal import Decimal

import pytest

from tcm import (
    BidAsk,
    Composite,
    Fill,
    FinraTaf,
    OnBuy,
    OnSell,
    PerShare,
    SecFee,
    Side,
    StampDuty,
    TransactionCostError,
    TransactionCostModel,
)
from tcm.types import FillEconomics, OrderView


def view(side: Side, qty: str = "1000", notional: str = "50000") -> OrderView:
    return OrderView(
        fills=(FillEconomics(Decimal(qty), Decimal(notional)),),
        side=side,
    )


def test_sec_fee_only_on_sells() -> None:
    fee = SecFee("0.0000278")
    assert fee.charge(view(Side.BUY)).amount == Decimal("0")
    sell = fee.charge(view(Side.SELL))
    assert sell.amount == Decimal("1.39")
    assert "Section 31" in sell.detail


def test_finra_taf_cap_binds() -> None:
    fee = FinraTaf(rate_per_share="0.000166", cap="8.30")
    small = fee.charge(view(Side.SELL, qty="1000", notional="50000"))
    assert small.amount == Decimal("0.166")
    large = fee.charge(view(Side.SELL, qty="100000", notional="5000000"))
    assert large.amount == Decimal("8.30")
    assert "cap" in large.detail
    assert fee.charge(view(Side.BUY, qty="100000", notional="5000000")).amount == Decimal("0")


def test_stamp_duty_only_on_buys() -> None:
    duty = StampDuty("0.5")
    buy = duty.charge(view(Side.BUY, qty="100", notional="10000"))
    assert buy.amount == Decimal("50")
    assert duty.charge(view(Side.SELL, qty="100", notional="10000")).amount == Decimal("0")


def test_on_buy_and_on_sell_wrappers() -> None:
    buy_only = OnBuy(StampDuty("0.5"))
    sell_only = OnSell(SecFee("0.0001"))
    order_buy = view(Side.BUY, notional="10000")
    order_sell = view(Side.SELL, notional="10000")
    assert buy_only.charge(order_buy).amount == Decimal("50")
    assert buy_only.charge(order_sell).amount == Decimal("0")
    assert sell_only.charge(order_sell).amount == Decimal("1")
    assert sell_only.charge(order_buy).amount == Decimal("0")


def test_agency_composite_in_the_model() -> None:
    agency = Composite(
        PerShare("0.005", minimum="1"),
        SecFee("0.0000278"),
        FinraTaf(rate_per_share="0.000166", cap="8.30"),
    )
    model = TransactionCostModel(agency=agency)
    buy = model.cost(
        Fill(
            symbol="AAPL",
            side="buy",
            quantity="1000",
            price="50",
            spread=BidAsk("49.99", "50.01"),
        )
    )
    sell = model.cost(
        Fill(
            symbol="AAPL",
            side="sell",
            quantity="1000",
            price="50",
            spread=BidAsk("49.99", "50.01"),
        )
    )
    # Buy: commission 5, no SEC/TAF. Spread 10.
    assert buy.agency_amount == Decimal("5")
    assert buy.spread == Decimal("10")
    assert buy.total == Decimal("15")
    # Sell: commission 5 + SEC 1.39 + TAF 0.166 = 6.556. Spread 10.
    assert sell.agency_amount == Decimal("6.556")
    assert sell.total == Decimal("16.556")
    assert sell.residual == Decimal("0")
    assert sell.market_impact == Decimal("0")


def test_negative_fee_rates_are_rejected() -> None:
    with pytest.raises(TransactionCostError):
        SecFee("-0.1")
    with pytest.raises(TransactionCostError):
        StampDuty("-0.5")
