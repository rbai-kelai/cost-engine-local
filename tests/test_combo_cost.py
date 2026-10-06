"""Combo SOD → DoD trades → t-cost fills."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from tcost_engine.combo.cost import cost_combo_sod
from tcost_engine.combo.sod import (
    cost_joined_trades,
    join_trades_prices,
    load_sod_panel,
    sod_trades,
    trades_to_fills,
)
from tcost_engine.commission import DEFAULT_COMMISH_MILS, PerShare
from tcost_engine.model import TransactionCostModel


def _sod_panel() -> pd.DataFrame:
    """Two names, three days of signed dollar SOD notionals."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            101: [1000.0, 1500.0, 1500.0],  # buy 500 on day 2, flat day 3
            202: [-800.0, -200.0, 0.0],  # cover 600 on day 2, cover 200 on day 3
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )


def _prices() -> pd.DataFrame:
    rows = []
    for day, close, vwap, bid, ask in [
        ("2024-01-02", 10.0, 10.0, 9.98, 10.02),
        ("2024-01-03", 10.0, 10.05, 9.99, 10.01),
        ("2024-01-04", 20.0, 19.90, 19.80, 20.00),
    ]:
        for infocode in (101, 202):
            rows.append(
                {
                    "marketdate": pd.Timestamp(day),
                    "infocode": infocode,
                    "bid": bid,
                    "ask": ask,
                    "vwap": vwap,
                    "close": close,
                }
            )
    # Drop VWAP for 202 on 2024-01-04 so that trade is skipped
    df = pd.DataFrame(rows)
    df.loc[(df["infocode"] == 202) & (df["marketdate"] == "2024-01-04"), "vwap"] = float(
        "nan"
    )
    return df


def test_sod_trades_skip_first_day_and_sign_sides() -> None:
    trades = sod_trades(_sod_panel())
    assert list(trades["date"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-03",
        "2024-01-03",
        "2024-01-04",
    ]
    by = trades.set_index(["date", "infocode"])
    assert by.loc[("2024-01-03", 101), "side"] == "buy"
    assert by.loc[("2024-01-03", 101), "delta_dollars"] == pytest.approx(500.0)
    assert by.loc[("2024-01-03", 202), "side"] == "buy"  # -800 → -200 is +600
    assert by.loc[("2024-01-03", 202), "delta_dollars"] == pytest.approx(600.0)
    assert by.loc[("2024-01-04", 202), "side"] == "buy"
    assert by.loc[("2024-01-04", 202), "delta_dollars"] == pytest.approx(200.0)


def test_trades_to_fills_costs_match_hand_calc() -> None:
    trades = sod_trades(_sod_panel())
    fills, stats = trades_to_fills(trades, _prices())
    # Missing VWAP falls back to close → all 3 trades keep a fill
    assert stats.n_trades == 3
    assert stats.n_fills == 3
    assert stats.n_dropped == 0

    model = TransactionCostModel(commish=PerShare(DEFAULT_COMMISH_MILS))
    blotter = model.cost_many(fills)

    # Combo fills carry zero spread (VWAP stack).
    slip = sum(
        (
            (f.price - f.close) * f.quantity  # type: ignore[operator]
            for f in fills
        ),
        Decimal(0),
    )
    mils = Decimal(10) / Decimal(10000)
    commish = mils * sum((f.quantity for f in fills), Decimal(0))  # type: ignore[arg-type]

    assert blotter.spread == Decimal(0)
    assert blotter.residual == slip
    assert blotter.commish == commish
    assert blotter.total == commish + slip
    assert blotter.market_impact == Decimal(0)
    # day2: 500/10.05 + 600/10.05; day3: 200/close20 (VWAP missing → close)
    qty_total = sum((f.quantity for f in fills), Decimal(0))  # type: ignore[arg-type]
    assert float(qty_total) == pytest.approx(
        500 / 10.05 + 600 / 10.05 + 200 / 20.0, rel=1e-9
    )


def test_vectorized_cost_matches_fill_model() -> None:
    trades = sod_trades(_sod_panel())
    prices = _prices()
    fills, stats = trades_to_fills(trades, prices)
    joined, stats2 = join_trades_prices(trades, prices)
    assert stats == stats2
    model = TransactionCostModel(commish=PerShare(DEFAULT_COMMISH_MILS))
    blotter = model.cost_many(fills)
    daily = cost_joined_trades(joined, mils=DEFAULT_COMMISH_MILS)
    assert float(daily["spread"].sum()) == pytest.approx(0.0)
    assert float(daily["commish"].sum()) == pytest.approx(float(blotter.commish), rel=1e-9)
    # Combo slippage is PnL-signed (= − blotter residual); tcost = commish + slippage.
    assert float(daily["intraday_slippage"].sum()) == pytest.approx(
        -float(blotter.residual), rel=1e-9
    )
    assert float(daily["total"].sum()) == pytest.approx(
        float(blotter.commish) + float(daily["intraday_slippage"].sum()), rel=1e-9
    )


def test_load_sod_panel_from_parquet(tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    _sod_panel().to_parquet(path)
    loaded = load_sod_panel(path)
    assert list(loaded.columns) == [101, 202]
    assert loaded.index.name == "date"
    assert len(loaded) == 3


def test_cost_combo_sod_with_injected_prices(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    _sod_panel().to_parquet(path)
    prices = _prices()

    def fake_pull(*_a, **_k):
        return prices

    monkeypatch.setattr("tcost_engine.combo.cost.pull_cost_prices", fake_pull)
    result = cost_combo_sod(path, start=date(2024, 1, 2), end=date(2024, 1, 4))
    assert result.n_fills == 3
    assert result.n_dropped == 0
    assert len(result.daily) == 2  # 2024-01-03 and 2024-01-04
    assert result.daily.iloc[0]["date"] == pd.Timestamp("2024-01-03")
    assert result.total_spread == Decimal(0)
    assert result.total_commish > 0
    # total = commish + PnL-signed intraday slippage
    assert float(result.total_cost) == pytest.approx(
        float(result.total_commish) + float(result.total_intraday_slippage), rel=1e-9
    )


def test_moc_fill_is_commish_only() -> None:
    """MOC exec at close → intraday_slippage = 0; tcost = mils only."""
    trades = sod_trades(_sod_panel())
    joined, stats = join_trades_prices(trades, _prices(), fill="moc")
    assert stats.n_fills == 3
    # qty = |delta| / close
    by = joined.set_index(["date", "infocode"])
    assert by.loc[(pd.Timestamp("2024-01-03"), 101), "qty"] == pytest.approx(500 / 10.0)
    assert by.loc[(pd.Timestamp("2024-01-03"), 202), "qty"] == pytest.approx(600 / 10.0)
    assert by.loc[(pd.Timestamp("2024-01-04"), 202), "qty"] == pytest.approx(200 / 20.0)

    daily = cost_joined_trades(joined, mils=DEFAULT_COMMISH_MILS, fill="moc")
    assert float(daily["intraday_slippage"].sum()) == pytest.approx(0.0)
    assert float(daily["spread"].sum()) == pytest.approx(0.0)
    mils = 10.0 / 10000.0
    qty = 500 / 10.0 + 600 / 10.0 + 200 / 20.0
    assert float(daily["commish"].sum()) == pytest.approx(mils * qty, rel=1e-9)
    assert float(daily["total"].sum()) == pytest.approx(mils * qty, rel=1e-9)


def test_intraday_slippage_is_pnl_signed() -> None:
    """Buy high / sell low vs close → negative intraday slippage."""
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "infocode": [101, 202],
            "side": ["buy", "sell"],
            "delta_dollars": [1000.0, -1000.0],
        }
    )
    prices = pd.DataFrame(
        [
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 101,
                "bid": 9.9,
                "ask": 10.1,
                "vwap": 10.10,  # buy above close
                "close": 10.0,
            },
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 202,
                "bid": 19.9,
                "ask": 20.1,
                "vwap": 19.90,  # sell below close
                "close": 20.0,
            },
        ]
    )
    joined, _ = join_trades_prices(trades, prices, fill="vwap")
    daily = cost_joined_trades(joined, mils=0, fill="vwap")
    # buy: +1×(10−10.10)×(1000/10.10) < 0
    # sell: −1×(20−19.90)×(1000/19.90) < 0
    assert float(daily["intraday_slippage"].sum()) < 0
    buy_qty = 1000 / 10.10
    sell_qty = 1000 / 19.90
    expect = 1.0 * (10.0 - 10.10) * buy_qty + (-1.0) * (20.0 - 19.90) * sell_qty
    assert float(daily["intraday_slippage"].sum()) == pytest.approx(expect, rel=1e-9)
    assert float(daily["total"].sum()) == pytest.approx(
        float(daily["commish"].sum()) + expect, rel=1e-9
    )


def test_cost_combo_sod_moc(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    _sod_panel().to_parquet(path)

    monkeypatch.setattr(
        "tcost_engine.combo.cost.pull_cost_prices", lambda *_a, **_k: _prices()
    )
    result = cost_combo_sod(
        path, start=date(2024, 1, 2), end=date(2024, 1, 4), fill="moc"
    )
    assert result.n_fills == 3
    assert result.total_intraday_slippage == Decimal(0)
    assert result.total_spread == Decimal(0)
    assert result.total_cost == result.total_commish
    assert result.total_commish > 0
