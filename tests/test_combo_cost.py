"""Combo SOD → DoD trades → t-cost fills."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from tcost_engine.combo.cost import cost_combo_sod, result_summary
from tcost_engine.combo.sod import (
    cost_joined_trades,
    join_trades_prices,
    load_sod_panel,
    sod_trades,
    trades_to_fills,
)
from tcost_engine.combo.stats import daily_turnover
from tcost_engine.commission import DEFAULT_COMMISH_MILS, PerShare
from tcost_engine.model import TransactionCostModel


def _sod_panel() -> pd.DataFrame:
    """Two names, three days of signed dollar SOD notionals."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            101: [1000.0, 1500.0, 1500.0],
            202: [-800.0, -200.0, 0.0],
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )


def _prices() -> pd.DataFrame:
    rows = []
    for day, close, vwap in [
        ("2024-01-02", 10.0, 10.0),
        ("2024-01-03", 10.0, 10.05),
        ("2024-01-04", 10.0, 9.95),
    ]:
        for infocode in (101, 202):
            rows.append(
                {
                    "marketdate": pd.Timestamp(day),
                    "infocode": infocode,
                    "bid": close - 0.02,
                    "ask": close + 0.02,
                    "vwap": vwap,
                    "close": close,
                    "close_adjusted": close,
                }
            )
    df = pd.DataFrame(rows)
    df.loc[(df["infocode"] == 202) & (df["marketdate"] == "2024-01-04"), "vwap"] = float(
        "nan"
    )
    return df


def test_sod_trades_delta_dollars_and_qty_via_unadj_close() -> None:
    trades, stats = sod_trades(_sod_panel(), _prices())
    assert stats.n_fills == 3
    assert stats.n_dropped == 0
    by = trades.set_index(["date", "infocode"])
    # adj=close: δ$ = Δn_adj × close_adj = 50 × 10
    assert by.loc[(pd.Timestamp("2024-01-03"), 101), "delta_dollars"] == pytest.approx(
        500.0
    )
    assert by.loc[(pd.Timestamp("2024-01-03"), 101), "delta_shares_adj"] == pytest.approx(
        50.0
    )
    joined, _ = join_trades_prices(trades, _prices())
    j = joined.set_index(["date", "infocode"])
    # qty = |δ$| / close_t = 500 / 10
    assert j.loc[(pd.Timestamp("2024-01-03"), 101), "qty"] == pytest.approx(50.0)
    assert j.loc[(pd.Timestamp("2024-01-03"), 202), "qty"] == pytest.approx(60.0)
    assert j.loc[(pd.Timestamp("2024-01-04"), 202), "qty"] == pytest.approx(20.0)


def test_back_adjusted_close_does_not_inflate_qty() -> None:
    """close_adj = close × f with f<1 must not inflate real share qty."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    panel = pd.DataFrame(
        {101: [1000.0, 1500.0]}, index=pd.DatetimeIndex(idx, name="date")
    )
    prices = pd.DataFrame(
        [
            {
                "marketdate": pd.Timestamp("2024-01-02"),
                "infocode": 101,
                "bid": 99.0,
                "ask": 101.0,
                "vwap": 100.0,
                "close": 100.0,
                "close_adjusted": 10.0,  # f=0.1
            },
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 101,
                "bid": 99.0,
                "ask": 101.0,
                "vwap": 100.5,
                "close": 100.0,
                "close_adjusted": 10.0,
            },
        ]
    )
    trades, _ = sod_trades(panel, prices)
    # Δn_adj = 1500/10 − 1000/10 = 50; δ$ = 50×10 = 500; qty = 500/100 = 5
    assert trades.iloc[0]["delta_dollars"] == pytest.approx(500.0)
    assert trades.iloc[0]["delta_shares_adj"] == pytest.approx(50.0)
    joined, _ = join_trades_prices(trades, prices)
    assert joined.iloc[0]["qty"] == pytest.approx(5.0)
    # Not the inflated adj-share count
    assert joined.iloc[0]["qty"] != pytest.approx(50.0)


def test_flat_shares_no_trade_when_price_moves() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    panel = pd.DataFrame({101: [1000.0, 1100.0]}, index=pd.DatetimeIndex(idx, name="date"))
    prices = pd.DataFrame(
        [
            {
                "marketdate": pd.Timestamp("2024-01-02"),
                "infocode": 101,
                "bid": 9.9,
                "ask": 10.1,
                "vwap": 10.0,
                "close": 10.0,
                "close_adjusted": 10.0,
            },
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 101,
                "bid": 10.9,
                "ask": 11.1,
                "vwap": 11.0,
                "close": 11.0,
                "close_adjusted": 11.0,
            },
        ]
    )
    trades, stats = sod_trades(panel, prices)
    assert len(trades) == 0
    assert stats.n_fills == 0
    assert stats.n_dropped == 0


def test_zero_sod_unheld_not_dropped_entry_ok_without_prior_price() -> None:
    """Unheld is 0 not NaN: never-held ignored; entry with no t−1 price still fills."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    # 101 held both days; 202 never held (zeros); 303 new listing on day 2.
    panel = pd.DataFrame(
        {
            101: [1000.0, 1000.0],
            202: [0.0, 0.0],
            303: [0.0, 500.0],
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )
    # Prices for 101 both days; 303 only on day 2 (no t−1 price); 202 absent.
    prices = pd.DataFrame(
        [
            {
                "marketdate": pd.Timestamp("2024-01-02"),
                "infocode": 101,
                "bid": 9.9,
                "ask": 10.1,
                "vwap": 10.0,
                "close": 10.0,
                "close_adjusted": 10.0,
            },
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 101,
                "bid": 9.9,
                "ask": 10.1,
                "vwap": 10.0,
                "close": 10.0,
                "close_adjusted": 10.0,
            },
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 303,
                "bid": 4.9,
                "ask": 5.1,
                "vwap": 5.0,
                "close": 5.0,
                "close_adjusted": 5.0,
            },
        ]
    )
    trades, stats = sod_trades(panel, prices)
    # One trade: enter 303. Never-held 202 is not a drop. Flat 101 shares → no trade.
    assert stats.n_fills == 1
    assert stats.n_dropped == 0
    assert stats.n_trades == 1
    assert list(trades["infocode"]) == [303]
    assert trades.iloc[0]["delta_dollars"] == pytest.approx(500.0)
    joined, jstats = join_trades_prices(trades, prices)
    assert jstats.n_fills == 1
    assert jstats.n_dropped == 0
    assert joined.iloc[0]["qty"] == pytest.approx(100.0)  # 500/5


def test_missing_price_name_counted_as_dropped() -> None:
    """Held SOD name absent from prices → dropped, not silently omitted."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    panel = pd.DataFrame(
        {101: [1000.0, 1500.0], 999: [500.0, 800.0]},
        index=pd.DatetimeIndex(idx, name="date"),
    )
    prices = _prices()  # only 101, 202
    trades, stats = sod_trades(panel, prices)
    assert stats.n_dropped >= 1
    assert 999 not in set(trades["infocode"])
    assert stats.n_trades == stats.n_fills + stats.n_dropped


def test_trades_to_fills_qty_is_abs_delta_dollars_over_close() -> None:
    trades, _ = sod_trades(_sod_panel(), _prices())
    fills, stats = trades_to_fills(trades, _prices())
    assert stats.n_fills == 3
    qty_total = sum((float(f.quantity) for f in fills), 0.0)
    assert qty_total == pytest.approx(50.0 + 60.0 + 20.0, rel=1e-9)


def test_vectorized_cost_matches_fill_model() -> None:
    prices = _prices()
    trades, _ = sod_trades(_sod_panel(), prices)
    fills, stats = trades_to_fills(trades, prices)
    joined, stats2 = join_trades_prices(trades, prices)
    assert stats == stats2
    model = TransactionCostModel(commish=PerShare(DEFAULT_COMMISH_MILS))
    blotter = model.cost_many(fills)
    daily = cost_joined_trades(joined, mils=DEFAULT_COMMISH_MILS)
    assert float(daily["commish"].sum()) == pytest.approx(float(blotter.commish), rel=1e-9)
    assert float(daily["spread"].sum()) == pytest.approx(float(blotter.spread), rel=1e-9)
    assert float(daily["intraday_slippage"].sum()) == pytest.approx(
        float(blotter.residual), rel=1e-9
    )


def test_half_spread_from_eod_bid_ask() -> None:
    """EOD bid/ask half-spread × qty; disabled via include_spread=False."""
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03"]),
            "infocode": [101],
            "side": ["buy"],
            "delta_shares_adj": [100.0],
            "delta_dollars": [1000.0],
        }
    )
    prices = pd.DataFrame(
        [
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 101,
                "bid": 9.96,
                "ask": 10.04,
                "vwap": 10.0,
                "close": 10.0,
            }
        ]
    )
    joined, _ = join_trades_prices(trades, prices, fill="vwap")
    # half = 0.04; qty = 100 → spread = 4
    daily = cost_joined_trades(joined, mils=0, fill="vwap", include_spread=True)
    assert float(daily["spread"].sum()) == pytest.approx(4.0, rel=1e-9)
    assert float(daily["total"].sum()) == pytest.approx(4.0, rel=1e-9)
    off = cost_joined_trades(joined, mils=0, fill="vwap", include_spread=False)
    assert float(off["spread"].sum()) == pytest.approx(0.0)


def test_load_sod_panel_from_parquet(tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    _sod_panel().to_parquet(path)
    loaded = load_sod_panel(path)
    assert list(loaded.columns) == [101, 202]


def test_cost_combo_sod_with_injected_prices(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    _sod_panel().to_parquet(path)
    monkeypatch.setattr(
        "tcost_engine.combo.cost.pull_cost_prices", lambda *_a, **_k: _prices()
    )
    result = cost_combo_sod(path, start=date(2024, 1, 2), end=date(2024, 1, 4))
    assert result.n_fills == 3
    assert result.daily.iloc[0]["date"] == pd.Timestamp("2024-01-03")
    assert float(result.total_commish) == pytest.approx(
        (10.0 / 10000.0) * (50 + 60 + 20), rel=1e-9
    )


def test_result_summary_uses_run_mils(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    _sod_panel().to_parquet(path)
    monkeypatch.setattr(
        "tcost_engine.combo.cost.pull_cost_prices", lambda *_a, **_k: _prices()
    )
    result = cost_combo_sod(
        path, start=date(2024, 1, 2), end=date(2024, 1, 4), mils=7
    )
    summary = result_summary(result)
    assert summary["mils"] == "7"
    assert result.mils == Decimal("7")


def test_moc_fill_is_commish_only() -> None:
    prices = _prices()
    trades, _ = sod_trades(_sod_panel(), prices)
    joined, stats = join_trades_prices(trades, prices, fill="moc")
    assert stats.n_fills == 3
    by = joined.set_index(["date", "infocode"])
    assert by.loc[(pd.Timestamp("2024-01-03"), 101), "qty"] == pytest.approx(50.0)
    daily = cost_joined_trades(joined, mils=DEFAULT_COMMISH_MILS, fill="moc")
    assert float(daily["intraday_slippage"].sum()) == pytest.approx(0.0)
    assert float(daily["commish"].sum()) == pytest.approx(
        (10.0 / 10000.0) * 130.0, rel=1e-9
    )
    # MOC never charges half-spread (even when bid/ask present).
    assert float(daily["spread"].sum()) == pytest.approx(0.0)
    assert float(daily["total"].sum()) == pytest.approx(
        float(daily["commish"].sum()), rel=1e-9
    )


def test_intraday_slippage_vs_same_day_close() -> None:
    trades = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "infocode": [101, 202],
            "side": ["buy", "sell"],
            "delta_shares_adj": [100.0, -50.0],
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
                "vwap": 10.10,
                "close": 10.0,
            },
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 202,
                "bid": 19.9,
                "ask": 20.1,
                "vwap": 19.90,
                "close": 20.0,
            },
        ]
    )
    joined, _ = join_trades_prices(trades, prices, fill="vwap")
    # qty = |δ$|/close → 1000/10 and 1000/20
    assert joined.set_index("infocode").loc[101, "qty"] == pytest.approx(100.0)
    assert joined.set_index("infocode").loc[202, "qty"] == pytest.approx(50.0)
    daily = cost_joined_trades(joined, mils=0, fill="vwap")
    expect = 1.0 * (10.10 - 10.0) * 100.0 + (-1.0) * (19.90 - 20.0) * 50.0
    assert float(daily["intraday_slippage"].sum()) == pytest.approx(expect, rel=1e-9)


def test_daily_turnover_excludes_price_drift() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    # Flat shares: SOD dollars rise with price → raw dollar TO > 0, share TO = 0
    panel = pd.DataFrame({101: [1000.0, 1100.0]}, index=pd.DatetimeIndex(idx, name="date"))
    prices = pd.DataFrame(
        [
            {
                "marketdate": pd.Timestamp("2024-01-02"),
                "infocode": 101,
                "bid": 9.9,
                "ask": 10.1,
                "vwap": 10.0,
                "close": 10.0,
                "close_adjusted": 10.0,
            },
            {
                "marketdate": pd.Timestamp("2024-01-03"),
                "infocode": 101,
                "bid": 10.9,
                "ask": 11.1,
                "vwap": 11.0,
                "close": 11.0,
                "close_adjusted": 11.0,
            },
        ]
    )
    raw = daily_turnover(panel)
    adj = daily_turnover(panel, prices)
    assert raw.loc[pd.Timestamp("2024-01-03")] == pytest.approx(0.1)
    assert adj.loc[pd.Timestamp("2024-01-03")] == pytest.approx(0.0)


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
    assert float(result.total_cost) == pytest.approx(
        float(result.total_commish), rel=1e-9
    )


def test_cost_combo_sod_no_spread(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    _sod_panel().to_parquet(path)
    monkeypatch.setattr(
        "tcost_engine.combo.cost.pull_cost_prices", lambda *_a, **_k: _prices()
    )
    result = cost_combo_sod(
        path,
        start=date(2024, 1, 2),
        end=date(2024, 1, 4),
        fill="moc",
        include_spread=False,
    )
    assert result.total_spread == Decimal(0)
    assert result.total_cost == result.total_commish
