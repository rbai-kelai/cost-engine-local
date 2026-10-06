"""Pre-tcost vs post-tcost debug_perturb (mosek-style tables)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import tcost_engine.combo.cost as combo_cost
import tcost_engine.combo.debug_perturb as dbg
from tcost_engine.combo.debug_perturb import (
    format_debug_perturb,
    run_debug_perturb,
    write_perturb_artifacts,
)
from tcost_engine.combo.pnl import lag1_dollar_pnl
from tcost_engine.combo.stats import daily_turnover, path_metrics, yearly_perf_frame


def _sod() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            101: [1000.0, 1000.0, 1000.0],
            202: [-1000.0, -1000.0, -1000.0],
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )


def _prices() -> pd.DataFrame:
    rows = []
    levels = [
        ("2024-01-02", 10.0, 20.0, 10.0, 20.0),
        ("2024-01-03", 11.0, 22.0, 11.0, 22.0),
        ("2024-01-04", 11.0, 22.0, 11.0, 22.0),
    ]
    for day, c1, c2, a1, a2 in levels:
        rows.append(
            {
                "marketdate": pd.Timestamp(day),
                "infocode": 101,
                "bid": c1 - 0.01,
                "ask": c1 + 0.01,
                "vwap": c1,
                "close": c1,
                "close_adjusted": a1,
            }
        )
        rows.append(
            {
                "marketdate": pd.Timestamp(day),
                "infocode": 202,
                "bid": c2 - 0.01,
                "ask": c2 + 0.01,
                "vwap": c2,
                "close": c2,
                "close_adjusted": a2,
            }
        )
    return pd.DataFrame(rows)


def test_lag1_pnl_matches_hand_calc() -> None:
    pnl = lag1_dollar_pnl(_sod(), _prices())
    day = pnl.set_index("date").loc[pd.Timestamp("2024-01-03")]
    assert day["pre_pnl"] == pytest.approx(0.0)
    assert day["gmv"] == pytest.approx(2000.0)
    day2 = pnl.set_index("date").loc[pd.Timestamp("2024-01-04")]
    assert day2["pre_pnl"] == pytest.approx(0.0)


def test_daily_turnover_two_way() -> None:
    sod = _sod()
    sod.loc[pd.Timestamp("2024-01-03"), 101] = 1500.0
    to = daily_turnover(sod)
    # Δ = 500 on GMV_prev = 2000 → 0.25
    assert to.loc[pd.Timestamp("2024-01-03")] == pytest.approx(0.25)


def test_path_metrics_known_series() -> None:
    # constant +1% daily → ann_ret ≈ 2.52, sharpe huge, maxDD 0
    r = pd.Series([0.01] * 10)
    m = path_metrics(r)
    assert m["ann_ret"] == pytest.approx(0.01 * 252)
    assert m["max_dd"] == pytest.approx(0.0)


def test_debug_perturb_post_equals_pre_minus_tcost(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    sod = _sod()
    sod.loc[pd.Timestamp("2024-01-03"), 101] = 1500.0
    sod.to_parquet(path)
    prices = _prices()

    monkeypatch.setattr(dbg, "pull_cost_prices", lambda *_a, **_k: prices)
    monkeypatch.setattr(combo_cost, "pull_cost_prices", lambda *_a, **_k: prices)

    result = run_debug_perturb(path, start=date(2024, 1, 2), end=date(2024, 1, 4))
    assert len(result.daily) >= 1
    assert result.total_post_pnl == pytest.approx(
        result.total_pre_pnl - result.total_tcost
    )
    assert list(result.pre_table.columns) == [
        "Year",
        "n_days",
        "medGMV",
        "Sharpe",
        "ret",
        "vol",
        "maxDD",
        "daily TO",
        "trd",
        "drag",
    ]
    text = format_debug_perturb(result)
    assert "Pre-tcost" in text
    assert "Post-tcost" in text
    assert "Sharpe" in text
    assert "trade PnL/GMV /day" in text
    assert "drag (cost/GMV)/day" in text
    assert "t-cost total" not in text
    assert "drag" in result.daily.columns
    assert "trade_pnl_gmv" in result.daily.columns
    assert "intraday_alpha" in result.daily.columns
    assert result.daily["intraday_alpha"].sum() == pytest.approx(
        result.daily["residual"].sum()
    )
    day = result.daily.iloc[0]
    if day["gmv"] > 0:
        assert day["drag"] == pytest.approx(day["tcost"] / day["gmv"])
        assert day["trade_pnl_gmv"] == pytest.approx(day["intraday_pnl"] / day["gmv"])

    out = write_perturb_artifacts(result, tmp_path / "perturb")
    assert (out / "README.md").is_file()
    assert (out / "pre_tcost_yearly.csv").is_file()
    assert (out / "post_tcost_yearly.csv").is_file()


def test_debug_perturb_moc_tcost_equals_commish(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "sod.parquet"
    sod = _sod()
    sod.loc[pd.Timestamp("2024-01-03"), 101] = 1500.0
    sod.to_parquet(path)
    prices = _prices()

    monkeypatch.setattr(dbg, "pull_cost_prices", lambda *_a, **_k: prices)
    monkeypatch.setattr(combo_cost, "pull_cost_prices", lambda *_a, **_k: prices)

    result = run_debug_perturb(
        path, start=date(2024, 1, 2), end=date(2024, 1, 4), fill="moc"
    )
    assert result.fill == "moc"
    assert float(result.cost.total_residual) == pytest.approx(0.0)
    assert result.total_tcost == pytest.approx(float(result.cost.total_commish))
    assert result.daily["intraday_alpha"].sum() == pytest.approx(0.0)
    assert result.total_post_pnl == pytest.approx(
        result.total_pre_pnl - float(result.cost.total_commish)
    )
    text = format_debug_perturb(result)
    assert "fill=MOC" in text
    assert "commish only" in text


def test_yearly_perf_has_footer() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2025-01-02", "2025-01-03"]),
            "pre_ret": [0.01, -0.005, 0.002, 0.003],
            "gmv": [1e8, 1e8, 1e8, 1e8],
            "daily_to": [0.2, 0.3, 0.25, 0.25],
        }
    )
    disp, num = yearly_perf_frame(daily, ret_col="pre_ret")
    assert "2024" in set(disp["Year"].astype(str)) or 2024 in set(disp["Year"])
    assert any("–" in str(y) or str(y) == "2024–25" for y in disp["Year"])
    assert len(num) == 3  # 2 years + pooled
