"""Pre-tcost vs post-tcost debug_perturb (mosek-style tables)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import tcost_engine.combo.cost as combo_cost
import tcost_engine.combo.debug_perturb as dbg
from tcost_engine.combo.debug_perturb import (
    MOC_FILL_LABEL,
    VWAP_FILL_LABEL,
    format_debug_perturb,
    result_summary,
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
    for day, c1, c2 in [
        ("2024-01-02", 10.0, 20.0),
        ("2024-01-03", 11.0, 22.0),
        ("2024-01-04", 11.0, 22.0),
    ]:
        for infocode, close in ((101, c1), (202, c2)):
            rows.append(
                {
                    "marketdate": pd.Timestamp(day),
                    "infocode": infocode,
                    "bid": close - 0.01,
                    "ask": close + 0.01,
                    "vwap": close,
                    "close": close,
                    "close_adjusted": close,
                }
            )
    return pd.DataFrame(rows)


def _run(monkeypatch, tmp_path: Path, **kwargs):
    path = tmp_path / "sod.parquet"
    sod = _sod()
    sod.loc[pd.Timestamp("2024-01-03"), 101] = 1500.0
    sod.loc[pd.Timestamp("2024-01-04"), 101] = 1500.0
    sod.to_parquet(path)
    prices = _prices()
    monkeypatch.setattr(dbg, "pull_cost_prices", lambda *_a, **_k: prices)
    monkeypatch.setattr(combo_cost, "pull_cost_prices", lambda *_a, **_k: prices)
    return run_debug_perturb(
        path, start=date(2024, 1, 2), end=date(2024, 1, 4), **kwargs
    )


def test_lag1_pnl_matches_hand_calc() -> None:
    pnl = lag1_dollar_pnl(_sod(), _prices())
    day = pnl.set_index("date").loc[pd.Timestamp("2024-01-03")]
    assert day["pre_pnl"] == pytest.approx(0.0)
    assert day["gmv"] == pytest.approx(2000.0)


def test_daily_turnover_two_way() -> None:
    sod = _sod()
    sod.loc[pd.Timestamp("2024-01-03"), 101] = 1500.0
    # Flat closes → share-based TO matches raw dollar TO
    to = daily_turnover(sod, _prices())
    assert to.loc[pd.Timestamp("2024-01-03")] == pytest.approx(0.25)


def test_path_metrics_known_series() -> None:
    r = pd.Series([0.01] * 10)
    m = path_metrics(r)
    assert m["ann_ret"] == pytest.approx(0.01 * 252)


def test_debug_perturb_reports_both_fill_headers(monkeypatch, tmp_path: Path) -> None:
    result = _run(monkeypatch, tmp_path)
    assert result.moc.label == MOC_FILL_LABEL
    assert result.vwap.label == VWAP_FILL_LABEL
    assert result.moc.total_post_pnl == pytest.approx(
        result.total_pre_pnl - result.moc.total_tcost
    )
    assert result.vwap.total_post_pnl == pytest.approx(
        result.total_pre_pnl - result.vwap.total_tcost
    )
    text = format_debug_perturb(result)
    assert "## MOC fill" in text
    assert "## VWAP fill" in text
    assert "share-based Δn" in text
    assert f"mils={result.mils}" in text
    out = write_perturb_artifacts(result, tmp_path / "perturb")
    assert (out / "README.md").is_file()
    assert (out / "moc_post_tcost_yearly.csv").is_file()
    assert (out / "vwap_post_tcost_yearly.csv").is_file()
    assert (out / "pre_tcost_yearly.csv").is_file()


def test_debug_perturb_moc_is_commish_only(monkeypatch, tmp_path: Path) -> None:
    result = _run(monkeypatch, tmp_path)
    assert result.moc.total_intraday_slippage == pytest.approx(0.0)
    assert result.moc.total_spread == pytest.approx(0.0)
    assert result.moc.total_tcost == pytest.approx(result.moc.total_commish)
    assert result.daily["moc_tcost"].sum() == pytest.approx(
        result.daily["commish"].sum()
    )


def test_debug_perturb_tcost_same_day_as_trade(monkeypatch, tmp_path: Path) -> None:
    result = _run(monkeypatch, tmp_path)
    by = result.daily.set_index("date")
    assert by.loc[pd.Timestamp("2024-01-03"), "moc_tcost"] > 0
    assert by.loc[pd.Timestamp("2024-01-03"), "vwap_tcost"] > 0
    assert by.loc[pd.Timestamp("2024-01-04"), "moc_tcost"] == pytest.approx(0.0)
    assert by.loc[pd.Timestamp("2024-01-04"), "vwap_tcost"] == pytest.approx(0.0)


def test_debug_perturb_vwap_is_commish_plus_spread_plus_slippage(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run(monkeypatch, tmp_path)
    assert result.daily["vwap_tcost"].sum() == pytest.approx(
        (
            result.daily["commish"]
            + result.daily["spread"]
            + result.daily["intraday_slippage"]
        ).sum()
    )
    text = format_debug_perturb(result)
    assert "half-spread" in text
    assert "\nspread" in text or any(
        line.strip().startswith("spread") for line in text.splitlines()
    )


def test_debug_perturb_no_spread_affects_vwap_only(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run(monkeypatch, tmp_path, include_spread=False)
    assert float(result.cost.total_spread) == pytest.approx(0.0)
    assert result.vwap.total_spread == pytest.approx(0.0)
    assert result.vwap.total_tcost == pytest.approx(
        result.vwap.total_commish + result.vwap.total_intraday_slippage
    )
    # MOC unchanged: still commish only
    assert result.moc.total_tcost == pytest.approx(result.moc.total_commish)


def test_debug_perturb_json_summary_has_both_scenarios(
    monkeypatch, tmp_path: Path
) -> None:
    result = _run(monkeypatch, tmp_path)
    summary = result_summary(result)
    assert summary["moc_fill"]["label"] == MOC_FILL_LABEL
    assert summary["vwap_fill"]["label"] == VWAP_FILL_LABEL
    assert summary["moc_fill"]["spread"] == 0.0
    assert summary["vwap_fill"]["tcost"] == pytest.approx(result.vwap.total_tcost)


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
    assert len(num) == 3
