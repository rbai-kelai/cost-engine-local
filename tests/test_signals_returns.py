"""Tests for close_lookback, ret, and TOP500 cross-sectionalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcost_engine.signals import (
    add_ret_signal,
    close_lookback,
    cross_sectionalize,
    long_short_book,
    portfolio_returns,
    ret_signal,
)


def test_close_lookback_series_is_shift() -> None:
    px = pd.Series([10.0, 11.0, 12.5], index=pd.RangeIndex(3))
    assert close_lookback(px).tolist() == [10.0, 11.0, 12.5]
    assert np.isnan(close_lookback(px, window=1).iloc[0])
    assert close_lookback(px, window=1).iloc[1:].tolist() == [10.0, 11.0]
    assert close_lookback(px, window=2).iloc[2] == 10.0


def test_close_lookback_rejects_negative_window() -> None:
    with pytest.raises(ValueError, match="window"):
        close_lookback(pd.Series([1.0]), window=-1)


def test_ret_signal_series_is_today_over_yday() -> None:
    px = pd.Series([100.0, 102.0, 99.0])
    got = ret_signal(px)
    assert np.isnan(got.iloc[0])
    assert got.iloc[1] == pytest.approx(102.0 / 100.0)
    assert got.iloc[2] == pytest.approx(99.0 / 102.0)


def test_ret_signal_panel_uses_adjusted_close_per_name() -> None:
    df = pd.DataFrame(
        {
            "marketdate": pd.to_datetime(
                ["2016-01-04", "2016-01-05", "2016-01-06"] * 2
            ),
            "infocode": [1, 1, 1, 2, 2, 2],
            "close_adjusted": [10.0, 11.0, 10.0, 50.0, 50.0, 55.0],
            "close": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "top500": [1, 1, 1, 1, 1, 1],
        }
    )
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    out = add_ret_signal(df, cross_section=False, book=False)
    a = out[out["infocode"] == 1].sort_values("marketdate")
    b = out[out["infocode"] == 2].sort_values("marketdate")

    assert np.isnan(a["ret"].iloc[0])
    assert a["ret"].iloc[1] == pytest.approx(11.0 / 10.0)
    assert a["ret"].iloc[2] == pytest.approx(10.0 / 11.0)
    assert b["ret"].iloc[1] == pytest.approx(1.0)
    assert b["ret"].iloc[2] == pytest.approx(55.0 / 50.0)


def test_close_lookback_panel() -> None:
    df = pd.DataFrame(
        {
            "marketdate": pd.to_datetime(["2016-01-04", "2016-01-05", "2016-01-06"]),
            "infocode": [7, 7, 7],
            "close_adjusted": [10.0, 20.0, 30.0],
        }
    )
    yday = close_lookback(df, window=1)
    assert np.isnan(yday.iloc[0])
    assert yday.iloc[1] == 10.0
    assert yday.iloc[2] == 20.0
    assert close_lookback(df).tolist() == [10.0, 20.0, 30.0]


def test_cross_sectionalize_ret_only_in_top500() -> None:
    # One date, three names: two in TOP500 with rets 1.0 and 1.2, one outside.
    df = pd.DataFrame(
        {
            "marketdate": pd.to_datetime(["2016-01-05"] * 3),
            "infocode": [1, 2, 3],
            "ret": [1.0, 1.2, 9.0],
            "top500": [1, 1, 0],
        }
    )
    cs = cross_sectionalize(df)
    # Outside universe stays NaN even with an extreme raw ret.
    assert np.isnan(cs.iloc[2])
    # Within TOP500: pct ranks → 0.5 and 1.0 (average method, n=2)
    assert cs.iloc[0] == pytest.approx(0.5)
    assert cs.iloc[1] == pytest.approx(1.0)
    assert cs.dropna().between(0.0, 1.0).all()


def test_add_ret_signal_cross_sections_by_default() -> None:
    dates = pd.to_datetime(["2016-01-04", "2016-01-05"])
    df = pd.DataFrame(
        {
            "marketdate": [dates[0], dates[1], dates[0], dates[1], dates[1]],
            "infocode": [1, 1, 2, 2, 3],
            "close_adjusted": [10.0, 11.0, 20.0, 20.0, 30.0],
            "top500": [1, 1, 1, 1, 0],
        }
    )
    out = add_ret_signal(df, book=False)
    assert "ret" in out.columns and "ret_cs" in out.columns
    day = out[out["marketdate"] == dates[1]].set_index("infocode")
    # rets: name1=1.1, name2=1.0, name3=NaN (no yday) and out of univ anyway
    assert day.loc[1, "ret"] == pytest.approx(1.1)
    assert day.loc[2, "ret"] == pytest.approx(1.0)
    assert np.isnan(day.loc[3, "ret_cs"])
    assert day.loc[1, "ret_cs"] == pytest.approx(1.0)  # higher ret
    assert day.loc[2, "ret_cs"] == pytest.approx(0.5)
    assert day.loc[[1, 2], "ret_cs"].between(0.0, 1.0).all()


def test_long_short_book_from_ret_cs() -> None:
    df = pd.DataFrame(
        {
            "marketdate": pd.to_datetime(["2016-01-05"] * 4),
            "infocode": [1, 2, 3, 4],
            "ret_cs": [0.25, 0.5, 0.75, 1.0],
            "top500": [1, 1, 1, 1],
        }
    )
    w = long_short_book(df)
    # raw = ret_cs; mean = 0.625; centered = [-0.375, -0.125, 0.125, 0.375]
    # long sum = 0.5 → weights 0.25, 0.75; short sum = -0.5 → weights -0.75, -0.25
    assert w.iloc[0] == pytest.approx(-0.75)
    assert w.iloc[1] == pytest.approx(-0.25)
    assert w.iloc[2] == pytest.approx(0.25)
    assert w.iloc[3] == pytest.approx(0.75)
    assert w.sum() == pytest.approx(0.0)
    assert w[w > 0].sum() == pytest.approx(1.0)
    assert w[w < 0].sum() == pytest.approx(-1.0)


def test_add_ret_signal_builds_daily_book() -> None:
    dates = pd.to_datetime(["2016-01-04", "2016-01-05"])
    # four names so demeaned ranks have both a long and a short leg
    rows = []
    for d in dates:
        for i, px in enumerate([10.0, 11.0, 12.0, 13.0], start=1):
            # day1 prices; day2 move so rets differ
            close = px if d == dates[0] else px * (1.0 + 0.01 * i)
            rows.append(
                {
                    "marketdate": d,
                    "infocode": i,
                    "close_adjusted": close,
                    "top500": 1,
                }
            )
    out = add_ret_signal(pd.DataFrame(rows))
    assert "weight" in out.columns
    day = out[out["marketdate"] == dates[1]]
    assert day["weight"].sum() == pytest.approx(0.0)
    assert day.loc[day["weight"] > 0, "weight"].sum() == pytest.approx(1.0)
    assert day.loc[day["weight"] < 0, "weight"].sum() == pytest.approx(-1.0)


def test_portfolio_returns_lags_weights() -> None:
    """Enter at close t with knowledge of ret(t); PnL is close(t+1)/close(t)-1."""
    dates = pd.to_datetime(["2016-01-04", "2016-01-05", "2016-01-06"])
    # two names; known weights on day0/day1 and simple rets on day1/day2
    df = pd.DataFrame(
        {
            "marketdate": dates.tolist() * 2,
            "infocode": [1, 1, 1, 2, 2, 2],
            "ret": [1.0, 1.10, 1.05, 1.0, 0.90, 1.02],
            "weight": [0.5, 0.5, 0.0, -0.5, -0.5, 0.0],
        }
    )
    # Book formed 01-04 earns 01-05 PnL: 0.5*(1.10-1) + (-0.5)*(0.90-1) = 0.10
    # Book formed 01-05 earns 01-06 PnL: 0.5*(1.05-1) + (-0.5)*(1.02-1) = 0.015
    port = portfolio_returns(df)
    assert port.loc[dates[0]] == pytest.approx(0.0)  # no prior book yet
    assert port.loc[dates[1]] == pytest.approx(0.10)
    assert port.loc[dates[2]] == pytest.approx(0.015)