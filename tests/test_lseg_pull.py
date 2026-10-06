"""Synthetic Datastream2 H5 fixture + pull tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("h5py")
pytest.importorskip("numpy")
pytest.importorskip("pandas")

import h5py
import numpy as np
import pandas as pd

from tcost_engine.lseg.pull import pull_ohlcv, pull_top500, write_frame


def _ns(d: date) -> int:
    return int(np.datetime64(d.isoformat(), "D").astype("datetime64[ns]").astype(np.int64))


@pytest.fixture()
def tiny_ds2(tmp_path: Path) -> Path:
    """Minimal ds2_data.h5 with 3 dates × 3 INFOCODEs, TOP500 + OHLCV."""
    path = tmp_path / "ds2_data.h5"
    dates = [date(2015, 12, 31), date(2016, 1, 4), date(2016, 1, 5)]
    infocodes = np.array([100, 200, 300], dtype=np.int64)
    tickers = np.array(["AAA", "BBB", "CCC"], dtype=object)
    # TOP500: none on 2015-12-31; AAA+BBB on 2016-01-04; AAA on 2016-01-05
    top500 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, np.nan],
        ],
        dtype=np.float64,
    )
    close = np.array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
        ],
        dtype=np.float64,
    )
    close_adj = close * 2.0
    volume = close * 1000.0
    # ticker index codes into vocabulary
    ticker_index = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=np.int32,
    )

    with h5py.File(path, "w") as h5:
        h5.create_dataset("metadata/TICKERS", data=tickers.astype("S"))
        axis1 = np.array([_ns(d) for d in dates], dtype=np.int64)
        for name, values in (
            ("CLOSE", close),
            ("OPEN", close - 0.5),
            ("HIGH", close + 0.5),
            ("LOW", close - 1.0),
            ("VOLUME", volume),
            ("CLOSE_ADJUSTED", close_adj),
            ("OPEN_ADJUSTED", close_adj - 1.0),
            ("HIGH_ADJUSTED", close_adj + 1.0),
            ("LOW_ADJUSTED", close_adj - 2.0),
            ("VOLUME_ADJUSTED", volume * 0.5),
            ("TOP500", top500),
            ("TICKER_INDEX", ticker_index.astype(np.float64)),
        ):
            g = h5.create_group(f"ds2_data/{name}")
            g.create_dataset("axis0", data=infocodes)
            g.create_dataset("axis1", data=axis1)
            g.create_dataset("block0_items", data=infocodes)
            g.create_dataset("block0_values", data=values)
    return path


def test_pull_top500_from_2016(tiny_ds2: Path) -> None:
    df = pull_top500(tiny_ds2, start=date(2016, 1, 1))
    assert list(df.columns) == ["marketdate", "infocode", "top500", "ticker"]
    assert len(df) == 3  # AAA+BBB on Jan 4, AAA on Jan 5
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert df["top500"].eq(1).all()
    assert df["marketdate"].min() == pd.Timestamp("2016-01-04")


def test_pull_ohlcv_both_scales_top500(tiny_ds2: Path, tmp_path: Path) -> None:
    df = pull_ohlcv(
        tiny_ds2,
        start=date(2016, 1, 1),
        adjustment="both",
        universe="top500",
    )
    assert len(df) == 3
    assert {"open", "close", "volume", "open_adjusted", "close_adjusted", "volume_adjusted"} <= set(
        df.columns
    )
    aaa = df[df["ticker"] == "AAA"].sort_values("marketdate")
    assert list(aaa["close"]) == [11.0, 12.0]
    assert list(aaa["close_adjusted"]) == [22.0, 24.0]

    out = write_frame(df, tmp_path / "ohlcv.csv")
    assert out.is_file()
    roundtrip = pd.read_csv(out)
    assert len(roundtrip) == 3


def test_pull_ohlcv_unadjusted_all(tiny_ds2: Path) -> None:
    df = pull_ohlcv(
        tiny_ds2,
        start=date(2015, 1, 1),
        adjustment="unadjusted",
        universe="all",
    )
    assert len(df) == 9  # 3 dates × 3 names
    assert "close_adjusted" not in df.columns
    assert "close" in df.columns
