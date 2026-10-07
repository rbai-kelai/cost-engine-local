"""High-level LSEG pulls: OHLCV (adj/unadj) and TOP500 constituents."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from tcost_engine.lseg.ds2 import (
    COST_PRICE_FIELDS,
    DS2_NAMESPACE,
    OHLCV_ADJUSTED,
    OHLCV_UNADJUSTED,
    as_python_date,
    date_slice,
    load_panel_index,
    open_h5,
    read_field,
    ticker_matrix,
)
from tcost_engine.lseg.paths import resolve_ds2_h5

Adjustment = Literal["unadjusted", "adjusted", "both"]
UniverseFilter = Literal["all", "top500"]


def pull_top500(
    h5_path: str | Path | None = None,
    *,
    start: date = date(2016, 1, 1),
    end: date | None = None,
    cache_dir: str | Path | None = None,
    include_ticker: bool = True,
):
    """Point-in-time TOP500 membership as a long DataFrame.

    Columns: ``marketdate``, ``infocode``, ``top500`` (=1), and ``ticker`` when
    the H5 carries ``TICKER_INDEX`` / ``metadata/TICKERS``.
    """
    import numpy as np
    import pandas as pd

    path = resolve_ds2_h5(h5_path, cache_dir=cache_dir)
    with open_h5(path) as h5:
        index = load_panel_index(h5)
        rows = date_slice(index.dates, start=start, end=end)
        flags = read_field(h5, "TOP500", rows)
        tickers = ticker_matrix(h5, rows, index) if include_ticker else None

    dates = index.dates[rows]
    # members only
    mask = flags == 1
    date_ix, id_ix = np.nonzero(mask)
    out = {
        "marketdate": pd.to_datetime(dates[date_ix]),
        "infocode": index.infocodes[id_ix].astype(np.int64),
        "top500": np.ones(date_ix.shape[0], dtype=np.int8),
    }
    if tickers is not None:
        out["ticker"] = tickers[date_ix, id_ix]
    return pd.DataFrame(out)


def pull_ohlcv(
    h5_path: str | Path | None = None,
    *,
    start: date = date(2016, 1, 1),
    end: date | None = None,
    adjustment: Adjustment = "both",
    universe: UniverseFilter = "top500",
    cache_dir: str | Path | None = None,
    include_ticker: bool = True,
):
    """Daily OHLCV in long form.

    Default mirrors a research pull off the AWS box: dates from *start*
    (2016-01-01) onward, restricted to names in ``TOP500`` that day, with both
    unadjusted and adjusted OHLCV columns.
    """
    import numpy as np
    import pandas as pd

    fields = _ohlcv_fields(adjustment)
    path = resolve_ds2_h5(h5_path, cache_dir=cache_dir)
    with open_h5(path) as h5:
        index = load_panel_index(h5)
        rows = date_slice(index.dates, start=start, end=end)
        panels = {field: read_field(h5, field, rows) for field in fields}
        top = read_field(h5, "TOP500", rows) if universe == "top500" else None
        tickers = ticker_matrix(h5, rows, index) if include_ticker else None

    dates = index.dates[rows]

    if universe == "top500":
        assert top is not None
        mask = top == 1
    else:
        # any finite close (prefer adjusted close when present)
        close_key = "CLOSE_ADJUSTED" if "CLOSE_ADJUSTED" in panels else "CLOSE"
        close = panels[close_key]
        mask = np.isfinite(close)

    date_ix, id_ix = np.nonzero(mask)
    out: dict[str, object] = {
        "marketdate": pd.to_datetime(dates[date_ix]),
        "infocode": index.infocodes[id_ix].astype(np.int64),
    }
    if tickers is not None:
        out["ticker"] = tickers[date_ix, id_ix]
    for field, values in panels.items():
        col = _column_name(field)
        out[col] = values[date_ix, id_ix].astype(np.float64)
    if universe == "top500":
        out["top500"] = np.ones(date_ix.shape[0], dtype=np.int8)
    return pd.DataFrame(out)


def pull_cost_prices(
    h5_path: str | Path | None = None,
    *,
    start: date = date(2016, 1, 1),
    end: date | None = None,
    infocodes: object | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
):
    """Long frame of BID / ASK / VWAP / CLOSE [/ CLOSE_ADJUSTED] for t-cost joins.

    Columns: ``marketdate``, ``infocode``, ``bid``, ``ask``, ``vwap``, ``close``,
    and ``close_adjusted`` when present in the H5. Rows keep names with a finite
    close that day. Pass *infocodes* (iterable of ints) to restrict columns to a
    SOD universe.

    Results are cached under ``~/.cache/tcost-engine/prices/`` (or *cache_dir*)
    keyed by H5 mtime, date range, and INFOCODE set. Pass ``refresh=True`` to
    bypass the cache.
    """
    import numpy as np
    import pandas as pd

    from tcost_engine.cacheutil import (
        default_cache_root,
        file_stamp,
        fingerprint,
        read_parquet_if_fresh,
        write_parquet_cache,
    )

    path = resolve_ds2_h5(h5_path, cache_dir=cache_dir)
    h5_key, h5_mtime = file_stamp(path)
    ids_key = (
        "ALL"
        if infocodes is None
        else fingerprint(tuple(sorted({int(x) for x in infocodes})))
    )
    end_key = "" if end is None else end.isoformat()
    meta = {
        "h5": h5_key,
        "h5_mtime": h5_mtime,
        "start": start.isoformat(),
        "end": end_key,
        "ids": ids_key,
        "fields": list(COST_PRICE_FIELDS),
    }
    fp = fingerprint(meta["h5"], meta["h5_mtime"], meta["start"], meta["end"], meta["ids"])
    cache_root = default_cache_root(cache_dir) / "prices"
    cache_path = cache_root / f"cost_prices_{fp}.parquet"
    meta_path = cache_root / f"cost_prices_{fp}.json"
    if not refresh:
        cached = read_parquet_if_fresh(cache_path, meta_path=meta_path, expected=meta)
        if cached is not None:
            print(f"price cache hit: {cache_path}", flush=True)
            return cached

    print(f"reading cost prices from {path} …", flush=True)
    with open_h5(path) as h5:
        index = load_panel_index(h5)
        rows = date_slice(index.dates, start=start, end=end)
        panels = {}
        for field in COST_PRICE_FIELDS:
            key = f"{DS2_NAMESPACE}/{field}"
            if key in h5:
                panels[field] = read_field(h5, field, rows)

    if "CLOSE" not in panels:
        raise KeyError("ds2 H5 missing CLOSE panel")

    dates = index.dates[rows]
    all_ids = index.infocodes.astype(np.int64)
    if infocodes is None:
        col_ix = np.arange(len(all_ids))
    else:
        wanted = np.asarray(sorted({int(x) for x in infocodes}), dtype=np.int64)
        col_ix = np.flatnonzero(np.isin(all_ids, wanted))
        if col_ix.size == 0:
            return pd.DataFrame(
                columns=[
                    "marketdate",
                    "infocode",
                    "bid",
                    "ask",
                    "vwap",
                    "close",
                    "close_adjusted",
                ]
            )

    close = panels["CLOSE"][:, col_ix]
    mask = np.isfinite(close)
    date_ix, local_ix = np.nonzero(mask)
    id_ix = col_ix[local_ix]
    out: dict[str, object] = {
        "marketdate": pd.to_datetime(dates[date_ix]),
        "infocode": all_ids[id_ix],
        "close": close[date_ix, local_ix].astype(np.float64),
    }
    for field, col in (("BID", "bid"), ("ASK", "ask"), ("VWAP", "vwap")):
        if field in panels:
            out[col] = panels[field][:, col_ix][date_ix, local_ix].astype(np.float64)
    if "CLOSE_ADJUSTED" in panels:
        out["close_adjusted"] = panels["CLOSE_ADJUSTED"][:, col_ix][
            date_ix, local_ix
        ].astype(np.float64)
    df = pd.DataFrame(out)
    try:
        write_parquet_cache(df, cache_path, meta_path=meta_path, meta=meta)
        print(f"wrote price cache: {cache_path} ({len(df):,} rows)", flush=True)
    except (OSError, ImportError, ValueError) as exc:
        print(f"WARNING: could not write price cache ({exc})", flush=True)
    return df


def write_frame(df, path: str | Path) -> Path:
    """Write a DataFrame to ``.parquet`` or ``.csv`` based on suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix in {".csv", ".txt"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output suffix {suffix!r}; use .parquet or .csv")
    return path


def _ohlcv_fields(adjustment: Adjustment) -> tuple[str, ...]:
    if adjustment == "unadjusted":
        return OHLCV_UNADJUSTED
    if adjustment == "adjusted":
        return OHLCV_ADJUSTED
    if adjustment == "both":
        return OHLCV_UNADJUSTED + OHLCV_ADJUSTED
    raise ValueError(f"unknown adjustment={adjustment!r}")


def _column_name(field: str) -> str:
    return field.lower()


def summarize_pull(df, *, label: str) -> str:
    if df.empty:
        return f"{label}: empty"
    dates = df["marketdate"]
    n_ids = df["infocode"].nunique()
    extra = ""
    if "ticker" in df.columns:
        extra = f", tickers={df['ticker'].nunique()}"
    return (
        f"{label}: rows={len(df):,} dates="
        f"{as_python_date(dates.min().date())}→{as_python_date(dates.max().date())} "
        f"infocodes={n_ids}{extra}"
    )
