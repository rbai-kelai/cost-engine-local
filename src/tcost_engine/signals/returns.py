"""Price lookbacks and return signals from adjusted closes."""

from __future__ import annotations

from typing import Hashable

import pandas as pd


def close_lookback(
    df: pd.DataFrame | pd.Series,
    window: int = 0,
    *,
    price_col: str = "close_adjusted",
    by: Hashable | list[Hashable] = "infocode",
) -> pd.Series:
    """Lagged close: ``close_lookback(df, window=w) = close.shift(w)``.

    ``window`` defaults to ``0`` (today); ``window=1`` is yesterday. For a long
    panel, lags are within each *by* group after sorting by ``marketdate`` when
    that column is present. Pass a single-name price *Series* to skip grouping.
    """
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")

    if isinstance(df, pd.Series):
        return _shift(df, window)

    if price_col not in df.columns:
        raise KeyError(
            f"missing price column {price_col!r}; "
            "pull adjusted OHLCV or pass price_col="
        )
    ordered = _ordered(df, by=by)
    return ordered.groupby(by, sort=False)[price_col].transform(
        lambda s: _shift(s, window)
    )


def ret_signal(
    df: pd.DataFrame | pd.Series,
    *,
    price_col: str = "close_adjusted",
    by: Hashable | list[Hashable] = "infocode",
) -> pd.Series:
    """Gross daily return: today's close / yesterday's close on adjusted px.

    ``ret = close_lookback(df) / close_lookback(df, window=1)``.
    """
    today = close_lookback(df, price_col=price_col, by=by)  # window=0
    yday = close_lookback(df, window=1, price_col=price_col, by=by)
    return today / yday


def cross_sectionalize(
    df: pd.DataFrame,
    col: str = "ret",
    *,
    by: Hashable = "marketdate",
    universe_col: str = "top500",
    out_col: str = "ret_cs",
    min_count: int = 2,
) -> pd.Series:
    """Cross-sectional percentile rank of *col*, only among TOP500 names.

    On each *by* date (default ``marketdate``), keep rows with
    ``universe_col == 1``, then ``Series.rank(method="average", pct=True)`` so
    values lie in ``(0, 1]``. Names outside the universe, or dates with fewer
    than *min_count* finite values, get NaN.
    """
    if col not in df.columns:
        raise KeyError(f"missing signal column {col!r}")
    if by not in df.columns:
        raise KeyError(f"missing date column {by!r}")
    if universe_col not in df.columns:
        raise KeyError(
            f"missing universe column {universe_col!r}; "
            "pull OHLCV with universe='top500' or pass universe_col="
        )

    in_univ = df[universe_col] == 1
    x = df[col].where(in_univ)

    def _pct_rank(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < min_count:
            return pd.Series(float("nan"), index=s.index)
        return s.rank(method="average", pct=True)

    return x.groupby(df[by], sort=False).transform(_pct_rank).rename(out_col)


def long_short_book(
    df: pd.DataFrame,
    signal_col: str = "ret_cs",
    *,
    by: Hashable = "marketdate",
    out_col: str = "weight",
    min_count: int = 2,
) -> pd.Series:
    """Daily long/short weights from *signal_col* (default ``ret_cs``).

    Per *by* date:

    1. Take finite signal values as raw weights (``weight = ret_cs``).
    2. Demean within the day so the book is dollar-neutral.
    3. Rescale so the long leg sums to ``+1`` and the short leg to ``-1``.

    Names with missing signal (e.g. outside TOP500) get NaN. Days with no
    long or no short after demeaning, or fewer than *min_count* names, are NaN.
    """
    if signal_col not in df.columns:
        raise KeyError(f"missing signal column {signal_col!r}")
    if by not in df.columns:
        raise KeyError(f"missing date column {by!r}")

    def _day_weights(s: pd.Series) -> pd.Series:
        out = pd.Series(float("nan"), index=s.index, dtype=float)
        mask = s.notna()
        if int(mask.sum()) < min_count:
            return out
        raw = s.loc[mask]
        centered = raw - raw.mean()
        long = centered[centered > 0]
        short = centered[centered < 0]
        if long.empty or short.empty:
            return out
        w = centered.copy()
        w.loc[centered > 0] = centered.loc[centered > 0] / long.sum()
        w.loc[centered < 0] = centered.loc[centered < 0] / (-short.sum())
        w.loc[centered == 0] = 0.0
        out.loc[mask] = w
        return out

    return (
        df[signal_col]
        .groupby(df[by], sort=False)
        .transform(_day_weights)
        .rename(out_col)
    )


def portfolio_returns(
    df: pd.DataFrame,
    *,
    weight_col: str = "weight",
    ret_col: str = "ret",
    by: Hashable = "marketdate",
    name_col: str = "infocode",
    lag_weights: int = 1,
) -> pd.Series:
    """Daily long/short portfolio simple returns.

    Timing (default ``lag_weights=1``):

    - Form the book at **today's close** with full knowledge of today's
      ``ret = close(t) / close(t-1)`` (and thus ``ret_cs`` / ``weight``).
    - PnL is the **next** day's close-to-close simple return:
      ``R_i(t+1) = close(t+1) / close(t) - 1 = ret_i(t+1) - 1``.
    - Portfolio return dated on the PnL day:
      ``port_ret(t) = Σ_i weight_i(t-1) × (ret_i(t) - 1)``.

    Missing weights or stock returns contribute zero. Pass ``lag_weights=0``
    only for same-day attribution (look-ahead).
    """
    if weight_col not in df.columns:
        raise KeyError(f"missing weight column {weight_col!r}")
    if ret_col not in df.columns:
        raise KeyError(f"missing return column {ret_col!r}")
    if by not in df.columns:
        raise KeyError(f"missing date column {by!r}")
    if name_col not in df.columns:
        raise KeyError(f"missing name column {name_col!r}")
    if lag_weights < 0:
        raise ValueError(f"lag_weights must be >= 0, got {lag_weights}")

    ordered = df.sort_values([name_col, by], kind="mergesort")
    w = ordered.groupby(name_col, sort=False)[weight_col].shift(lag_weights)
    simple = ordered[ret_col] - 1.0
    contrib = w.fillna(0.0) * simple.fillna(0.0)
    port = contrib.groupby(ordered[by], sort=True).sum()
    port.name = "port_ret"
    return port


def add_ret_signal(
    df: pd.DataFrame,
    *,
    price_col: str = "close_adjusted",
    by: Hashable | list[Hashable] = "infocode",
    out_col: str = "ret",
    cross_section: bool = True,
    cs_out_col: str = "ret_cs",
    universe_col: str = "top500",
    book: bool = True,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Add ``ret``, TOP500 ``ret_cs``, and a daily long/short ``weight`` book.

    ``ret = close_lookback(df) / close_lookback(df, window=1)`` on adjusted
    prices. When *cross_section* is true (default), percentile-rank ``ret``
    within each ``marketdate`` among ``universe_col == 1`` names (``(0, 1]``).
    When *book* is true (default), set ``weight`` from ``ret_cs`` via
    :func:`long_short_book` (demeaned, long ``+1`` / short ``-1``).

    Portfolio P&L is :func:`portfolio_returns` (lagged weights × simple stock
    returns), not a column on this frame.
    """
    ordered = _ordered(df, by=by)
    out = ordered.copy()
    out[out_col] = ret_signal(out, price_col=price_col, by=by)
    if cross_section:
        out[cs_out_col] = cross_sectionalize(
            out, col=out_col, universe_col=universe_col, out_col=cs_out_col
        )
        if book:
            out[weight_col] = long_short_book(out, signal_col=cs_out_col, out_col=weight_col)
    elif book:
        raise ValueError("book=True requires cross_section=True (weights come from ret_cs)")
    return out


def _shift(px: pd.Series, window: int) -> pd.Series:
    if window == 0:
        return px.copy()
    return px.shift(window)


def _ordered(df: pd.DataFrame, *, by: Hashable | list[Hashable]) -> pd.DataFrame:
    keys = [by] if isinstance(by, str) else list(by)
    if "marketdate" in df.columns:
        return df.sort_values([*keys, "marketdate"], kind="mergesort")
    return df
