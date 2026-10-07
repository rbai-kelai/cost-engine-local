"""Load combo SOD dollar panels and turn DoD deltas into t-cost fills."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from tcost_engine.s3util import ensure_local_s3
from tcost_engine.types import Fill, FullSpreadBps, Side


@dataclass(frozen=True)
class FillBuildStats:
    """How many trade rows became fills vs were dropped."""

    n_trades: int
    n_fills: int
    n_dropped: int


def load_sod_panel(path: str | Path, *, cache_dir: str | Path | None = None):
    """Wide SOD panel: DatetimeIndex ``date`` × INFOCODE columns (dollar notionals).

    *path* may be a local parquet or an ``s3://`` URL pulled via ``aws s3 cp``
    (same pattern as pm-risk ``ensure_local_s3_parquet``).
    """
    import pandas as pd

    local = _resolve_path(path, cache_dir=cache_dir)
    df = pd.read_parquet(local)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            raise ValueError(
                f"SOD parquet needs a DatetimeIndex or a 'date' column; got index "
                f"{type(df.index).__name__} and columns {list(df.columns)[:8]}"
            )
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    df.columns = [int(c) for c in df.columns]
    df = df.sort_index()
    return df.astype("float64")


def rebalance_dollars(panel, prices):
    """Wide adjusted-share Δn and rebalance dollars δ$ = Δn_adj × close_adj_t.

    SOD stores unheld names as **0** (not NaN). Zero dollars → zero shares even
    when ``close_adj`` is missing, so entries after a no-price prior day still
    produce Δn when day-t prices exist::

        shares = 0 if SOD == 0 else SOD / close_adj
        Δn_adj_t = shares_t − shares_{t−1}
        δ$_t     = Δn_adj_t × close_adj_t

    Returns ``(delta_shares_adj, delta_dollars)`` wide frames for
    ``panel.iloc[1:]`` (first SOD day has no prior book).
    """
    import numpy as np
    import pandas as pd

    if panel is None or panel.empty or len(panel.index) < 2:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return empty, empty
    if prices is None or len(prices) == 0:
        raise ValueError("rebalance_dollars requires a prices frame (for close_adj)")

    px = prices.copy()
    px["marketdate"] = pd.to_datetime(px["marketdate"]).dt.normalize()
    adj_col = "close_adjusted" if "close_adjusted" in px.columns else "close"
    if adj_col not in px.columns:
        raise KeyError("prices need 'close_adjusted' or 'close'")

    adj = (
        px.pivot_table(
            index="marketdate", columns="infocode", values=adj_col, aggfunc="last"
        )
        .sort_index()
    )
    ids = [int(c) for c in panel.columns]
    sod = panel.sort_index().copy()
    sod.columns = ids
    adj_w = adj.reindex(columns=ids).reindex(sod.index)
    sod_arr = sod.to_numpy(dtype=np.float64)
    adj_arr = adj_w.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Unheld (SOD==0) → 0 shares; do not propagate NaN from missing adj.
        shares_arr = np.where(sod_arr == 0, 0.0, sod_arr / adj_arr)
    shares = pd.DataFrame(shares_arr, index=sod.index, columns=ids)
    dn = (shares - shares.shift(1)).iloc[1:]
    d_dollars = dn * adj_w.reindex(dn.index)
    return dn, d_dollars


def sod_trades(panel, prices) -> tuple[object, FillBuildStats]:
    """Share-based day-over-day trades from a dollar SOD panel + LSEG closes.

    Uses ``rebalance_dollars`` (δ$ = Δn_adj × close_adj_t). Real shares for
    costing: ``qty = |δ$| / close_t`` in ``join_trades_prices``.

    Name-days with a **held** book (nonzero SOD on t or t−1) but missing
    prices are counted as dropped. Never-held zeros are not trades and not
    drops. Returns ``(trades, FillBuildStats)``.
    """
    import numpy as np
    import pandas as pd

    empty = pd.DataFrame(
        columns=["date", "infocode", "side", "delta_shares_adj", "delta_dollars"]
    )
    if panel is None or panel.empty or len(panel.index) < 2:
        return empty, FillBuildStats(n_trades=0, n_fills=0, n_dropped=0)
    if prices is None or len(prices) == 0:
        raise ValueError("sod_trades requires a prices frame (for close_adj)")

    sod = panel.sort_index().copy()
    sod.columns = [int(c) for c in sod.columns]
    dn, d_dollars = rebalance_dollars(sod, prices)

    sod_t = sod.iloc[1:]
    sod_prev = sod.shift(1).iloc[1:]
    # Unheld is 0, not NaN — only nonzero SOD counts as a book.
    has_book = (sod_t.fillna(0.0) != 0) | (sod_prev.fillna(0.0) != 0)
    missing_px = has_book & dn.isna()
    trade_ok = dn.notna() & (dn != 0)

    n_dropped = int(missing_px.to_numpy().sum())
    n_fills = int(trade_ok.to_numpy().sum())

    long_dn = dn.where(trade_ok).stack(future_stack=True).rename("delta_shares_adj")
    long_dd = d_dollars.where(trade_ok).stack(future_stack=True).rename("delta_dollars")
    out = pd.concat([long_dn, long_dd], axis=1).dropna(how="any").reset_index()
    out = out.rename(columns={out.columns[0]: "date", out.columns[1]: "infocode"})
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_dollars"])
    out = out[out["delta_dollars"] != 0]
    if out.empty:
        return empty, FillBuildStats(
            n_trades=n_dropped, n_fills=0, n_dropped=n_dropped
        )

    out["infocode"] = out["infocode"].astype(np.int64)
    out["side"] = np.where(out["delta_dollars"] > 0, "buy", "sell")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["delta_dollars"] = out["delta_dollars"].astype(np.float64)
    out["delta_shares_adj"] = out["delta_shares_adj"].astype(np.float64)
    n_fills = int(len(out))
    n_trades = n_fills + n_dropped
    return (
        out[["date", "infocode", "side", "delta_shares_adj", "delta_dollars"]],
        FillBuildStats(n_trades=n_trades, n_fills=n_fills, n_dropped=n_dropped),
    )


def join_trades_prices(trades, prices, *, fill: str = "vwap"):
    """Merge DoD trades to LSEG prices; drop rows missing usable prices.

    Trade dated ``t`` carries ``delta_dollars = Δn_adj × close_adj_t`` from
    ``sod_trades``. Real shares and same-day VWAP vs close:

    * ``qty = |δ$_t| / close_t`` (unadjusted close)
    * ``exec_px = VWAP_t`` (fallback close) or ``close_t`` under MOC
    * costs **dated on t** (same day as lag-1 ``pre_t``)
    * MOC: exec at close → zero slippage
    * ``bid`` / ``ask`` passed through when present (EOD half-spread proxy)

    Returns ``(ok_frame, FillBuildStats)`` with columns
    ``date, infocode, side, delta_shares_adj, delta_dollars, vwap, close, qty``
    plus ``bid``, ``ask`` when available on *prices*.
    """
    import numpy as np
    import pandas as pd

    mode = str(fill).strip().lower()
    if mode not in ("vwap", "moc"):
        raise ValueError(f"fill must be 'vwap' or 'moc', got {fill!r}")

    n_trades = int(len(trades)) if trades is not None else 0
    empty_cols = [
        "date",
        "infocode",
        "side",
        "delta_shares_adj",
        "delta_dollars",
        "vwap",
        "close",
        "qty",
        "bid",
        "ask",
    ]
    if n_trades == 0:
        return (
            pd.DataFrame(columns=empty_cols),
            FillBuildStats(n_trades=0, n_fills=0, n_dropped=0),
        )

    px = prices.copy()
    px["marketdate"] = pd.to_datetime(px["marketdate"]).dt.normalize()
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    merged = t.merge(
        px,
        left_on=["date", "infocode"],
        right_on=["marketdate", "infocode"],
        how="left",
    )
    vwap = merged["vwap"].to_numpy(dtype=np.float64, copy=False)
    close = merged["close"].to_numpy(dtype=np.float64, copy=False)
    d_dollars = merged["delta_dollars"].to_numpy(dtype=np.float64, copy=False)
    if mode == "moc":
        exec_px = close
    else:
        exec_px = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close)
    ok = (
        np.isfinite(exec_px)
        & (exec_px > 0)
        & np.isfinite(close)
        & (close > 0)
        & np.isfinite(d_dollars)
        & (d_dollars != 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        qty = np.where(ok, np.abs(d_dollars) / close, np.nan)
    ok &= np.isfinite(qty) & (qty > 0)
    n_fills = int(ok.sum())
    n_dropped = n_trades - n_fills
    if n_fills == 0:
        return (
            pd.DataFrame(columns=empty_cols),
            FillBuildStats(n_trades=n_trades, n_fills=0, n_dropped=n_dropped),
        )
    cols = ["date", "infocode", "side", "delta_dollars", "delta_shares_adj"]
    out = merged.loc[ok, cols].copy()
    out["vwap"] = exec_px[ok]
    out["close"] = close[ok]
    out["qty"] = qty[ok]
    if "bid" in merged.columns:
        out["bid"] = merged.loc[ok, "bid"].to_numpy(dtype=np.float64, copy=False)
    else:
        out["bid"] = np.nan
    if "ask" in merged.columns:
        out["ask"] = merged.loc[ok, "ask"].to_numpy(dtype=np.float64, copy=False)
    else:
        out["ask"] = np.nan
    return out, FillBuildStats(n_trades=n_trades, n_fills=n_fills, n_dropped=n_dropped)


def trades_to_fills(
    trades, prices, *, include_spread: bool = True
) -> tuple[list[Fill], FillBuildStats]:
    """Join trades to LSEG cost prices and build ``Fill`` rows.

    ``qty = |δ$| / close_t``. ``Fill.close`` is same-day close. Spread is the
    LSEG EOD ``BidAsk`` half-spread when *include_spread* and bid/ask are usable;
    otherwise ``FullSpreadBps(0)``.
    """
    import math

    import pandas as pd

    from tcost_engine.types import BidAsk

    joined, stats = join_trades_prices(trades, prices)
    if stats.n_fills == 0:
        return [], stats

    zero_spread = FullSpreadBps("0")
    fills: list[Fill] = []
    for row in joined.itertuples(index=False):
        day = pd.Timestamp(row.date).strftime("%Y-%m-%d")
        spread = zero_spread
        if include_spread:
            bid = float(getattr(row, "bid", float("nan")))
            ask = float(getattr(row, "ask", float("nan")))
            if (
                math.isfinite(bid)
                and math.isfinite(ask)
                and bid > 0
                and ask > 0
                and ask >= bid
            ):
                spread = BidAsk(
                    format(Decimal(str(bid)), "f"),
                    format(Decimal(str(ask)), "f"),
                )
        fills.append(
            Fill(
                side=Side.BUY if row.side == "buy" else Side.SELL,
                quantity=format(Decimal(str(row.qty)), "f"),
                price=format(Decimal(str(row.vwap)), "f"),
                spread=spread,
                symbol=str(int(row.infocode)),
                order_id=f"{day}-{int(row.infocode)}",
                close=format(Decimal(str(row.close)), "f"),
            )
        )
    return fills, stats


def cost_joined_trades(
    joined, *, mils: object = 10, fill: str = "vwap", include_spread: bool = True
):
    """Vectorized daily t-cost frame from ``join_trades_prices`` output.

    Cost-signed, dated on trade day ``t`` (same day as lag-1 ``pre_t``):

        qty                = |δ$_t| / close_t
        commish            = (mils / 10_000) × qty
        spread             = ((ask − bid) / 2) × qty   # EOD half-spread proxy
        intraday_slippage  = side × (VWAP_t − close_t) × qty

    ``δ$`` is the adjusted-share rebalance dollar. Paper assumes a free switch
    at ``close_t``; reality works the trade at VWAP and pays half-spread on
    top (VWAP alone is optimistic vs taking liquidity). Under MOC, exec =
    close → commission only (slippage 0, spread 0; *include_spread* ignored).
    Missing or invalid bid/ask → spread 0 for that fill.
    """
    import numpy as np
    import pandas as pd

    from tcost_engine.types import to_decimal

    mode = str(fill).strip().lower()
    if mode not in ("vwap", "moc"):
        raise ValueError(f"fill must be 'vwap' or 'moc', got {fill!r}")

    if joined is None or len(joined) == 0:
        return pd.DataFrame(
            columns=[
                "date",
                "commish",
                "spread",
                "intraday_slippage",
                "market_impact",
                "total",
                "trade_notional",
                "n_fills",
            ]
        )

    mils_d = float(to_decimal(mils, name="mils")) / 10000.0
    qty = joined["qty"].to_numpy(dtype=np.float64, copy=False)
    exec_px = joined["vwap"].to_numpy(dtype=np.float64, copy=False)
    close = joined["close"].to_numpy(dtype=np.float64, copy=False)
    side = np.where(joined["side"].to_numpy() == "buy", 1.0, -1.0)
    work = joined[["date"]].copy()
    work["commish"] = mils_d * qty
    # MOC = fill at close: commission only. Spread applies to VWAP fills only.
    if (
        mode != "moc"
        and include_spread
        and "bid" in joined.columns
        and "ask" in joined.columns
    ):
        bid = joined["bid"].to_numpy(dtype=np.float64, copy=False)
        ask = joined["ask"].to_numpy(dtype=np.float64, copy=False)
        half = np.where(
            np.isfinite(bid)
            & np.isfinite(ask)
            & (bid > 0)
            & (ask > 0)
            & (ask >= bid),
            (ask - bid) / 2.0,
            0.0,
        )
        work["spread"] = half * qty
    else:
        work["spread"] = 0.0
    if mode == "moc":
        work["intraday_slippage"] = 0.0
    else:
        work["intraday_slippage"] = side * (exec_px - close) * qty
    work["trade_notional"] = close * qty  # = |δ$|
    work["n_fills"] = 1
    daily = (
        work.groupby("date", sort=True)
        .agg(
            commish=("commish", "sum"),
            spread=("spread", "sum"),
            intraday_slippage=("intraday_slippage", "sum"),
            trade_notional=("trade_notional", "sum"),
            n_fills=("n_fills", "sum"),
        )
        .reset_index()
    )
    daily["market_impact"] = 0.0
    daily["total"] = daily["commish"] + daily["spread"] + daily["intraday_slippage"]
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily[
        [
            "date",
            "commish",
            "spread",
            "intraday_slippage",
            "market_impact",
            "total",
            "trade_notional",
            "n_fills",
        ]
    ]


def _resolve_path(path: str | Path, *, cache_dir: str | Path | None = None) -> Path:
    text = str(path)
    if not text.startswith("s3://"):
        local = Path(path)
        if not local.is_file():
            raise FileNotFoundError(f"SOD file not found: {local}")
        return local

    rest = text[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"not a valid s3 url: {path}")

    # On kelai-team-robert the Stage-C SOD is mirrored under /data/robert/<key>.
    # Never aws-cp over that mirror — use it as-is when present.
    box_path = Path("/data/robert") / key
    if box_path.is_file():
        print(f"SOD parquet local: {box_path}", flush=True)
        return box_path

    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home() / ".cache" / "tcost-engine" / "sod"
    )
    dest = root / bucket / key
    return ensure_local_s3(text, dest, refresh=True, label="SOD parquet")
