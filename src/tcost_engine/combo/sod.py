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


def sod_trades(panel):
    """Day-over-day signed dollar trades from a wide SOD panel.

    ``delta_$ = SOD(t) − SOD(t−1)``. The first date has no prior book and yields
    no trades. Returns a long DataFrame: ``date``, ``infocode``, ``side``,
    ``delta_dollars``.
    """
    import numpy as np
    import pandas as pd

    if panel.empty or len(panel.index) < 2:
        return pd.DataFrame(columns=["date", "infocode", "side", "delta_dollars"])

    prev = panel.shift(1)
    delta = panel - prev
    delta = delta.iloc[1:]
    stacked = delta.stack(future_stack=True)
    stacked = stacked.replace([np.inf, -np.inf], np.nan).dropna()
    stacked = stacked[stacked != 0]
    if stacked.empty:
        return pd.DataFrame(columns=["date", "infocode", "side", "delta_dollars"])

    out = stacked.reset_index()
    out.columns = ["date", "infocode", "delta_dollars"]
    out["infocode"] = out["infocode"].astype(np.int64)
    out["side"] = np.where(out["delta_dollars"] > 0, "buy", "sell")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out[["date", "infocode", "side", "delta_dollars"]]


def join_trades_prices(trades, prices, *, fill: str = "vwap"):
    """Merge DoD trades to LSEG prices; drop rows missing usable prices.

    *fill*:
      - ``\"vwap\"``: execute at VWAP (fallback to close if VWAP missing)
      - ``\"moc\"``: market-on-close — execute at close (no VWAP−close residual)

    Returns ``(ok_frame, FillBuildStats)``. ``ok_frame`` has columns
    ``date, infocode, side, delta_dollars, vwap, close, qty`` where ``vwap``
    holds the execution price used for qty / residual.
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
        "delta_dollars",
        "vwap",
        "close",
        "qty",
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
    delta = merged["delta_dollars"].to_numpy(dtype=np.float64, copy=False)
    if mode == "moc":
        exec_px = close
    else:
        # Prefer VWAP; fall back to close so we don't drop names missing VWAP only.
        exec_px = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close)
    ok = np.isfinite(exec_px) & (exec_px > 0) & np.isfinite(close) & (close > 0)
    qty = np.where(ok, np.abs(delta) / exec_px, np.nan)
    ok &= np.isfinite(qty) & (qty > 0)
    n_fills = int(ok.sum())
    n_dropped = n_trades - n_fills
    if n_fills == 0:
        return (
            pd.DataFrame(columns=empty_cols),
            FillBuildStats(n_trades=n_trades, n_fills=0, n_dropped=n_dropped),
        )
    out = merged.loc[ok, ["date", "infocode", "side", "delta_dollars"]].copy()
    # Column name stays ``vwap`` for qty/residual helpers (= exec price).
    out["vwap"] = exec_px[ok]
    out["close"] = close[ok]
    out["qty"] = qty[ok]
    return out, FillBuildStats(n_trades=n_trades, n_fills=n_fills, n_dropped=n_dropped)


def trades_to_fills(trades, prices) -> tuple[list[Fill], FillBuildStats]:
    """Join trades to LSEG cost prices and build ``Fill`` rows.

    ``qty = |delta_$| / VWAP``. Missing vwap/close rows are dropped.
    Spread on combo fills is zero (VWAP already embeds liquidity).
    ``order_id`` is ``YYYY-MM-DD-{infocode}`` for daily aggregation.

    Prefer ``join_trades_prices`` + vectorized costing for large books.
    """
    import pandas as pd

    joined, stats = join_trades_prices(trades, prices)
    if stats.n_fills == 0:
        return [], stats

    zero_spread = FullSpreadBps("0")
    fills: list[Fill] = []
    for row in joined.itertuples(index=False):
        day = pd.Timestamp(row.date).strftime("%Y-%m-%d")
        fills.append(
            Fill(
                side=Side.BUY if row.side == "buy" else Side.SELL,
                quantity=format(Decimal(str(row.qty)), "f"),
                price=format(Decimal(str(row.vwap)), "f"),
                spread=zero_spread,
                symbol=str(int(row.infocode)),
                order_id=f"{day}-{int(row.infocode)}",
                close=format(Decimal(str(row.close)), "f"),
            )
        )
    return fills, stats


def cost_joined_trades(joined, *, mils: object = 10, fill: str = "vwap"):
    """Vectorized daily t-cost frame from ``join_trades_prices`` output.

    Intraday slippage is *PnL-signed*:

        intraday_slippage = side × (close − exec) × qty

    (buy@exec>close / sell@exec<close → negative). Combo tcost is

        total = commish + intraday_slippage

    Under MOC, exec = close → slippage 0 and total = mils only.
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
    work["spread"] = 0.0
    if mode == "moc":
        work["intraday_slippage"] = 0.0
    else:
        # PnL-signed: buy high / sell low vs close → negative slippage.
        work["intraday_slippage"] = side * (close - exec_px) * qty
    work["trade_notional"] = exec_px * qty
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


def _resolve_path(path: str | Path, *, cache_dir: str | Path | None) -> Path:
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
