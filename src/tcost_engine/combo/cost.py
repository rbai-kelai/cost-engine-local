"""Run the t-cost model on combo SOD day-over-day trades."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from tcost_engine.commission import DEFAULT_COMMISH_MILS
from tcost_engine.combo.sod import (
    cost_joined_trades,
    join_trades_prices,
    load_sod_panel,
    sod_trades,
)
from tcost_engine.lseg.pull import pull_cost_prices
from tcost_engine.types import Number, to_decimal


@dataclass(frozen=True)
class ComboCostResult:
    """Daily t-cost summary for a combo SOD book."""

    daily: object  # pandas.DataFrame
    total_commish: Decimal
    total_spread: Decimal
    total_intraday_slippage: Decimal
    total_cost: Decimal
    trade_notional: Decimal
    n_fills: int
    n_dropped: int
    n_trades: int


def cost_combo_sod(
    sod_path: str | Path,
    *,
    h5_path: str | Path | None = None,
    start: date | None = None,
    end: date | None = None,
    mils: Number = DEFAULT_COMMISH_MILS,
    cache_dir: str | Path | None = None,
    panel=None,
    prices=None,
    refresh: bool = False,
    fill: str = "vwap",
) -> ComboCostResult:
    """Cost DoD trades from a combo SOD dollar panel against LSEG prices.

    Trades: ``delta_$ = SOD(t) − SOD(t−1)``.
    *fill* ``\"vwap\"``: ``qty = |delta_$| / VWAP``;
    ``intraday_slippage = side×(close−VWAP)×qty``; ``total = mils + slippage``.
    *fill* ``\"moc\"``: ``qty = |delta_$| / close``; slippage 0; total = mils.

    Pass *panel* / *prices* to skip reloading (used by debug_perturb).
    Daily totals are cached under ``~/.cache/tcost-engine/combo_cost/`` unless
    ``refresh=True``.
    """
    import pandas as pd

    from tcost_engine.cacheutil import (
        default_cache_root,
        file_stamp,
        fingerprint,
        read_parquet_if_fresh,
        write_parquet_cache,
    )
    fill_mode = str(fill).strip().lower()
    if fill_mode not in ("vwap", "moc"):
        raise ValueError(f"fill must be 'vwap' or 'moc', got {fill!r}")
    sod_resolved = _resolved_sod_path(sod_path, cache_dir=cache_dir)
    sod_key, sod_mtime = file_stamp(sod_resolved)
    h5_key, h5_mtime = _h5_cache_stamp(h5_path, cache_dir=cache_dir)
    mils_d = to_decimal(mils, name="mils")
    meta = {
        "sod": sod_key,
        "sod_mtime": sod_mtime,
        "h5": h5_key,
        "h5_mtime": h5_mtime,
        "start": "" if start is None else start.isoformat(),
        "end": "" if end is None else end.isoformat(),
        "mils": str(mils_d),
        "spread": "0",
        "fill": fill_mode,
        # Bump when combo total / slippage convention changes.
        "tcost_defn": "commish+intraday_slippage",
    }
    fp = fingerprint(
        meta["sod"],
        meta["sod_mtime"],
        meta["h5"],
        meta["h5_mtime"],
        meta["start"],
        meta["end"],
        meta["mils"],
        meta["fill"],
        meta["tcost_defn"],
    )
    cache_root = default_cache_root(cache_dir) / "combo_cost"
    cache_path = cache_root / f"daily_{fp}.parquet"
    meta_path = cache_root / f"daily_{fp}.json"
    if not refresh:
        cached = read_parquet_if_fresh(cache_path, meta_path=meta_path, expected=meta)
        if cached is not None:
            sidecar = _read_sidecar(meta_path)
            print(f"combo-cost cache hit: {cache_path}", flush=True)
            return ComboCostResult(
                daily=cached,
                total_commish=Decimal(str(sidecar.get("total_commish", 0))),
                total_spread=Decimal(str(sidecar.get("total_spread", 0))),
                total_intraday_slippage=Decimal(
                    str(
                        sidecar.get(
                            "total_intraday_slippage",
                            sidecar.get("total_residual", 0),
                        )
                    )
                ),
                total_cost=Decimal(str(sidecar.get("total_cost", 0))),
                trade_notional=Decimal(str(sidecar.get("trade_notional", 0))),
                n_fills=int(sidecar.get("n_fills", 0)),
                n_dropped=int(sidecar.get("n_dropped", 0)),
                n_trades=int(sidecar.get("n_trades", 0)),
            )

    if panel is None:
        panel = load_sod_panel(sod_path, cache_dir=cache_dir)
        if start is not None:
            panel = panel.loc[panel.index >= pd.Timestamp(start)]
        if end is not None:
            panel = panel.loc[panel.index <= pd.Timestamp(end)]
    if panel.empty:
        return _empty_result()

    print(f"building DoD trades from SOD {panel.shape} …", flush=True)
    trades = sod_trades(panel)
    if trades.empty:
        return _empty_result()
    print(f"  trades={len(trades):,}", flush=True)

    if prices is None:
        trade_start = trades["date"].min().date()
        trade_end = trades["date"].max().date()
        print(
            f"pulling LSEG BID/ASK/VWAP/CLOSE {trade_start}→{trade_end} …",
            flush=True,
        )
        prices = pull_cost_prices(
            h5_path,
            start=trade_start,
            end=trade_end,
            infocodes=panel.columns.tolist(),
            cache_dir=cache_dir,
            refresh=refresh,
        )
        print(f"  price rows={len(prices):,}", flush=True)

    print(f"joining trades → prices (vectorized, fill={fill_mode}) …", flush=True)
    joined, stats = join_trades_prices(trades, prices, fill=fill_mode)
    print(f"  fills={stats.n_fills:,}", flush=True)
    if stats.n_dropped:
        print(f"  dropped={stats.n_dropped:,} (no price)", flush=True)
    if stats.n_fills == 0:
        return ComboCostResult(
            daily=_empty_daily(),
            total_commish=Decimal(0),
            total_spread=Decimal(0),
            total_intraday_slippage=Decimal(0),
            total_cost=Decimal(0),
            trade_notional=Decimal(0),
            n_fills=0,
            n_dropped=stats.n_dropped,
            n_trades=stats.n_trades,
        )

    print("aggregating daily t-costs …", flush=True)
    daily = cost_joined_trades(joined, mils=mils, fill=fill_mode)
    total_commish = Decimal(str(float(daily["commish"].sum())))
    total_spread = Decimal(str(float(daily["spread"].sum())))
    total_intraday_slippage = Decimal(str(float(daily["intraday_slippage"].sum())))
    total_cost = Decimal(str(float(daily["total"].sum())))
    trade_notional = Decimal(str(float(daily["trade_notional"].sum())))
    result = ComboCostResult(
        daily=daily,
        total_commish=total_commish,
        total_spread=total_spread,
        total_intraday_slippage=total_intraday_slippage,
        total_cost=total_cost,
        trade_notional=trade_notional,
        n_fills=stats.n_fills,
        n_dropped=stats.n_dropped,
        n_trades=stats.n_trades,
    )
    sidecar = {
        **meta,
        "total_commish": str(total_commish),
        "total_spread": str(total_spread),
        "total_intraday_slippage": str(total_intraday_slippage),
        "total_cost": str(total_cost),
        "trade_notional": str(trade_notional),
        "n_fills": stats.n_fills,
        "n_dropped": stats.n_dropped,
        "n_trades": stats.n_trades,
    }
    try:
        write_parquet_cache(daily, cache_path, meta_path=meta_path, meta=sidecar)
        print(f"wrote combo-cost cache: {cache_path}", flush=True)
    except (OSError, ImportError, ValueError) as exc:
        print(f"WARNING: could not write combo-cost cache ({exc})", flush=True)
    return result


def _resolved_sod_path(sod_path: str | Path, *, cache_dir: str | Path | None) -> Path:
    from tcost_engine.combo.sod import _resolve_path

    return _resolve_path(sod_path, cache_dir=cache_dir)


def _h5_cache_stamp(
    h5_path: str | Path | None, *, cache_dir: str | Path | None
) -> tuple[str, float]:
    """Stamp for cache invalidation without forcing an S3 download."""
    import subprocess

    from tcost_engine.cacheutil import file_stamp
    from tcost_engine.lseg.paths import _best_local_ds2, resolve_ds2_h5

    if h5_path is None:
        local = _best_local_ds2()
        if local is not None:
            return file_stamp(local)
        return ("auto", 0.0)
    text = str(h5_path)
    if not text.startswith("s3://"):
        return file_stamp(h5_path)
    try:
        return file_stamp(resolve_ds2_h5(h5_path, cache_dir=cache_dir))
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError):
        return (text, 0.0)


def _read_sidecar(meta_path: Path) -> dict:
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _empty_daily():
    import pandas as pd

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


def _empty_result() -> ComboCostResult:
    return ComboCostResult(
        daily=_empty_daily(),
        total_commish=Decimal(0),
        total_spread=Decimal(0),
        total_intraday_slippage=Decimal(0),
        total_cost=Decimal(0),
        trade_notional=Decimal(0),
        n_fills=0,
        n_dropped=0,
        n_trades=0,
    )


def result_summary(result: ComboCostResult) -> dict[str, object]:
    """JSON-serializable totals for CLI ``--json``."""
    mils = to_decimal(DEFAULT_COMMISH_MILS, name="mils")
    return {
        "mils": str(mils),
        "n_trades": result.n_trades,
        "n_fills": result.n_fills,
        "n_dropped": result.n_dropped,
        "trade_notional": str(result.trade_notional),
        "commish": str(result.total_commish),
        "spread": str(result.total_spread),
        "intraday_slippage": str(result.total_intraday_slippage),
        "total": str(result.total_cost),
        "n_days": int(len(result.daily)),
    }
