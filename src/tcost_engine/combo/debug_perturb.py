"""Debugger perturb: pre-tcost vs post-tcost Stage-C-style performance tables.

Mirrors signal-sel-opt ``debug_perturb_stage_ab_mosek`` UX: yearly
Year / n_days / medGMV / Sharpe / ret / vol / maxDD / daily TO tables,
side-by-side pre | post, artifacts under an outdir, stdout markdown.

PyCharm:
  Run config: debug_perturb
  Script path: debug_perturb.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from tcost_engine.commission import DEFAULT_COMMISH_MILS
from tcost_engine.combo.cost import ComboCostResult, cost_combo_sod
from tcost_engine.combo.pnl import lag1_dollar_pnl
from tcost_engine.combo.sod import load_sod_panel
from tcost_engine.combo.stats import (
    daily_turnover,
    path_metrics,
    side_by_side,
    yearly_perf_frame,
)
from tcost_engine.lseg.pull import pull_cost_prices
from tcost_engine.types import Number

DEFAULT_OUTDIR = Path("outputs/perturb_tcost")


@dataclass(frozen=True)
class DebugPerturbResult:
    """Daily series + Stage-C-style yearly tables for pre/post t-cost."""

    daily: object  # pandas.DataFrame
    pre_table: object  # display DataFrame
    post_table: object  # display DataFrame
    pre_numeric: object
    post_numeric: object
    cost: ComboCostResult
    total_pre_pnl: float
    total_tcost: float
    total_post_pnl: float
    mean_gmv: float
    pre_pooled: dict[str, float]
    post_pooled: dict[str, float]
    fill: str = "vwap"


def run_debug_perturb(
    sod_path: str | Path,
    *,
    h5_path: str | Path | None = None,
    start: date | None = None,
    end: date | None = None,
    mils: Number = DEFAULT_COMMISH_MILS,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    fill: str = "vwap",
) -> DebugPerturbResult:
    """Compare lag-1 SOD dollar PnL before and after modeled t-costs.

    Pre:  ``pnl_t = Σ SOD_{t−1} × (close_adj_t / close_adj_{t−1} − 1)``
    Intraday slippage ``Σ side×(close−exec)×qty`` is PnL-signed (buy high / sell
    low vs close → negative). Under MOC it is 0.
    Tcost: ``commish + intraday_slippage``.
    Post: ``pre_pnl_t − tcost_t``.
    Returns / GMV / TO use Stage C ``prev_gmv`` conventions.
    """
    import numpy as np
    import pandas as pd

    fill_mode = str(fill).strip().lower()
    if fill_mode not in ("vwap", "moc"):
        raise ValueError(f"fill must be 'vwap' or 'moc', got {fill!r}")

    empty_cols = [
        "date",
        "pre_pnl",
        "gmv",
        "pre_ret",
        "tcost",
        "commish",
        "spread",
        "intraday_slippage",
        "post_pnl",
        "post_ret",
        "daily_to",
        "cum_pre_pnl",
        "cum_post_pnl",
    ]
    empty = pd.DataFrame(columns=empty_cols)
    empty_tbl, empty_num = yearly_perf_frame(empty, ret_col="pre_ret")

    print(f"loading SOD {sod_path} …", flush=True)
    panel = load_sod_panel(sod_path, cache_dir=cache_dir)
    if start is not None:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel.loc[panel.index <= pd.Timestamp(end)]
    print(
        f"  SOD panel {panel.shape[0]:,} days × {panel.shape[1]:,} names "
        f"({panel.index.min().date()} → {panel.index.max().date()})",
        flush=True,
    )
    if len(panel.index) < 2:
        cost = cost_combo_sod(
            sod_path,
            h5_path=h5_path,
            start=start,
            end=end,
            mils=mils,
            cache_dir=cache_dir,
            panel=panel,
            refresh=refresh,
            fill=fill_mode,
        )
        nan_m = path_metrics([])
        return DebugPerturbResult(
            daily=empty,
            pre_table=empty_tbl,
            post_table=empty_tbl,
            pre_numeric=empty_num,
            post_numeric=empty_num,
            cost=cost,
            total_pre_pnl=0.0,
            total_tcost=0.0,
            total_post_pnl=0.0,
            mean_gmv=0.0,
            pre_pooled=nan_m,
            post_pooled=nan_m,
            fill=fill_mode,
        )

    print(
        f"pulling LSEG prices {panel.index.min().date()}→{panel.index.max().date()} "
        f"(this can take a few minutes) …",
        flush=True,
    )
    prices = pull_cost_prices(
        h5_path,
        start=panel.index.min().date(),
        end=panel.index.max().date(),
        infocodes=panel.columns.tolist(),
        cache_dir=cache_dir,
        refresh=refresh,
    )
    print(f"  price rows={len(prices):,}", flush=True)

    print("computing lag-1 pre-tcost PnL …", flush=True)
    pre = lag1_dollar_pnl(panel, prices)
    to = daily_turnover(panel).reset_index()
    to.columns = ["date", "daily_to"]
    to["date"] = pd.to_datetime(to["date"]).dt.normalize()

    print(f"computing post-tcost (model, fill={fill_mode}) …", flush=True)
    cost = cost_combo_sod(
        sod_path,
        h5_path=h5_path,
        start=start,
        end=end,
        mils=mils,
        cache_dir=cache_dir,
        panel=panel,
        prices=prices,
        refresh=refresh,
        fill=fill_mode,
    )
    tcost = cost.daily.copy()
    if not tcost.empty:
        tcost["date"] = pd.to_datetime(tcost["date"]).dt.normalize()
        tcost = tcost.rename(columns={"total": "tcost"})
        # Wire: tcost = commish + intraday_slippage (same as cost.daily total).
        tcost["tcost"] = tcost["commish"] + tcost["intraday_slippage"]
        cols = ["date", "tcost", "commish", "spread", "intraday_slippage"]
        tcost = tcost[cols]
    else:
        tcost = pd.DataFrame(
            columns=["date", "tcost", "commish", "spread", "intraday_slippage"]
        )

    daily = pre.merge(tcost, on="date", how="left").merge(to, on="date", how="left")
    for col in ("tcost", "commish", "spread", "intraday_slippage", "daily_to"):
        daily[col] = daily[col].fillna(0.0)
    daily["intraday_pnl"] = daily["intraday_slippage"]
    daily["post_pnl"] = daily["pre_pnl"] - daily["tcost"]
    daily["post_ret"] = np.where(daily["gmv"] > 0, daily["post_pnl"] / daily["gmv"], np.nan)
    gmv_pos = daily["gmv"] > 0
    daily["drag"] = np.where(gmv_pos, daily["tcost"] / daily["gmv"], np.nan)
    daily["commish_gmv"] = np.where(gmv_pos, daily["commish"] / daily["gmv"], np.nan)
    daily["intraday_slippage_gmv"] = np.where(
        gmv_pos, daily["intraday_slippage"] / daily["gmv"], np.nan
    )
    daily["trade_pnl_gmv"] = daily["intraday_slippage_gmv"]
    daily["cum_pre_pnl"] = daily["pre_pnl"].cumsum()
    daily["cum_post_pnl"] = daily["post_pnl"].cumsum()

    pre_tbl, pre_num = yearly_perf_frame(daily, ret_col="pre_ret")
    post_tbl, post_num = yearly_perf_frame(daily, ret_col="post_ret")

    return DebugPerturbResult(
        daily=daily,
        pre_table=pre_tbl,
        post_table=post_tbl,
        pre_numeric=pre_num,
        post_numeric=post_num,
        cost=cost,
        total_pre_pnl=float(np.nansum(daily["pre_pnl"])),
        total_tcost=float(np.nansum(daily["tcost"])),
        total_post_pnl=float(np.nansum(daily["post_pnl"])),
        mean_gmv=float(np.nanmean(daily["gmv"])) if len(daily) else 0.0,
        pre_pooled=path_metrics(daily["pre_ret"]),
        post_pooled=path_metrics(daily["post_ret"]),
        fill=fill_mode,
    )


def format_debug_perturb(result: DebugPerturbResult) -> str:
    """Mosek-style stdout: side-by-side yearly tables + t-cost drag line."""
    pre = result.pre_table
    post = result.post_table
    block = side_by_side(
        pre.to_string(index=False) if len(pre) else "(empty)",
        post.to_string(index=False) if len(post) else "(empty)",
        left_title="### Pre-tcost (gross)",
        right_title="### Post-tcost (net of model)",
    )
    import pandas as pd

    drag_mean, drag_med = _daily_frac_stats(result, "drag")
    com_mean, com_med = _daily_frac_stats(result, "commish_gmv")
    slip_mean, slip_med = _daily_frac_stats(result, "intraday_slippage_gmv")
    pre_sh = result.pre_pooled.get("sharpe", float("nan"))
    post_sh = result.post_pooled.get("sharpe", float("nan"))
    fill = getattr(result, "fill", "vwap") or "vwap"
    if fill == "moc":
        cost_desc = f"mils={DEFAULT_COMMISH_MILS}, fill=MOC"
    else:
        cost_desc = (
            f"mils={DEFAULT_COMMISH_MILS}, fill=VWAP "
            f"(tcost = commish + intraday_slippage)"
        )
    cost_tbl = pd.DataFrame(
        {
            "mean": [
                _fmt_bps_cell(com_mean),
                _fmt_bps_cell(slip_mean),
                _fmt_bps_cell(drag_mean),
            ],
            "median": [
                _fmt_bps_cell(com_med),
                _fmt_bps_cell(slip_med),
                _fmt_bps_cell(drag_med),
            ],
        },
        index=["commish", "intraday slippage", "tcost"],
    )
    lines = [
        "# Perturb: pre-tcost vs post-tcost",
        "",
        f"Debugger perturb — Stage C lag-1 SOD book × LSEG costs ({cost_desc}).",
        "",
        "## Pre-tcost | Post-tcost",
        "",
        block,
        "",
        "## T-cost / trading (bps/day = cost÷GMV)",
        "",
        f"Sharpe pre → post   {_fmt_num(pre_sh)} → {_fmt_num(post_sh)}",
        "",
        cost_tbl.to_string(),
        "",
    ]
    return "\n".join(lines)


def _daily_frac_stats(result: DebugPerturbResult, col: str) -> tuple[float, float]:
    """Mean and median of a daily fraction column (e.g. tcost/gmv)."""
    import numpy as np

    daily = result.daily
    if daily is None or len(daily) == 0 or col not in getattr(daily, "columns", []):
        return float("nan"), float("nan")
    x = daily[col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.median(x))


def _fmt_bps_day(x: float) -> str:
    """Format a daily fraction as bps/day."""
    if x != x:
        return "n/a"
    return f"{10000.0 * float(x):.4f} bps/day"


def _fmt_bps_cell(x: float) -> str:
    """Format a daily fraction as bps for a table cell."""
    if x != x:
        return "n/a"
    return f"{10000.0 * float(x):.4f}"


def _fmt_num(x: float, nd: int = 2) -> str:
    if x != x:
        return "n/a"
    return f"{float(x):.{nd}f}"


def write_perturb_artifacts(result: DebugPerturbResult, outdir: str | Path) -> Path:
    """Write mosek-style perturb folder: tables, daily series, README."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result.pre_table.to_csv(out / "pre_tcost_yearly.csv", index=False)
    result.post_table.to_csv(out / "post_tcost_yearly.csv", index=False)
    result.pre_numeric.to_csv(out / "pre_tcost_yearly_numeric.csv", index=False)
    result.post_numeric.to_csv(out / "post_tcost_yearly_numeric.csv", index=False)
    daily_path = out / "daily_pre_post.parquet"
    try:
        result.daily.to_parquet(daily_path, index=False)
    except (ImportError, ValueError, OSError):
        daily_path = out / "daily_pre_post.csv"
        result.daily.to_csv(daily_path, index=False)
    md = out / "README.md"
    md.write_text(
        format_debug_perturb(result)
        + "\n## Artifacts\n\n"
        + "- `pre_tcost_yearly.csv` / `post_tcost_yearly.csv` — display tables\n"
        + "- `*_yearly_numeric.csv` — raw sharpe/ret/vol/maxdd/to\n"
        + f"- `{daily_path.name}` — daily pre/post series\n",
        encoding="utf-8",
    )
    return out


def result_summary(result: DebugPerturbResult) -> dict[str, object]:
    drag_mean, drag_med = _daily_frac_stats(result, "drag")
    com_mean, com_med = _daily_frac_stats(result, "commish_gmv")
    slip_mean, slip_med = _daily_frac_stats(result, "intraday_slippage_gmv")

    def _bps(x: float) -> float | None:
        return None if x != x else 10000.0 * x

    return {
        "n_days": int(len(result.daily)),
        "mean_gmv": result.mean_gmv,
        "pre_pnl": result.total_pre_pnl,
        "post_pnl": result.total_post_pnl,
        "commish_bps_per_day_mean": _bps(com_mean),
        "commish_bps_per_day_median": _bps(com_med),
        "intraday_slippage_bps_per_day_mean": _bps(slip_mean),
        "intraday_slippage_bps_per_day_median": _bps(slip_med),
        "tcost_bps_per_day_mean": _bps(drag_mean),
        "tcost_bps_per_day_median": _bps(drag_med),
        "pre_sharpe": result.pre_pooled.get("sharpe"),
        "post_sharpe": result.post_pooled.get("sharpe"),
        "pre_ann_ret": result.pre_pooled.get("ann_ret"),
        "post_ann_ret": result.post_pooled.get("ann_ret"),
        "n_fills": result.cost.n_fills,
        "n_dropped": result.cost.n_dropped,
        "mils": str(DEFAULT_COMMISH_MILS),
        "fill": getattr(result, "fill", "vwap"),
    }
