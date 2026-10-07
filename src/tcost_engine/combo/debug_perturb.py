"""Debugger perturb: pre-tcost vs post-tcost Stage-C-style performance tables.

One run produces two after-tcost fill scenarios side by side:

* **MOC fill** — close fill; ``tcost = commish`` (no slippage, no spread)
* **VWAP fill** — ``tcost = commish + half-spread + VWAP−close slippage``

Mirrors signal-sel-opt ``debug_perturb_stage_ab_mosek`` UX: yearly
Year / n_days / medGMV / Sharpe / ret / vol / maxDD / daily TO tables,
artifacts under an outdir, stdout markdown.

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

MOC_FILL_LABEL = "MOC fill"
VWAP_FILL_LABEL = "VWAP fill"


@dataclass(frozen=True)
class FillScenarioResult:
    """Yearly post-tcost tables + totals for one fill assumption."""

    label: str
    post_table: object  # display DataFrame
    post_numeric: object
    total_tcost: float
    total_post_pnl: float
    post_pooled: dict[str, float]
    total_commish: float
    total_spread: float
    total_intraday_slippage: float


@dataclass(frozen=True)
class DebugPerturbResult:
    """Daily series + Stage-C-style yearly tables for pre / MOC / VWAP."""

    daily: object  # pandas.DataFrame
    pre_table: object  # display DataFrame
    pre_numeric: object
    moc: FillScenarioResult
    vwap: FillScenarioResult
    cost: ComboCostResult  # VWAP-path ComboCostResult (shared fills)
    total_pre_pnl: float
    mean_gmv: float
    pre_pooled: dict[str, float]
    mils: object = DEFAULT_COMMISH_MILS
    include_spread: bool = True


def run_debug_perturb(
    sod_path: str | Path,
    *,
    h5_path: str | Path | None = None,
    start: date | None = None,
    end: date | None = None,
    mils: Number = DEFAULT_COMMISH_MILS,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    include_spread: bool = True,
    fill: str | None = None,
) -> DebugPerturbResult:
    """Compare lag-1 SOD dollar PnL before and after modeled t-costs.

    Pre (paper): ``pnl_t = Σ SOD_{t−1} × (close_adj_t / close_adj_{t−1} − 1)``
    — old book through close ``t``, free switch into SOD_t at close.

    One run reports two post-tcost scenarios (shared SOD / prices / fills):

    * **MOC fill**: close fill; ``tcost = commish`` (slippage 0, spread 0)
    * **VWAP fill**: ``tcost = commish + spread + side×(VWAP−close)×qty``
      (*include_spread* toggles half-spread; default ON)

    Returns / GMV / TO use Stage C ``prev_gmv`` conventions.

    *fill* is accepted for backward compatibility and ignored — both scenarios
    are always produced.
    """
    import numpy as np
    import pandas as pd

    if fill is not None:
        fill_mode = str(fill).strip().lower()
        if fill_mode not in ("vwap", "moc"):
            raise ValueError(f"fill must be 'vwap' or 'moc', got {fill!r}")

    empty_cols = [
        "date",
        "pre_pnl",
        "gmv",
        "pre_ret",
        "commish",
        "spread",
        "intraday_slippage",
        "moc_tcost",
        "moc_post_pnl",
        "moc_post_ret",
        "vwap_tcost",
        "vwap_post_pnl",
        "vwap_post_ret",
        "daily_to",
        "cum_pre_pnl",
        "moc_cum_post_pnl",
        "vwap_cum_post_pnl",
    ]
    empty = pd.DataFrame(columns=empty_cols)
    empty_tbl, empty_num = yearly_perf_frame(empty, ret_col="pre_ret")
    empty_moc = FillScenarioResult(
        label=MOC_FILL_LABEL,
        post_table=empty_tbl,
        post_numeric=empty_num,
        total_tcost=0.0,
        total_post_pnl=0.0,
        post_pooled=path_metrics([]),
        total_commish=0.0,
        total_spread=0.0,
        total_intraday_slippage=0.0,
    )
    empty_vwap = FillScenarioResult(
        label=VWAP_FILL_LABEL,
        post_table=empty_tbl,
        post_numeric=empty_num,
        total_tcost=0.0,
        total_post_pnl=0.0,
        post_pooled=path_metrics([]),
        total_commish=0.0,
        total_spread=0.0,
        total_intraday_slippage=0.0,
    )

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
            fill="vwap",
            include_spread=include_spread,
        )
        nan_m = path_metrics([])
        return DebugPerturbResult(
            daily=empty,
            pre_table=empty_tbl,
            pre_numeric=empty_num,
            moc=empty_moc,
            vwap=empty_vwap,
            cost=cost,
            total_pre_pnl=0.0,
            mean_gmv=0.0,
            pre_pooled=nan_m,
            mils=mils,
            include_spread=include_spread,
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
    to = daily_turnover(panel, prices).reset_index()
    to.columns = ["date", "daily_to"]
    to["date"] = pd.to_datetime(to["date"]).dt.normalize()

    # One VWAP cost pass; MOC is derived (commish only) from the same fills.
    print("computing post-tcost (MOC fill + VWAP fill) …", flush=True)
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
        fill="vwap",
        include_spread=include_spread,
    )
    tcost = cost.daily.copy()
    if not tcost.empty:
        tcost["date"] = pd.to_datetime(tcost["date"]).dt.normalize()
        cols = ["date", "commish", "spread", "intraday_slippage"]
        tcost = tcost[cols]
    else:
        tcost = pd.DataFrame(columns=["date", "commish", "spread", "intraday_slippage"])

    daily = pre.merge(tcost, on="date", how="left").merge(to, on="date", how="left")
    daily = daily.sort_values("date").reset_index(drop=True)
    for col in ("commish", "spread", "intraday_slippage", "daily_to"):
        daily[col] = daily[col].fillna(0.0)

    # MOC fill: close fill → commission only.
    daily["moc_tcost"] = daily["commish"]
    daily["moc_post_pnl"] = daily["pre_pnl"] - daily["moc_tcost"]
    daily["moc_post_ret"] = np.where(
        daily["gmv"] > 0, daily["moc_post_pnl"] / daily["gmv"], np.nan
    )

    # VWAP fill: mils + half-spread + VWAP−close (cost-signed).
    daily["vwap_tcost"] = (
        daily["commish"] + daily["spread"] + daily["intraday_slippage"]
    )
    daily["intraday_pnl"] = -daily["intraday_slippage"]
    daily["vwap_post_pnl"] = daily["pre_pnl"] - daily["vwap_tcost"]
    daily["vwap_post_ret"] = np.where(
        daily["gmv"] > 0, daily["vwap_post_pnl"] / daily["gmv"], np.nan
    )

    gmv_pos = daily["gmv"] > 0
    daily["moc_tcost_gmv"] = np.where(gmv_pos, daily["moc_tcost"] / daily["gmv"], np.nan)
    daily["vwap_tcost_gmv"] = np.where(
        gmv_pos, daily["vwap_tcost"] / daily["gmv"], np.nan
    )
    daily["commish_gmv"] = np.where(gmv_pos, daily["commish"] / daily["gmv"], np.nan)
    daily["spread_gmv"] = np.where(gmv_pos, daily["spread"] / daily["gmv"], np.nan)
    daily["intraday_slippage_gmv"] = np.where(
        gmv_pos, daily["intraday_slippage"] / daily["gmv"], np.nan
    )
    daily["trade_pnl_gmv"] = np.where(
        gmv_pos, daily["intraday_pnl"] / daily["gmv"], np.nan
    )
    daily["cum_pre_pnl"] = daily["pre_pnl"].cumsum()
    daily["moc_cum_post_pnl"] = daily["moc_post_pnl"].cumsum()
    daily["vwap_cum_post_pnl"] = daily["vwap_post_pnl"].cumsum()

    pre_tbl, pre_num = yearly_perf_frame(daily, ret_col="pre_ret")
    moc_post_tbl, moc_post_num = yearly_perf_frame(daily, ret_col="moc_post_ret")
    vwap_post_tbl, vwap_post_num = yearly_perf_frame(daily, ret_col="vwap_post_ret")

    moc_tcost = float(np.nansum(daily["moc_tcost"]))
    vwap_tcost = float(np.nansum(daily["vwap_tcost"]))
    moc_post = float(np.nansum(daily["moc_post_pnl"]))
    vwap_post = float(np.nansum(daily["vwap_post_pnl"]))
    total_commish = float(np.nansum(daily["commish"]))
    total_spread = float(np.nansum(daily["spread"]))
    total_slip = float(np.nansum(daily["intraday_slippage"]))

    return DebugPerturbResult(
        daily=daily,
        pre_table=pre_tbl,
        pre_numeric=pre_num,
        moc=FillScenarioResult(
            label=MOC_FILL_LABEL,
            post_table=moc_post_tbl,
            post_numeric=moc_post_num,
            total_tcost=moc_tcost,
            total_post_pnl=moc_post,
            post_pooled=path_metrics(daily["moc_post_ret"]),
            total_commish=total_commish,
            total_spread=0.0,
            total_intraday_slippage=0.0,
        ),
        vwap=FillScenarioResult(
            label=VWAP_FILL_LABEL,
            post_table=vwap_post_tbl,
            post_numeric=vwap_post_num,
            total_tcost=vwap_tcost,
            total_post_pnl=vwap_post,
            post_pooled=path_metrics(daily["vwap_post_ret"]),
            total_commish=total_commish,
            total_spread=total_spread,
            total_intraday_slippage=total_slip,
        ),
        cost=cost,
        total_pre_pnl=float(np.nansum(daily["pre_pnl"])),
        mean_gmv=float(np.nanmean(daily["gmv"])) if len(daily) else 0.0,
        pre_pooled=path_metrics(daily["pre_ret"]),
        mils=mils,
        include_spread=include_spread,
    )


def format_debug_perturb(result: DebugPerturbResult) -> str:
    """Stdout: pre vs post yearly tables for MOC fill and VWAP fill."""
    mils_disp = getattr(result, "mils", DEFAULT_COMMISH_MILS)
    spread_note = (
        "half-spread ON"
        if getattr(result, "include_spread", True)
        else "half-spread OFF (--no-spread)"
    )
    lines = [
        "# Perturb: pre-tcost vs post-tcost",
        "",
        f"Debugger perturb — Stage C lag-1 SOD book × LSEG costs "
        f"(mils={mils_disp}; {spread_note}; share-based Δn). "
        "One run reports both fill scenarios below.",
        "",
    ]
    lines.extend(_format_scenario_block(result, result.moc))
    lines.append("")
    lines.extend(_format_scenario_block(result, result.vwap))
    lines.append("")
    return "\n".join(lines)


def _format_scenario_block(
    result: DebugPerturbResult, scenario: FillScenarioResult
) -> list[str]:
    """Markdown section for one fill scenario."""
    import pandas as pd

    pre = result.pre_table
    post = scenario.post_table
    block = side_by_side(
        pre.to_string(index=False) if len(pre) else "(empty)",
        post.to_string(index=False) if len(post) else "(empty)",
        left_title="### Pre-tcost (gross)",
        right_title="### Post-tcost (net)",
    )
    pre_sh = result.pre_pooled.get("sharpe", float("nan"))
    post_sh = scenario.post_pooled.get("sharpe", float("nan"))

    if scenario.label == MOC_FILL_LABEL:
        cost_desc = "tcost = commish (fill at close; no slippage, no spread)"
        tcost_mean, tcost_med = _daily_frac_stats(result, "moc_tcost_gmv")
        com_mean, com_med = _daily_frac_stats(result, "commish_gmv")
        cost_tbl = pd.DataFrame(
            {
                "mean": [_fmt_bps_cell(com_mean), _fmt_bps_cell(tcost_mean)],
                "median": [_fmt_bps_cell(com_med), _fmt_bps_cell(tcost_med)],
            },
            index=["commish", "tcost"],
        )
    else:
        cost_desc = "tcost = commish + half-spread + VWAP−close_t"
        tcost_mean, tcost_med = _daily_frac_stats(result, "vwap_tcost_gmv")
        com_mean, com_med = _daily_frac_stats(result, "commish_gmv")
        spr_mean, spr_med = _daily_frac_stats(result, "spread_gmv")
        slip_mean, slip_med = _daily_frac_stats(result, "intraday_slippage_gmv")
        cost_tbl = pd.DataFrame(
            {
                "mean": [
                    _fmt_bps_cell(com_mean),
                    _fmt_bps_cell(spr_mean),
                    _fmt_bps_cell(slip_mean),
                    _fmt_bps_cell(tcost_mean),
                ],
                "median": [
                    _fmt_bps_cell(com_med),
                    _fmt_bps_cell(spr_med),
                    _fmt_bps_cell(slip_med),
                    _fmt_bps_cell(tcost_med),
                ],
            },
            index=["commish", "spread", "intraday slippage", "tcost"],
        )

    return [
        f"## {scenario.label}",
        "",
        cost_desc,
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
    ]


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
    """Write perturb folder: pre + MOC/VWAP post tables, daily series, README."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result.pre_table.to_csv(out / "pre_tcost_yearly.csv", index=False)
    result.pre_numeric.to_csv(out / "pre_tcost_yearly_numeric.csv", index=False)
    result.moc.post_table.to_csv(out / "moc_post_tcost_yearly.csv", index=False)
    result.moc.post_numeric.to_csv(
        out / "moc_post_tcost_yearly_numeric.csv", index=False
    )
    result.vwap.post_table.to_csv(out / "vwap_post_tcost_yearly.csv", index=False)
    result.vwap.post_numeric.to_csv(
        out / "vwap_post_tcost_yearly_numeric.csv", index=False
    )
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
        + "- `pre_tcost_yearly.csv` — pre-tcost display table\n"
        + "- `moc_post_tcost_yearly.csv` — MOC fill post-tcost table\n"
        + "- `vwap_post_tcost_yearly.csv` — VWAP fill post-tcost table\n"
        + "- `*_yearly_numeric.csv` — raw sharpe/ret/vol/maxdd/to\n"
        + f"- `{daily_path.name}` — daily pre / MOC / VWAP series\n",
        encoding="utf-8",
    )
    return out


def result_summary(result: DebugPerturbResult) -> dict[str, object]:
    """JSON-friendly summary with both fill scenarios."""

    def _bps_pair(col: str) -> tuple[float | None, float | None]:
        mean, med = _daily_frac_stats(result, col)

        def _bps(x: float) -> float | None:
            return None if x != x else 10000.0 * x

        return _bps(mean), _bps(med)

    com_mean, com_med = _bps_pair("commish_gmv")
    spr_mean, spr_med = _bps_pair("spread_gmv")
    slip_mean, slip_med = _bps_pair("intraday_slippage_gmv")
    moc_mean, moc_med = _bps_pair("moc_tcost_gmv")
    vwap_mean, vwap_med = _bps_pair("vwap_tcost_gmv")

    return {
        "n_days": int(len(result.daily)),
        "mean_gmv": result.mean_gmv,
        "pre_pnl": result.total_pre_pnl,
        "pre_sharpe": result.pre_pooled.get("sharpe"),
        "pre_ann_ret": result.pre_pooled.get("ann_ret"),
        "mils": str(getattr(result, "mils", DEFAULT_COMMISH_MILS)),
        "include_spread": bool(getattr(result, "include_spread", True)),
        "n_fills": result.cost.n_fills,
        "n_dropped": result.cost.n_dropped,
        "moc_fill": {
            "label": result.moc.label,
            "post_pnl": result.moc.total_post_pnl,
            "tcost": result.moc.total_tcost,
            "commish": result.moc.total_commish,
            "spread": result.moc.total_spread,
            "intraday_slippage": result.moc.total_intraday_slippage,
            "tcost_bps_per_day_mean": moc_mean,
            "tcost_bps_per_day_median": moc_med,
            "post_sharpe": result.moc.post_pooled.get("sharpe"),
            "post_ann_ret": result.moc.post_pooled.get("ann_ret"),
            "formula": "tcost = commish (fill at close)",
        },
        "vwap_fill": {
            "label": result.vwap.label,
            "post_pnl": result.vwap.total_post_pnl,
            "tcost": result.vwap.total_tcost,
            "commish": result.vwap.total_commish,
            "spread": result.vwap.total_spread,
            "intraday_slippage": result.vwap.total_intraday_slippage,
            "commish_bps_per_day_mean": com_mean,
            "commish_bps_per_day_median": com_med,
            "spread_bps_per_day_mean": spr_mean,
            "spread_bps_per_day_median": spr_med,
            "intraday_slippage_bps_per_day_mean": slip_mean,
            "intraday_slippage_bps_per_day_median": slip_med,
            "tcost_bps_per_day_mean": vwap_mean,
            "tcost_bps_per_day_median": vwap_med,
            "post_sharpe": result.vwap.post_pooled.get("sharpe"),
            "post_ann_ret": result.vwap.post_pooled.get("ann_ret"),
            "formula": "tcost = commish + half-spread + VWAP−close",
        },
    }
