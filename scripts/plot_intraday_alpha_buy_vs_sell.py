#!/usr/bin/env python3
"""Plot buy vs sell intraday alpha (= −intraday slippage) for Stage C SOD.

Intraday slippage (cost-signed): side × (VWAP − close) × qty
  side = +1 buy (δ$ > 0), −1 sell (δ$ < 0)
Intraday alpha (PnL-signed): −slippage

Daily series: sum(alpha_$)/GMV in bps for buys and sells separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcost_engine.combo.sod import join_trades_prices, load_sod_panel, sod_trades

SOD = Path("/tmp/stage_c_sod.parquet")
PX = Path.home() / ".cache/tcost-engine/prices/cost_prices_315ba3e4c30a06f8.parquet"
DAILY = ROOT / "outputs/perturb_tcost/daily_pre_post.parquet"
OUT = ROOT / "outputs/perturb_tcost/intraday_alpha_buy_vs_sell.png"


def main() -> int:
    print("loading prices…", flush=True)
    prices = pd.read_parquet(PX)
    print(f"  prices {len(prices):,}", flush=True)

    print("loading SOD…", flush=True)
    panel = load_sod_panel(SOD)
    print(f"  SOD {panel.shape}", flush=True)

    print("building trades + fills…", flush=True)
    trades, sod_stats = sod_trades(panel, prices)
    joined, join_stats = join_trades_prices(trades, prices, fill="vwap")
    print(
        f"  trades={sod_stats.n_trades:,} fills={join_stats.n_fills:,} "
        f"dropped={sod_stats.n_dropped + join_stats.n_dropped:,}",
        flush=True,
    )

    side = np.where(joined["side"].to_numpy() == "buy", 1.0, -1.0)
    qty = joined["qty"].to_numpy(dtype=np.float64)
    vwap = joined["vwap"].to_numpy(dtype=np.float64)
    close = joined["close"].to_numpy(dtype=np.float64)
    slip = side * (vwap - close) * qty  # cost-signed
    alpha = -slip  # PnL-signed

    j = joined.copy()
    j["slip"] = slip
    j["alpha"] = alpha
    j["date"] = pd.to_datetime(j["date"]).dt.normalize()

    gmv = panel.abs().sum(axis=1)
    gmv.index = pd.to_datetime(gmv.index).normalize()

    daily_ref = pd.read_parquet(DAILY)
    daily_ref["date"] = pd.to_datetime(daily_ref["date"]).dt.normalize()

    def side_daily(side_name: str) -> pd.DataFrame:
        sub = j[j["side"] == side_name]
        d = sub.groupby("date").agg(
            alpha=("alpha", "sum"),
            slip=("slip", "sum"),
            n=("alpha", "size"),
        )
        d["gmv"] = gmv.reindex(d.index)
        d["alpha_bps_gmv"] = 1e4 * d["alpha"] / d["gmv"]
        return d

    buy_d = side_daily("buy")
    sell_d = side_daily("sell")

    # Align to full sample calendar from daily_pre_post
    cal = daily_ref.set_index("date").index
    buy_d = buy_d.reindex(cal)
    sell_d = sell_d.reindex(cal)
    buy_d["gmv"] = gmv.reindex(cal)
    sell_d["gmv"] = gmv.reindex(cal)
    for d in (buy_d, sell_d):
        d["alpha"] = d["alpha"].fillna(0.0)
        d["slip"] = d["slip"].fillna(0.0)
        d["alpha_bps_gmv"] = 1e4 * d["alpha"] / d["gmv"]

    all_slip = j.groupby("date")["slip"].sum().reindex(cal).fillna(0.0)
    sanity = daily_ref.set_index("date")["intraday_slippage"].reindex(cal)
    corr = float(all_slip.corr(sanity))
    diff = float((all_slip - sanity).abs().sum())
    print(f"sanity vs daily_pre_post: corr={corr:.6f} sum|Δ|={diff:,.2f}", flush=True)

    def summarize(name: str, d: pd.DataFrame) -> None:
        bps = d["alpha_bps_gmv"]
        print(
            f"{name:4s}  mean α bps/day={bps.mean():+.4f}  "
            f"median={bps.median():+.4f}  "
            f"total α $={d['alpha'].sum():+,.0f}  "
            f"n_days={bps.notna().sum()}",
            flush=True,
        )

    print("\n=== Intraday alpha (= −slip) buy vs sell ===", flush=True)
    summarize("BUY", buy_d)
    summarize("SELL", sell_d)
    total_d = buy_d["alpha"] + sell_d["alpha"]
    total_bps = 1e4 * total_d / buy_d["gmv"]
    print(
        f"ALL   mean α bps/day={total_bps.mean():+.4f}  "
        f"median={total_bps.median():+.4f}  "
        f"total α $={total_d.sum():+,.0f}",
        flush=True,
    )

    # --- plot ---
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2.2, 1.4, 1.2]}
    )
    fig.suptitle(
        "Intraday alpha (PnL) = −side×(VWAP−close)×qty\n"
        "Daily α / GMV (bps) — buys vs sells",
        fontsize=12,
        y=0.98,
    )

    ax = axes[0]
    # light rolling for readability
    win = 21
    buy_roll = buy_d["alpha_bps_gmv"].rolling(win, min_periods=5).mean()
    sell_roll = sell_d["alpha_bps_gmv"].rolling(win, min_periods=5).mean()
    ax.plot(
        buy_d.index,
        buy_d["alpha_bps_gmv"],
        color="#2a6fdb",
        alpha=0.22,
        lw=0.6,
        label="Buy daily",
    )
    ax.plot(
        sell_d.index,
        sell_d["alpha_bps_gmv"],
        color="#c44e52",
        alpha=0.22,
        lw=0.6,
        label="Sell daily",
    )
    ax.plot(buy_d.index, buy_roll, color="#2a6fdb", lw=1.6, label=f"Buy {win}d MA")
    ax.plot(sell_d.index, sell_roll, color="#c44e52", lw=1.6, label=f"Sell {win}d MA")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.axhline(
        buy_d["alpha_bps_gmv"].mean(),
        color="#2a6fdb",
        ls="--",
        lw=1,
        alpha=0.8,
    )
    ax.axhline(
        sell_d["alpha_bps_gmv"].mean(),
        color="#c44e52",
        ls="--",
        lw=1,
        alpha=0.8,
    )
    ax.set_ylabel("α bps / GMV")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    buy_m, sell_m = buy_d["alpha_bps_gmv"].mean(), sell_d["alpha_bps_gmv"].mean()
    buy_med, sell_med = (
        buy_d["alpha_bps_gmv"].median(),
        sell_d["alpha_bps_gmv"].median(),
    )
    ax.text(
        0.01,
        0.02,
        (
            f"Buy  mean {buy_m:+.3f} / med {buy_med:+.3f} bps/day   "
            f"Σα ${buy_d['alpha'].sum()/1e6:+.2f}M\n"
            f"Sell mean {sell_m:+.3f} / med {sell_med:+.3f} bps/day   "
            f"Σα ${sell_d['alpha'].sum()/1e6:+.2f}M"
        ),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, lw=0.4),
    )

    ax = axes[1]
    ax.plot(
        buy_d.index,
        buy_d["alpha"].cumsum() / 1e6,
        color="#2a6fdb",
        lw=1.5,
        label="Buy cum α",
    )
    ax.plot(
        sell_d.index,
        sell_d["alpha"].cumsum() / 1e6,
        color="#c44e52",
        lw=1.5,
        label="Sell cum α",
    )
    ax.plot(
        buy_d.index,
        total_d.cumsum() / 1e6,
        color="#333333",
        lw=1.2,
        ls="--",
        label="Total cum α",
    )
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_ylabel("Cumulative α ($M)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    bins = np.linspace(-6, 6, 81)
    ax.hist(
        buy_d["alpha_bps_gmv"].dropna(),
        bins=bins,
        color="#2a6fdb",
        alpha=0.55,
        density=True,
        label="Buy",
    )
    ax.hist(
        sell_d["alpha_bps_gmv"].dropna(),
        bins=bins,
        color="#c44e52",
        alpha=0.55,
        density=True,
        label="Sell",
    )
    ax.axvline(buy_m, color="#2a6fdb", ls="--", lw=1.2)
    ax.axvline(sell_m, color="#c44e52", ls="--", lw=1.2)
    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("Daily α bps / GMV")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT}", flush=True)

    # also write a small CSV for reuse
    csv_out = OUT.with_suffix(".csv")
    out_df = pd.DataFrame(
        {
            "date": cal,
            "buy_alpha_$": buy_d["alpha"].to_numpy(),
            "sell_alpha_$": sell_d["alpha"].to_numpy(),
            "buy_alpha_bps_gmv": buy_d["alpha_bps_gmv"].to_numpy(),
            "sell_alpha_bps_gmv": sell_d["alpha_bps_gmv"].to_numpy(),
            "gmv": buy_d["gmv"].to_numpy(),
        }
    )
    out_df.to_csv(csv_out, index=False)
    print(f"saved {csv_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
