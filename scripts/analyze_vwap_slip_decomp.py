#!/usr/bin/env python3
"""Buy/sell VWAP slip and sign(δ)×r decomposition (Stage C SOD)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcost_engine.combo.sod import join_trades_prices, load_sod_panel, sod_trades
from tcost_engine.lseg.pull import pull_cost_prices

SOD = (
    "/data/robert/stage_c_pinnet_mktbeta_pos_20261005_gto3e-4_to28/"
    "stage_c_pinnet_pos_2021_2026_gto3e-4_to28.parquet"
)
START, END = date(2021, 1, 1), date(2026, 10, 1)


def nw_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(x, weights=w))


def main() -> int:
    print("loading SOD …", flush=True)
    panel = load_sod_panel(SOD)
    panel = panel.loc[
        (panel.index >= pd.Timestamp(START)) & (panel.index <= pd.Timestamp(END))
    ]
    print(f"  panel {panel.shape[0]} days × {panel.shape[1]} names", flush=True)

    print("loading cached prices …", flush=True)
    prices = pull_cost_prices(
        None,
        start=panel.index.min().date(),
        end=panel.index.max().date(),
        infocodes=panel.columns.tolist(),
        refresh=False,
    )
    print(f"  price rows={len(prices):,}", flush=True)

    print("building share-based trades + fills …", flush=True)
    trades, tstats = sod_trades(panel, prices)
    joined, jstats = join_trades_prices(trades, prices, fill="vwap")
    dropped = tstats.n_dropped + jstats.n_dropped
    print(
        f"  trades={tstats.n_fills:,} fills={jstats.n_fills:,} dropped={dropped:,}",
        flush=True,
    )

    side = np.where(joined["side"].to_numpy() == "buy", 1.0, -1.0)
    vwap = joined["vwap"].to_numpy(dtype=np.float64)
    close = joined["close"].to_numpy(dtype=np.float64)
    qty = joined["qty"].to_numpy(dtype=np.float64)
    notional = np.abs(joined["delta_dollars"].to_numpy(dtype=np.float64))
    slip = side * (vwap - close) * qty
    r_vwap_close = (close - vwap) / close
    ctrl = side * r_vwap_close  # = -slip/notional
    slip_per_notional = slip / notional
    identity_err = float(np.nanmax(np.abs(slip_per_notional + ctrl)))
    print(f"identity max|slip/$ + sign(δ)×r| = {identity_err:.3e}", flush=True)

    gmv = panel.abs().sum(axis=1)
    gmv_by_date = gmv.copy()
    gmv_by_date.index = pd.to_datetime(gmv_by_date.index).normalize()

    j = joined.copy()
    j["slip"] = slip
    j["notional"] = notional
    j["r"] = r_vwap_close
    j["ctrl"] = ctrl
    j["date"] = pd.to_datetime(j["date"]).dt.normalize()

    daily = j.groupby("date").agg(
        slip=("slip", "sum"),
        notional=("notional", "sum"),
        n=("slip", "size"),
    )
    daily["gmv"] = gmv_by_date.reindex(daily.index)
    daily["slip_bps_gmv"] = 1e4 * daily["slip"] / daily["gmv"]
    daily["to"] = daily["notional"] / daily["gmv"]
    j["ctrl_dollars"] = j["ctrl"] * j["notional"]
    daily["ctrl_bps_gmv"] = (
        1e4 * j.groupby("date")["ctrl_dollars"].sum() / daily["gmv"]
    )

    def side_daily(side_name: str) -> pd.DataFrame:
        sub = j[j["side"] == side_name]
        d = sub.groupby("date").agg(slip=("slip", "sum"), notional=("notional", "sum"))
        d["gmv"] = gmv_by_date.reindex(d.index)
        d["slip_bps_gmv"] = 1e4 * d["slip"] / d["gmv"]
        d["slip_bps_notional"] = 1e4 * d["slip"] / d["notional"]
        return d

    buy_d = side_daily("buy")
    sell_d = side_daily("sell")

    print("\n=== Headline (match debug_perturb /GMV bps) ===")
    print(f"mean slip bps/day (all): {daily['slip_bps_gmv'].mean():.4f}")
    print(f"median slip bps/day:     {daily['slip_bps_gmv'].median():.4f}")
    print(f"mean daily TO:           {daily['to'].mean():.4%}")

    print("\n=== Buy vs Sell (bps of book GMV / day) ===")
    print(
        f"BUY  mean slip bps/day:  {buy_d['slip_bps_gmv'].mean():.4f}"
        f"  median {buy_d['slip_bps_gmv'].median():.4f}"
    )
    print(
        f"SELL mean slip bps/day:  {sell_d['slip_bps_gmv'].mean():.4f}"
        f"  median {sell_d['slip_bps_gmv'].median():.4f}"
    )
    print(f"BUY  mean slip bps/notional: {buy_d['slip_bps_notional'].mean():.4f}")
    print(f"SELL mean slip bps/notional: {sell_d['slip_bps_notional'].mean():.4f}")
    print(f"BUY  mean daily notional $:  {buy_d['notional'].mean():,.0f}")
    print(f"SELL mean daily notional $: {sell_d['notional'].mean():,.0f}")

    print("\n=== Trade-level (notional-weighted) ===")
    masks = {
        "ALL": np.ones(len(j), dtype=bool),
        "BUY": j["side"].eq("buy").to_numpy(),
        "SELL": j["side"].eq("sell").to_numpy(),
    }
    for name, mask in masks.items():
        w = notional[mask]
        print(
            f"{name:4s}  slip/notional bps={1e4 * nw_mean(slip_per_notional[mask], w):+.4f}"
            f"  E[r]={1e4 * nw_mean(r_vwap_close[mask], w):+.4f} bps"
            f"  E[sign(δ)×r]={1e4 * nw_mean(ctrl[mask], w):+.4f} bps"
            f"  n={int(mask.sum()):,}"
        )

    # close-to-close adjusted return on trade day
    px = prices.copy()
    px["marketdate"] = pd.to_datetime(px["marketdate"]).dt.normalize()
    adj_col = "close_adjusted" if "close_adjusted" in px.columns else "close"
    adj = (
        px.pivot_table(
            index="marketdate", columns="infocode", values=adj_col, aggfunc="last"
        )
        .sort_index()
    )
    ret_cc = adj / adj.shift(1) - 1.0
    ret_long = ret_cc.stack(future_stack=True).rename("ret_cc").reset_index()
    ret_long.columns = ["date", "infocode", "ret_cc"]
    ret_long["date"] = pd.to_datetime(ret_long["date"]).dt.normalize()
    ret_long["infocode"] = ret_long["infocode"].astype(np.int64)
    jm = j.merge(ret_long, on=["date", "infocode"], how="left")
    jm = jm[np.isfinite(jm["ret_cc"].to_numpy())]
    sign = np.where(jm["side"].to_numpy() == "buy", 1.0, -1.0)
    jm["sign_r"] = sign * jm["ret_cc"].to_numpy()
    jm["sign_r_dollars"] = jm["sign_r"] * jm["notional"]

    daily2 = jm.groupby("date").agg(
        slip=("slip", "sum"),
        notional=("notional", "sum"),
        sign_r_dollars=("sign_r_dollars", "sum"),
    )
    daily2["gmv"] = gmv_by_date.reindex(daily2.index)
    daily2["slip_bps"] = 1e4 * daily2["slip"] / daily2["gmv"]
    daily2["sign_r_bps"] = 1e4 * daily2["sign_r_dollars"] / daily2["gmv"]

    x = daily2["sign_r_bps"].to_numpy()
    y = daily2["slip_bps"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(resid**2) / ss_tot if ss_tot > 0 else np.nan

    x_m = daily["ctrl_bps_gmv"].to_numpy()
    y_m = daily["slip_bps_gmv"].to_numpy()
    okm = np.isfinite(x_m) & np.isfinite(y_m)
    Xm = np.column_stack([np.ones(int(okm.sum())), x_m[okm]])
    betam, *_ = np.linalg.lstsq(Xm, y_m[okm], rcond=None)
    residm = y_m[okm] - Xm @ betam
    r2m = 1 - np.sum(residm**2) / np.sum((y_m[okm] - y_m[okm].mean()) ** 2)

    print("\n=== Mechanical: slip ≡ −sign(δ)×r_vwap→close ===")
    print(f"mean ctrl bps/day (= −slip): {daily['ctrl_bps_gmv'].mean():.4f}")
    print(
        f"OLS slip ~ a + b·ctrl: a={betam[0]:.6f} b={betam[1]:.6f} R²={r2m:.6f}"
    )
    print(f"mean residual after ctrl: {residm.mean():.6f} bps")

    print("\n=== Selection: slip vs sign(δ)×r_cc (daily /GMV bps) ===")
    print(f"mean sign(δ)×r_cc bps/day: {daily2['sign_r_bps'].mean():.4f}")
    print(f"corr(slip, signδ×r_cc): {np.corrcoef(x, y)[0, 1]:.4f}")
    print(
        f"OLS slip ~ a + b·(signδ×r_cc): a={beta[0]:.4f} b={beta[1]:.4f} R²={r2:.4f}"
    )
    print(f"mean residual after control: {resid.mean():.4f} bps")
    print(f"median residual: {np.median(resid):.4f} bps")
    print(f"std residual: {resid.std():.4f} bps")
    slip_mean = float(y.mean())
    explained = float(beta[1] * x.mean())
    print(
        f"E[slip]={slip_mean:.4f}  intercept(leftover)={beta[0]:.4f}"
        f"  b·E[signδ r]={explained:.4f}"
    )

    for side_name in ("buy", "sell"):
        sub = jm[jm["side"] == side_name]
        d = sub.groupby("date").agg(slip=("slip", "sum"), srd=("sign_r_dollars", "sum"))
        d["gmv"] = gmv_by_date.reindex(d.index)
        print(
            f"{side_name.upper():4s} E[slip]={1e4 * (d['slip'] / d['gmv']).mean():.4f}"
            f"  E[signδ×r_cc]={1e4 * (d['srd'] / d['gmv']).mean():.4f} bps/day"
        )

    print("\n=== Directional selection into VWAP→close ===")
    buy_mask = j["side"].eq("buy").to_numpy()
    sell_mask = j["side"].eq("sell").to_numpy()
    buy_r = nw_mean(r_vwap_close[buy_mask], notional[buy_mask])
    sell_r = nw_mean(r_vwap_close[sell_mask], notional[sell_mask])
    print(
        f"BUY  NW E[r_vwap→close]={1e4 * buy_r:+.4f} bps"
        f"  (close>VWAP ⇒ buy slip negative)"
    )
    print(
        f"SELL NW E[r_vwap→close]={1e4 * sell_r:+.4f} bps"
        f"  (close<VWAP ⇒ sell slip negative)"
    )

    print("\n=== VERDICT INPUTS ===")
    print(f"headline mean slip: {daily['slip_bps_gmv'].mean():.4f} bps/day")
    print(
        f"of which BUY: {buy_d['slip_bps_gmv'].mean():.4f}"
        f"  SELL: {sell_d['slip_bps_gmv'].mean():.4f}"
    )
    print(
        f"cc-selection R²={r2:.3f}; residual mean after cc control="
        f"{resid.mean():.4f} bps"
    )
    print(f"mechanical VWAP→close explains slip with R²={r2m:.3f} (tautology)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
