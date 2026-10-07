"""Buy/sell + with/against-day + residual-after-sign(δ)×r VWAP slip decomposition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOD = Path(
    "/data/robert/stage_c_pinnet_mktbeta_pos_20261005_gto3e-4_to28/"
    "stage_c_pinnet_pos_2021_2026_gto3e-4_to28.parquet"
)
PX = Path.home() / ".cache/tcost-engine/prices/cost_prices_315ba3e4c30a06f8.parquet"
DAILY = Path("outputs/perturb_tcost/daily_pre_post.parquet")

sys.path.insert(0, "src")
from tcost_engine.combo.sod import join_trades_prices, load_sod_panel, sod_trades  # noqa: E402


def main() -> None:
    print("loading prices…", flush=True)
    prices = pd.read_parquet(PX)
    print(
        f"  prices {len(prices):,}  "
        f"{prices['marketdate'].min().date()}→{prices['marketdate'].max().date()}",
        flush=True,
    )

    print("loading SOD…", flush=True)
    panel = load_sod_panel(SOD)
    print(f"  SOD {panel.shape}", flush=True)

    print("building trades…", flush=True)
    trades, sod_stats = sod_trades(panel, prices)
    print(f"  trades={len(trades):,} dropped={sod_stats.n_dropped:,}", flush=True)

    print("joining fills…", flush=True)
    joined, join_stats = join_trades_prices(trades, prices, fill="vwap")
    print(f"  fills={join_stats.n_fills:,} dropped={join_stats.n_dropped:,}", flush=True)

    side = np.where(joined["side"].to_numpy() == "buy", 1.0, -1.0)
    qty = joined["qty"].to_numpy(dtype=np.float64)
    vwap = joined["vwap"].to_numpy(dtype=np.float64)
    close = joined["close"].to_numpy(dtype=np.float64)
    delta = joined["delta_dollars"].to_numpy(dtype=np.float64)
    slip = side * (vwap - close) * qty
    notional = np.abs(delta)

    px = prices.copy()
    px["marketdate"] = pd.to_datetime(px["marketdate"]).dt.normalize()
    px = px.sort_values(["infocode", "marketdate"])
    px["r_t"] = px.groupby("infocode")["close_adjusted"].pct_change()

    j = joined.merge(
        px[["marketdate", "infocode", "r_t"]],
        left_on=["date", "infocode"],
        right_on=["marketdate", "infocode"],
        how="left",
    )
    r = j["r_t"].to_numpy(dtype=np.float64)
    ok_r = np.isfinite(r) & (r != 0) & np.isfinite(slip)

    sign_d = np.sign(delta)
    sign_r = np.sign(r)
    with_day = ok_r & (sign_d * sign_r > 0)
    against_day = ok_r & (sign_d * sign_r < 0)
    sdr = sign_d * r

    daily = pd.read_parquet(DAILY)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    mean_gmv = float(daily["gmv"].mean())
    med_gmv = float(daily["gmv"].median())
    n_days = int(len(daily))

    def bps_of_notional(x: float, n: float) -> float:
        return 1e4 * float(x) / float(n) if n else float("nan")

    def bps_per_day_gmv(x: float) -> float:
        return 1e4 * float(x) / (mean_gmv * n_days)

    def summarize(mask: np.ndarray, label: str) -> dict:
        s = float(slip[mask].sum())
        n = float(notional[mask].sum())
        nf = int(mask.sum())
        return {
            "bucket": label,
            "n_fills": nf,
            "notional_$": n,
            "slip_$": s,
            "slip_bps_notional": bps_of_notional(s, n),
            "slip_bps_per_day_gmv": bps_per_day_gmv(s),
            "pct_fills": 100.0 * nf / len(slip),
            "pct_notional": 100.0 * n / float(notional.sum()),
            "pct_slip": 100.0 * s / float(slip.sum()) if slip.sum() != 0 else float("nan"),
        }

    rows = [
        summarize(np.ones(len(slip), dtype=bool), "ALL"),
        summarize(joined["side"].to_numpy() == "buy", "BUY"),
        summarize(joined["side"].to_numpy() == "sell", "SELL"),
        summarize(with_day, "WITH_DAY (signδ·sign r>0)"),
        summarize(against_day, "AGAINST_DAY (signδ·sign r<0)"),
        summarize(~ok_r, "NO_R (missing/0 r)"),
    ]

    frac = np.where(notional > 0, slip / notional, np.nan)
    mask = ok_r & np.isfinite(frac)
    x = sdr[mask]
    y = frac[mask]
    w = notional[mask]
    W = float(w.sum())
    xbar = float((w * x).sum() / W)
    ybar = float((w * y).sum() / W)
    varx = float((w * (x - xbar) ** 2).sum() / W)
    vary = float((w * (y - ybar) ** 2).sum() / W)
    covxy = float((w * (x - xbar) * (y - ybar)).sum() / W)
    b = covxy / varx if varx > 0 else float("nan")
    a = ybar - b * xbar
    resid = y - (a + b * x)
    explained_dollars = (b * x) * w
    intercept_dollars = a * w
    resid_dollars = resid * w
    slip_ok = slip[mask]
    b0 = float((w * x * y).sum() / (w * x * x).sum())
    explained0_dollars = (b0 * x) * w
    resid0_dollars = slip_ok - explained0_dollars

    print("\n=== VWAP−close slip decomposition ===")
    print(
        f"fills={len(slip):,}  meanGMV=${mean_gmv:,.0f}  "
        f"medGMV=${med_gmv:,.0f}  n_days={n_days}"
    )
    print(
        f"total slip=${slip.sum():,.0f}  "
        f"({bps_per_day_gmv(float(slip.sum())):+.4f} bps/day /GMV)"
    )
    print(f"total notional=${notional.sum():,.0f}")
    print()
    sumdf = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print(sumdf.to_string(index=False))

    print("\n=== OLS: slip/|δ$| = a + b·(signδ·r_t)  (notional-weighted) ===")
    corr = covxy / np.sqrt(varx * vary) if varx > 0 and vary > 0 else float("nan")
    print(f"a={a:.6f}  b={b:.4f}  corr={corr:.4f}")
    print(
        f"explained(β·sδr) $={explained_dollars.sum():,.0f}  "
        f"({bps_per_day_gmv(float(explained_dollars.sum())):+.4f} bps/day)"
    )
    print(
        f"intercept $={intercept_dollars.sum():,.0f}  "
        f"({bps_per_day_gmv(float(intercept_dollars.sum())):+.4f} bps/day)"
    )
    print(
        f"residual $={resid_dollars.sum():,.0f}  "
        f"({bps_per_day_gmv(float(resid_dollars.sum())):+.4f} bps/day)"
    )
    print(
        f"check sum="
        f"{explained_dollars.sum() + intercept_dollars.sum() + resid_dollars.sum():,.0f} "
        f"vs slip_ok={slip_ok.sum():,.0f}"
    )

    print("\n=== Zero-intercept: slip/|δ$| = b0·(signδ·r_t) ===")
    print(f"b0={b0:.4f}")
    print(
        f"δ×r component $={explained0_dollars.sum():,.0f}  "
        f"({bps_per_day_gmv(float(explained0_dollars.sum())):+.4f} bps/day)"
    )
    print(
        f"residual after sign(δ)×r $={resid0_dollars.sum():,.0f}  "
        f"({bps_per_day_gmv(float(resid0_dollars.sum())):+.4f} bps/day)"
    )
    print(
        f"pct of slip explained by δ×r="
        f"{100 * explained0_dollars.sum() / slip_ok.sum():.1f}%"
    )

    print("\n=== Cross: side × with/against ===")
    cross_rows = []
    for side_name, sm in [
        ("buy", joined["side"].to_numpy() == "buy"),
        ("sell", joined["side"].to_numpy() == "sell"),
    ]:
        for day_name, dm in [("with", with_day), ("against", against_day)]:
            cross_rows.append(summarize(sm & dm, f"{side_name.upper()}×{day_name}"))
    print(pd.DataFrame(cross_rows).to_string(index=False))

    print("\n=== Mean slip frac (slip/|δ$|, bps) ===")
    for label, m in [
        ("ALL", np.ones(len(slip), dtype=bool)),
        ("BUY", joined["side"].to_numpy() == "buy"),
        ("SELL", joined["side"].to_numpy() == "sell"),
        ("WITH", with_day),
        ("AGAINST", against_day),
    ]:
        m2 = m & np.isfinite(frac)
        print(
            f"{label:8s}  mean={1e4 * np.average(frac[m2], weights=notional[m2]):+.3f} bps  "
            f"median_unw={1e4 * np.median(frac[m2]):+.3f} bps  n={m2.sum():,}"
        )

    j["slip"] = slip
    j["notional"] = notional
    by = j.groupby("date").agg(slip=("slip", "sum"), notional=("notional", "sum"))
    merged = daily.set_index("date").join(by, how="inner")
    print("\n=== Sanity vs daily_pre_post intraday_slippage ===")
    print("corr", merged["intraday_slippage"].corr(merged["slip"]))
    print(
        "sum daily",
        merged["intraday_slippage"].sum(),
        "sum rebuilt",
        merged["slip"].sum(),
        "diff",
        merged["intraday_slippage"].sum() - merged["slip"].sum(),
    )

    print("\n=== VERDICT INPUTS ===")
    buy_m = joined["side"].to_numpy() == "buy"
    sell_m = joined["side"].to_numpy() == "sell"
    print("buy_slip_bps_day", bps_per_day_gmv(float(slip[buy_m].sum())))
    print("sell_slip_bps_day", bps_per_day_gmv(float(slip[sell_m].sum())))
    print("with_bps_day", bps_per_day_gmv(float(slip[with_day].sum())))
    print("against_bps_day", bps_per_day_gmv(float(slip[against_day].sum())))
    print("residual0_bps_day", bps_per_day_gmv(float(resid0_dollars.sum())))
    print("explained0_bps_day", bps_per_day_gmv(float(explained0_dollars.sum())))
    print("with_notional_share", float(notional[with_day].sum() / notional[ok_r].sum()))
    print(
        "against_notional_share",
        float(notional[against_day].sum() / notional[ok_r].sum()),
    )


if __name__ == "__main__":
    main()
