"""Lag-1 dollar PnL for a combo SOD dollar panel (pre-tcost)."""

from __future__ import annotations


def lag1_dollar_pnl(panel, prices, *, price_col: str = "close_adjusted"):
    """Daily dollar PnL: prior SOD notionals × today's close-to-close return.

    ``pnl_t = Σ_i SOD_{i,t-1} × (P_{i,t} / P_{i,t-1} − 1)``

    *panel* is wide (DatetimeIndex × INFOCODE). *prices* is long with
    ``marketdate``, ``infocode``, and *price_col* (default ``close_adjusted``,
    falling back to ``close`` when the adjusted column is absent).
    """
    import numpy as np
    import pandas as pd

    col = price_col if price_col in prices.columns else "close"
    if col not in prices.columns:
        raise KeyError(
            f"prices need {price_col!r} or 'close'; got {list(prices.columns)}"
        )

    px = prices.copy()
    px["marketdate"] = pd.to_datetime(px["marketdate"]).dt.normalize()
    wide = (
        px.pivot_table(index="marketdate", columns="infocode", values=col, aggfunc="last")
        .sort_index()
    )
    # Align to SOD columns / dates
    common_ids = [c for c in panel.columns if c in wide.columns]
    if not common_ids:
        return pd.DataFrame(
            columns=["date", "pre_pnl", "gmv", "pre_ret"],
        )

    sod = panel[common_ids].sort_index()
    pxw = wide[common_ids].reindex(sod.index)
    ret = pxw / pxw.shift(1) - 1.0
    prior = sod.shift(1)
    # First day has no prior book
    pnl = (prior * ret).sum(axis=1, min_count=1)
    gmv = prior.abs().sum(axis=1)
    out = pd.DataFrame(
        {
            "date": sod.index,
            "pre_pnl": pnl.to_numpy(dtype=np.float64),
            "gmv": gmv.to_numpy(dtype=np.float64),
        }
    )
    out["pre_ret"] = np.where(out["gmv"] > 0, out["pre_pnl"] / out["gmv"], np.nan)
    # Drop the first SOD day (no lag-1 PnL)
    out = out.iloc[1:].reset_index(drop=True)
    return out
