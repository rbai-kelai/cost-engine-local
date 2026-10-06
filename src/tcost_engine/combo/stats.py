"""Path metrics aligned with Stage C / TYS (signal-sel-opt)."""

from __future__ import annotations

import math

PERIODS_PER_YEAR = 252


def path_metrics(ret, *, periods_per_year: int = PERIODS_PER_YEAR) -> dict[str, float]:
    """Ann. ret/vol/Sharpe and wealth maxDD on a daily return series."""
    import numpy as np
    import pandas as pd

    x = pd.Series(ret).dropna().astype(float)
    if len(x) < 2:
        return {
            "n": float(len(x)),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "ann_ret": float("nan"),
            "max_dd": float("nan"),
        }
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    ann_ret = mu * periods_per_year
    ann_vol = sd * math.sqrt(periods_per_year) if sd > 0 else float("nan")
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else float("nan")
    wealth = (1.0 + x).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    max_dd = float((-dd).max()) if len(dd) else float("nan")
    return {
        "n": float(len(x)),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "ann_ret": ann_ret,
        "max_dd": max_dd,
    }


def daily_turnover(panel):
    """2-way daily TO: ``‖SOD_t − SOD_{t-1}‖₁ / ‖SOD_{t-1}‖₁`` (not halved)."""
    import numpy as np
    import pandas as pd

    if panel.empty or len(panel.index) < 2:
        return pd.Series(dtype=float)
    prev = panel.shift(1)
    gmv_prev = prev.abs().sum(axis=1)
    delta = (panel - prev).abs().sum(axis=1)
    to = delta / gmv_prev.replace(0.0, np.nan)
    out = to.iloc[1:].replace([np.inf, -np.inf], np.nan)
    out.index = pd.to_datetime(out.index).normalize()
    out.name = "daily_to"
    return out


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _pct2(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _num(x: float, nd: int = 2) -> str:
    return f"{float(x):.{nd}f}"


def _to(x: float, nd: int = 1) -> str:
    return f"{100.0 * float(x):.{nd}f}%"


def yearly_perf_frame(
    daily,
    *,
    ret_col: str,
    gmv_col: str = "gmv",
    to_col: str = "daily_to",
    date_col: str = "date",
    periods_per_year: int = PERIODS_PER_YEAR,
):
    """Year rows + pooled footer; returns (display_df, numeric_df)."""
    import numpy as np
    import pandas as pd

    if daily is None or len(daily) == 0:
        empty = pd.DataFrame(
            columns=[
                "Year",
                "n_days",
                "medGMV",
                "Sharpe",
                "ret",
                "vol",
                "maxDD",
                "daily TO",
            ]
        )
        return empty, empty.copy()

    df = daily.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df["year"] = df[date_col].dt.year

    num_rows: list[dict] = []
    for year, g in df.groupby("year"):
        m = path_metrics(g[ret_col], periods_per_year=periods_per_year)
        med_gmv = float(np.nanmedian(g[gmv_col])) if gmv_col in g else float("nan")
        mean_to = float(np.nanmean(g[to_col])) if to_col in g else float("nan")
        num_rows.append(
            {
                "year": int(year),
                "n_days": int(m["n"]),
                "median_gmv": med_gmv,
                "sharpe": m["sharpe"],
                "ann_ret": m["ann_ret"],
                "ann_vol": m["ann_vol"],
                "max_dd": m["max_dd"],
                "daily_to": mean_to,
            }
        )
    numeric = pd.DataFrame(num_rows).sort_values("year") if num_rows else pd.DataFrame()

    pooled = path_metrics(df[ret_col], periods_per_year=periods_per_year)
    years = sorted(int(y) for y in df["year"].unique())
    if len(years) >= 2:
        pooled_label = f"{years[0]}–{str(years[-1])[-2:]}"
    elif years:
        pooled_label = str(years[0])
    else:
        pooled_label = "pooled"
    pooled_row = {
        "year": pooled_label,
        "n_days": int(pooled["n"]),
        "median_gmv": float(np.nanmedian(df[gmv_col])) if gmv_col in df else float("nan"),
        "sharpe": pooled["sharpe"],
        "ann_ret": pooled["ann_ret"],
        "ann_vol": pooled["ann_vol"],
        "max_dd": pooled["max_dd"],
        "daily_to": float(np.nanmean(df[to_col])) if to_col in df else float("nan"),
    }

    def _display_row(r: dict, *, year_as: object) -> dict:
        med = r["median_gmv"]
        return {
            "Year": year_as,
            "n_days": r["n_days"] if isinstance(year_as, int) else "",
            "medGMV": f"${med / 1e6:.1f}M" if med == med else "",
            "Sharpe": _num(r["sharpe"]) if r["sharpe"] == r["sharpe"] else "",
            "ret": _pct(r["ann_ret"]) if r["ann_ret"] == r["ann_ret"] else "",
            "vol": _pct(r["ann_vol"]) if r["ann_vol"] == r["ann_vol"] else "",
            "maxDD": _pct2(r["max_dd"]) if r["max_dd"] == r["max_dd"] else "",
            "daily TO": _to(r["daily_to"]) if r["daily_to"] == r["daily_to"] else "",
        }

    disp_rows = [_display_row(r, year_as=int(r["year"])) for r in num_rows]
    disp_rows.append(_display_row(pooled_row, year_as=pooled_label))
    display = pd.DataFrame(disp_rows)

    numeric_all = pd.concat(
        [numeric, pd.DataFrame([{**pooled_row, "year": pooled_label}])],
        ignore_index=True,
    ) if len(numeric) else pd.DataFrame([pooled_row])
    return display, numeric_all


def side_by_side(
    left: str,
    right: str,
    *,
    gap: int = 4,
    left_title: str = "",
    right_title: str = "",
) -> str:
    """Place two monospaced blocks on the same horizontal row."""
    left_lines = left.splitlines() or [""]
    right_lines = right.splitlines() or [""]
    if left_title:
        left_lines = [left_title, ""] + left_lines
    if right_title:
        right_lines = [right_title, ""] + right_lines
    width = max(len(line) for line in left_lines)
    n = max(len(left_lines), len(right_lines))
    left_lines += [""] * (n - len(left_lines))
    right_lines += [""] * (n - len(right_lines))
    sep = " " * gap
    return "\n".join(
        f"{left_lines[i].ljust(width)}{sep}{right_lines[i]}" for i in range(n)
    )
