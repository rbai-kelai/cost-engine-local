# tcost-engine

Costs a trade with the Northfield / diBartolomeo decomposition used across the
execution literature (also echoed by Deutsche Bank and Bocconi surveys):

```text
total = commish + spread + market_impact + residual
```

| Term | Status | Meaning |
| --- | --- | --- |
| Commish | Modeled | Fixed broker commission at **10 mils/share** (`$0.001`; 1 mil = `$0.0001`), plus optional fees. |
| Spread | Modeled | Bid-ask half-spread. LSEG EOD bid/ask are the proxy for that day's intraday spread. |
| Market impact | Deferred (`0`) | Size-dependent price move from *this* trade (temporary / permanent; often √size). |
| Residual | Modeled | VWAP vs close benchmark slippage: `side × (VWAP − close) × qty`. |

Amounts are in the price currency. Inputs are `Decimal`, `str`, or `int`. Floats
are rejected so a value like `0.005` cannot pick up a binary fraction. Positive
amounts are costs paid by the trader. A rebate or a maker spread capture is
negative.

`Fill.price` is the VWAP (execution). `Fill.close` is the close benchmark for
slippage. Spread uses EOD bid/ask as an intraday-width proxy — not fill-time
quotes. Size-dependent market impact is still deferred.

## Commish

Commish is charged once per order. Fills that share an `order_id` are one order.
A fill with no `order_id` is its own order. Use `cost_many` for a blotter so a
minimum or flat fee is not charged again on every fill.

### Broker commission

| Schedule | Charge |
| --- | --- |
| `PerShare(mils, minimum=0)` | `max(mils / 10,000 × order quantity, minimum)`. One mil is `$0.0001`/share. |
| `FlatFee(amount)` | `amount` once per order |
| `BpsOfNotional(bps, minimum=0)` | `max(execution notional × bps / 10,000, minimum)` |
| `PercentOfNotional(percent, minimum=0)` | `max(execution notional × percent / 100, minimum)`. 1% is 100 bps. |
| `PerFill(schedule)` | Apply `schedule` to each fill, then sum. |
| `Composite(*schedules)` | Sum of schedules on the same order. |
| `AtLeast(schedule, minimum)` | Floor on the combined order charge. |
| `NoCommission()` | Zero. |

Execution notional is `quantity × VWAP`. A minimum is a floor on what the trader
pays. Leave it at 0 when the rate is a rebate. The house fixed commission is
**10 mils** (`$0.001` per share).

### Fees and taxes

Compose with commission via `Composite`. Side-aware fees need the order side
(sells for US SEC / FINRA; buys for stamp duty).

| Schedule | Charge |
| --- | --- |
| `SecFee(rate)` | `rate × sell notional` (SEC Section 31). Buys are zero. Pass the live published rate. |
| `FinraTaf(rate_per_share, cap)` | Per-share sell fee with an optional cap. Buys are zero. Defaults are illustrative. |
| `StampDuty(percent=0.5)` | `percent / 100 × buy notional` (UK SDRT-style). Sells are zero. |
| `OnBuy(schedule)` / `OnSell(schedule)` | Apply any schedule on one side only. |

When an order has several fills, the order commish charge is split across them
pro rata by quantity so the fill totals add up.

## Spread

Spread is the quoted width, not VWAP−close slippage. With LSEG Datastream2,
`BID` / `ASK` are closing prints used as a **proxy for that day's intraday
spread**. Deutsche Bank’s PM guidebook treats bid-ask spread and price impact
as the two *intraday* cost levers; we model the first from EOD quotes and leave
size-dependent impact at zero.

For a bid/ask, the one-way cost of taking liquidity is the half-spread:

```text
half_spread = (ask - bid) / 2
taker cost  = half_spread × quantity
```

A buy lifting the ask and a sell hitting the bid are each half a spread from the
mid, so the cost is the same on both sides. A locked quote (`bid == ask`) costs
nothing. A crossed quote is rejected.

`FullSpreadBps` is the quoted width `(ask - bid) / reference × 10,000`. The taker
pays half of it. `OneWaySpreadBps` is already that one-way cost and is not
halved. If you omit the reference price, the execution price is the reference,
and the taker cost in bps of execution notional equals that one-way spread exactly.

| Liquidity | Spread cost |
| --- | --- |
| Taker | `+ half_spread × quantity` |
| Maker | `- maker_capture × half_spread × quantity` |
| Midpoint | `0` |

`maker_capture` is a fraction from 0 to 1 on the model. The default is 0: a maker
pays no spread and is not credited with earning one. At 1, the maker earns the
full half-spread.

## Residual — VWAP vs close slippage

When `Fill.close` is set, blotter residual is the **cost-signed** close-benchmark
slippage of a VWAP fill (also `OrderCost.slippage` / `BlotterCost.slippage`):

```text
slippage = side × (VWAP − close) × quantity   # positive = adverse
```

with buy = `+1` and sell = `−1`. Omit `close` to leave residual at zero.

Combo SOD / `debug_perturb` instead report **PnL-signed** `intraday_slippage`
(`side × (close − VWAP) × qty`) and set `tcost = commish + intraday_slippage`.

## Deferred term

Market impact (Almgren-style / Northfield linear + √size with takeover boundary
conditions) remains a stub at zero so a size-dependent model can be added later
without redefining the explicit terms.

## Basis points

```text
bps = cost / execution_notional × 10,000
```

The numbers on `OrderCost` and `BlotterCost` are exact. The text report rounds
bps to 0.0001 for display.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Cursor / VS Code is set to use `.venv/bin/python` (see `.vscode/settings.json`). After creating the venv, reload the window if the interpreter banner is still showing.

## Library

```python
from tcost_engine import BidAsk, Fill, PerShare, TransactionCostModel
from tcost_engine.report import format_report

model = TransactionCostModel(commish=PerShare("10"))  # house fixed 10 mils
fill = Fill(
    symbol="AAPL",
    side="buy",
    quantity="1000",
    price="50.02",          # VWAP
    close="50.00",          # close benchmark
    spread=BidAsk("50.00", "50.02"),  # EOD bid/ask proxy
)
result = model.cost(fill)
print(result.commish_amount)  # 1   (10 mils × 1000 / 10000)
print(result.spread)          # 10
print(result.market_impact)   # 0
print(result.residual)        # 20  (= slippage)
print(result.slippage)        # 20
print(result.total)           # 31
print(format_report(model.cost_many([fill])))
```

`commission=` is accepted as a synonym for `commish=`.

## CLI

```bash
tcost-engine cost --symbol AAPL --side buy --qty 1000 --price 50.02 \
  --close 50.00 --bid 50.00 --ask 50.02

tcost-engine blotter examples/blotter.csv

tcost-engine cost --side sell --qty 1000 --price 50 --close 50.10 \
  --bid 49.99 --ask 50.01 --sec-fee 0.0000278 --finra-taf
```

`--mils` defaults to **10**. Pass `--mils 0` for a spread/slippage-only run.
Pass exactly one spread: `--bid` and `--ask`, or `--full-spread-bps`, or
`--one-way-spread-bps`. Add `--json` for decimal amounts as strings.

`examples/blotter.csv` is three orders with VWAP, close, and EOD bid/ask. At
10 mils, totals are commish 1.3, spread 30, impact 0, residual (slippage) 45,
total 76.3.

## Combo SOD t-costs

Cost day-over-day trades from a wide combo SOD dollar panel (DatetimeIndex ×
INFOCODE) against LSEG Datastream2 `VWAP` / `CLOSE`. Combo fills carry **no**
half-spread (VWAP already embeds liquidity).

```text
delta_$             = SOD(t) − SOD(t−1)
qty                 = |delta_$| / exec_px(t)     # VWAP, or close under --fill moc
intraday_slippage   = side × (close − exec_px) × qty   # PnL-signed
tcost               = 10-mil commish + intraday_slippage
```

`side` is `+1` buy / `−1` sell. Buy above close or sell below close → **negative**
intraday slippage. Under `--fill moc`, exec = close so slippage is 0 and
tcost = commish only.

Default SOD (local on `kelai-team-robert`; not overwritten by S3 sync):

`/data/robert/stage_c_pinnet_mktbeta_pos_20261005_gto3e-4_to28/stage_c_pinnet_pos_2021_2026_gto3e-4_to28.parquet`

```bash
pip install -e ".[lseg]"

tcost-engine cost-combo \
  --sod /data/robert/stage_c_pinnet_mktbeta_pos_20261005_gto3e-4_to28/stage_c_pinnet_pos_2021_2026_gto3e-4_to28.parquet \
  --start 2021-01-01 --end 2026-10-01 \
  --fill vwap \
  -o data/combo_tcost_daily.parquet
```

`--mils` defaults to 10. `--fill` is `vwap` (default) or `moc`. Rows missing a
usable price are skipped. Run on the team box (Datastream2 H5), or pass `--h5`.

### debug_perturb — pre vs post t-cost (mosek-style)

Stage C–style yearly tables side-by-side (**pre-tcost | post-tcost**), plus a
commish / intraday slippage / tcost summary in **bps/day** (mean and median).
Artifacts under `outputs/perturb_tcost/`.

From a Mac, PyCharm run config **debug_perturb** hops to
`robert@kelai-team-robert` for the H5 + local SOD, then rsyncs
`outputs/perturb_tcost/` back. On the box, pass `--local`.

Caches under `~/.cache/tcost-engine/` (prices + daily combo-cost). Pass
`--refresh` to rebuild.

```text
pre_pnl_t             = Σ SOD_{t-1} × (close_adj_t / close_adj_{t-1} − 1)
intraday_slippage_t   = Σ side × (close − VWAP) × qty
tcost_t               = commish + intraday_slippage
post_pnl_t            = pre_pnl_t − tcost_t
ret_t                 = pnl_t / ‖SOD_{t-1}‖₁      # Stage C prev_gmv
TO_t                  = ‖SOD_t − SOD_{t-1}‖₁ / ‖SOD_{t-1}‖₁
```

```bash
python debug_perturb.py --start 2021-01-01 --end 2026-10-01 --fill vwap
python debug_perturb.py --fill moc --outdir outputs/perturb_tcost_moc
# on the box:
python debug_perturb.py --local --start 2021-01-01 --end 2026-10-01
```

## LSEG Datastream2 pulls

Mirrors the AWS research-box ability to read the published Datastream2 H5
(`kelai-team-robert:/data/robert/lseg/Datastream2/ds2_data.h5`, also
`s3://kelaidata/data/LSEG/Datastream2/ds2_data.h5`).

Install the optional extra:

```bash
pip install -e ".[lseg]"
```

Pull **OHLCV unadjusted + adjusted** for names in the point-in-time **TOP500**
universe from 2016 onward (default):

```bash
tcost-engine lseg ohlcv -o data/ohlcv_top500_2016.parquet
tcost-engine lseg ohlcv --adjustment unadjusted --universe all -o data/ohlcv_all.csv
```

Pull **TOP500 constituents** from 2016 onward:

```bash
tcost-engine lseg top500 -o data/top500_constituents_2016.parquet
```

On the AWS box the newest dated H5 under
`/data/robert/lseg/Datastream2/` is picked automatically (the undated
`ds2_data.h5` can lag). Elsewhere pass `--h5` to a local copy or the S3 URL
(requires AWS credentials; downloads land under `~/.cache/tcost-engine/lseg`).

Published pulls on the box:

```text
/data/robert/lseg/pulls/top500_constituents_2016.parquet
/data/robert/lseg/pulls/ohlcv_top500_2016.parquet
```

| Source | Path |
| --- | --- |
| AWS box | `/data/robert/lseg/Datastream2/ds2_data.h5` |
| S3 prod | `s3://kelaidata/data/LSEG/Datastream2/ds2_data.h5` |
| Snowflake (upstream) | `KELAI.LSEG_CANARY.BASE_DATA_US_DT`, `KELAI.LSEG_CANARY.TOPN_UNIVERSES_DT` |

Panels: `OPEN`/`HIGH`/`LOW`/`CLOSE`/`VOLUME` and `*_ADJUSTED`, plus `TOP500`.

## Return signals

Gross daily return on adjusted closes, then cross-sectionalized within the
TOP500 universe:

```text
ret    = close_lookback(df) / close_lookback(df, window=1)
ret_cs = rank_pct(ret | top500 == 1)   # within each marketdate, in (0, 1]
weight = long_short_book(ret_cs)       # enter at today's close; long +1 / short -1
port   = Σ w(t) × (close(t+1)/close(t) - 1)   # PnL = next day's close-to-close
       = Σ w(t-1) × (ret(t) - 1)              # series dated on the PnL day
```

``close_lookback(df, window=w) = close.shift(w)`` within each name (``window``
defaults to ``0`` = today; ``window=1`` = yesterday).

Signal and weights at ``t`` use full knowledge of ``close(t) / close(t-1)``.
The book is assumed filled at that close; PnL is the next session's return.

```bash
pip install -e ".[signals]"
```

```python
from tcost_engine.signals import add_ret_signal, portfolio_returns

panel = add_ret_signal(ohlcv_df)       # ret, ret_cs, weight (as of close t)
port = portfolio_returns(panel)        # daily Series: port_ret (PnL on t+1)
```

## Tests

```bash
pytest
```

## Literature notes

Inspiration drawn from:

- **Northfield Transaction Cost Model** / **diBartolomeo (2007)** — total cost =
  agency (our `commish`) + bid/ask + market impact + trend; `commish` and spread
  are estimable in advance; impact needs size-dependent functional form with
  boundary conditions.
- **Deutsche Bank, *A Portfolio Manager’s Guidebook to Trade Execution* (2015)** —
  for discretionary execution, the two intraday cost metrics are bid-ask spread
  and price impact; half-spread is the immediate cost of aggressive liquidity-taking.
- **Bocconi / BSIC survey** — explicit (commission, taxes) vs implicit (impact);
  effective cost vs mid; opportunity cost and implementation shortfall as separate
  ideas from the touch spread.
- **Frazzini, Israel, Moskowitz (2017)** — live-trade calibrated impact curves;
  cite ~5 mils/share as a US institutional baseline. This engine’s house rate
  is **10 mils/share**.
