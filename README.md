# tcost-engine

Costs a trade with the Northfield / diBartolomeo decomposition used across the
execution literature (also echoed by Deutsche Bank and Bocconi surveys):

```text
total = agency + spread + market_impact + residual
```

This release costs **agency** and **spread**. Market impact and residual are
reported as zero so they can be added later without redefining the explicit
terms.

| Term | Status | Meaning |
| --- | --- | --- |
| Agency | Modeled | Broker commission + exchange / regulatory / transfer fees. Explicit and known in advance. |
| Spread | Modeled | Bid-ask half-spread for taking liquidity. Transparent. |
| Market impact | Deferred (`0`) | Size-dependent price move from *this* trade (temporary / permanent; often √size). |
| Residual | Deferred (`0`) | Trend cost (other flow) and opportunity cost of slow or incomplete fills. |

Amounts are in the price currency. Inputs are `Decimal`, `str`, or `int`. Floats
are rejected so a value like `0.005` cannot pick up a binary fraction. Positive
amounts are costs paid by the trader. A rebate or a maker spread capture is
negative.

This is not an implementation-shortfall model. The gap between the execution
price and the touch is not a cost here — that is where market impact belongs
later. For a taker at the touch, the spread term *is* the mid-to-touch /
effective half-spread piece of arrival-price cost.

## Agency

Agency is charged once per order. Fills that share an `order_id` are one order.
A fill with no `order_id` is its own order. Use `cost_many` for a blotter so a
minimum or flat fee is not charged again on every fill.

### Broker commission

| Schedule | Charge |
| --- | --- |
| `PerShare(rate, minimum=0)` | `max(rate × order quantity, minimum)` |
| `FlatFee(amount)` | `amount` once per order |
| `BpsOfNotional(bps, minimum=0)` | `max(execution notional × bps / 10,000, minimum)` |
| `PercentOfNotional(percent, minimum=0)` | `max(execution notional × percent / 100, minimum)`. 1% is 100 bps. |
| `PerFill(schedule)` | Apply `schedule` to each fill, then sum. |
| `Composite(*schedules)` | Sum of schedules on the same order. |
| `AtLeast(schedule, minimum)` | Floor on the combined order charge. |
| `NoCommission()` | Zero. |

Execution notional is `quantity × execution price`. A minimum is a floor on what
the trader pays. Leave it at 0 when the rate is a rebate. Frazzini–Israel–Moskowitz
(2017) cite about `$0.005` per share as a representative US institutional commission.

### Fees and taxes

Compose with commission via `Composite`. Side-aware fees need the order side
(sells for US SEC / FINRA; buys for stamp duty).

| Schedule | Charge |
| --- | --- |
| `SecFee(rate)` | `rate × sell notional` (SEC Section 31). Buys are zero. Pass the live published rate. |
| `FinraTaf(rate_per_share, cap)` | Per-share sell fee with an optional cap. Buys are zero. Defaults are illustrative. |
| `StampDuty(percent=0.5)` | `percent / 100 × buy notional` (UK SDRT-style). Sells are zero. |
| `OnBuy(schedule)` / `OnSell(schedule)` | Apply any schedule on one side only. |

When an order has several fills, the order agency charge is split across them
pro rata by quantity so the fill totals add up.

## Spread

Spread is the quoted width, not slippage past the touch. Deutsche Bank’s PM
guidebook treats bid-ask spread and price impact as the two *intraday* cost
levers; we model the first and leave the second at zero.

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

## Deferred terms

Market impact (Almgren-style / Northfield linear + √size with takeover boundary
conditions) and residual trend / opportunity cost are stubs that return zero.
They appear in reports so the four-term identity stays visible.

## Basis points

```text
bps = cost / execution_notional × 10,000
```

The numbers on `OrderCost` and `BlotterCost` are exact. The text report rounds
bps to 0.0001 for display.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Library

```python
from tcost_engine import BidAsk, Composite, Fill, FinraTaf, PerShare, SecFee, TransactionCostModel
from tcost_engine.report import format_report

agency = Composite(PerShare("0.005", minimum="1"), SecFee("0.0000278"), FinraTaf())
model = TransactionCostModel(agency=agency)
fill = Fill(
    symbol="AAPL",
    side="buy",
    quantity="1000",
    price="50.02",
    spread=BidAsk("50.00", "50.02"),
)
result = model.cost(fill)
print(result.agency_amount)   # 5  (SEC/TAF are sell-only)
print(result.spread)          # 10
print(result.market_impact)   # 0
print(result.residual)        # 0
print(result.total)           # 15
print(format_report(model.cost_many([fill])))
```

`commission=` is accepted as a synonym for `agency=`.

## CLI

```bash
tcost-engine cost --symbol AAPL --side buy --qty 1000 --price 50.02 \
  --bid 50.00 --ask 50.02 --per-share 0.005

tcost-engine blotter examples/blotter.csv --per-share 0.005 --min-commission 1

tcost-engine cost --side sell --qty 1000 --price 50 --bid 49.99 --ask 50.01 \
  --per-share 0.005 --sec-fee 0.0000278 --finra-taf
```

Pass exactly one spread: `--bid` and `--ask`, or `--full-spread-bps`, or
`--one-way-spread-bps`. Add `--json` for decimal amounts as strings.
`--min-commission` applies to the per-share schedule and requires `--per-share`.

`examples/blotter.csv` is three orders. With `$0.005` per share and a `$1` order
minimum, the blotter totals are agency 7, spread 30, impact 0, residual 0, total 37.

## Tests

```bash
pytest
```

## Literature notes

Inspiration drawn from:

- **Northfield Transaction Cost Model** / **diBartolomeo (2007)** — total cost =
  agency + bid/ask + market impact + trend; agency and spread are estimable in
  advance; impact needs size-dependent functional form with boundary conditions.
- **Deutsche Bank, *A Portfolio Manager’s Guidebook to Trade Execution* (2015)** —
  for discretionary execution, the two intraday cost metrics are bid-ask spread
  and price impact; half-spread is the immediate cost of aggressive liquidity-taking.
- **Bocconi / BSIC survey** — explicit (commission, taxes) vs implicit (impact);
  effective cost vs mid; opportunity cost and implementation shortfall as separate
  ideas from the touch spread.
- **Frazzini, Israel, Moskowitz (2017)** — live-trade calibrated impact curves;
  ~`$0.005`/share commission as a practical US agency baseline.
