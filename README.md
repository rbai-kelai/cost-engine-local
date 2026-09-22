# Transaction cost model

Costs a trade as **commission + spread**. Market impact is not part of the model. It is reported as zero so the total stays:

```text
total = commission + spread + market_impact
market_impact = 0
```

Amounts are in the price currency. Inputs are `Decimal`, `str`, or `int`. Floats are rejected so a value like `0.005` cannot pick up a binary fraction. Positive amounts are costs paid by the trader. A rebate or a maker spread capture is negative.

This is not an implementation-shortfall model. The gap between the execution price and the touch is not a cost here. That gap is where market impact would go later, and it is left out on purpose.

## Commission

Commission is charged once per order. Fills that share an `order_id` are one order. A fill with no `order_id` is its own order. Use `cost_many` for a blotter so a minimum or a flat fee is not charged again on every fill.

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

Execution notional is `quantity × execution price`. A minimum is a floor on what the trader pays. Leave it at 0 when the rate is a rebate.

When an order has several fills, the order commission is split across them pro rata by quantity so the fill totals add up. The order charge is the one that matches the schedule.

## Spread

Spread is the quoted width, not slippage past the touch.

For a bid/ask, the one-way cost of taking liquidity is the half-spread:

```text
half_spread = (ask - bid) / 2
taker cost  = half_spread × quantity
```

A buy lifting the ask and a sell hitting the bid are each half a spread from the mid, so the cost is the same on both sides. A locked quote (`bid == ask`) costs nothing. A crossed quote is rejected.

`FullSpreadBps` is the quoted width `(ask - bid) / reference × 10,000`. The taker pays half of it. `OneWaySpreadBps` is already that one-way cost and is not halved. If you omit the reference price, the execution price is the reference, and the taker cost in bps of execution notional equals that one-way spread exactly.

| Liquidity | Spread cost |
| --- | --- |
| Taker | `+ half_spread × quantity` |
| Maker | `- maker_capture × half_spread × quantity` |
| Midpoint | `0` |

`maker_capture` is a fraction from 0 to 1 on the model. The default is 0: a maker pays no spread and is not credited with earning one. At 1, the maker earns the full half-spread.

## Basis points

```text
bps = cost / execution_notional × 10,000
```

The numbers on `OrderCost` and `BlotterCost` are exact. The text report rounds bps to 0.0001 for display.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Library

```python
from tcm import BidAsk, Fill, PerShare, TransactionCostModel
from tcm.report import format_report

model = TransactionCostModel(commission=PerShare("0.005", minimum="1"))
fill = Fill(
    symbol="AAPL",
    side="buy",
    quantity="1000",
    price="50.02",
    spread=BidAsk("50.00", "50.02"),
)
result = model.cost(fill)
print(result.commission_amount)  # 5
print(result.spread)             # 10
print(result.market_impact)      # 0
print(result.total)              # 15
print(format_report(model.cost_many([fill])))
```

A per-share fee plus an extra basis point:

```python
from tcm import BpsOfNotional, Composite, PerShare

commission = Composite(PerShare("0.005"), BpsOfNotional("1"))
```

## CLI

```bash
tcm cost --symbol AAPL --side buy --qty 1000 --price 50.02 \
  --bid 50.00 --ask 50.02 --per-share 0.005

tcm blotter examples/blotter.csv --per-share 0.005 --min-commission 1
```

Pass exactly one spread: `--bid` and `--ask`, or `--full-spread-bps`, or `--one-way-spread-bps`. Add `--json` for decimal amounts as strings. `--min-commission` applies to the per-share schedule and requires `--per-share`.

`examples/blotter.csv` is three orders. With `$0.005` per share and a `$1` order minimum, the blotter totals are commission 7, spread 30, market impact 0, total 37.

## Tests

```bash
pytest
```
