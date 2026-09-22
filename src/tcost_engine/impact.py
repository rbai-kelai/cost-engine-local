"""Market impact and residual trading costs.

Not modeled yet. The literature (Northfield / diBartolomeo; Deutsche Bank;
Bocconi survey of Almgren, Frazzini–Israel–Moskowitz) treats them as separate
from agency fees and the bid-ask spread:

    total = agency + spread + market_impact + residual

Agency and spread are computed elsewhere. These two terms return zero so a
later impact or trend/opportunity model can be plugged in beside them without
redefining the explicit costs.

Market impact is the price move required to induce the other side of *this*
trade (temporary and permanent components; often linear and/or square-root
in size). Residual covers trend cost (other participants' order flow between
decision and fill) and opportunity cost of delayed or incomplete fills. Neither
is inferred from the execution price here.
"""

from __future__ import annotations

from decimal import Decimal


def market_impact() -> tuple[Decimal, str]:
    """Currency impact cost. Always zero until an impact model is added."""
    return Decimal(0), "not modeled (size-dependent price move deferred)"


def residual_cost() -> tuple[Decimal, str]:
    """Trend / opportunity cost. Always zero until a residual model is added."""
    return Decimal(0), "not modeled (trend / opportunity cost deferred)"
