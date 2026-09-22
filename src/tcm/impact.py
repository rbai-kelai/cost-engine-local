"""Market impact.

Not modeled. Commission and spread are the whole cost:

    total = commission + spread + market_impact()

market_impact() is zero so a later impact term can be added beside the other
two without redefining them. Impact is not inferred from the execution price.
"""

from __future__ import annotations

from decimal import Decimal


def market_impact() -> tuple[Decimal, str]:
    """Currency impact cost. Always zero until an impact model is added."""
    return Decimal(0), "not modeled"
