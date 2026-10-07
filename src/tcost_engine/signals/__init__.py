"""Price lookbacks and return signals from adjusted closes."""

from tcost_engine.signals.returns import (
    add_ret_signal,
    close_lookback,
    cross_sectionalize,
    long_short_book,
    portfolio_returns,
    ret_signal,
)

__all__ = [
    "add_ret_signal",
    "close_lookback",
    "cross_sectionalize",
    "long_short_book",
    "portfolio_returns",
    "ret_signal",
]
