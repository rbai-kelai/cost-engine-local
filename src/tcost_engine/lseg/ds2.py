"""Low-level Datastream2 H5 access (pandas-fixed panels via h5py)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

DS2_NAMESPACE = "ds2_data"
TICKER_VOCABULARY = "metadata/TICKERS"
_TICKER_MISSING_CODE = -1

OHLCV_UNADJUSTED = ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")
OHLCV_ADJUSTED = (
    "OPEN_ADJUSTED",
    "HIGH_ADJUSTED",
    "LOW_ADJUSTED",
    "CLOSE_ADJUSTED",
    "VOLUME_ADJUSTED",
)
# Unadjusted quote/VWAP/close for t-costs; adjusted close for lag-1 PnL.
COST_PRICE_FIELDS = ("BID", "ASK", "VWAP", "CLOSE", "CLOSE_ADJUSTED")
TOP_FIELDS = ("TOP100", "TOP500", "TOP1000", "TOP2000", "TOP3000")


@dataclass(frozen=True)
class Ds2PanelIndex:
    """Shared date / INFOCODE axes for the ds2 panels."""

    dates: object  # np.ndarray[datetime64[D]]
    infocodes: object  # np.ndarray[int64]
    tickers_by_code: tuple[str, ...]  # vocabulary; TICKER_INDEX stores codes into this


def _require_h5py_numpy():
    try:
        import h5py  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "LSEG pulls need the optional [lseg] extra: "
            'pip install -e ".[lseg]"  (h5py, numpy, pandas, boto3)'
        ) from exc


def open_h5(path: str | Path):
    _require_h5py_numpy()
    import h5py

    return h5py.File(str(path), "r")


def load_panel_index(h5) -> Ds2PanelIndex:
    import numpy as np

    close = h5[f"{DS2_NAMESPACE}/CLOSE"]
    dates_ns = close["axis1"][:]
    dates = np.array(
        [np.datetime64(int(x), "ns").astype("datetime64[D]") for x in dates_ns]
    )
    infocodes = close["axis0"][:].astype(np.int64)
    if TICKER_VOCABULARY in h5:
        raw = h5[TICKER_VOCABULARY].asstr()[...]
        vocab = tuple(str(t).upper() for t in raw)
    else:
        vocab = ()
    return Ds2PanelIndex(dates=dates, infocodes=infocodes, tickers_by_code=vocab)


def date_slice(
    dates,
    *,
    start: date | None = None,
    end: date | None = None,
) -> slice:
    import numpy as np

    lo = 0
    hi = len(dates)
    if start is not None:
        start_d = np.datetime64(start.isoformat(), "D")
        lo = int(np.searchsorted(dates, start_d, side="left"))
    if end is not None:
        end_d = np.datetime64(end.isoformat(), "D")
        hi = int(np.searchsorted(dates, end_d, side="right"))
    if lo >= hi:
        raise ValueError(
            f"no ds2 dates in range "
            f"[{start.isoformat() if start else '…'}, "
            f"{end.isoformat() if end else '…'}]"
        )
    return slice(lo, hi)


def read_field(h5, field: str, row_slice: slice):
    """Return ``block0_values[row_slice, :]`` for *field*."""
    key = f"{DS2_NAMESPACE}/{field}"
    if key not in h5:
        raise KeyError(f"ds2 H5 missing group {key}")
    return h5[key]["block0_values"][row_slice, :]


def ticker_matrix(h5, row_slice: slice, index: Ds2PanelIndex):
    """Decode ``TICKER_INDEX`` codes → object array of ticker strings ('' if missing)."""
    import numpy as np

    key = f"{DS2_NAMESPACE}/TICKER_INDEX"
    if key not in h5 or not index.tickers_by_code:
        return None
    codes = h5[key]["block0_values"][row_slice, :].astype(np.int32)
    vocab = np.asarray(index.tickers_by_code, dtype=object)
    out = np.empty(codes.shape, dtype=object)
    out[:] = ""
    valid = (codes >= 0) & (codes < len(vocab)) & (codes != _TICKER_MISSING_CODE)
    out[valid] = vocab[codes[valid]]
    return out


def as_python_date(d) -> date:
    return date.fromisoformat(str(d))


def parse_iso_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()
