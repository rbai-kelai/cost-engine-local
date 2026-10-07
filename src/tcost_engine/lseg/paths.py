"""Default locations for the Datastream2 H5 the AWS box already uses."""

from __future__ import annotations

from pathlib import Path

from tcost_engine.s3util import ensure_local_s3

DEFAULT_DS2_DIR = Path("/data/robert/lseg/Datastream2")
DEFAULT_DS2_H5_LOCAL = DEFAULT_DS2_DIR / "ds2_data.h5"
DEFAULT_DS2_H5_S3 = "s3://kelaidata/data/LSEG/Datastream2/ds2_data.h5"
DEFAULT_DS2_H5_S3_CANARY = "s3://kelaidata/data_canary/LSEG/Datastream2/ds2_data.h5"


def resolve_ds2_h5(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a local H5 path.

    Resolution order when *path* is omitted:

    1. Newest usable file under ``/data/robert/lseg/Datastream2/``
       (dated ``ds2_data_YYYYMMDD.h5`` preferred over a stale undated copy)
    2. ``s3://kelaidata/data/LSEG/Datastream2/ds2_data.h5`` via ``aws s3 cp``
    """
    if path is None:
        local = _best_local_ds2()
        if local is not None:
            return local
        path = DEFAULT_DS2_H5_S3

    text = str(path)
    if text.startswith("s3://"):
        return _fetch_s3(text, cache_dir=cache_dir)

    local = Path(path)
    if not local.is_file():
        raise FileNotFoundError(
            f"ds2 H5 not found at {local}. On the AWS box use a file under "
            f"{DEFAULT_DS2_DIR}, or pass {DEFAULT_DS2_H5_S3}."
        )
    return local


def _best_local_ds2() -> Path | None:
    """Pick the best local snapshot: prefer files that carry TICKER_INDEX."""
    if not DEFAULT_DS2_DIR.is_dir():
        return None
    candidates = sorted(DEFAULT_DS2_DIR.glob("ds2_data*.h5"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None

    for path in reversed(candidates):
        if _has_ticker_index(path):
            return path
    return candidates[-1]


def _has_ticker_index(path: Path) -> bool:
    try:
        import h5py
    except ImportError:
        return path.name != "ds2_data.h5"

    try:
        with h5py.File(path, "r") as h5:
            return "metadata" in h5 and "ds2_data/TICKER_INDEX" in h5
    except OSError:
        return False


def _fetch_s3(url: str, *, cache_dir: str | Path | None) -> Path:
    """Pull H5 with ``aws s3 cp`` (pm-risk style), cache under ~/.cache or *cache_dir*."""
    rest = url[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"not a valid s3 url: {url}")

    root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "tcost-engine" / "lseg"
    dest = root / bucket / key
    return ensure_local_s3(url, dest, refresh=True, label="Datastream2 h5")
