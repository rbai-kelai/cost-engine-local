"""Disk cache helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tcost_engine.cacheutil import (
    fingerprint,
    read_parquet_if_fresh,
    write_parquet_cache,
)


def test_parquet_cache_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    path = tmp_path / "x.parquet"
    meta_path = tmp_path / "x.json"
    meta = {"k": "v", "n": 1}
    write_parquet_cache(df, path, meta_path=meta_path, meta={**meta, "extra": 9})
    # Expected keys only — extras in sidecar are OK
    hit = read_parquet_if_fresh(path, meta_path=meta_path, expected=meta)
    assert hit is not None
    assert list(hit["a"]) == [1, 2]
    miss = read_parquet_if_fresh(path, meta_path=meta_path, expected={"k": "v", "n": 2})
    assert miss is None


def test_fingerprint_stable() -> None:
    assert fingerprint(1, "a") == fingerprint(1, "a")
    assert fingerprint(1, "a") != fingerprint(1, "b")
