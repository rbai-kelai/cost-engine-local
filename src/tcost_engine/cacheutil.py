"""Disk cache helpers for expensive LSEG / combo pulls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def default_cache_root(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return Path.home() / ".cache" / "tcost-engine"


def fingerprint(*parts: object) -> str:
    """Stable short hash of cache key parts."""
    h = hashlib.sha1()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def file_stamp(path: str | Path | None) -> tuple[str, float]:
    """(resolved path, mtime) for cache invalidation; missing → ('', 0)."""
    if path is None:
        return ("", 0.0)
    p = Path(path)
    try:
        return (str(p.resolve()), float(p.stat().st_mtime))
    except OSError:
        return (str(p), 0.0)


def read_parquet_if_fresh(path: Path, *, meta_path: Path, expected: dict) -> object | None:
    """Return DataFrame if cache parquet + sidecar meta match *expected* keys."""
    if not path.is_file() or not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    # Sidecar may carry extra stats; only the expected keys must match.
    if any(meta.get(k) != v for k, v in expected.items()):
        return None
    import pandas as pd

    return pd.read_parquet(path)


def write_parquet_cache(df, path: Path, *, meta_path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    meta_path.write_text(json.dumps(meta, sort_keys=True, default=str))
