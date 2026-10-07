"""S3 pull helpers — same pattern as pm-risk ``engine/pnl.py`` (``aws s3 cp``)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def aws_s3_cp(src: str, dest: str) -> None:
    """``aws s3 cp src dest`` (raises on failure)."""
    subprocess.run(["aws", "s3", "cp", src, dest], check=True)


def s3_last_modified(uri: str):
    """S3 object LastModified as ``pandas.Timestamp``, or None if head fails."""
    import pandas as pd

    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    try:
        raw = subprocess.check_output(
            ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
            text=True,
        )
        return pd.Timestamp(json.loads(raw)["LastModified"])
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError, FileNotFoundError):
        return None


def ensure_local_s3(
    uri: str,
    dest: Path,
    *,
    refresh: bool = True,
    label: str = "s3 object",
) -> Path:
    """Copy ``s3://…`` to *dest* when missing or older than S3 (pm-risk style)."""
    if not str(uri).startswith("s3://"):
        path = Path(uri)
        if not path.is_file():
            raise FileNotFoundError(uri)
        return path

    dest = Path(dest)
    if not refresh:
        if not dest.is_file():
            raise FileNotFoundError(f"no local cache for {uri} at {dest}")
        return dest

    s3_mtime = s3_last_modified(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stale = True
    if dest.is_file() and s3_mtime is not None and dest.stat().st_mtime >= s3_mtime.timestamp() - 2:
        stale = False
    elif dest.is_file() and s3_mtime is None:
        stale = False
        print(f"{label} S3 head missed {uri}; using {dest}", flush=True)
    if stale:
        print(f"fetching {label} → {dest} …", flush=True)
        aws_s3_cp(uri, str(dest))
    else:
        print(f"{label} current: {dest}", flush=True)
    return dest
