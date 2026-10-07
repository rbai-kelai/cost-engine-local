#!/usr/bin/env python3
"""Debugger perturb: pre-tcost vs post-tcost (mosek-style yearly tables).

PyCharm (Mac): hops to ``robert@kelai-team-robert`` like pm-risk — S3 pulls and
Datastream2 H5 live on the box. Use ``--local`` only when already on the box.

PyCharm:
  Run config: debug_perturb
  Script path: debug_perturb.py
  Working directory: <repo root>
  Python: .venv

CLI:
  python debug_perturb.py --start 2021-01-01 --end 2026-10-01
  python debug_perturb.py --local   # on kelai-team-robert
  # Always reports both: MOC fill (commish only) and VWAP fill
  # (mils + half-spread + VWAP−close).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BOX_HOST = "robert@kelai-team-robert"
BOX_DIR = "/data/robert/repos/cost-engine"
BOX_PYTHON = "/data/robert/venvs/ki-ops/bin/python"
SCRIPT = "debug_perturb.py"
DEFAULT_OUTDIR = "outputs/perturb_tcost"


def _on_team_box() -> bool:
    return Path("/data/robert/lseg/Datastream2").is_dir()


def _run_on_box(argv: list[str]) -> int:
    print(f"prices h5 / aws S3 not on this machine; running on {BOX_HOST}", flush=True)
    sync = subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".venv",
            "--exclude",
            "__pycache__",
            "--exclude",
            ".git",
            "--exclude",
            ".idea",
            "--exclude",
            "outputs",
            "--exclude",
            "data",
            f"{ROOT}/",
            f"{BOX_HOST}:{BOX_DIR}/",
        ],
        check=False,
    )
    if sync.returncode:
        return sync.returncode

    remote_args = [a for a in argv if a not in ("--local",)]
    # Force --local on the box so it does not hop again. Unbuffered so progress
    # lines show up in the Mac PyCharm console while H5/costing run (~5–10 min).
    cmd = (
        f"cd {shlex.quote(BOX_DIR)} && "
        f"PYTHONUNBUFFERED=1 {shlex.quote(BOX_PYTHON)} -u {SCRIPT} --local"
    )
    if remote_args:
        cmd += " " + " ".join(shlex.quote(a) for a in remote_args)
    rc = subprocess.call(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", BOX_HOST, cmd]
    )
    if rc:
        return rc

    # Pull perturb artifacts back (outdir from argv or default).
    outdir = DEFAULT_OUTDIR
    if "--outdir" in argv:
        i = argv.index("--outdir")
        if i + 1 < len(argv):
            outdir = argv[i + 1]
    dest = ROOT / outdir
    dest.mkdir(parents=True, exist_ok=True)
    pull = subprocess.run(
        ["rsync", "-az", f"{BOX_HOST}:{BOX_DIR}/{outdir}/", f"{dest}/"],
        check=False,
    )
    if pull.returncode:
        print(f"WARNING: failed to copy {outdir}/ from {BOX_HOST}", flush=True)
        return pull.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    local = "--local" in argv
    if not local and not _on_team_box():
        return _run_on_box(argv)

    from tcost_engine.cli import main as cli_main

    if not argv or argv[0] != "debug-perturb":
        argv = ["debug-perturb", *argv]
    # Strip hop-only flag before argparse.
    argv = [a for a in argv if a != "--local"]
    return int(cli_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
