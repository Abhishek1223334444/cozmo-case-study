"""One command per capture: `cozmo run <capture> [--tier lidar] [-o out/<name>]`."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


def _unpack(path: Path, work: Path) -> Path:
    if path.suffix == ".zip":
        dest = work / path.stem
        if not dest.exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(dest)
        return dest
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cozmo", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one capture into a plan")
    r.add_argument("capture", type=Path, help="Stray Scanner folder or .zip")
    r.add_argument("--tier", choices=["lidar"], default="lidar")
    r.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out = args.out or Path("out") / args.capture.stem
    out.mkdir(parents=True, exist_ok=True)
    src = _unpack(args.capture, Path("out") / "unpacked")

    from . import lidar
    from .render import render

    plan, dbg = lidar.run(src, cache_dir=Path("out") / "cache")
    plan.save(out / "plan.json")
    render(plan, out / "plan.png", dbg["wall_uv"])
    print(json.dumps({"rooms": len(plan.rooms), "footprint_m2": plan.footprint_area.to_json(), "out": str(out)}))
    print(json.dumps(plan.diagnostics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
