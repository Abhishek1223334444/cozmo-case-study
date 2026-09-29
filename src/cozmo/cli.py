"""Local capture-to-plan pipeline and reproduction commands."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path


def unpack(path, work):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return path
    import hashlib

    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()[:16]
    dest = Path(work) / (path.stem + "-" + digest)
    marker = dest / ".complete"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                target = (dest / member.filename).resolve()
                if (
                    not target.is_relative_to(dest.resolve())
                    or (member.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Unsafe archive member")
            z.extractall(dest)
        marker.write_text(digest)
    return dest


def process(
    capture,
    tier="lidar",
    out=None,
    drift=True,
    damage=False,
    max_frames=48,
    rotation=0,
    models=Path("models"),
    device="auto",
):
    from .damage import assess, surfaces
    from .render import render
    from .viewer import write_viewer

    started = time.time()
    capture = Path(capture)
    out = Path(out or Path("out") / (capture.stem + "-" + tier))
    out.mkdir(parents=True, exist_ok=True)
    source = unpack(capture, Path("out/unpacked"))
    if tier == "lidar":
        from .lidar import run

        plan, debug = run(source, Path("out/cache"), drift)
    else:
        from .rgb import run

        plan, debug = run(
            source, tier, Path("out/cache"), Path(models) / "depth", max_frames, rotation, device
        )
    surfaces(plan)
    if damage:
        assess(plan, debug, out, Path(models) / "damage", device=device)
    else:
        plan.diagnostics["damage_assessment"] = {
            "status": "not_run",
            "reason": "Enable --damage for local CLIPSeg candidate detection",
        }
    from .evaluation import validate_geometry

    plan.quality["geometry_checks"] = validate_geometry(plan.to_json())
    plan.provenance["runtime_environment"] = {"python": sys.version.split()[0]}
    import hashlib

    source_hash = hashlib.sha256()
    for file in sorted(Path(__file__).parent.iterdir()):
        if file.suffix in {".py", ".html", ".json"}:
            source_hash.update(file.name.encode())
            source_hash.update(file.read_bytes())
    plan.provenance["implementation_sha256"] = source_hash.hexdigest()
    plan.diagnostics["total_runtime_s"] = round(time.time() - started, 3)
    plan.save(out / "plan.json")
    render(plan, out / "plan.png", debug.get("wall_uv"))
    render(plan, out / "plan.svg")
    write_viewer(plan.to_json(), out / "index.html")
    (out / "run.json").write_text(
        json.dumps(
            {
                "capture": str(capture),
                "tier": tier,
                "drift": drift,
                "damage": damage,
                "max_frames": max_frames,
                "rotation": rotation,
                "models": str(models),
                "device": device,
            },
            indent=2,
        )
    )
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cozmo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Generate JSON, PNG, SVG and interactive HTML")
    r.add_argument("capture", type=Path)
    r.add_argument("--tier", choices=["lidar", "video", "photos"], default="lidar")
    r.add_argument("-o", "--out", type=Path)
    r.add_argument("--no-drift", action="store_true", help="Uncorrected LiDAR ablation")
    r.add_argument("--damage", action="store_true")
    r.add_argument("--max-frames", type=int, default=48)
    r.add_argument(
        "--rotation",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Clockwise RGB rotation; supplied raw videos need 90",
    )
    r.add_argument("--models", type=Path, default=Path("models"))
    r.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p = sub.add_parser(
        "prepare-sample", help="Derive isolated photo folders with weak-label provenance"
    )
    p.add_argument("capture", type=Path)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--rotation", type=int, default=90, choices=[0, 90, 180, 270])
    f = sub.add_parser("fetch-models")
    f.add_argument("kind", choices=["depth", "damage", "matcher", "all"], default="all", nargs="?")
    f.add_argument("--directory", type=Path, default=Path("models"))
    v = sub.add_parser("validate")
    v.add_argument("plan", type=Path)
    e = sub.add_parser("evaluate")
    e.add_argument("plan", type=Path)
    e.add_argument("truth", type=Path)
    e.add_argument("-o", "--out", type=Path, default=Path("out/evaluation.json"))
    b = sub.add_parser("benchmark")
    b.add_argument("--samples", type=Path, default=Path("samples"))
    b.add_argument("--out", type=Path, default=Path("out/benchmark"))
    s = sub.add_parser("serve")
    s.add_argument("--directory", type=Path, default=Path("out"))
    s.add_argument("--port", type=int, default=8765)
    a = parser.parse_args(argv)
    try:
        if a.command == "run":
            if not 2 <= a.max_frames <= 160:
                raise ValueError("--max-frames must be between 2 and 160")
            plan = process(
                a.capture,
                a.tier,
                a.out,
                not a.no_drift,
                a.damage,
                a.max_frames,
                a.rotation,
                a.models,
                a.device,
            )
            print(
                json.dumps(
                    {"rooms": len(plan.rooms), "tier": plan.tier, "quality": plan.quality}, indent=2
                )
            )
        elif a.command == "prepare-sample":
            from .sample import prepare

            print(prepare(unpack(a.capture, Path("out/unpacked")), a.plan, a.out, a.rotation))
        elif a.command == "fetch-models":
            from .models import fetch

            for kind in ["depth", "damage", "matcher"] if a.kind == "all" else [a.kind]:
                print(fetch(kind, a.directory))
        elif a.command == "validate":
            from .evaluation import validate

            result = validate(a.plan)
            print(json.dumps(result, indent=2))
            return 0 if result["valid"] else 1
        elif a.command == "evaluate":
            from .evaluation import evaluate

            result = evaluate(json.loads(a.plan.read_text()), json.loads(a.truth.read_text()))
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        elif a.command == "benchmark":
            from .evaluation import benchmark

            benchmark(a.samples, a.out)
        elif a.command == "serve":
            from functools import partial
            from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

            print(f"Results at http://127.0.0.1:{a.port}", flush=True)
            ThreadingHTTPServer(
                ("127.0.0.1", a.port), partial(SimpleHTTPRequestHandler, directory=str(a.directory))
            ).serve_forever()
        return 0
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"cozmo: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
