"""Separate geometric consistency, determinism and independent accuracy evidence."""

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
from shapely.geometry import Polygon


def validate_geometry(plan):
    errors, overlaps = [], []
    ids = {r["id"] for r in plan["rooms"]}
    if len(ids) != len(plan["rooms"]):
        errors.append("Duplicate room IDs")
    for i, room in enumerate(plan["rooms"]):
        polygon = Polygon(room["polygon"])
        if not polygon.is_valid or polygon.area <= 0:
            errors.append(f"Room {room['id']}: invalid polygon")
            continue
        if abs(polygon.area - room["floor_area"]["value"]) > 0.01:
            errors.append(f"Room {room['id']}: area does not match polygon")
        V = np.asarray(room["polygon"])
        for k, wall in enumerate(room["walls"]):
            length = np.linalg.norm(np.array(wall["end"]) - wall["start"])
            if abs(length - wall["length"]["value"]) > 0.002:
                errors.append(f"Room {room['id']}, wall {k}: length does not match endpoints")
            if (
                k >= len(V)
                or np.linalg.norm(np.array(wall["start"]) - V[k]) > 0.002
                or np.linalg.norm(np.array(wall["end"]) - V[(k + 1) % len(V)]) > 0.002
            ):
                errors.append(f"Room {room['id']}, wall {k}: endpoints do not match polygon")
        for opening in room["openings"]:
            k = opening["wall"]
            if (
                k >= len(room["walls"])
                or opening["offset"]["value"] + opening["width"]["value"]
                > room["walls"][k]["length"]["value"] + 0.025
            ):
                errors.append(f"Room {room['id']}: opening outside wall")
            if opening["connects_to"] is not None and opening["connects_to"] not in ids:
                errors.append(f"Room {room['id']}: opening references missing room")
        for other in plan["rooms"][i + 1 :]:
            q = Polygon(other["polygon"])
            if q.is_valid:
                area = polygon.intersection(q).area
                if area > 0.005:
                    overlaps.append({"rooms": [room["id"], other["id"]], "area_m2": round(area, 4)})
    return {
        "valid": not errors and not overlaps,
        "errors": errors,
        "overlaps": overlaps,
        "accuracy_validated": False,
    }


def validate(path):
    plan = json.loads(Path(path).read_text())
    schema = json.loads((Path(__file__).parent / "schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(plan)
    measurements = flatten(plan)
    for key, m in measurements.items():
        if (
            not all(math.isfinite(v) for v in [m["value"], m["sigma"], *m["ci95"]])
            or not m["ci95"][0] <= m["value"] <= m["ci95"][1]
        ):
            raise ValueError(f"Invalid interval: {key}")
    return {"schema_valid": True, **validate_geometry(plan)}


def flatten(plan):
    dimensions = {"footprint_area": plan["footprint_area"]}
    for room in plan["rooms"]:
        prefix = f"room-{room['id']}"
        for field in ["floor_area", "ceiling_height", "perimeter"]:
            if room[field] is not None:
                dimensions[f"{prefix}/{field}"] = room[field]
        for w in room["walls"]:
            dimensions[f"{prefix}/wall-{w['index']}/length"] = w["length"]
        for i, o in enumerate(room["openings"]):
            for field in ["width", "height"]:
                dimensions[f"{prefix}/opening-{i}/{field}"] = o[field]
    return dimensions


def evaluate(plan, truth):
    if truth.get("source") not in ["laser", "tape"]:
        raise ValueError(
            "Accuracy evaluation requires source=laser or tape; sensor reconstructions are not ground truth"
        )
    if not truth.get("measurements"):
        raise ValueError("No independently measured dimensions supplied")
    predicted = flatten(plan)
    rows = []
    for key, target in truth["measurements"].items():
        if not isinstance(target, (float, int)) or not math.isfinite(target) or target <= 0:
            raise ValueError(f"Ground truth must contain positive measured numbers: {key}")
        m = predicted.get(key)
        if m is None:
            rows.append({"dimension": key, "status": "missed", "truth": target})
            continue
        error = abs(m["value"] - target)
        tolerance = None
        if "/wall-" in key:
            tolerance = {"photos": 0.08 * target, "video": 0.03 * target}.get(
                plan["tier"], truth.get("lidar_wall_tolerance_m")
            )
        elif "/opening-" in key and key.endswith("/width"):
            tolerance = 0.02
        elif key.endswith("/ceiling_height"):
            tolerance = 0.015
        elif key == "footprint_area" and plan["tier"] == "photos":
            tolerance = 0.08 * target
        rows.append(
            {
                "dimension": key,
                "truth": target,
                "prediction": m["value"],
                "absolute_error": error,
                "relative_error": error / target,
                "interval_contains_truth": m["ci95"][0] <= target <= m["ci95"][1],
                "tolerance": tolerance,
                "passes": None if tolerance is None else error <= tolerance,
                "status": "matched",
            }
        )
    opening_truth = {k for k in truth["measurements"] if "/opening-" in k and k.endswith("/width")}
    opening_pred = {k for k in predicted if "/opening-" in k and k.endswith("/width")}
    opening_success = sum(r.get("passes") is True for r in rows if r["dimension"] in opening_truth)
    phantom = opening_pred - opening_truth
    denominator = len(opening_truth) + len(phantom)
    observed = [r for r in rows if r["status"] == "matched"]
    return {
        "source": truth["source"],
        "matching": "Explicit dimension IDs supplied by evaluator; inspect correspondence manually",
        "rows": rows,
        "interval_coverage": sum(r["interval_contains_truth"] for r in observed) / len(observed)
        if observed
        else None,
        "coverage_count": len(observed),
        "coverage_is_calibration_claim": False,
        "opening_gate": {
            "status": "not_evaluable" if not truth.get("openings_exhaustive") else "evaluated",
            "success_fraction": opening_success / denominator
            if denominator and truth.get("openings_exhaustive")
            else None,
            "phantoms": sorted(phantom) if truth.get("openings_exhaustive") else [],
            "misses": sorted(opening_truth - opening_pred),
            "passes": opening_success / denominator >= 0.85
            if denominator and truth.get("openings_exhaustive")
            else None,
        },
    }


def benchmark(samples, out):
    from .cli import process

    samples, out = Path(samples), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    captures = sorted(p for p in samples.iterdir() if p.is_dir() and (p / "odometry.csv").exists())
    if not captures:
        raise ValueError("No extracted sample captures found")
    records = []
    for capture in captures:
        print(f"Benchmark {capture.name}: raw", flush=True)
        before = process(capture, out=out / capture.name / "before", drift=False)
        print(f"Benchmark {capture.name}: corrected", flush=True)
        after = process(capture, out=out / capture.name / "after", drift=True)
        d = after.diagnostics["drift_correction"]
        records.append(
            {
                "capture": capture.name,
                "rooms_before": len(before.rooms),
                "rooms_after": len(after.rooms),
                "area_before_m2": before.footprint_area.value,
                "area_after_m2": after.footprint_area.value,
                "drift": d,
                "geometry": after.quality["geometry_checks"],
                "runtime_s": after.diagnostics["total_runtime_s"],
                "absolute_accuracy": "not_evaluable_no_ground_truth",
            }
        )
    report = {
        "captures": records,
        "ground_truth": "not_supplied",
        "accuracy_gates": "not_evaluable",
        "repeatability": "not_evaluable_without_identified_independent_repeat_capture",
        "incumbent_comparison": "not_evaluable_no_app_exports",
        "interval_calibration": "not_evaluable_no_independent_ground_truth",
        "warning": "Plane residual is internal consistency, not measurement error against ground truth.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Supplied-sample benchmark",
        "",
        report["warning"],
        "",
        "| Capture | Rooms before → after | Area before → after (m²) | Plane residual before → after (mm) | Geometry valid |",
        "|---|---:|---:|---:|---|",
    ]
    for r in records:
        d = r["drift"]
        lines.append(
            f"| {r['capture']} | {r['rooms_before']} → {r['rooms_after']} | {r['area_before_m2']:.2f} → {r['area_after_m2']:.2f} | {d['heldout_plane_residual_before_m'] * 1000:.2f} → {d['heldout_plane_residual_after_m'] * 1000:.2f} | {r['geometry']['valid']} |"
        )
    lines += [
        "",
        "Accuracy, repeatability, interval calibration and incumbent-comparison gates remain unverified: independent measurements, matched repeat captures and app exports were not supplied.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(out / "report.md")
