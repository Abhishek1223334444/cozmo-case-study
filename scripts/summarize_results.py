"""Build a concise report and landing page from actual saved runs."""

import html
import json
from pathlib import Path

root = Path("out")
paths = sorted(root.glob("benchmark/*/after/plan.json"))
paths += [
    p
    for p in [
        root / "photos-single/plan.json",
        root / "video-single/plan.json",
        root / "video-single-dense/plan.json",
        root / "lidar-single-full/plan.json",
    ]
    if p.exists()
]
paths += [
    p
    for p in [
        root / "lidar-floor-full/plan.json",
        root / "photos-property/plan.json",
        root / "video-property/plan.json",
        root / "lidar-property-full/plan.json",
    ]
    if p.exists()
]
rows = []
labels = {
    "benchmark/1a8384c3f6/after": "Property scan, floor only",
    "benchmark/c00a170fe1/after": "Short scan",
    "benchmark/c7d28f72c6/after": "Property scan, with ceiling",
    "photos-single": "Short scan, derived photos",
    "video-single": "Short scan, 80-frame selection",
    "video-single-dense": "Short scan, 160-frame selection",
    "lidar-single-full": "Short scan, with damage assessment",
    "lidar-floor-full": "Property scan floor only, with damage assessment",
    "photos-property": "Property scan, derived photos",
    "video-property": "Property scan, 80-frame selection",
    "lidar-property-full": "Property scan, with damage assessment",
}
for path in paths:
    plan = json.loads(path.read_text())
    key = str(path.parent.relative_to(root))
    damage = plan["diagnostics"].get("damage_assessment", {})
    rows.append(
        {
            "path": key,
            "label": labels.get(key, key),
            "capture": plan["capture"],
            "tier": plan["tier"],
            "rooms": len(plan["rooms"]),
            "area_m2": plan["footprint_area"]["value"],
            "runtime_s": plan["diagnostics"].get("total_runtime_s"),
            "geometry_valid": plan["quality"].get("geometry_checks", {}).get("valid"),
            "stitch_status": plan["quality"].get(
                "stitch_status", "sensor-frame layout; adjacency unverified"
            ),
            "components": plan["diagnostics"].get("components"),
            "damage_status": damage.get("status", "not_run"),
            "damage_views_checked": damage.get("views_checked"),
            "damage_candidates": len(plan.get("damage", [])),
            "accuracy_gate": "unverified",
        }
    )


def damage_text(r):
    if r["damage_status"] == "not_run":
        return "not run"
    return f"{r['damage_candidates']} candidates ({r['damage_views_checked']} views)"


lines = [
    "# Actual sample runs",
    "",
    "These are development runs. No independent ground truth was supplied. Runtime may include cached intermediates; see run options and logs.",
    "",
    "| Tier | Run | Rooms/regions | Sum area (m²) | Runtime (s) | Geometry checks | Stitch status | Damage |",
    "|---|---|---:|---:|---:|---|---|---|",
]
for r in rows:
    lines.append(
        f"| {r['tier']} | [{r['label']}](../out/{r['path']}/index.html) | {r['rooms']} | {r['area_m2']:.2f} | {r['runtime_s']:.1f} | {r['geometry_valid']} | {r['stitch_status']} | {damage_text(r)} |"
    )
lines += [
    "",
    "RGB areas are sums of inferred observed envelopes; they are not validated whole-property footprints.",
    "Photo folders derived from video used weak LiDAR room labels for input preparation; inference consumed RGB only.",
    "Damage candidates come from zero-shot CLIPSeg prompts and require human confirmation; zero candidates does not establish absence of damage.",
    "Opening/height accuracy, repeatability, calibrated intervals, and incumbent comparison remain unverified.",
]
Path("docs/benchmark-report.md").write_text("\n".join(lines) + "\n")
(root / "runs.json").write_text(json.dumps(rows, indent=2))
cards = []
for r in rows:
    label = html.escape(r["tier"].upper() + " · " + r["label"])
    geometry_status = "pass" if r["geometry_valid"] else "FAIL"
    cards.append(
        f'<a class="card" href="{r["path"]}/index.html"><img src="{r["path"]}/plan.png" alt="Floor plan preview"><h2>{label}</h2><p>{r["rooms"]} rooms / regions · {r["area_m2"]:.1f} m²</p><small>Geometry checks: {geometry_status}<br>Damage: {html.escape(damage_text(r))}<br>{html.escape(r["stitch_status"])}</small></a>'
    )
(root / "index.html").write_text(
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cozmo · Sample results</title><style>body{font-family:system-ui;background:#f3f5f2;color:#20312d;margin:0;padding:40px}main{max-width:1250px;margin:auto}h1{font-size:36px;letter-spacing:-1.5px}p{line-height:1.6;color:#67776c}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}.card{display:block;background:white;border:1px solid #dbe3dd;border-radius:12px;padding:20px;text-decoration:none;color:inherit}.card img{width:100%;height:260px;object-fit:contain}h2{font-size:16px}.card small{color:#8a7748}.notice{padding:16px;background:#f8f4e4;border-radius:8px;margin:25px 0}@media(max-width:600px){body{padding:16px}}</style></head><body><main><h1>cozmo. Sample results</h1><p>Phone captures, reconstructed locally. Open a result to inspect dimensions, uncertainty and evidence.</p><div class="notice">Experimental prototype. LiDAR plans and damage assessment run on all three samples. RGB whole-property stitching remains unresolved. Centimetre accuracy has not been independently validated.</div><div class="grid">"""
    + "".join(cards)
    + """</div></main></body></html>"""
)
print("Wrote out/index.html and docs/benchmark-report.md")
