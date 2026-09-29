"""Build a concise report and landing page from actual saved runs."""
from pathlib import Path
import html
import json

root=Path("out")
paths=sorted(root.glob("benchmark/*/after/plan.json"))
paths += [p for p in [root/"photos-single/plan.json",root/"video-single/plan.json",root/"video-single-dense/plan.json",root/"lidar-single-full/plan.json"] if p.exists()]
paths += [p for p in [root/"photos-property/plan.json",root/"video-property/plan.json",root/"lidar-property-full/plan.json"] if p.exists()]
rows=[]
for path in paths:
    plan=json.loads(path.read_text())
    rows.append({"path":str(path.parent.relative_to(root)),"capture":plan["capture"],"tier":plan["tier"],
                 "rooms":len(plan["rooms"]),"area_m2":plan["footprint_area"]["value"],
                 "runtime_s":plan["diagnostics"].get("total_runtime_s"),
                 "geometry_valid":plan["quality"].get("geometry_checks",{}).get("valid"),
                 "stitch_status":plan["quality"].get("stitch_status","sensor-frame layout; adjacency unverified"),
                 "components":plan["diagnostics"].get("components"),"accuracy_gate":"unverified"})
lines=["# Actual sample runs","","These are development runs. No independent ground truth was supplied. Runtime may include cached intermediates; see run options and logs.","",
       "| Tier | Capture | Rooms/regions | Sum area (m²) | Runtime (s) | Geometry checks | Stitch status |","|---|---|---:|---:|---:|---|---|"]
for r in rows:
    lines.append(f"| {r['tier']} | {r['capture']} | {r['rooms']} | {r['area_m2']:.2f} | {r['runtime_s']:.1f} | {r['geometry_valid']} | {r['stitch_status']} |")
lines += ["","RGB areas are sums of inferred observed envelopes; they are not validated whole-property footprints.",
          "Photo folders derived from video used weak LiDAR room labels for input preparation; inference consumed RGB only.",
          "Opening/height accuracy, repeatability, calibrated intervals, and incumbent comparison remain unverified."]
Path("docs/benchmark-report.md").write_text("\n".join(lines)+"\n")
(root/"runs.json").write_text(json.dumps(rows,indent=2))
cards=[]
for r in rows:
    label=html.escape(r['tier'].upper()+" · "+r['capture'])
    cards.append(f'<a class="card" href="{r["path"]}/index.html"><img src="{r["path"]}/plan.png" alt="Floor plan preview"><h2>{label}</h2><p>{r["rooms"]} rooms / regions · {r["area_m2"]:.1f} m²</p><small>{html.escape(r["stitch_status"])}</small></a>')
(root/"index.html").write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cozmo · Sample results</title><style>body{font-family:system-ui;background:#f3f5f2;color:#20312d;margin:0;padding:40px}main{max-width:1250px;margin:auto}h1{font-size:36px;letter-spacing:-1.5px}p{line-height:1.6;color:#67776c}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}.card{display:block;background:white;border:1px solid #dbe3dd;border-radius:12px;padding:20px;text-decoration:none;color:inherit}.card img{width:100%;height:260px;object-fit:contain}h2{font-size:16px}.card small{color:#8a7748}.notice{padding:16px;background:#f8f4e4;border-radius:8px;margin:25px 0}</style></head><body><main><h1>cozmo. Sample results</h1><p>Phone captures, reconstructed locally. Open a result to inspect dimensions, uncertainty and evidence.</p><div class="notice">Experimental prototype. LiDAR plans run on all three samples. RGB whole-property stitching remains unresolved. Centimetre accuracy has not been independently validated.</div><div class="grid">'''+"".join(cards)+'''</div></main></body></html>''')
print("Wrote out/index.html and docs/benchmark-report.md")
