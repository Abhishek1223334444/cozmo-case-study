# Cozmo: phone captures to dimensioned floor plans

Turns a phone scan of a property into a dimensioned, multi-room floor plan with
visible-damage candidates. Input is a [Stray Scanner](https://github.com/strayrobots/scanner)
LiDAR capture, a handheld video, or folders of photos. Everything runs locally.

## Pipeline

```
capture ─► load ─► drift correction ─► 3D point cloud ─► floor, ceiling, wall direction
        ─► free-space map ─► room segmentation ─► wall fitting ─► openings
        ─► measurements + intervals ─► surfaces ─► damage (optional) ─► checks ─► outputs
```

### LiDAR tier

| Step | What happens | Module |
|---|---|---|
| 1. Load | Read RGB video, depth, confidence, per-frame camera poses and intrinsics | `capture.py` |
| 2. Drift correction | Fit persistent wall/floor planes, solve smooth per-keyframe translations, keep the correction only if held-out plane residuals improve | `drift.py` |
| 3. Point cloud | Back-project every depth frame into one world-frame cloud with normals (cached) | `cache.py`, `geometry.py` |
| 4. Floor, ceiling, walls | Height histograms give floor and ceiling; wall normals give the dominant (Manhattan) axis | `geometry.py` |
| 5. Free space | For each frame, cast a 180-bin fan from the camera to the farthest hit and accumulate seen-through floor cells | `rooms.py` |
| 6. Rooms | Distance-transform sweep finds one seed per room (doorways disconnect first), seeds grow back over free space, splits that are not doorways are merged, rooms never entered are dropped | `rooms.py` |
| 7. Walls | Trace each room outline, simplify to axis-aligned walls, snap every wall onto the measured wall surface, resolve overlaps between rooms | `layout.py` |
| 8. Openings | Scan each wall for gaps: full-height and seen through is a door, wall below is a window | `layout.py` |
| 9. Measurements | Wall lengths, floor area, perimeter, ceiling height and opening sizes, each with an uncertainty interval; doors link rooms into an adjacency graph | `lidar.py`, `plan.py` |

### Photo and video tiers

No depth or poses are used, only RGB.

1. Metric depth per image from Depth Anything V2 (metric indoor).
2. SIFT features matched with LightGlue, relative poses from PnP, pose-graph optimisation.
3. Connected views are levelled and fused; each photo folder (or video segment) becomes a room envelope.
4. Views that do not connect are kept apart and marked as display placement only.

Modules: `models.py`, `matching.py`, `rgb.py`.

### Damage (`--damage`)

1. Sample about 12 frames across the capture.
2. Score each frame with CLIPSeg for "water stain", "crack" and a "clean wall" reference.
3. Keep regions above 0.55 that beat the reference by 0.15.
4. Project each region onto its wall using depth and pose, and measure its area in m².
5. Save evidence images, merge repeats of the same region, and add a scope line item. Water stains also raise a concealed-moisture inspection flag.

Module: `damage.py`. Every result is a candidate that needs human confirmation.

### Outputs and checks

Each run writes `plan.json` (following `src/cozmo/schema.json`), `plan.png`, `plan.svg`,
an offline interactive `index.html` and `run.json` with the exact options. Geometry
checks (valid polygons, consistent lengths and areas, openings inside walls, no room
overlaps) are stored in the plan. Modules: `plan.py`, `render.py`, `viewer.py`, `evaluation.py`.

## Quick start

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.

```sh
uv sync --locked
uv run python scripts/download_samples.py
uv run cozmo fetch-models all
```

One command per capture:

```sh
uv run cozmo run samples/c7d28f72c6 --damage -o out/lidar-property-full
uv run cozmo run samples/c00a170fe1/rgb.mp4 --tier video --rotation 90 --max-frames 80 -o out/video-single
uv run cozmo run path/to/photos --tier photos -o out/my-photos
```

Reproduce every reported result, then browse them at `http://127.0.0.1:8877/`:

```sh
uv run python scripts/reproduce.py --rgb --damage --property
uv run cozmo serve --directory out --port 8877
```

Other commands: `cozmo validate plan.json`, `cozmo evaluate plan.json truth.json`
(needs laser or tape measurements), `cozmo benchmark` (drift on/off ablation),
`uv run pytest -q`.

## Limits

Walls are assumed roughly perpendicular, and uncertainty intervals are not calibrated
against ground truth. Photo and video stitching is incomplete. See
`docs/technical-report.md` and `docs/compliance.md` for details.
