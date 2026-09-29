# Compliance matrix

Statuses describe implemented evidence, not promised interview scores.

| Requirement | Implementation / artifact | Status |
|---|---|---|
| Stock capture route + device matrix | `docs/capture-protocol.md` | Written; new-device protocol not field-tested |
| One command per capture | `src/cozmo/cli.py`, README | Implemented |
| LiDAR wall/height/area plan | `lidar.py`, `layout.py`; `out/benchmark/*/after/plan.json` | Runs on all three supplied scans; accuracy unverified |
| Stitched LiDAR property plan | Shared sensor coordinate frame, room segmentation and drift correction | Implemented, adjacency/detection not independently scored |
| Photo folders | `rgb.py`; `out/photos-single` | Runs; strongest subset per room; whole-property stitch unresolved on sample |
| Video | `rgb.py`; `out/video-single` | Runs; disconnected observed regions remain; full room segmentation incomplete |
| Drift correction + ablation | `drift.py`; `out/benchmark/*/{before,after}` | Implemented; translation only; internal consistency improved |
| Openings | `layout.find_openings` | LiDAR geometric candidates; RGB openings incomplete; 2 cm/85% gate unverified |
| Ceiling height | `layout.fit_room` | Returned only when supported; no 1.5 cm accuracy evidence |
| Every reported measurement has intervals | `plan.M`, `schema.json` | Implemented; intervals explicitly uncalibrated |
| Published schema | `src/cozmo/schema.json` | Local contract supplied; interviewer schema missing |
| Visible damage and metric regions | `damage.py`, `--damage`; `out/lidar-single-full` | Experimental local CLIPSeg; no labelled damage test set; sample produced zero candidates |
| Concealed flags + firing rule | `damage.py` | Rule implemented for visible water-stain candidates; not evidence of hidden damage |
| Surface-keyed scope | `damage.py` | Provisional inspection quantities for candidates; full repair/pricing scope incomplete |
| Multi-room sample | Larger supplied scans | Available; physical room labels/adjacency not independently verified |
| Same rooms at all tiers | `sample.py`, `out/inputs/*/provenance.json` | Derived development inputs; not independent photo/video captures |
| Laser/tape ground truth | `evaluate` command; `docs/ground-truth.example.json` | Missing data; no fabricated measurements |
| Repeat capture gate | Supplied recordings | Not established as matched independent repeats |
| Consumer app comparison | Required same-room exports | Missing exports; not evaluated |
| Fix declaration and reproducible delta | `docs/fix-declaration.md`, benchmark | Diagnostic fix shipped; scored accuracy fail→pass loop not established |
| Process history | Git commits | Incremental history retained; new work isolated on `case-study-build` |
| Fresh setup / offline inference | `uv.lock`, model pins, sample downloader | Local sample runs verified; 15-minute clean-machine install not independently timed |
| Cold walk-in all tiers | Capture route + CLI | Not ready to claim all scored gates pass |
