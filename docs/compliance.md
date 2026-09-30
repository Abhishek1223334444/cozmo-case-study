# Compliance matrix

Status of each requirement in the brief, with where to find it.

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
| Every reported measurement has intervals | `plan.M`, `schema.json` | Implemented; intervals not yet calibrated |
| Published schema | `src/cozmo/schema.json` | Local contract supplied; interviewer schema missing |
| Visible damage and metric regions | `damage.py`, `--damage`; `out/lidar-*-full` | Local CLIPSeg on all three LiDAR captures; no labelled damage test set; zero candidates found |
| Concealed flags + firing rule | `damage.py` | Rule implemented for visible water-stain candidates; not evidence of hidden damage |
| Surface-keyed scope | `damage.py` | Provisional inspection quantities for candidates; full repair/pricing scope incomplete |
| Multi-room sample | Larger supplied scans | Available; physical room labels/adjacency not independently verified |
| Same rooms at all tiers | `sample.py`, `out/inputs/*/provenance.json` | Derived development inputs; not independent photo/video captures |
| Laser/tape ground truth | `evaluate` command; `docs/ground-truth.example.json` | No measurements available yet |
| Repeat capture gate | Supplied recordings | Not established as matched independent repeats |
| Consumer app comparison | Required same-room exports | Missing exports; not evaluated |
| Fix declaration and reproducible delta | `docs/fix-declaration.md`, benchmark | Drift fix shipped, residual down 28–33%; accuracy gate not measurable without ground truth |
| Process history | Git commits | Incremental commit history on `main` |
| Fresh setup / offline inference | `uv.lock`, model pins, sample downloader | Runs locally on the samples; 15-minute clean-machine install not timed |
| Cold walk-in all tiers | Capture route + CLI | All tiers run from one command; accuracy gates unverified |
