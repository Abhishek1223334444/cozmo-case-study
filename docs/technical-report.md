# Technical report

## 1. Scope and evidence

The supplied captures contain 1,715, 5,251 and 9,745 paired depth/confidence frames,
RGB video, intrinsics and camera poses. This implementation runs without the candidate
owning an iPhone. The source case study is
[Cozmo AI, August 2026](https://docs.google.com/document/d/14YP72G24IjRHdHmW1l0MYYnRxsF52-9qU8rFJKGAC3A/edit).
The development data is the [supplied folder](https://drive.google.com/drive/folders/1rvcx0uEIwU6mIlEi8m5SF88jHK6ubAOu).
No independent tape/laser dimensions, damage labels, incumbent exports or published
JSON schema are present in the inspected archives. Scored claims requiring them are
not made. The compliance matrix is explicit about incomplete product requirements.

## 2. LiDAR geometry and stitching

Depth values are millimetres, decoded without 8-bit conversion. Intrinsics are scaled
from actual RGB resolution and read per frame when available. Poses follow the
sample's OpenCV camera convention. World +Y is up. Invalid/low-confidence pixels and
depth discontinuities are removed, and spatially downsampled clouds are fused.

The floor and ceiling are supported horizontal layers; dominant wall normals define
a Manhattan frame. Per-frame visibility fans establish free space. Morphological
seeds split rooms and doorway-like boundaries prevent inappropriate merging. Rooms
not entered by the camera are separately tracked. Room outlines are fitted to wall
surfaces, then overlapping polygons are resolved. Door/window candidates require
geometric gaps, supporting jambs and, for doors, observed space beyond the wall.
Unbounded jambs are rejected. These heuristics are not validated semantic detectors.

Rooms share one scan coordinate system. Adjacency candidates arise from detected
doors into another segmented room. Correct room identity and adjacency still need
independent annotation; a non-overlap check alone does not establish correctness.

## 3. Drift fix and ablation

The inherited implementation used poses as supplied. The new path estimates fixed
structural planes, gathers per-keyframe point-to-plane translation residuals, and
solves a temporally regularized translation trajectory. A conservative cap prevents
large moves. Corrections must improve held-out point residuals or are rejected.
The fixed plane model is estimated on the same scan; this holdout measures internal
consistency only. It cannot detect a uniform sensor bias. Rotation drift is not fixed.

`cozmo benchmark` produces regenerable before/after plans and a readable table.
The declared target for the two longer captures was at least 20% reduction in mean
held-out structural-plane residual. Observed reductions were about 28% and 33%.
On the short capture the residual fell from 17.82 to 10.90 mm. Room counts and
footprints also changed, demonstrating that lower residual does not itself prove a
better floor plan. The scored accuracy fix-loop gate cannot be assessed without truth.

## 4. Photo/video paths and uncertainty

Only RGB enters these inference paths. A pinned Depth Anything V2 metric indoor small
model supplies depth priors. RootSIFT and LightGlue match images; PnP estimates
relative poses. A maximum-support graph initializes independent components and robust
pose-graph optimization uses redundant edges. The first upright image supplies a
vertical prior refined from surface normals. Focal length is assumed from image size.

For photo folders, the strongest registered subset per room forms an estimated
oriented envelope. Discarded views are reported. Video currently produces observed
component envelopes; complete multi-room segmentation is unfinished. Disconnected
components remain explicitly unplaced, with display-only offsets. Overlaps and
unresolved stitching are quality failures, not hidden by visually packing rooms.
This baseline does not establish the photo ±8%, video ±3%, or adjacency gates.

The output includes provisional intervals on every reported measurement. LiDAR
allowances include a 20 mm systematic plane term, a 1% scale term, and larger terms
for unmeasured walls. RGB allowances are much wider (12–18% one-sigma scale priors,
plus minimum absolute uncertainty); these are assumptions, not measured performance.
Correlated room-area uncertainty is conservatively summed. No interval is labelled
calibrated. `evaluate` reports empirical coverage only when independent truth is given.

## 5. Damage, scope and reproducibility

Local CLIPSeg prompts identify water-stain/crack candidates, opposed by a clean-wall
prompt. Components above threshold are back-projected, associated with nearby wall
surfaces and merged across observations. Extent uses a projected convex hull and is
labelled an upper region extent, not exact damaged area. Scores are not probabilities.
Evidence images, model revisions and thresholds accompany candidates. Water-stain
candidates trigger a named concealed-moisture inspection rule, never a hidden-damage
diagnosis. Surface-linked scope items currently request inspection/confirmation.

Inference does not call hosted models. Dependencies are locked, model revisions and
download hashes are pinned, sensor caches are keyed to inputs/configuration, and
run options are saved. Unit tests exercise depth units, camera intrinsics, sparse frame
IDs, geometry consistency, missed/phantom opening scoring, archive safety and viewer
escaping. A local browser render was inspected. A clean-machine 15-minute setup and
independent cold walk-in performance have not been timed or established.

## 6. Known failures and next evidence

Mirrors and glass produce spurious depth; blank walls and fast camera motion weaken
registration; furniture can masquerade as walls or ceilings; non-Manhattan rooms and
steps violate simplifying assumptions. Scarce ceiling coverage yields missing height.
Derived photo folders have weak room labels from LiDAR geometry and are disclosed
as development inputs. They must not be represented as independently captured photos.

The next validation requires laser/tape measurements with explicit surface IDs,
independent repeat captures, damage annotations and a consumer-app export on the same
rooms. A real fix-loop accuracy claim then requires choosing the worst measured gate,
declaring a prediction before changing it, and reproducing both runs. Current artifacts
support an engineering prototype and diagnostic improvement, not a completed set of
case-study gates.
