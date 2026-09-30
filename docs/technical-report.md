# Technical report

## 1. Scope and data

I built and tested the pipeline on the three supplied Stray Scanner captures, which
have 1,715, 5,251 and 9,745 frames of depth, confidence, RGB video, intrinsics and
camera poses. No iPhone was needed to develop it. The supplied data has no tape or
laser measurements, damage labels, consumer-app exports or published JSON schema, so
the accuracy gates that depend on them are not assessed here. `docs/compliance.md`
gives the status of each requirement.

## 2. LiDAR geometry and stitching

Depth is read as 16-bit millimetres. Intrinsics are scaled from the RGB resolution to
the depth resolution and read per frame when available. Poses use the OpenCV camera
convention (I checked this: with it the floor collapses to a 2 cm layer, with ARKit's
convention it smears). World +Y is up. I drop low-confidence pixels and depth
discontinuities, downsample each frame and fuse everything into one cloud.

The floor and ceiling come from the dominant horizontal layers, and wall normals give
the main wall direction (Manhattan frame). For free space, each frame casts a fan from
the camera to the farthest hit in each direction. Rooms are split with a distance-
transform sweep, so narrow doorways disconnect before rooms do, and two regions are
only kept apart if the boundary between them sits in a wall line. Rooms the camera
never entered are tracked but not drawn. Each room outline is then fitted to the
measured wall surfaces and overlapping rooms are trimmed. A door needs a full-height
gap with jambs on both sides and observed space beyond it; a window needs wall below
the gap.

All rooms share the scan's coordinate system, so stitching comes from the sensor
poses. Two rooms are adjacent when a detected door leads from one into the other.

## 3. Drift fix and ablation

The first version used the phone's poses as recorded. Drift correction now fits
persistent wall and floor planes, measures how far each keyframe's points sit from
them, and solves a smooth per-keyframe translation, capped so it cannot move the
camera far. A correction is kept only if it lowers the residual on held-out points.

`cozmo benchmark` writes before and after plans for every capture and a summary table.
I predicted at least a 20% drop in held-out plane residual on the two longer captures
and measured about 28% and 33%. On the short capture it fell from 17.82 to 10.90 mm.
On one capture the correction also separated two rooms that had merged (7 to 8 rooms).

## 4. Photo and video tiers

These tiers only use RGB. Depth Anything V2 (metric indoor, small) gives per-image
depth. RootSIFT features are matched with LightGlue and PnP gives relative poses. A
maximum-support spanning graph initialises each connected group of views, and a
robust pose-graph optimisation uses the extra edges. The first upright image sets the
vertical, refined from surface normals. Focal length is assumed from image size.

For photo folders, the best-registered views in each folder form that room's outline,
and views that do not register are listed in the output. For video, connected groups
are split into rooms using visibility and wall geometry, or kept as one observed
region when the group is small. Groups that do not connect are placed side by side
and marked as display placement only, rather than packed together to look complete.
Room overlaps are reported as failures.

## 5. Uncertainty

Every measurement has a value, a sigma and a 95% interval. For LiDAR the sigma
combines a 20 mm plane term, a 1% scale term and a larger term for walls with no
measured surface. RGB uses much wider scale priors (12–18% one sigma) plus a minimum
absolute term. The footprint sigma is the sum of the room sigmas, since their errors
are correlated.
These values are assumptions; `cozmo evaluate` reports real coverage once tape or
laser measurements are supplied.

## 6. Damage, scope and reproducibility

CLIPSeg scores each sampled frame for "water stain" and "crack" against a "clean wall"
prompt. Regions above the threshold are back-projected with depth, assigned to the
nearest wall and merged across frames. The area is the convex hull of the projected
points, so it is an upper bound. Each candidate carries its evidence images, model
revision and thresholds. A water-stain candidate triggers a concealed-moisture
inspection flag with the rule name, and every candidate gets a scope line asking for
inspection before repair.

Inference runs locally. Dependencies are locked, model revisions and download hashes
are pinned, caches are keyed to the input and settings, and each run saves its options.
Unit tests cover depth units, intrinsics, sparse frame IDs, geometry consistency,
missed and phantom opening scoring, archive safety and viewer escaping. I checked the
viewer in a browser.

## 7. Known failures and limits

- Mirrors and glass give false depth; blank walls and fast motion break registration.
- Furniture can be mistaken for walls or ceilings; angled walls and steps break the
  Manhattan assumption. Ceilings that were not scanned have no height.
- Rooms can be over-segmented and openings missed.
- Drift correction fixes translation only, not rotation or a global scale bias. The
  held-out residual measures internal consistency, not absolute accuracy.
- Photo and video stitching is incomplete; the property photo run has overlapping rooms.
- The photo folders were cut from the sample videos using rough LiDAR room labels, so
  they are development inputs, not independent photo captures.
- Intervals are uncalibrated and damage scores are not probabilities.
- A 15-minute clean-machine setup and a cold walk-in run have not been timed.

To close these gaps I need laser or tape measurements keyed to walls and openings,
a repeat capture of the same rooms, labelled damage, and a consumer-app export of the
same rooms. With those, the fix loop can target the worst measured gate directly.
