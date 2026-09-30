# Capture protocol: stock apps, no custom iPhone app

The supplied Stray Scanner ZIPs are sufficient to run development without an iPhone.
This page describes the proposed walk-in capture route; it has not been field-tested
on a new device in this submission.

## Photos: native Camera

Use the rear 1× camera. Turn on the lights, open interior doors, and keep furniture
stationary. Hold the phone upright. Take 6–8 images per room, moving around its
perimeter with substantial overlap, not just rotating at one point. Include floor
and wall junctions, the ceiling, and both sides of each doorway. Photograph each
shared doorway from both rooms. Avoid close-ups, motion blur, and looking straight
into mirrors or windows. Export originals into `photos/<room-name>/`.

Run `uv run cozmo run photos --tier photos -o out/new-photos`.
The 2-image case can be attempted but is frequently underconstrained. Missing visual
connections produce separate components that are not placed relative to each other. Learned metric scale
is a prior and has not met the ±8% accuracy requirement on independent measurements.

## Video: native Camera

Use 1080p or 4K at 30 fps, rear 1× camera, portrait orientation, without changing
lenses or zoom. Walk slowly around each room, cross doorways at walking speed, and
look back through each connecting doorway. Return to the start for a loop. Cover
floor/wall junctions and the ceiling. Avoid fast turns, excessive tilt and long
close-ups. Export the original MOV/MP4, without messaging-app compression.

Run `uv run cozmo run walk.mov --tier video -o out/new-video`.
Check image orientation in the preview. `--rotation 90` rotates clockwise and is
required for the supplied raw Stray Scanner RGB streams, not necessarily native video.

## LiDAR: Stray Scanner

Install [Stray Scanner](https://github.com/strayrobots/scanner) using its App Store
link on a LiDAR-equipped iPhone Pro. Start one continuous recording. Walk through
all rooms and connectors, look at each wall and opening, and deliberately sweep
both the floor and ceiling. Stay roughly 0.5–4 m from surfaces. Complete a loop back
to the starting room. Do not pause and resume unrelated coordinate systems.

Export the complete capture folder or ZIP: `rgb.mp4`, `depth/`, `confidence/`,
`odometry.csv`, `imu.csv`, `camera_matrix.csv`. Do not send only a screenshot or mesh.
Run `uv run cozmo run capture.zip --damage -o out/new-lidar`.

## Hardware and accuracy matrix

| Input | Capture hardware in brief | Local processing | Accuracy evidence |
|---|---|---|---|
| Photos | iPhone 15 or newer; ordinary RGB camera | CPU; optional CUDA/MPS depth inference | Experimental; no verified ±8% or adjacency gate |
| Video | iPhone 15 or newer | CPU; optional CUDA/MPS depth inference | Experimental; no verified ±3% gate |
| LiDAR | iPhone 15 Pro/Pro Max or later LiDAR-equipped Pro model | CPU geometry; optional accelerator for damage model | Sample plans generated; cm-level accuracy unverified without laser/tape truth |

Mirror/glass returns and low light remain known failure cases. Photograph damaged
surfaces with enough context to locate them. Concealed-moisture flags request an
inspection; they are not a diagnosis of hidden damage.
