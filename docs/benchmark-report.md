# Actual sample runs

These are development runs. No independent ground truth was supplied. Runtime may include cached intermediates; see run options and logs.

| Tier | Run | Rooms/regions | Sum area (m²) | Runtime (s) | Geometry checks | Stitch status | Damage |
|---|---|---:|---:|---:|---|---|---|
| lidar | [Property scan, floor only](../out/benchmark/1a8384c3f6/after/index.html) | 8 | 48.94 | 8.7 | True | sensor-frame layout; adjacency unverified | not run |
| lidar | [Short scan](../out/benchmark/c00a170fe1/after/index.html) | 3 | 19.67 | 2.5 | True | sensor-frame layout; adjacency unverified | not run |
| lidar | [Property scan, with ceiling](../out/benchmark/c7d28f72c6/after/index.html) | 8 | 46.54 | 13.3 | True | sensor-frame layout; adjacency unverified | not run |
| photos | [Short scan, derived photos](../out/photos-single/index.html) | 3 | 18.97 | 6.1 | True | unresolved | not run |
| video | [Short scan, 80-frame selection](../out/video-single/index.html) | 6 | 26.10 | 31.2 | True | unresolved | not run |
| video | [Short scan, 160-frame selection](../out/video-single-dense/index.html) | 6 | 10.58 | 130.8 | True | unresolved | not run |
| lidar | [Short scan, with damage assessment](../out/lidar-single-full/index.html) | 3 | 19.67 | 11.0 | True | sensor-frame layout; adjacency unverified | 0 candidates (11 views) |
| lidar | [Property scan floor only, with damage assessment](../out/lidar-floor-full/index.html) | 8 | 48.94 | 14.4 | True | sensor-frame layout; adjacency unverified | 0 candidates (11 views) |
| photos | [Property scan, derived photos](../out/photos-property/index.html) | 8 | 79.36 | 11.0 | False | unresolved | not run |
| video | [Property scan, 80-frame selection](../out/video-property/index.html) | 3 | 13.30 | 18.5 | True | unresolved | not run |
| lidar | [Property scan, with damage assessment](../out/lidar-property-full/index.html) | 8 | 46.54 | 19.1 | True | sensor-frame layout; adjacency unverified | 0 candidates (11 views) |

RGB areas are sums of inferred observed envelopes; they are not validated whole-property footprints.
Photo folders derived from video used weak LiDAR room labels for input preparation; inference consumed RGB only.
Damage candidates come from zero-shot CLIPSeg prompts and require human confirmation; zero candidates does not establish absence of damage.
Opening/height accuracy, repeatability, calibrated intervals, and incumbent comparison remain unverified.
