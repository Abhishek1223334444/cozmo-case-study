# Actual sample runs

These are development runs. No independent ground truth was supplied. Runtime may include cached intermediates; see run options and logs.

| Tier | Run | Rooms/regions | Sum area (m²) | Runtime (s) | Geometry checks | Stitch status |
|---|---|---:|---:|---:|---|---|
| lidar | [Property scan, floor only](../out/benchmark/1a8384c3f6/after/index.html) | 8 | 48.94 | 8.6 | True | sensor-frame layout; adjacency unverified |
| lidar | [Short scan](../out/benchmark/c00a170fe1/after/index.html) | 3 | 19.67 | 2.6 | True | sensor-frame layout; adjacency unverified |
| lidar | [Property scan, with ceiling](../out/benchmark/c7d28f72c6/after/index.html) | 8 | 46.54 | 12.6 | True | sensor-frame layout; adjacency unverified |
| photos | [Short scan, derived photos](../out/photos-single/index.html) | 3 | 18.97 | 6.4 | True | unresolved |
| video | [Short scan, 80-frame selection](../out/video-single/index.html) | 6 | 26.10 | 30.4 | True | unresolved |
| video | [Short scan, 160-frame selection](../out/video-single-dense/index.html) | 6 | 10.58 | 132.0 | True | unresolved |
| lidar | [Short scan, with damage assessment](../out/lidar-single-full/index.html) | 3 | 19.67 | 16.6 | True | sensor-frame layout; adjacency unverified |
| photos | [Property scan, derived photos](../out/photos-property/index.html) | 8 | 79.36 | 11.5 | False | unresolved |
| video | [Property scan, 80-frame selection](../out/video-property/index.html) | 3 | 13.30 | 19.3 | True | unresolved |
| lidar | [Property scan, with damage assessment](../out/lidar-property-full/index.html) | 8 | 46.54 | 25.9 | True | sensor-frame layout; adjacency unverified |

RGB areas are sums of inferred observed envelopes; they are not validated whole-property footprints.
Photo folders derived from video used weak LiDAR room labels for input preparation; inference consumed RGB only.
Opening/height accuracy, repeatability, calibrated intervals, and incumbent comparison remain unverified.
