# cozmo — phone captures to dimensioned floor plans

Work in progress for the Cozmo AI case study. Current state: **LiDAR tier** runs end to end
on Stray Scanner captures.

## Setup (macOS / Linux, Python 3.11 via uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync
```

## Run

```bash
uv run cozmo run path/to/capture.zip        # or an unzipped Stray Scanner folder
# -> out/<capture>/plan.json  (every measurement: value, sigma, 95% interval)
# -> out/<capture>/plan.png   (dimensioned, stitched floor plan)
```

## Pipeline (LiDAR tier)

1. `capture.py` — load depth (mm, 256x192), confidence, poses (OpenCV camera convention), intrinsics.
2. `geometry.py` — fuse to a gravity-aligned cloud with normals; floor/ceiling layers; Manhattan alignment.
3. `rooms.py` — free-space map from per-frame view fans; morphological-sweep room seeds;
   merge regions whose boundary is not a doorway in a wall line; keep rooms the camera entered.
4. `layout.py` — rectilinear outline per room, each wall snapped to its measured surface
   (5 mm histogram peak of inward-facing points); per-room floor/ceiling; doors and windows as
   gaps in the wall surface.
5. `lidar.py` — error model and assembly; `render.py` — plan drawing; `cli.py` — one command.

`experiments/` holds the scripts used to develop each stage.
