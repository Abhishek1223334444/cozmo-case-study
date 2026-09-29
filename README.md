# Cozmo — phone captures to dimensioned plans

A local, runnable case-study prototype using the interviewer's three Stray Scanner
samples. **No iPhone is needed to run the supplied examples.**

LiDAR reconstruction, room segmentation, dimensions, openings, translation-drift
correction and interactive plan exports run on the supplied captures. Photo/video
paths use local learned depth and image matching; their stitching remains experimental
and is explicitly marked unresolved when views do not connect. This is **not a
claim of passing the case study's accuracy or cold walk-in gates**.

## Run the existing build

```sh
cd /Users/apple/cozmo-case-study
uv run cozmo serve --directory out --port 8877
```

Open `http://127.0.0.1:8877/` for the results gallery, or
`http://127.0.0.1:8877/benchmark/c7d28f72c6/after/` for the larger scan with
ceiling coverage. Each output directory also has an `index.html` that opens directly
in a browser, plus `plan.json`, `plan.svg`, `plan.png`, and exact run options.

## Fresh installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then:

```sh
uv sync --locked
uv run python scripts/download_samples.py
uv run cozmo fetch-models all
```

Python 3.11 is selected by `.python-version`. Model weights and samples are fetched
separately; they are not committed. Setup needs internet. Processing runs locally
without an API key, remote inference or this author's infrastructure. LiDAR geometry
does not need model weights unless `--damage` is enabled. Disk space: allow 4–6 GB
for the environment, samples, weights and outputs. CPU works; depth inference can
use `--device cuda` or `--device mps` on supported hardware.

The existing `samples` link points to `/Users/apple/cozmo/samples` to reuse downloaded
data. On another machine the download script creates an ordinary `samples` directory.

## One command per capture

```sh
# LiDAR: a ZIP or extracted Stray Scanner directory
uv run cozmo run samples/c7d28f72c6 -o out/lidar-property

# Include visible-damage candidates, surface projection and inspection rules
uv run cozmo run samples/c00a170fe1 --damage -o out/lidar-single-full

# Video: only RGB is consumed, never the neighbouring sensor files
uv run cozmo run samples/c00a170fe1/rgb.mp4 --tier video --rotation 90 --max-frames 80 -o out/video-single

# Photos: 2–8 images per room directory, no depth or pose sidecars
uv run cozmo run path/to/photos --tier photos -o out/my-photos

# Recreate isolated photo inputs from the supplied RGB video
uv run cozmo prepare-sample samples/c00a170fe1 --plan out/benchmark/c00a170fe1/after/plan.json -o out/inputs/single
uv run cozmo run out/inputs/single/photos --tier photos -o out/photos-single
```

`prepare-sample` uses camera trajectories and inferred room polygons **only to assign
development photo folders**. It writes that provenance explicitly. Photo inference
receives only the exported images. These are weakly labelled development examples,
not an independent photo benchmark. Supplied raw video is sideways and needs
`--rotation 90`; native video may already have correct orientation metadata.

## Reproduce and evaluate

```sh
uv run python scripts/reproduce.py --rgb --damage --property
uv run cozmo validate out/benchmark/c7d28f72c6/after/plan.json
uv run pytest -q
uv run ruff check src scripts tests

# After independently measuring and matching the physical dimensions:
uv run cozmo evaluate out/lidar-property/plan.json my-laser-measurements.json
```

The benchmark writes uncorrected and corrected plans and reports internal plane
residuals. `--no-drift` selects the baseline. Replaying one recording tests deterministic
processing; it does not test capture repeatability. The evaluation command requires
independent laser/tape data and counts missed/phantom opening widths when the opening
inventory is declared exhaustive. See `docs/ground-truth.example.json` for the format.

Every numerical measurement includes a value, units, standard-deviation allowance,
and a nominal 95% interval. These are **uncalibrated model intervals**, not empirically
verified confidence coverage. Geometry fitting residuals cannot establish absolute
measurement accuracy. The local schema is in `src/cozmo/schema.json`; the published
interviewer schema was not included in the linked materials.

## Architecture

- `capture.py`, `cache.py`: Stray Scanner decoding, per-frame intrinsics, content-aware cache keys.
- `drift.py`: plane anchors, temporally smooth translations, held-out consistency check.
- `geometry.py`, `rooms.py`, `layout.py`, `lidar.py`: fused points, room segmentation, wall fitting, opening candidates.
- `models.py`, `matching.py`, `rgb.py`: pinned local depth weights, RootSIFT/LightGlue, PnP, pose-graph optimization, provisional room envelopes.
- `damage.py`: CLIPSeg candidates, metric surface regions, evidence images and rule-based inspection flags.
- `plan.py`, `schema.json`, `render.py`, `viewer.*`: common output contract and interactive/exported plans.
- `evaluation.py`: schema/geometry checks, explicit dimension matching, benchmark and drift ablation.

## Current limits

LiDAR layouts assume approximately perpendicular walls. Rooms can be over-segmented,
occluded walls inferred, and openings missed. Translation correction does not solve
rotation drift or global depth bias. Unobserved ceilings are returned as `null`.

RGB depth has learned scale and assumed intrinsics; sparse views, mirrors, blank walls
and close-ups cause registration failures. Disconnected components get clearly marked
display offsets, never invented adjacency. RGB room envelopes and openings do not yet
meet the full stitched-property contract. Video components may contain multiple rooms
without correctly separating them.

Saved sample-run counts, geometry failures and runtime measurements are listed in
`docs/benchmark-report.md`. Cached and cold runtimes are not directly comparable.

Damage outputs are candidates requiring confirmation, not calibrated classifications.
The supplied data has no labelled staged-damage benchmark; zero candidates does not
mean no damage. Concealed-moisture flags only request inspection. Scope quantities are
provisional and do not constitute a completed restoration estimate.

Independent laser/tape truth, matched repeat captures, incumbent-app exports and the
interviewer's schema are absent. Their gates remain unverified. See
`docs/compliance.md`, `docs/technical-report.md`, `docs/capture-protocol.md`, and
`docs/fix-declaration.md` for the submission evidence and limits.

## Result bundle

`deliverables/cozmo-case-study.zip` contains source, locked dependencies, docs, saved
plans, a standalone results gallery, an installable wheel and incremental Git history.
It excludes the large sample archives, model weights and caches; the pinned download
commands above restore those. After extracting, open `out/index.html` in a browser.

To regenerate the bundle after reproducing results and committing reviewed changes:

```sh
uv build --offline
uv run --offline python scripts/package_results.py
```

`MANIFEST.json` records the source revision and hashes of every packaged file.
