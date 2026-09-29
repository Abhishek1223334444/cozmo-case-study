"""LiDAR tier: Stray Scanner depth + ARKit poses -> dimensioned, stitched plan."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from . import geometry as g
from .cache import fused_cloud
from .capture import Capture
from .layout import Opening, RoomLayout, find_openings, fit_room, inward_sign, resolve_overlaps
from .plan import M, OpeningOut, PlanOut, RoomOut, WallOut, quad
from .rooms import free_space, segment_rooms

RES = 0.02  # plan raster, m

# Error model priors (1-sigma). To be calibrated against tape/laser ground truth.
SIGMA_SURFACE_SYS = 0.02  # provisional systematic allowance, not calibrated accuracy
SIGMA_UNMEASURED = 0.05  # wall placed from free-space raster only
SCALE_REL = 0.01  # provisional relative scale allowance
SIGMA_JAMB = 0.008  # one jamb edge located from surface points
SIGMA_JAMB_WEAK = 0.03


def _wall_sigma(w) -> float:
    if not w.snapped:
        return SIGMA_UNMEASURED
    return quad(w.spread / math.sqrt(max(w.n_points, 1)), SIGMA_SURFACE_SYS)


def _room_out(room: RoomLayout, openings: list[Opening], labels, frame, adjacency_out) -> RoomOut:
    V = room.vertices
    k = len(V)
    sig = [_wall_sigma(w) for w in room.walls]
    lengths = room.wall_lengths
    walls = []
    for i in range(k):
        # length of wall i is set by its two perpendicular neighbours
        s = quad(sig[(i - 1) % k], sig[(i + 1) % k], SCALE_REL * lengths[i])
        walls.append(
            WallOut(
                i,
                V[i].round(4).tolist(),
                V[(i + 1) % k].round(4).tolist(),
                M(lengths[i], s),
                room.walls[i].snapped,
            )
        )
    area = room.area
    area_sigma = quad(*(lengths[i] * sig[i] for i in range(k)), 2 * SCALE_REL * area)
    perim_sigma = quad(*(2 * s for s in sig), SCALE_REL * room.perimeter)

    ceiling = None
    notes = []
    if room.ceiling_height is not None:
        h = room.ceiling_height
        ceiling = M(
            h,
            quad(
                room.ceiling_spread / math.sqrt(max(room.n_ceiling, 1)),
                room.floor_spread / math.sqrt(max(room.n_floor, 1)),
                SIGMA_SURFACE_SYS,
                SCALE_REL * h,
            ),
        )
    else:
        notes.append("ceiling not observed in this capture")
    unmeasured = sum(not w.snapped for w in room.walls)
    if unmeasured:
        notes.append(f"{unmeasured} of {k} walls placed from free space only (no surface fit)")

    ops = []
    for o in openings:
        a, b = V[o.wall], V[(o.wall + 1) % k]
        d = (b - a) / np.linalg.norm(b - a)
        mid = a + d * (o.start + o.width / 2)
        across = 1 - room.walls[o.wall].axis
        outward = -inward_sign(V, o.wall, across)
        connects = None
        if o.kind == "door":
            for depth in (0.3, 0.5, 0.8):
                q = mid.copy()
                q[across] += outward * depth
                r, c = frame.to_cell(q[None])
                if 0 <= r[0] < labels.shape[0] and 0 <= c[0] < labels.shape[1]:
                    lab = int(labels[r[0], c[0]])
                    if lab and lab != room.id:
                        connects = lab
                        break
            if connects is not None:
                adjacency_out.add(tuple(sorted((room.id, connects))))
        jamb = SIGMA_JAMB if o.jamb_points > 20 else SIGMA_JAMB_WEAK
        ops.append(
            OpeningOut(
                o.kind,
                o.wall,
                M(o.start, quad(sig[(o.wall - 1) % k], jamb)),
                M(o.width, quad(jamb, jamb, SCALE_REL * o.width)),
                M(o.top - o.bottom, quad(0.03, SCALE_REL * o.top)),
                M(o.bottom, 0.02) if o.kind == "window" else None,
                connects,
            )
        )
    return RoomOut(
        room.id,
        f"Room {room.id}",
        V.round(4).tolist(),
        walls,
        M(area, area_sigma, "m2"),
        M(room.perimeter, perim_sigma),
        ceiling,
        ops,
        notes,
    )


def run(
    capture_dir: str | Path, cache_dir: str | Path = "out/cache", drift: bool = True
) -> tuple[PlanOut, dict]:
    """Returns the plan and a bag of intermediates for rendering/debugging."""
    t0 = time.time()
    cap = Capture(capture_dir)
    drift_report = {"method": "off", "applied": False}
    if drift:
        from .drift import correct

        drift_report = correct(cap, cache_dir)
    P, N, _ = fused_cloud(cap, cache_dir)
    floor_y = g.find_floor(P, N)
    ceil_y = g.find_ceiling(P, N, floor_y)
    h = P[:, 1] - floor_y
    vertical = np.abs(N[:, 1]) < g.VERTICAL_NY
    horizontal = np.abs(N[:, 1]) > g.HORIZONTAL_NY
    angle = g.manhattan_angle(N, vertical)
    uv = g.PlanFrame(angle, np.zeros(2), RES, (1, 1)).to_plan(P)
    frame = g.PlanFrame.fit(uv[np.abs(h) < 0.05], angle, RES)

    wsel = vertical & (h > 0.2) & (h < 2.0)
    wall_raster = frame.rasterize(uv[wsel]) >= 3
    wall_n = frame.to_plan(np.c_[N[wsel, 0], np.zeros(wsel.sum()), N[wsel, 2]])
    traj = frame.to_plan(np.array([f.T_wc[:3, 3] for f in cap.frames]))

    cache = Path(cache_dir) / f"{cap.fingerprint}_{cap.cache_tag}_free_{int(RES * 1000)}mm.npy"
    if cache.exists() and np.load(cache).shape == frame.shape:
        free = np.load(cache)
    else:
        free = free_space(cap, frame, floor_y, ceil_y)
        np.save(cache, free)

    labels, unvisited = segment_rooms(free, wall_raster, uv[wsel], wall_n, traj, frame)

    lays = []
    for rid in range(1, labels.max() + 1):
        others = (labels > 0) & (labels != rid)
        lays.append(
            fit_room(rid, labels == rid, frame, P, N, uv, floor_y, vertical, horizontal, others)
        )
    n_overlaps = resolve_overlaps(lays, labels, frame)

    adjacency: set[tuple[int, int]] = set()
    rooms_out, layouts = [], []
    for lay in lays:
        ops = find_openings(lay, P, uv, free, frame)
        layouts.append((lay, ops))
        rooms_out.append(_room_out(lay, ops, labels, frame, adjacency))

    footprint = sum(r.floor_area.value for r in rooms_out)
    footprint_sigma = sum(r.floor_area.sigma for r in rooms_out)  # shared scale/bias is correlated
    path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    plan = PlanOut(
        capture=cap.root.name,
        tier="lidar",
        rooms=rooms_out,
        adjacency=[{"rooms": list(p), "via": "door"} for p in sorted(adjacency)],
        footprint_area=M(footprint, footprint_sigma, "m2"),
        frame={
            "plan_xy_from_world_xz_rotation_deg": round(math.degrees(angle), 3),
            "floor_world_y": round(floor_y, 4),
            "units": "m",
        },
        diagnostics={
            "frames": len(cap),
            "duration_s": round(cap.frames[-1].timestamp - cap.frames[0].timestamp, 1),
            "camera_path_m": round(path_len, 2),
            "loop_gap_m": round(float(np.linalg.norm(traj[-1] - traj[0])), 3),
            "rooms_seen_not_entered": int(len(np.unique(unvisited)) - 1),
            "drift_correction": drift_report,
            "overlaps_resolved": n_overlaps,
            "runtime_s": round(time.time() - t0, 1),
        },
    )
    plan.quality = {
        "status": "experimental",
        "metric_scale": "lidar_sensor",
        "interval_calibration": "unvalidated_without_independent_ground_truth",
        "warnings": [
            "Manhattan-world geometry assumes approximately perpendicular walls.",
            "Openings are geometric candidates, not validated detections.",
        ],
    }
    plan.provenance = {
        "input_fingerprint": cap.fingerprint,
        "capture_path": str(cap.root.resolve()),
        "input_modalities": ["rgb", "depth", "confidence", "poses", "intrinsics"],
    }
    debug = dict(
        capture=cap,
        points=P,
        normals=N,
        frame=frame,
        labels=labels,
        traj=traj,
        wall_uv=uv[wsel],
        layouts=layouts,
        free=free,
    )
    return plan, debug
