"""Room segmentation on a 2D plan raster.

Pipeline: free-space map from per-frame view fans -> morphological sweep for
room seeds -> geodesic growth -> merge regions whose shared boundary is not a
doorway in a wall line -> keep rooms the camera actually walked through.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .capture import Capture
from .geometry import PlanFrame

FAN_BINS = 180  # 2 degree azimuth bins per frame


def free_space(
    cap: Capture, frame: PlanFrame, floor_y: float, ceil_y: float | None, frame_step: int = 5
) -> np.ndarray:
    """Count, per plan cell, how many frames saw through it.

    For each frame, the farthest hit per azimuth bin between shin and ceiling
    height bounds a fan of free space (rays pass over furniture to the wall).
    """
    acc = np.zeros(frame.shape, np.uint16)
    top = (ceil_y - floor_y - 0.15) if ceil_y is not None else 2.2
    for i in range(0, len(cap), frame_step):
        P, _ = cap.points_world(i, min_conf=1, max_depth=5.0, stride=2)
        h = P[:, 1] - floor_y
        P = P[(h > 0.15) & (h < top)]
        if len(P) < 50:
            continue
        cam = frame.to_plan(cap.frames[i].T_wc[None, :3, 3])[0]
        d = frame.to_plan(P) - cam
        b = ((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * FAN_BINS).astype(int) % FAN_BINS
        far = np.zeros(FAN_BINS)
        np.maximum.at(far, b, np.hypot(d[:, 0], d[:, 1]))
        occ = np.flatnonzero(far > 0)
        a = (occ + 0.5) / FAN_BINS * 2 * np.pi - np.pi
        tips = (cam + np.stack([np.cos(a), np.sin(a)], 1) * far[occ, None] - frame.origin) / frame.res
        apex = (cam - frame.origin) / frame.res
        mask = np.zeros(frame.shape, np.uint8)
        for j in np.flatnonzero(np.diff(occ) == 1):
            tri = np.array([apex, tips[j], tips[j + 1]]) * 16
            cv2.fillConvexPoly(mask, tri.astype(np.int32), 1, shift=4)
        acc += mask
    return acc


def _grow(labels: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    lab = labels.copy()
    while True:
        d = ndi.grey_dilation(lab, size=(3, 3))
        new = (lab == 0) & allowed & (d > 0)
        if not new.any():
            return lab
        lab[new] = d[new]


def _sweep_seeds(interior: np.ndarray, res: float, min_seed_m2: float = 0.3) -> np.ndarray:
    """Erode progressively; a seed is replaced by several only on a genuine split."""
    dist = ndi.distance_transform_edt(interior) * res
    seeds, _ = ndi.label(dist > 0.10)
    for t in np.arange(0.14, 1.5, 0.04):
        comps, nc = ndi.label(dist > t)
        if nc == 0:
            break
        areas = ndi.sum(np.ones_like(dist), comps, range(1, nc + 1)) * res * res
        nxt = seeds.max() + 1
        for sid in np.unique(seeds)[1:]:
            region = seeds == sid
            big = [c for c in np.unique(comps[region & (comps > 0)]) if areas[c - 1] >= min_seed_m2]
            if len(big) >= 2:
                seeds[region] = 0
                for c in big:
                    seeds[comps == c] = nxt
                    nxt += 1
    return seeds


def _boundary_is_door(ws, a, b, wall_uv, wall_n, frame: PlanFrame) -> bool:
    """A doorway sits in a wall line: wall faces parallel to the cut continue past
    both jambs, or past one jamb when the opening is door-sized (door by a corner).
    A cut through a hallway has neither."""
    ma, mb = ws == a, ws == b
    k = np.ones((3, 3))
    cut = (ndi.binary_dilation(ma, k) & mb) | (ndi.binary_dilation(mb, k) & ma)
    pts = frame.cell_center(*np.nonzero(cut))
    ext = np.ptp(pts, 0)
    ax = int(np.argmax(ext))
    lo, hi, mid = pts[:, ax].min(), pts[:, ax].max(), np.median(pts[:, 1 - ax])
    parallel = np.abs(wall_n[:, 1 - ax]) > 0.8
    spans = []
    for sgn, e in ((-1, lo), (1, hi)):
        along = (wall_uv[:, ax] - e) * sgn
        m = parallel & (along > 0.02) & (along < 0.40) & (np.abs(wall_uv[:, 1 - ax] - mid) < 0.25)
        spans.append(np.ptp(wall_uv[m, ax]) if m.sum() > 20 else 0.0)
    return min(spans) > 0.08 or (max(spans) > 0.08 and ext[ax] < 1.3)


def _merge_non_doors(ws, wall_uv, wall_n, frame: PlanFrame) -> np.ndarray:
    changed = True
    while changed:
        changed = False
        for a in np.unique(ws)[1:]:
            ma = ws == a
            ring = ndi.binary_dilation(ma, np.ones((3, 3))) & ~ma
            nb, cnt = np.unique(ws[ring], return_counts=True)
            for b, n in zip(nb, cnt):
                if b > a and n >= 5 and not _boundary_is_door(ws, a, b, wall_uv, wall_n, frame):
                    ws[ws == b] = a
                    changed = True
                    break
            if changed:
                break
    return ws


def segment_rooms(
    free: np.ndarray,
    wall: np.ndarray,
    wall_uv: np.ndarray,
    wall_n: np.ndarray,
    traj_uv: np.ndarray,
    frame: PlanFrame,
    min_room_m2: float = 1.0,
    min_visit_frames: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Label raster of rooms (0 = not a room). Also returns labels of rooms that
    were seen (e.g. through a doorway) but never entered, for reporting."""
    res = frame.res
    interior = ndi.binary_opening((free >= 2) & ~wall, np.ones((5, 5), bool))
    ws = _grow(_sweep_seeds(interior, res), interior)
    ids, cnt = np.unique(ws[ws > 0], return_counts=True)
    ws[np.isin(ws, ids[cnt * res * res < min_room_m2])] = 0
    ws = _grow(ws, interior)
    ws = _merge_non_doors(ws, wall_uv, wall_n, frame)

    r, c = frame.to_cell(traj_uv)
    ok = (r >= 0) & (r < ws.shape[0]) & (c >= 0) & (c < ws.shape[1])
    ids, cnt = np.unique(ws[r[ok], c[ok]], return_counts=True)
    visited = ids[(ids > 0) & (cnt >= min_visit_frames)]
    unvisited = np.where(np.isin(ws, visited), 0, ws)
    ws = np.where(np.isin(ws, visited), ws, 0)
    # relabel 1..n in a stable order (by first visit)
    order = [v for v in dict.fromkeys(ws[r[ok], c[ok]]) if v > 0]
    out = np.zeros_like(ws)
    for k, v in enumerate(order, 1):
        out[ws == v] = k
    return out, unvisited
