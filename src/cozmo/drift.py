"""Plane-anchored translation correction with temporal regularization.

Reduces internal inconsistency; it cannot establish metric accuracy or fix rotation.
Accept a correction only if it improves held-out plane residuals.
"""
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.linalg import solve_banded
from . import geometry as g
from .cache import fused_cloud


def correct(cap, cache_dir):
    P, N, _ = fused_cloud(cap, cache_dir)
    angle = g.manhattan_angle(N, np.abs(N[:, 1]) < g.VERTICAL_NY)
    c, s = np.cos(angle), np.sin(angle)
    A = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Q, U = P @ A.T, N @ A.T
    planes = []
    for axis in range(3):
        vals = Q[np.abs(U[:, axis]) > 0.94, axis]
        if len(vals) < 100:
            planes.append(np.array([])); continue
        hist, edges = np.histogram(vals, np.arange(vals.min() - .01, vals.max() + .02, .01))
        smoothed = gaussian_filter1d(hist.astype(float), 1)
        peaks, _ = find_peaks(smoothed, distance=12, prominence=max(25, smoothed.max() * .045))
        planes.append(edges[peaks] + .005)
    indices = np.arange(0, len(cap), max(10, len(cap) // 160))
    offsets, weights, held = [], [], []
    for i in indices:
        pts, normals = cap.points_world(int(i), stride=3)
        pts, normals = pts @ A.T, normals @ A.T
        delta, weight, validation = [], [], []
        for axis, anchors in enumerate(planes):
            vals = pts[np.abs(normals[:, axis]) > .94, axis]
            if not len(anchors) or len(vals) < 40:
                delta.append(0.); weight.append(0.); validation.append(np.array([])); continue
            residual = anchors[abs(vals[:, None] - anchors).argmin(1)] - vals
            residual = residual[abs(residual) < .09]
            train, test = residual[::2], residual[1::2]
            delta.append(float(np.median(train)) if len(train) >= 20 else 0.)
            weight.append(min(1., len(train) / 200) if len(train) >= 20 else 0.)
            validation.append(test)
        offsets.append(delta); weights.append(weight); held.append(validation)
    offsets, weights = np.array(offsets), np.array(weights)
    shifts = np.zeros_like(offsets)
    for axis in range(3):
        w = weights[:, axis]; n = len(w)
        ab = np.zeros((3, n)); ab[1] = w + .15 + 4
        ab[1, 0] -= 2; ab[1, -1] -= 2
        ab[0, 1:] = -2; ab[2, :-1] = -2
        shifts[:, axis] = solve_banded((1, 1), ab, w * offsets[:, axis])
    shifts = np.clip(shifts, -.08, .08)
    before, after = [], []
    for k, validation in enumerate(held):
        for axis, residual in enumerate(validation):
            before.extend(abs(residual)); after.extend(abs(residual - shifts[k, axis]))
    b = float(np.mean(before)) if before else None
    a = float(np.mean(after)) if after else None
    accepted = bool(b is not None and a < b * .995)
    corrections = np.zeros((len(cap), 3))
    if accepted:
        interp = np.stack([np.interp(np.arange(len(cap)), indices, shifts[:, j]) for j in range(3)], 1)
        corrections = interp @ A
        for f, shift in zip(cap.frames, corrections):
            f.T_wc[:3, 3] += shift
        cap.cache_tag = "planes-v1-" + hashlib.sha256(corrections.tobytes()).hexdigest()[:12]
    return {"method": "plane_anchored_translation", "applied": accepted,
            "keyframes": len(indices), "planes_per_axis": [len(p) for p in planes],
            "heldout_plane_residual_before_m": b, "heldout_plane_residual_after_m": a,
            "max_translation_m": float(np.linalg.norm(corrections, axis=1).max()),
            "limitation": "Internal consistency only; no independent accuracy claim. Rotation drift is not corrected."}
