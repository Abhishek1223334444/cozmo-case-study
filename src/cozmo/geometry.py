"""Scene-level geometry from a fused, gravity-aligned point cloud.

Everything here works in a 2D "plan" frame: world XZ rotated so the dominant
wall direction lies along the plan axes (Manhattan alignment). Heights stay in
world Y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

HORIZONTAL_NY = 0.9
VERTICAL_NY = 0.15


def _peak_refine(values: np.ndarray, center: float, half_width: float) -> float:
    near = values[np.abs(values - center) < half_width]
    return float(np.median(near)) if len(near) else center


def find_floor(P: np.ndarray, N: np.ndarray, bin_m: float = 0.01) -> float:
    """World-Y of the floor: the lowest strongly supported upward-facing layer."""
    horiz = np.abs(N[:, 1]) > HORIZONTAL_NY
    y = P[horiz, 1]
    if len(y) < 20 or np.ptp(y) < bin_m:
        raise ValueError("Insufficient observed horizontal surfaces to estimate a floor")
    hist, edges = np.histogram(y, bins=np.arange(y.min(), y.max() + bin_m, bin_m))
    strong = np.flatnonzero(hist > 0.25 * hist.max())
    return _peak_refine(y, edges[strong[0]] + bin_m / 2, 0.03)


def find_ceiling(
    P: np.ndarray, N: np.ndarray, floor_y: float, min_height: float = 1.9, min_points: int = 2000
) -> float | None:
    """World-Y of the dominant downward-facing layer above head height, if seen."""
    horiz = (np.abs(N[:, 1]) > HORIZONTAL_NY) & (P[:, 1] > floor_y + min_height)
    y = P[horiz, 1]
    if len(y) < min_points:
        return None
    hist, edges = np.histogram(y, bins=np.arange(y.min(), y.max() + 0.01, 0.01))
    return _peak_refine(y, edges[np.argmax(hist)] + 0.005, 0.03)


def manhattan_angle(N: np.ndarray, vertical: np.ndarray) -> float:
    """Dominant wall orientation in radians, in [0, pi/2)."""
    ang = np.arctan2(N[vertical, 2], N[vertical, 0]) % (np.pi / 2)
    if len(ang) < 20:
        raise ValueError("Insufficient observed wall surfaces to establish plan orientation")
    hist, edges = np.histogram(ang, bins=360, range=(0, np.pi / 2))
    hist = ndi.gaussian_filter1d(hist.astype(float), 2, mode="wrap")
    a0 = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    d = (ang - a0 + np.pi / 4) % (np.pi / 2) - np.pi / 4
    d = d[np.abs(d) < np.radians(2)]
    return float((a0 + np.mean(d)) % (np.pi / 2))


@dataclass
class PlanFrame:
    """World XZ <-> aligned plan UV, plus a raster grid over the plan."""

    angle: float
    origin: np.ndarray
    res: float
    shape: tuple[int, int]

    def to_plan(self, P: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        x, z = P[:, 0], P[:, 2]
        return np.stack([c * x + s * z, -s * x + c * z], 1)

    def to_world_xz(self, uv: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        u, v = uv[:, 0], uv[:, 1]
        return np.stack([c * u - s * v, s * u + c * v], 1)

    def to_cell(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ij = np.floor((uv - self.origin) / self.res).astype(int)
        return ij[:, 1], ij[:, 0]

    def cell_center(self, rows, cols) -> np.ndarray:
        return self.origin + (np.stack([cols, rows], -1) + 0.5) * self.res

    def rasterize(self, uv: np.ndarray) -> np.ndarray:
        r, c = self.to_cell(uv)
        ok = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        g = np.zeros(self.shape, np.int32)
        np.add.at(g, (r[ok], c[ok]), 1)
        return g

    @classmethod
    def fit(cls, uv: np.ndarray, angle: float, res: float, margin: float = 0.5) -> PlanFrame:
        lo = np.percentile(uv, 0.1, 0) - margin
        hi = np.percentile(uv, 99.9, 0) + margin
        shape = tuple(np.ceil((hi - lo) / res).astype(int)[::-1])
        return cls(angle, lo, res, shape)


def voxel_downsample(P: np.ndarray, N: np.ndarray, voxel: float = 0.01):
    """Average points and normals per voxel. Returns (points, normals, counts)."""
    key = np.floor(P / voxel).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    flat = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    Ps = np.zeros((len(counts), 3))
    Ns = np.zeros((len(counts), 3))
    np.add.at(Ps, inv, P)
    np.add.at(Ns, inv, N)
    Ps /= counts[:, None]
    Ns /= np.linalg.norm(Ns, axis=1, keepdims=True) + 1e-12
    return Ps, Ns, counts
