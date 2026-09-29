"""Disk cache of the fused, voxel-downsampled cloud so iterations stay fast."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .capture import Capture
from .geometry import voxel_downsample


def fused_cloud(cap: Capture, cache_dir: str | Path = "out/cache", voxel: float = 0.01):
    path = Path(cache_dir) / f"{cap.root.name}_v{int(voxel * 1000)}mm.npz"
    if path.exists():
        z = np.load(path)
        return z["P"].astype(np.float64), z["N"].astype(np.float64), z["C"]
    P, N = cap.fused_points(frame_step=5, stride=2)
    P, N, C = voxel_downsample(P, N, voxel)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, P=P.astype(np.float32), N=N.astype(np.float32), C=C.astype(np.int32))
    return P, N, C
