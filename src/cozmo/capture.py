"""Load a Stray Scanner capture (rgb.mp4, depth/, confidence/, odometry.csv, imu.csv).

Conventions
-----------
World frame is ARKit's: gravity-aligned, +Y up, metres.
Camera poses are camera-to-world (T_WC). Stray Scanner writes them in the OpenCV
camera convention (+X right, +Y down, looks down +Z) -- verified empirically: with
that convention the floor collapses to a 2 cm-thick layer, with ARKit's it smears.
Depth PNGs are 16-bit millimetres at 256x192; intrinsics in odometry.csv are
for the 1920x1440 RGB stream and are rescaled to depth resolution here.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation

RGB_W, RGB_H = 1920, 1440


@dataclass
class Frame:
    index: int
    timestamp: float
    T_wc: np.ndarray  # 4x4 camera-to-world, OpenCV camera convention
    K_rgb: np.ndarray  # 3x3 intrinsics at RGB resolution


class Capture:
    def __init__(self, root: str | Path):
        root = Path(root)
        # Accept either the capture folder itself or a parent holding exactly one capture.
        if not (root / "odometry.csv").exists():
            subs = [p for p in root.iterdir() if (p / "odometry.csv").exists()]
            if len(subs) != 1:
                raise FileNotFoundError(f"no single Stray Scanner capture under {root}")
            root = subs[0]
        self.root = root
        video = cv2.VideoCapture(str(root / "rgb.mp4"))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rgb_w = width if width > 0 else RGB_W
        self.rgb_h = height if height > 0 else RGB_H
        video.release()
        self.frames = self._load_odometry()
        if not self.frames:
            raise ValueError("Capture contains no camera poses")
        first = self.depth(0)
        self.depth_h, self.depth_w = first.shape
        self.cache_tag = "raw"
        h = hashlib.sha256((root / "odometry.csv").read_bytes())
        for folder in ("depth", "confidence"):
            for p in sorted((root / folder).glob("*.png")):
                h.update(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}".encode())
        self.fingerprint = h.hexdigest()[:20]

    def _load_odometry(self) -> list[Frame]:
        frames = []
        with open(self.root / "odometry.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                row = [c.strip() for c in row]
                ts, idx = float(row[0]), int(row[1])
                x, y, z, qx, qy, qz, qw = map(float, row[2:9])
                if len(row) >= 13 and all(row[9:13]):
                    fx, fy, cx, cy = map(float, row[9:13])
                else:
                    k = np.loadtxt(self.root / "camera_matrix.csv", delimiter=",")
                    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T[:3, 3] = [x, y, z]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
                frames.append(Frame(idx, ts, T, K))
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres, 0 where invalid."""
        d = np.asarray(Image.open(self.root / "depth" / f"{self.frames[i].index:06d}.png"), dtype=np.float32)
        return d / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        """ARKit confidence: 0 low, 1 medium, 2 high."""
        return np.asarray(Image.open(self.root / "confidence" / f"{self.frames[i].index:06d}.png"))

    def K_depth(self, i: int) -> np.ndarray:
        K = self.frames[i].K_rgb.copy()
        K[0] *= self.depth_w / self.rgb_w
        K[1] *= self.depth_h / self.rgb_h
        return K

    def points_world(
        self, i: int, min_conf: int = 2, max_depth: float = 4.0, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Back-project frame i's depth map to world-frame points and unit normals (N, 3).

        Normals come from central differences on the depth grid and are oriented
        towards the camera. Pixels on depth discontinuities are dropped.
        """
        d = self.depth(i)
        c = self.confidence(i)
        K = self.K_depth(i)
        v, u = np.mgrid[0 : self.depth_h, 0 : self.depth_w]
        x = (u - K[0, 2]) / K[0, 0] * d
        y = (v - K[1, 2]) / K[1, 1] * d
        P = np.stack([x, y, d], -1)

        du = P[1:-1, 2:] - P[1:-1, :-2]
        dv = P[2:, 1:-1] - P[:-2, 1:-1]
        n = np.cross(du, dv)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        Pc = P[1:-1, 1:-1]
        n[(n * Pc).sum(-1) > 0] *= -1  # face the camera

        dc = d[1:-1, 1:-1]
        neigh = np.stack([d[1:-1, 2:], d[1:-1, :-2], d[2:, 1:-1], d[:-2, 1:-1]], -1)
        smooth = np.abs(neigh - dc[..., None]).max(-1) < 0.05 * dc + 0.02
        ok = (dc > 0.1) & (dc < max_depth) & (c[1:-1, 1:-1] >= min_conf) & smooth
        ok &= (neigh > 0).all(-1)
        if stride > 1:
            mask = np.zeros_like(ok)
            mask[::stride, ::stride] = True
            ok &= mask

        R, t = self.frames[i].T_wc[:3, :3], self.frames[i].T_wc[:3, 3]
        return Pc[ok] @ R.T + t, n[ok] @ R.T

    def fused_points(
        self, frame_step: int = 5, min_conf: int = 2, max_depth: float = 4.0, stride: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse every `frame_step`-th frame into one world-frame cloud with normals."""
        pts, nrm = zip(
            *(
                self.points_world(i, min_conf, max_depth, stride)
                for i in range(0, len(self), frame_step)
            )
        )
        return np.concatenate(pts), np.concatenate(nrm)
