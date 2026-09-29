"""Derive RGB-only development inputs from supplied captures, with provenance."""

import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from .capture import Capture
from .geometry import PlanFrame


def prepare(capture, plan_path, out, rotation=90):
    cap = Capture(capture)
    plan = json.loads(Path(plan_path).read_text())
    frame = PlanFrame(
        np.radians(plan["frame"]["plan_xy_from_world_xz_rotation_deg"]), np.zeros(2), 0.02, (1, 1)
    )
    trajectory = frame.to_plan(np.array([f.T_wc[:3, 3] for f in cap.frames]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoCapture(str(cap.root / "rgb.mp4"))
    records = []
    for room in plan["rooms"]:
        polygon = Polygon(room["polygon"])
        indices = [i for i, p in enumerate(trajectory) if polygon.covers(Point(p))]
        if len(indices) < 2:
            continue
        chosen = np.unique(
            np.array(indices)[np.linspace(0, len(indices) - 1, min(8, len(indices))).astype(int)]
        )
        folder = out / "photos" / f"room-{room['id']:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for i in chosen:
            video.set(cv2.CAP_PROP_POS_FRAMES, cap.frames[i].index)
            ok, image = video.read()
            if not ok:
                continue
            image = np.rot90(image, -(rotation // 90)).copy()
            p = folder / f"frame-{cap.frames[i].index:06d}.jpg"
            cv2.imwrite(str(p), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append(
                {
                    "file": str(p.relative_to(out)),
                    "source_frame": cap.frames[i].index,
                    "room": room["id"],
                }
            )
    video.release()
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(cap.root.resolve()),
                "derived_from_video": True,
                "room_grouping": "weak labels from LiDAR camera trajectory and inferred polygons",
                "limitation": "Development inputs, not independently captured photos or ground truth; no poses/depth provided to photo inference.",
                "rotation_clockwise_degrees": rotation,
                "photos": records,
            },
            indent=2,
        )
    )
    return out / "photos"
