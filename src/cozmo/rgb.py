"""RGB-only metric-depth + feature registration baseline.

No sensor depth, poses, IMU or calibration sidecars are opened by this module.
Disconnected components remain explicitly unplaced; a display offset is never
reported as a measured room relationship.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy import ndimage as ndi
from shapely.geometry import Polygon

from .models import MetricDepth
from .plan import M, PlanOut, RoomOut, WallOut
from . import geometry as g


@dataclass
class View:
    name: str
    room: str
    rgb: np.ndarray
    timestamp: float
    depth: np.ndarray | None = None
    K: np.ndarray | None = None
    pose: np.ndarray | None = None
    keypoints: np.ndarray | None = None
    descriptors: np.ndarray | None = None


def resize(rgb, maximum=640):
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (round(w * maximum / max(h, w)), round(h * maximum / max(h, w))))


def read_views(path, tier, max_frames=48, rotation=0):
    path = Path(path)
    views = []
    if tier == "video":
        if path.is_dir():
            candidates = sorted(path.rglob("rgb.mp4"))
            if len(candidates) != 1:
                raise ValueError("Video input must contain exactly one rgb.mp4, or be a video file")
            path = candidates[0]
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if total < 2:
            raise ValueError(f"Cannot decode video: {path}")
        for frame in np.unique(np.linspace(0, total - 1, min(max_frames, total)).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, bgr = cap.read()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = np.rot90(rgb, -(rotation // 90)).copy()
                views.append(View(f"frame-{frame:06d}", "walkthrough", resize(rgb), frame / fps))
        cap.release()
    else:
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in extensions)
        # Input is deliberately images only, never a Stray Scanner folder.
        if path.is_dir() and list(path.rglob("odometry.csv")):
            raise ValueError("Photo tier requires an isolated photo folder; use prepare-sample first")
        for i, p in enumerate(files):
            rgb = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
            rgb = np.rot90(rgb, -(rotation // 90)).copy()
            views.append(View(str(p), p.parent.name, resize(rgb), float(i)))
        counts = {r: sum(v.room == r for v in views) for r in {v.room for v in views}}
        if any(n > 8 for n in counts.values()):
            raise ValueError("Photo tier accepts at most 8 stills per room folder")
    if not views:
        raise ValueError("No decodable images found")
    return views


def camera_points(view, pixels):
    h, w = view.depth.shape
    xy = np.clip(np.rint(pixels).astype(int), [0, 0], [w - 1, h - 1])
    z = view.depth[xy[:, 1], xy[:, 0]]
    return np.c_[(pixels[:, 0] - view.K[0, 2]) / view.K[0, 0] * z,
                 (pixels[:, 1] - view.K[1, 2]) / view.K[1, 1] * z, z]


def match(a, b, correspondences=None):
    if correspondences is None:
        if a.descriptors is None or b.descriptors is None or len(b.descriptors) < 2:
            return None
        matches = cv2.BFMatcher().knnMatch(a.descriptors, b.descriptors, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < .78 * n.distance]
        pa = np.array([a.keypoints[m.queryIdx] for m in good], np.float32)
        pb = np.array([b.keypoints[m.trainIdx] for m in good], np.float32)
    else:
        pa,pb = correspondences
    if len(pa) < 12:
        return None
    xyz = camera_points(a, pa).astype(np.float32)
    cv2.setRNGSeed(17)
    ok, r, t, inliers = cv2.solvePnPRansac(xyz, pb, b.K, None, iterationsCount=300,
                                         reprojectionError=8., confidence=.999,
                                         flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 10 or len(inliers) < .2 * len(pa):
        return None
    ids = inliers[:, 0]
    # Require spatial support, not a tiny patch or repeated cabinet handle.
    if np.ptp(pb[ids], axis=0).prod() < b.rgb.shape[0] * b.rgb.shape[1] * .02:
        return None
    r, t = cv2.solvePnPRefineLM(xyz[ids], pb[ids], b.K, None, r, t)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(r)[0]; T[:3, 3] = t.ravel()
    error3d = np.linalg.norm(xyz[ids] @ T[:3, :3].T + T[:3, 3] - camera_points(b, pb[ids]), axis=1)
    residual = float(np.median(error3d))
    if residual > max(.6, float(np.median(xyz[ids, 2])) * .3) or np.linalg.norm(t) > 8:
        return None
    return T, len(ids), residual


def register(views, matcher_dir=None, cache_dir=Path("out/cache"), tier="photos"):
    sift = cv2.SIFT_create(nfeatures=2200)
    for v in views:
        k, v.descriptors = sift.detectAndCompute(cv2.cvtColor(v.rgb, cv2.COLOR_RGB2GRAY), None)
        v.keypoints = np.array([p.pt for p in k], np.float32)
    edges = []
    learned = None
    if matcher_dir is not None:
        from .matching import Matcher
        learned = Matcher(views, matcher_dir, cache_dir)
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            result = match(views[i], views[j])
            if learned and ((tier == "photos" and len(views) <= 32) or j-i <= 3 or result is not None):
                result = match(views[i], views[j], learned.pair(i,j)) or result
            if result:
                T, count, error = result
                edges.append((i, j, T, count, error))
        if learned and (i+1)%8==0:
            print(f"  registration {i+1}/{len(views)} views, {len(edges)} edges",flush=True)
    # Maximum-support spanning forest establishes each independent coordinate frame.
    remaining = set(range(len(views))); components = []
    while remaining:
        root = min(remaining); remaining.remove(root)
        component = [root]; views[root].pose = np.eye(4)
        while True:
            options = [e for e in edges if (e[0] in component and e[1] in remaining)
                       or (e[1] in component and e[0] in remaining)]
            if not options:
                break
            i, j, T, _, _ = max(options, key=lambda e: e[3] / (e[4] + .1))
            if i in component:
                views[j].pose = views[i].pose @ np.linalg.inv(T); nxt = j
            else:
                views[i].pose = views[j].pose @ T; nxt = i
            component.append(nxt); remaining.remove(nxt)
        components.append(component)
    reports = []
    for component in components:
        selected = [e for e in edges if e[0] in component and e[1] in component]
        if len(component) < 3 or len(selected) < len(component):
            reports.append({"views": len(component), "edges": len(selected), "loop_optimization": False})
            continue
        moving = component[1:]
        initial = np.concatenate([np.r_[Rotation.from_matrix(views[i].pose[:3, :3]).as_rotvec(), views[i].pose[:3, 3]] for i in moving])
        def matrices(x):
            poses = {component[0]: views[component[0]].pose}
            for i, row in zip(moving, x.reshape(-1, 6)):
                T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(row[:3]).as_matrix(); T[:3, 3] = row[3:]
                poses[i] = T
            return poses
        def residuals(x):
            poses = matrices(x); residual = []
            for i, j, target, count, error in selected:
                pred = np.linalg.inv(poses[j]) @ poses[i]
                rot = Rotation.from_matrix(target[:3, :3].T @ pred[:3, :3]).as_rotvec()
                delta = pred[:3, 3] - target[:3, 3]
                weight = np.sqrt(min(count, 100) / 30) / (1 + error)
                residual.extend(np.r_[rot * 3, delta] * weight)
            return residual
        before = float(np.sqrt(np.mean(np.square(residuals(initial)))))
        sparsity = lil_matrix((len(selected)*6,len(moving)*6),dtype=int)
        positions = {v:k for k,v in enumerate(moving)}
        for row,(i,j,*_) in enumerate(selected):
            for node in (i,j):
                if node in positions:
                    col = positions[node]*6; sparsity[row*6:(row+1)*6,col:col+6]=1
        fit = least_squares(residuals, initial, loss="soft_l1", f_scale=.1, max_nfev=35,jac_sparsity=sparsity.tocsr())
        for i, pose in matrices(fit.x).items():
            views[i].pose = pose
        reports.append({"views": len(component), "edges": len(selected), "loop_optimization": True,
                        "constraint_rms_before": before, "constraint_rms_after": float(np.sqrt(np.mean(np.square(residuals(fit.x))))),
                        "optimizer_converged": bool(fit.success)})
    return components, edges, reports


def cloud(view, step=4):
    h, w = view.depth.shape
    yy, xx = np.mgrid[1:h-1:step, 1:w-1:step]
    xy = np.c_[xx.ravel(), yy.ravel()].astype(float)
    pts = camera_points(view, xy)
    dx = camera_points(view, xy + [1, 0]) - camera_points(view, xy - [1, 0])
    dy = camera_points(view, xy + [0, 1]) - camera_points(view, xy - [0, 1])
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=1)
    normals /= norm[:, None] + 1e-10
    normals[(normals * pts).sum(1) > 0] *= -1
    ok = (norm > 1e-7) & (pts[:, 2] > .15) & (pts[:, 2] < 10)
    return pts[ok] @ view.pose[:3, :3].T + view.pose[:3, 3], normals[ok] @ view.pose[:3, :3].T


def level(component, views):
    pts, normals = cloud(views[component[0]])
    # First image is assumed upright. Infer a nearby vertical direction from normals.
    up = np.array([0., -1., 0.])
    near = normals[normals @ up > .88]
    if len(near) > 100:
        up = np.median(near, axis=0); up /= np.linalg.norm(up)
    rotation, _ = Rotation.align_vectors([[0., 1., 0.]], [up])
    R = rotation.as_matrix()
    for i in component:
        views[i].pose[:3, :3] = R @ views[i].pose[:3, :3]
        views[i].pose[:3, 3] = R @ views[i].pose[:3, 3]


def room_from_points(points, normals, rid, name, tier):
    """Robust oriented envelope, explicitly an inferred layout, with broad priors."""
    vertical = abs(normals[:, 1]) < .3
    angle = g.manhattan_angle(normals, vertical) if vertical.sum() > 50 else 0.
    frame = g.PlanFrame(angle, np.zeros(2), .04, (1, 1))
    uv = frame.to_plan(points)
    horizontal = abs(normals[:, 1]) > .8
    floor_points = points[horizontal & (normals[:, 1] > 0), 1]
    floor = float(np.percentile(floor_points, 15)) if len(floor_points) > 100 else float(np.percentile(points[:, 1], 3))
    sel = vertical & (points[:, 1] > floor + .25) & (points[:, 1] < floor + 2.2)
    sample = uv[sel] if sel.sum() > 100 else uv
    lo, hi = np.percentile(sample, [2, 98], axis=0)
    if np.any(hi - lo < .3):
        lo, hi = np.percentile(uv, [1, 99], axis=0)
    if np.any(hi - lo < .01):
        raise ValueError("Images show too little geometric extent to construct a plan")
    rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    polygon = frame.to_world_xz(rect)
    relative = .18 if tier == "photos" else .12
    walls = []
    for k in range(4):
        a, b = polygon[k], polygon[(k + 1) % 4]; length = float(np.linalg.norm(b - a))
        walls.append(WallOut(k, a.tolist(), b.tolist(), M(length, max(.12, length * relative)), False))
    area = float(np.prod(hi - lo)); perim = sum(w.length.value for w in walls)
    ceilings = points[horizontal & (normals[:, 1] < 0) & (points[:, 1] > floor + 1.8), 1]
    height = float(np.median(ceilings) - floor) if len(ceilings) > 200 else None
    return RoomOut(rid, name, polygon.tolist(), walls, M(area, area * relative * 2, "m2"),
                   M(perim, perim * relative), M(height, max(.2, height * relative)) if height else None, [],
                   ["Envelope inferred from learned depth; furniture and unseen walls can bias extent."] +
                   (["Only a partial surface patch was reconstructed; this is not a complete room."] if min(hi-lo)<.5 else []))


def run(path, tier, cache_dir=Path("out/cache"), model_dir=Path("models/depth"), max_frames=48, rotation=0, device="auto"):
    started = time.time(); views = read_views(path, tier, max_frames, rotation)
    depth_model = MetricDepth(model_dir, device)
    for i, v in enumerate(views):
        print(f"  depth {i+1}/{len(views)}: {v.name}", flush=True)
        v.depth = depth_model.predict(v.rgb, cache_dir)
        h, w = v.rgb.shape[:2]; f = .85 * max(w, h)
        v.K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1.]])
    components, edges, graph_report = register(views, Path(model_dir).parent/"matcher",cache_dir,tier)
    rooms, cloud_parts, room_components, observations = [], [], {}, []
    display_offset = 0.
    for ci, component in enumerate(components):
        level(component, views)
        if len(components)>1:
            all_points = np.concatenate([cloud(views[i])[0] for i in component])
            shift = display_offset-float(np.min(all_points[:,0]))
            for i in component:
                views[i].pose[0,3] += shift
            display_offset = float(np.max(all_points[:,0]))+shift+1.5
        # Keep each photo folder a room; a video component is an observed envelope.
        groups = {}
        for i in component:
            groups.setdefault(views[i].room, []).append(i)
        component_points = []
        for name, ids in groups.items():
            clouds = [cloud(views[i]) for i in ids]
            P = np.concatenate([p for p, _ in clouds]); N = np.concatenate([n for _, n in clouds])
            room = room_from_points(P, N, len(rooms)+1, name if tier == "photos" else f"Observed region {len(rooms)+1}", tier)
            if len(components) > 1:
                room.notes.append("Display placement only: disconnected from other components.")
            rooms.append(room); room_components[room.id] = ci
            component_points.append(P); observations.extend((i, room.id) for i in ids)
        cloud_parts.extend(component_points)
    area = sum(r.floor_area.value for r in rooms)
    overlaps = []
    for i, a in enumerate(rooms):
        for b in rooms[i+1:]:
            if room_components[a.id] != room_components[b.id]:
                continue
            intersection = Polygon(a.polygon).intersection(Polygon(b.polygon)).area
            if intersection > .1:
                overlaps.append({"rooms": [a.id, b.id], "area_m2": round(intersection, 3)})
    warnings = ["Learned metric scale and assumed focal length are not independently calibrated.",
                "RGB layout is an experimental room-envelope baseline; openings are not inferred."]
    if len(components) > 1:
        warnings.append("Visual overlap did not connect all inputs; component offsets are for display only.")
    if overlaps:
        warnings.append("Inferred room envelopes overlap; a valid whole-property stitch is not established.")
    plan = PlanOut(Path(path).stem, tier, rooms, [], M(area, sum(r.floor_area.sigma for r in rooms), "m2"),
                   {"units": "m", "vertical": "estimated from upright RGB", "component_ids": room_components},
                   diagnostics={"frames": len(views), "registered_edges": len(edges), "components": len(components),
                                "pose_graph": graph_report, "overlaps": overlaps, "runtime_s": time.time()-started},
                   quality={"status": "experimental", "metric_scale": "learned_prior", "warnings": warnings,
                            "stitch_status": "unresolved" if len(components)>1 or overlaps or tier=="video" else "registered_rooms_adjacency_unverified",
                            "interval_calibration": "unvalidated_without_independent_ground_truth"},
                   provenance={"model": depth_model.source, "input_modalities": ["rgb"],
                               "sensor_sidecars_used": False, "rotation_clockwise_degrees": rotation,
                               "input_fingerprint": hashlib.sha256(b"".join(v.rgb.tobytes() for v in views)).hexdigest()})
    return plan, {"wall_uv": np.concatenate(cloud_parts)[:, [0, 2]], "views": views, "observations": observations}
