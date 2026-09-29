"""Text-conditioned visible-damage candidates and traceable inspection rules.

CLIPSeg scores are not probabilities. Regions require human confirmation.
Concealed damage can only be flagged for inspection, not observed behind walls.
"""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from .plan import M

PROMPTS = ["a water stain on a wall", "a crack in a wall", "a clean undamaged wall"]
CLASSES = ["water_stain", "crack"]


def surfaces(plan):
    result = []
    for r in plan.rooms:
        for w in r.walls:
            area = None if r.ceiling_height is None else M(w.length.value * r.ceiling_height.value,
                    np.hypot(w.length.sigma * r.ceiling_height.value, r.ceiling_height.sigma * w.length.value), "m2")
            result.append({"id": f"room-{r.id}/wall-{w.index}", "room_id": r.id, "kind": "wall",
                           "wall_index": w.index, "gross_area": area})
        result.append({"id": f"room-{r.id}/floor", "room_id": r.id, "kind": "floor", "gross_area": r.floor_area})
        result.append({"id": f"room-{r.id}/ceiling", "room_id": r.id, "kind": "ceiling",
                       "gross_area": r.floor_area if r.ceiling_height else None})
    plan.surfaces = result


def assess(plan, debug, out, model_dir=Path("models/damage"), max_views=12, device="auto"):
    import torch
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    from .models import device_name
    from .rgb import camera_points
    device = device_name() if device == "auto" else device
    processor = CLIPSegProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    source = json.loads((Path(model_dir) / "source.json").read_text())
    evidence = Path(out) / "evidence"; evidence.mkdir(parents=True, exist_ok=True)
    observations = []
    if plan.tier == "lidar":
        cap = debug["capture"]; video = cv2.VideoCapture(str(cap.root / "rgb.mp4"))
        for i in np.unique(np.linspace(0, len(cap)-1, min(max_views, len(cap))).astype(int)):
            video.set(cv2.CAP_PROP_POS_FRAMES, cap.frames[i].index)
            ok, bgr = video.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Stray RGB is landscape sensor orientation; infer in upright orientation,
            # then rotate the mask back before using sensor intrinsics.
            observations.append((str(cap.frames[i].index), rgb, cap.depth(i), cap.K_depth(i), cap.frames[i].T_wc,
                                 cap.confidence(i), 90))
        video.release()
    else:
        for i in np.unique(np.linspace(0, len(debug["views"])-1, min(max_views, len(debug["views"]))).astype(int)):
            v = debug["views"][i]
            observations.append((str(i), v.rgb, v.depth, v.K, v.pose, np.full(v.depth.shape, 2), 0))
    regions = []
    for name, rgb, depth, K, pose, conf, rotation in observations:
        upright = np.rot90(rgb, -(rotation // 90)).copy()
        inputs = processor(text=PROMPTS, images=[Image.fromarray(upright)]*3, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            scores = model(**inputs).logits.sigmoid().cpu().numpy()
        scores = [cv2.resize(s, (upright.shape[1], upright.shape[0])) for s in scores]
        for klass, score in zip(CLASSES, scores[:2]):
            mask = ((score > .55) & (score > scores[2] + .15)).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for lab in range(1, n):
                if stats[lab, cv2.CC_STAT_AREA] < .001 * mask.size:
                    continue
                component = (labels == lab).astype(np.uint8)
                raw = np.rot90(component, rotation // 90).copy()
                small = cv2.resize(raw, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                yy, xx = np.nonzero((small > 0) & (depth > .1) & (conf >= 1))
                if len(xx) < 12:
                    continue
                z = depth[yy, xx]
                p = np.c_[(xx-K[0,2])/K[0,0]*z, (yy-K[1,2])/K[1,1]*z, z] @ pose[:3,:3].T + pose[:3,3]
                if plan.tier == "lidar":
                    uv = debug["frame"].to_plan(p); height = p[:, 1] - plan.frame["floor_world_y"]
                else:
                    uv = p[:, [0,2]]; height = None
                best = None
                for room in plan.rooms:
                    if plan.tier != "lidar":
                        floor = debug.get("room_floor_y",{}).get(room.id)
                        if floor is None:
                            continue
                        height = p[:,1]-floor
                    for wall in room.walls:
                        a, b = np.array(wall.start), np.array(wall.end); direction = (b-a) / wall.length.value
                        along = (uv-a) @ direction
                        distance = abs((uv-a) @ np.array([-direction[1], direction[0]]))
                        inside = (distance < (.15 if plan.tier == "lidar" else .4)) & (along > 0) & (along < wall.length.value) & (height > 0)
                        if inside.sum() < 12:
                            continue
                        candidate = (int(inside.sum()), room, wall, along[inside], height[inside])
                        if best is None or candidate[0] > best[0]:
                            best = candidate
                if best is None:
                    continue
                _, room, wall, along, height = best
                poly = MultiPoint(np.c_[along, height]).convex_hull
                if poly.geom_type != "Polygon" or poly.area < .002:
                    continue
                filename = f"{name}-{klass}-{lab}.jpg"
                overlay = upright.copy(); overlay[component > 0] = (overlay[component > 0]*.4 + np.array([239,94,60])*.6).astype(np.uint8)
                Image.fromarray(overlay).resize((max(1,round(640*upright.shape[1]/upright.shape[0])),640)).save(evidence/filename)
                regions.append({"surface_id": f"room-{room.id}/wall-{wall.index}", "class": klass,
                                "polygon": poly, "score": float(score[component>0].mean()), "evidence": f"evidence/{filename}"})
    # Merge overlapping same-class observations on the same surface to avoid double counting.
    for surface_id, klass in sorted({(r["surface_id"], r["class"]) for r in regions}):
        rs = [r for r in regions if r["surface_id"] == surface_id and r["class"] == klass]
        union = unary_union([r["polygon"] for r in rs])
        polygons = list(union.geoms) if hasattr(union, "geoms") else [union]
        for poly in polygons:
            ident = f"damage-{len(plan.damage)+1}"
            plan.damage.append({"id": ident, "surface_id": surface_id, "class": klass,
                                "status": "candidate_requires_confirmation", "score": max(r["score"] for r in rs),
                                "score_is_probability": False, "polygon_surface_m": list(map(list, poly.exterior.coords)),
                                "extent_area": M(poly.area, max(.01, poly.area*.3), "m2"),
                                "extent_method": "projected convex hull; upper region extent, not exact damaged area",
                                "evidence": sorted({r["evidence"] for r in rs})})
            plan.scope.append({"id": f"scope-{len(plan.scope)+1}", "surface_id": surface_id,
                               "damage_id": ident, "action": "inspect_and_confirm_before_repair",
                               "quantity": M(poly.area, max(.01,poly.area*.3), "m2"), "status": "provisional"})
            if klass == "water_stain":
                plan.concealed_damage.append({"surface_id": surface_id, "damage_id": ident,
                                             "rule": "visible_water_stain_candidate_v1",
                                             "flag": "inspect_for_possible_concealed_moisture",
                                             "diagnosis": False})
    plan.diagnostics["damage_assessment"] = {"status": "completed_experimental", "views_checked": len(observations),
                                             "model": source, "candidate_count": len(plan.damage),
                                             "threshold": .55, "negative_prompt_margin": .15,
                                             "limitation": "No labelled damage benchmark; absence of detections does not establish absence of damage."}
