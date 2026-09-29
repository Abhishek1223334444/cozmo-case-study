"""Per-room metric layout: rectilinear outline snapped to wall surfaces, heights."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import PlanFrame

MIN_EDGE = 0.25  # m; shorter jogs in the raster outline are treated as noise
SIMPLIFY = 0.10  # m; polygon simplification tolerance on the raster outline
COLLAPSE = 0.04  # m; after snapping, steps shorter than this are merged away


@dataclass
class Wall:
    axis: int  # 0: wall runs along plan-u (constant v); 1: along plan-v (constant u)
    coord: float  # the constant coordinate, m
    support: float = 0.0  # fraction of the wall's length backed by measured surface
    spread: float = 0.0  # robust std of surface points about `coord`, m
    n_points: int = 0
    snapped: bool = False


@dataclass
class RoomLayout:
    id: int
    walls: list[Wall]
    vertices: np.ndarray  # (k, 2) plan-UV, counter-clockwise
    floor_y: float | None = None
    ceiling_y: float | None = None
    floor_spread: float = 0.0
    ceiling_spread: float = 0.0
    n_ceiling: int = 0
    n_floor: int = 0
    extras: dict = field(default_factory=dict)

    @property
    def wall_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.roll(self.vertices, -1, 0) - self.vertices, axis=1)

    @property
    def area(self) -> float:
        x, y = self.vertices.T
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def perimeter(self) -> float:
        return float(self.wall_lengths.sum())

    @property
    def ceiling_height(self) -> float | None:
        if self.ceiling_y is None or self.floor_y is None:
            return None
        return self.ceiling_y - self.floor_y


def _mask_contour(mask: np.ndarray, frame: PlanFrame) -> np.ndarray:
    k = int(round(0.25 / frame.res)) | 1
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, SIMPLIFY / frame.res, True)[:, 0, :]  # (x=col, y=row)
    return frame.cell_center(c[:, 1], c[:, 0])


def _rectilinear(poly: np.ndarray) -> list[Wall]:
    """Snap a simple polygon to alternating axis-aligned walls."""
    edges = []
    for p, q in zip(poly, np.roll(poly, -1, 0)):
        d = q - p
        axis = 0 if abs(d[0]) >= abs(d[1]) else 1
        coord = (p[1] + q[1]) / 2 if axis == 0 else (p[0] + q[0]) / 2
        edges.append([axis, coord, float(np.hypot(*d))])

    def merge_runs(es):
        out = []
        for e in es:
            if out and out[-1][0] == e[0]:
                a = out[-1]
                w = a[2] + e[2]
                out[-1] = [a[0], (a[1] * a[2] + e[1] * e[2]) / max(w, 1e-9), w]
            else:
                out.append(e)
        if len(out) > 1 and out[0][0] == out[-1][0]:
            a, e = out.pop(), out[0]
            w = a[2] + e[2]
            out[0] = [a[0], (a[1] * a[2] + e[1] * e[2]) / max(w, 1e-9), w]
        return out

    edges = _prune(merge_runs(edges), MIN_EDGE, merge_runs)
    walls = [Wall(int(a), float(c)) for a, c, _ in edges]
    return _ccw(walls)


def _ccw(walls: list[Wall]) -> list[Wall]:
    x, y = _vertices(walls).T
    signed = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return walls if signed > 0 else walls[::-1]


def inward_sign(vertices: np.ndarray, i: int, across: int) -> float:
    """+1/-1: direction along plan axis `across` pointing into the room from wall i
    (interior is left of each edge of a counter-clockwise outline)."""
    d = vertices[(i + 1) % len(vertices)] - vertices[i]
    left = np.array([-d[1], d[0]])
    return float(np.sign(left[across]))


def _prune(edges, min_len, merge_runs):
    while len(edges) > 4:
        lens = _lengths_from_walls(edges)
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            break
        edges.pop(i)
        edges = merge_runs(edges)
    return edges


def _collapse(walls: list[Wall]) -> list[Wall]:
    """After snapping, drop near-zero steps; when two same-axis walls become
    neighbours keep the better-measured one's position."""

    def merge_runs(ws):
        out = []
        for w in ws:
            if out and out[-1].axis == w.axis:
                out[-1] = max(out[-1], w, key=lambda x: (x.snapped, x.n_points))
            else:
                out.append(w)
        if len(out) > 1 and out[0].axis == out[-1].axis:
            last = out.pop()
            out[0] = max(out[0], last, key=lambda x: (x.snapped, x.n_points))
        return out

    return _prune(merge_runs(list(walls)), COLLAPSE, merge_runs)


def _vertices(walls: list) -> np.ndarray:
    """Corner i joins wall i-1 and wall i."""
    vs = []
    for prev, cur in zip(np.roll(np.array(walls, dtype=object), 1), walls):
        pa, pc = (prev.axis, prev.coord) if isinstance(prev, Wall) else (prev[0], prev[1])
        ca, cc = (cur.axis, cur.coord) if isinstance(cur, Wall) else (cur[0], cur[1])
        # axis 0 wall fixes v; axis 1 wall fixes u
        u = pc if pa == 1 else cc
        v = pc if pa == 0 else cc
        vs.append((u, v))
    return np.array(vs)


def _lengths_from_walls(walls) -> np.ndarray:
    v = _vertices(walls)
    return np.linalg.norm(np.roll(v, -1, 0) - v, axis=1)


def _snap_walls(
    walls: list[Wall], wall_uv, wall_n, others: np.ndarray, frame: PlanFrame, search=0.35, bin_m=0.005
):
    """Move each wall onto the measured surface: a dense layer of vertical points
    whose normal faces into the room. Furniture fronts sit inward of the wall, so
    the outermost dense layer wins -- unless reaching it would sweep through space
    that belongs to another room."""
    verts = _vertices(walls)
    for i, w in enumerate(walls):
        a, b = verts[i], verts[(i + 1) % len(walls)]
        along = w.axis  # coordinate index that varies along the wall (0: u, 1: v)
        across = 1 - along
        lo, hi = sorted((a[along], b[along]))
        length = hi - lo
        if length < 0.3:
            continue
        inward = inward_sign(verts, i, across)
        m = (
            (np.abs(wall_uv[:, across] - w.coord) < search)
            & (wall_uv[:, along] > lo + 0.05)
            & (wall_uv[:, along] < hi - 0.05)
            & (wall_n[:, across] * inward > 0.85)
        )
        x = wall_uv[m, across]
        if len(x) < 50:
            continue
        hist, edges = np.histogram(x, bins=np.arange(w.coord - search, w.coord + search + bin_m, bin_m))
        hist = np.convolve(hist, [1, 2, 1], "same")
        strong = np.flatnonzero(hist >= 0.4 * hist.max())
        centers = edges[strong] + bin_m / 2
        peak = None
        for cand in centers[np.argsort(centers * inward)]:  # outermost first
            if _swept_fraction(others, frame, along, lo, hi, w.coord, cand) < 0.15:
                peak = cand
                break
        if peak is None:
            continue
        sel = np.abs(x - peak) < 0.02
        near = x[sel]
        coord = float(np.median(near))
        cover = np.unique(np.floor(wall_uv[m, along][sel] / 0.05))
        w.coord = coord
        w.support = float(min(1.0, len(cover) * 0.05 / length))
        w.spread = float(1.4826 * np.median(np.abs(near - coord)))
        w.n_points = int(len(near))
        w.snapped = w.support >= 0.2


def _swept_fraction(others, frame: PlanFrame, along, lo, hi, c0, c1) -> float:
    """Share of the strip between wall positions c0 and c1 owned by other rooms."""
    if abs(c1 - c0) < frame.res:
        return 0.0
    ta = np.arange(lo + 0.05, hi - 0.05, frame.res)
    tc = np.arange(min(c0, c1), max(c0, c1), frame.res)
    if len(ta) == 0 or len(tc) == 0:
        return 0.0
    A, C = np.meshgrid(ta, tc)
    q = np.zeros((A.size, 2))
    q[:, along], q[:, 1 - along] = A.ravel(), C.ravel()
    r, c = frame.to_cell(q)
    ok = (r >= 0) & (r < others.shape[0]) & (c >= 0) & (c < others.shape[1])
    return float(others[r[ok], c[ok]].mean()) if ok.any() else 0.0


def _layer(y: np.ndarray, bin_m=0.005) -> tuple[float, float, int]:
    hist, edges = np.histogram(y, bins=np.arange(y.min(), y.max() + 2 * bin_m, bin_m))
    pk = edges[np.argmax(hist)] + bin_m / 2
    near = y[np.abs(y - pk) < 0.02]
    med = float(np.median(near))
    return med, float(1.4826 * np.median(np.abs(near - med))), int(len(near))


def fit_room(
    room_id: int,
    mask: np.ndarray,
    frame: PlanFrame,
    P: np.ndarray,
    N: np.ndarray,
    uv: np.ndarray,
    floor_y: float,
    vertical: np.ndarray,
    horizontal: np.ndarray,
    others: np.ndarray | None = None,
) -> RoomLayout:
    walls = _rectilinear(_mask_contour(mask, frame))
    # wall points near this room at wall heights; normals in plan frame
    r, c = frame.to_cell(uv)
    ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    near_room = np.zeros(len(P), bool)
    near_room[ok] = grown[r[ok], c[ok]]
    h = P[:, 1] - floor_y
    sel = near_room & vertical & (h > 0.2) & (h < 2.2)
    n_uv = frame.to_plan(np.c_[N[sel, 0], np.zeros(sel.sum()), N[sel, 2]])
    if others is None:
        others = np.zeros_like(mask)
    _snap_walls(walls, uv[sel], n_uv, others, frame)
    walls = _ccw(_collapse(walls))
    room = RoomLayout(room_id, walls, _vertices(walls))

    in_room = np.zeros(len(P), bool)
    in_room[ok] = mask[r[ok], c[ok]]
    fl = in_room & horizontal & (np.abs(h) < 0.05)
    if fl.sum() > 200:
        room.floor_y, room.floor_spread, room.n_floor = _layer(P[fl, 1])
    else:
        room.floor_y = floor_y
    ce = in_room & horizontal & (h > 1.9)
    if ce.sum() > 500:
        room.ceiling_y, room.ceiling_spread, room.n_ceiling = _layer(P[ce, 1])
    return room


@dataclass
class Opening:
    kind: str  # "door" | "window"
    wall: int  # index into RoomLayout.walls
    start: float  # distance from the wall's first vertex, m
    width: float
    bottom: float  # height above floor, m (0 for doors)
    top: float  # head height above floor, m
    seen_through: float  # fraction of the gap where the space beyond was observed
    jamb_points: int  # points backing the two jamb edges (for the error model)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    d = np.diff(np.r_[0, flags.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_openings(
    room: RoomLayout,
    P: np.ndarray,
    uv: np.ndarray,
    free: np.ndarray,
    frame: PlanFrame,
    bin_m: float = 0.02,
    plane_tol: float = 0.08,
) -> list[Opening]:
    """Scan each wall for gaps in its surface. A full-height gap that the camera
    saw through is a door; a mid-height gap with wall below and above is a window."""
    floor_y = room.floor_y
    top_h = (room.ceiling_height or 2.4) - 0.05
    h_all = P[:, 1] - floor_y
    verts = room.vertices
    out: list[Opening] = []
    for i, w in enumerate(room.walls):
        a, b = verts[i], verts[(i + 1) % len(verts)]
        along, across = w.axis, 1 - w.axis
        length = abs(b[along] - a[along])
        if length < 0.5:
            continue
        direction = np.sign(b[along] - a[along])
        outward = -inward_sign(verts, i, across)
        s = (uv[:, along] - a[along]) * direction  # distance from vertex a
        m = (np.abs(uv[:, across] - w.coord) < plane_tol) & (s > -0.05) & (s < length + 0.05)
        s, hh = s[m], h_all[m]
        nb = int(np.ceil(length / bin_m))
        idx = np.clip((s / bin_m).astype(int), 0, nb - 1)

        def band(lo, hi):
            occ = np.zeros(nb, int)
            k = (hh > lo) & (hh < hi)
            np.add.at(occ, idx[k], 1)
            return occ > 0

        low, mid, head = band(0.15, 0.75), band(0.9, 1.8), band(2.05, top_h)

        # was the space just beyond the wall observed (seen through the gap)?
        t = (np.arange(nb) + 0.5) * bin_m
        beyond = np.zeros(nb)
        for depth in (0.25, 0.4, 0.6):
            q = np.zeros((nb, 2))
            q[:, along] = a[along] + direction * t
            q[:, across] = w.coord + outward * depth
            r, c = frame.to_cell(q)
            ok = (r >= 0) & (r < free.shape[0]) & (c >= 0) & (c < free.shape[1])
            vals = np.zeros(nb)
            vals[ok] = free[r[ok], c[ok]]
            beyond = np.maximum(beyond, vals)
        seen = beyond >= 3

        # close 1-2 bin speckle in the gap masks
        gap_full = ndi_close(~low & ~mid)
        gap_mid = ndi_close(~mid)
        mid_pts = s[(hh > 0.9) & (hh < 1.8)]
        for kind, gap, need_seen in (("door", gap_full, 0.5), ("window", gap_mid & ~gap_full, 0.0)):
            for st, en in _runs(gap):
                width_bins = (en - st) * bin_m
                if not (0.45 <= width_bins <= 2.4):
                    continue
                if st == 0 or en == nb:  # gap touching a corner: can't bound both jambs
                    continue
                frac_seen = float(seen[st:en].mean())
                if kind == "door" and frac_seen < need_seen:
                    continue
                if kind == "window" and not (low[max(0, st - 2) : en + 2].mean() > 0.6):
                    continue
                # refine jambs from the actual surface points either side of the gap
                left = mid_pts[(mid_pts < st * bin_m + bin_m) & (mid_pts > st * bin_m - 0.15)]
                right = mid_pts[(mid_pts > en * bin_m - bin_m) & (mid_pts < en * bin_m + 0.15)]
                j0 = float(np.percentile(left, 98)) if len(left) > 5 else st * bin_m
                j1 = float(np.percentile(right, 2)) if len(right) > 5 else en * bin_m
                if j0 < 0 or j1 > length or j1 <= j0:
                    continue  # unbounded jamb: do not report an opening outside its wall
                head_pts = hh[(s > j0) & (s < j1) & (hh > 1.5)]
                top = float(np.percentile(head_pts, 2)) if len(head_pts) > 20 else top_h
                bottom = 0.0
                if kind == "window":
                    sill = hh[(s > j0) & (s < j1) & (hh < 1.2)]
                    bottom = float(np.percentile(sill, 98)) if len(sill) > 20 else 0.9
                out.append(
                    Opening(kind, i, j0, j1 - j0, bottom, top, frac_seen, int(len(left) + len(right)))
                )
    return out


def ndi_close(mask: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.binary_closing(mask, np.ones(k, bool)) | mask


def _walls_from_polygon(coords: np.ndarray, originals: list[Wall]) -> list[Wall]:
    """Rebuild a rectilinear wall list from polygon vertices, keeping the surface
    statistics of any original wall the new edge still lies on."""
    walls = []
    for p, q in zip(coords, np.roll(coords, -1, 0)):
        d = q - p
        if np.hypot(*d) < 1e-6:
            continue
        axis = 0 if abs(d[0]) >= abs(d[1]) else 1
        coord = float(p[1] if axis == 0 else p[0])
        same = [w for w in originals if w.axis == axis and abs(w.coord - coord) < 0.01]
        if same:
            walls.append(Wall(axis, coord, same[0].support, same[0].spread, same[0].n_points, same[0].snapped))
        else:
            walls.append(Wall(axis, coord))
    merged = []
    for w in walls:
        if merged and merged[-1].axis == w.axis:
            continue
        merged.append(w)
    if len(merged) > 1 and merged[0].axis == merged[-1].axis:
        merged.pop()
    return _ccw(merged)


def resolve_overlaps(rooms: list[RoomLayout], labels: np.ndarray, frame: PlanFrame) -> int:
    """Hand each overlap to the room whose free-space region covers more of it and
    cut it from the other. Returns the number of overlaps resolved."""
    from shapely.geometry import MultiPolygon, Polygon

    fixed = 0
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            A, B = Polygon(rooms[i].vertices), Polygon(rooms[j].vertices)
            inter = A.intersection(B)
            if inter.area < 1e-4:
                continue
            minx, miny, maxx, maxy = inter.bounds
            uu, vv = np.meshgrid(np.arange(minx, maxx, frame.res / 2), np.arange(miny, maxy, frame.res / 2))
            q = np.c_[uu.ravel(), vv.ravel()]
            r, c = frame.to_cell(q)
            ok = (r >= 0) & (r < labels.shape[0]) & (c >= 0) & (c < labels.shape[1])
            lab = labels[r[ok], c[ok]]
            keep_i = (lab == rooms[i].id).sum() >= (lab == rooms[j].id).sum()
            loser = rooms[j] if keep_i else rooms[i]
            winner_poly = A if keep_i else B
            cut = Polygon(loser.vertices).difference(winner_poly)
            if isinstance(cut, MultiPolygon):
                cut = max(cut.geoms, key=lambda p: p.area)
            if cut.is_empty or cut.area < 0.2:
                continue
            coords = np.array(cut.exterior.coords)[:-1]
            loser.walls = _walls_from_polygon(coords, loser.walls)
            loser.vertices = _vertices(loser.walls)
            fixed += 1
    return fixed
