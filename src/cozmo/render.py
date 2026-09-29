"""Render a PlanOut as a dimensioned floor plan (PNG/SVG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .plan import Z95, PlanOut  # noqa: E402

ROOM_FILL = ["#e8eef7", "#eef5e9", "#f7efe4", "#f3e9f3", "#e7f3f3", "#f6f3df", "#efe9e4", "#e9ecf5"]
WALL = "#2b2b2b"
DOOR = "#c0392b"
WINDOW = "#2e86c1"


def _fmt(m) -> str:
    return (
        f"{m.value:.2f}±{Z95 * m.sigma * 100:.0f}cm" if m.sigma * Z95 >= 0.005 else f"{m.value:.3f}"
    )


def render(plan: PlanOut, path: str | Path, wall_uv: np.ndarray | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    if wall_uv is not None and len(wall_uv):
        s = wall_uv[:: max(1, len(wall_uv) // 150_000)]
        ax.scatter(s[:, 0], s[:, 1], s=0.05, c="0.8", zorder=0, rasterized=True)

    for k, room in enumerate(plan.rooms):
        V = np.array(room.polygon)
        ax.add_patch(Polygon(V, closed=True, fc=ROOM_FILL[k % len(ROOM_FILL)], ec="none", zorder=1))
        centroid = V.mean(0)
        for w in room.walls:
            a, b = np.array(w.start), np.array(w.end)
            ax.plot(
                *np.c_[a, b],
                color=WALL,
                lw=2.4 if w.measured else 1.2,
                ls="-" if w.measured else "--",
                zorder=3,
            )
            if w.length.value < 0.4:
                continue
            mid = (a + b) / 2
            d = (b - a) / np.linalg.norm(b - a)
            n = np.array([-d[1], d[0]])
            if np.dot(centroid - mid, n) < 0:
                n = -n
            ang = np.degrees(np.arctan2(d[1], d[0]))
            if ang > 90 or ang < -90:
                ang += 180
            ax.text(
                *(mid + n * 0.14),
                _fmt(w.length),
                fontsize=6.5,
                ha="center",
                va="center",
                rotation=ang,
                zorder=5,
            )
        for o in room.openings:
            a, b = np.array(room.walls[o.wall].start), np.array(room.walls[o.wall].end)
            d = (b - a) / np.linalg.norm(b - a)
            p0, p1 = a + d * o.offset.value, a + d * (o.offset.value + o.width.value)
            ax.plot(*np.c_[p0, p1], color="white", lw=4, zorder=4)
            ax.plot(*np.c_[p0, p1], color=DOOR if o.kind == "door" else WINDOW, lw=2.2, zorder=4.5)
        lines = [room.name, f"{room.floor_area.value:.2f} m²"]
        if room.ceiling_height:
            lines.append(f"h {room.ceiling_height.value:.2f} m")
        ax.text(
            *centroid,
            "\n".join(lines),
            fontsize=8,
            ha="center",
            va="center",
            weight="bold",
            zorder=6,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    fp = plan.footprint_area
    ax.set_title(
        f"{plan.capture} · {plan.tier} tier · {len(plan.rooms)} rooms · footprint {fp.value:.1f}±{Z95 * fp.sigma:.1f} m²\n"
        "solid wall = surface-fitted, dashed = inferred · red = door · blue = window/opening",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
