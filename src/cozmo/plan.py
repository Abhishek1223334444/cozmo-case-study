"""Output contract: every measurement carries a value, a sigma and a 95% interval."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "cozmo.plan/0.2"
Z95 = 1.96


@dataclass
class M:
    """A measurement in metres (or m^2), with a 1-sigma and a 95% interval."""

    value: float
    sigma: float
    unit: str = "m"

    def to_json(self) -> dict:
        if not math.isfinite(self.value) or not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("Measurements must be finite with nonnegative uncertainty")
        return {
            "value": round(self.value, 4),
            "ci95": [round(max(0, self.value - Z95 * self.sigma), 4), round(self.value + Z95 * self.sigma, 4)],
            "sigma": round(self.sigma, 4),
            "unit": self.unit,
            "interval_status": "uncalibrated_model_interval",
        }


def quad(*sigmas: float) -> float:
    return math.sqrt(sum(s * s for s in sigmas))


@dataclass
class WallOut:
    index: int
    start: list[float]  # plan-XY of first vertex, m
    end: list[float]
    length: M
    measured: bool  # backed by a surface fit (False: inferred from free space only)


@dataclass
class OpeningOut:
    kind: str
    wall: int
    offset: M  # from the wall's start vertex to the near jamb
    width: M
    height: M
    sill: M | None
    connects_to: int | None  # room id beyond a door, if any


@dataclass
class RoomOut:
    id: int
    name: str
    polygon: list[list[float]]  # plan-XY, m, counter-clockwise
    walls: list[WallOut]
    floor_area: M
    perimeter: M
    ceiling_height: M | None
    openings: list[OpeningOut]
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanOut:
    capture: str
    tier: str
    rooms: list[RoomOut]
    adjacency: list[dict]
    footprint_area: M
    frame: dict  # how plan-XY relates to the capture's world frame
    diagnostics: dict = field(default_factory=dict)
    damage: list = field(default_factory=list)
    scope: list = field(default_factory=list)
    concealed_damage: list = field(default_factory=list)
    surfaces: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    schema: str = SCHEMA_VERSION

    def to_json(self) -> dict:
        def conv(o):
            if isinstance(o, M):
                return o.to_json()
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [conv(v) for v in o]
            if hasattr(o, "__dataclass_fields__"):
                return {k: conv(getattr(o, k)) for k in o.__dataclass_fields__}
            if isinstance(o, float):
                return round(o, 4)
            return o

        return conv(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2, allow_nan=False))


__all__ = ["M", "quad", "WallOut", "OpeningOut", "RoomOut", "PlanOut", "asdict"]
