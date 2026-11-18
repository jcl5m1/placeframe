"""Route a tracked query to one map tile.

A large place can be built as overlapping tiles that share one world frame (fleet/tiler.py in the pipeline
repository). Each tile has a core (the frames it was cut around) and an overlap of frames shared with its
neighbours. If the overlap is wider than what a camera sees, a tracked query needs only one tile: the tile whose
core holds the predicted camera position. The current tile stays in use until the position is more than
`hysteresis_m` outside its core, so a phone near a border does not switch back and forth, and a switch happens
while the phone is still well inside the old tile's overlap. Without a prediction (first frame, lost track) the
caller queries every tile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileCore:
    """A tile's core area on the floor plane (x, z in the map frame; y is vertical)."""

    tile_id: str
    x_min: float
    z_min: float
    x_max: float
    z_max: float

    def distance(self, x: float, z: float) -> float:
        """Metres from (x, z) to this core; 0 inside it."""
        dx = max(self.x_min - x, 0.0, x - self.x_max)
        dz = max(self.z_min - z, 0.0, z - self.z_max)
        return float(np.hypot(dx, dz))


class TileRouter:
    def __init__(self, cores: list[TileCore], hysteresis_m: float) -> None:
        if not cores:
            raise ValueError("no tiles")
        self.cores = {c.tile_id: c for c in cores}
        self.hysteresis_m = hysteresis_m

    def choose(self, current: str | None, map_from_camera: np.ndarray | None) -> str | None:
        """The tile for a query whose predicted pose is `map_from_camera`; None: no prediction, query every tile."""
        if map_from_camera is None:
            return None
        x, z = float(map_from_camera[0, 3]), float(map_from_camera[2, 3])
        if current in self.cores and self.cores[current].distance(x, z) <= self.hysteresis_m:
            return current
        return min(self.cores.values(), key=lambda c: c.distance(x, z)).tile_id

    @classmethod
    def from_plan(cls, plan: dict, tile_ids: dict[str, str], hysteresis_m: float | None = None) -> TileRouter:
        """From fleet/tiler.py's tiles.json; `tile_ids` maps the plan's tile names to map ids. Default hysteresis:
        half the overlap, so a switch happens with half the overlap still around the phone in the old tile."""
        cores = [TileCore(tile_ids[t["name"]], *t["core_bbox_xz"]) for t in plan["tiles"] if t["name"] in tile_ids]
        return cls(cores, plan["overlap_m"] / 2 if hysteresis_m is None else hysteresis_m)
