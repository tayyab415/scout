"""A house reduced to what the planner and memory need: walkable grid, rooms and objects.

Built once per house from the MuJoCo scene (see `scout.world.house`) and saved as a small .npz,
so exploration experiments run without a simulator or GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Grid

# Object types used as goals, with the phrase used when talking about them.
GOAL_TYPES = {
    "Bed": "bed", "Toilet": "toilet", "Sofa": "sofa", "Television": "TV", "DiningTable": "dining table",
    "Fridge": "fridge", "Desk": "desk", "Bathtub": "bathtub", "Sink": "sink", "Dresser": "dresser",
    "ArmChair": "armchair", "CounterTop": "kitchen counter", "Microwave": "microwave", "Toaster": "toaster",
    "CoffeeMachine": "coffee machine", "LaundryHamper": "laundry hamper", "Laptop": "laptop",
    "HousePlant": "house plant", "FloorLamp": "floor lamp", "Painting": "painting",
}


@dataclass
class Room:
    id: str
    type: str
    polygon: np.ndarray  # (k, 2) world xy

    def contains(self, xy) -> bool:
        x, y = float(xy[0]), float(xy[1])
        p = self.polygon
        inside = False
        for (x1, y1), (x2, y2) in zip(p, np.roll(p, -1, axis=0)):
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
                inside = not inside
        return inside


@dataclass
class Obj:
    type: str
    xy: np.ndarray
    room: str | None


@dataclass
class Layout:
    name: str
    grid: Grid
    rooms: list[Room]
    objects: list[Obj] = field(default_factory=list)
    walls: np.ndarray | None = None  # cells where a wall crosses eye height (block sight)

    def room_at(self, xy) -> Room | None:
        for r in self.rooms:
            if r.contains(xy):
                return r
        return None

    def objects_of(self, type_: str) -> list[Obj]:
        return [o for o in self.objects if o.type == type_]

    def goal_types(self) -> list[str]:
        return sorted({o.type for o in self.objects if o.type in GOAL_TYPES})

    # --- serialisation -------------------------------------------------------------------------
    def save(self, path) -> None:
        np.savez_compressed(
            path, name=self.name, free=self.grid.free, origin=self.grid.origin, res=self.grid.res,
            walls=self.walls if self.walls is not None else np.zeros_like(self.grid.free),
            room_ids=np.array([r.id for r in self.rooms]), room_types=np.array([r.type for r in self.rooms]),
            room_sizes=np.array([len(r.polygon) for r in self.rooms]),
            room_points=np.concatenate([r.polygon for r in self.rooms]) if self.rooms else np.zeros((0, 2)),
            obj_types=np.array([o.type for o in self.objects]),
            obj_xy=np.array([o.xy for o in self.objects]).reshape(-1, 2),
            obj_rooms=np.array([o.room or "" for o in self.objects]),
        )

    @classmethod
    def load(cls, path) -> "Layout":
        z = np.load(path)
        grid = Grid(z["free"].astype(bool), z["origin"].astype(float), float(z["res"]))
        rooms, k = [], 0
        for rid, rt, n in zip(z["room_ids"], z["room_types"], z["room_sizes"]):
            rooms.append(Room(str(rid), str(rt), z["room_points"][k:k + n]))
            k += n
        objs = [Obj(str(t), xy, str(r) or None) for t, xy, r in zip(z["obj_types"], z["obj_xy"], z["obj_rooms"])]
        return cls(str(z["name"]), grid, rooms, objs, walls=z["walls"].astype(bool))
