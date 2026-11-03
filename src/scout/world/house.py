"""Load MolmoSpaces ProcTHOR houses into MuJoCo, alone or with the HumEnv/Motivo body.

House files live under `<root>/scenes/<name>/` and reference shared THOR assets at
`<root>/objects/thor/` (see `scripts/fetch_houses.py`).
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

from .grid import Grid, largest_component
from .layout import Layout, Obj, Room

BODY_RADIUS = 0.25          # m, clearance kept around the body
BODY_Z = (0.1, 1.7)         # m, height band that must be clear to walk
EYE_Z = 1.55                # m, head-camera height
CEILING_GROUP = 2           # ceilings get their own render group so outside cameras can hide them (THOR uses 3-5)


def house_files(root: str, name: str) -> tuple[str, str, str]:
    d = f"{root}/scenes/{name}"
    xml = [p for p in glob.glob(f"{d}/*.xml") if "ceiling" not in p][0]
    js = [p for p in glob.glob(f"{d}/*.json") if "metadata" not in p][0]
    return xml, xml[:-4] + "_ceiling.xml", js


def _load_spec(path: str):
    """House spec with absolute asset paths, door leaves removed and every other joint welded."""
    import mujoco
    spec = mujoco.MjSpec.from_file(path)
    d = os.path.dirname(path)
    for m in spec.meshes:
        if m.file:
            m.file = os.path.normpath(os.path.join(d, m.file))
    for t in spec.textures:
        if t.file:
            t.file = os.path.normpath(os.path.join(d, t.file))
    for b in list(spec.bodies):
        try:
            if any("doorwaydoor" in (j.name or "") for j in b.joints):
                spec.delete(b)
        except Exception:  # already removed with its parent
            pass
    for j in list(spec.joints):
        spec.delete(j)
    spec.compiler.boundmass, spec.compiler.boundinertia = 1e-3, 1e-6
    return spec


def _house_spec(root: str, name: str, ceiling: bool):
    """The `*_ceiling.xml` file is the whole house plus ceiling bodies, so load one file or the other, never both."""
    xml, cxml, _ = house_files(root, name)
    use_ceiling = ceiling and os.path.exists(cxml)
    spec = _load_spec(cxml if use_ceiling else xml)
    if use_ceiling:
        for b in spec.bodies:
            if (b.name or "").startswith("ceiling_"):
                for g in b.geoms:
                    g.group = CEILING_GROUP
    return spec


def build_house(root: str, name: str, ceiling: bool = True):
    """Static house only (for free-camera rendering and occupancy)."""
    import mujoco
    world = mujoco.MjSpec()
    world.compiler.boundmass, world.compiler.boundinertia = 1e-3, 1e-6
    world.attach(_house_spec(root, name, ceiling), prefix="h_", frame=world.worldbody.add_frame(pos=[0, 0, 0]))
    world.visual.headlight.ambient = [0.5] * 3
    world.visual.headlight.diffuse = [0.5] * 3
    world.visual.headlight.specular = [0.05] * 3
    world.visual.global_.fovy = 75
    return world.compile()


def build_body_world(root: str, name: str, ceiling: bool = True, fovy: float = 75, eye_pitch_deg: float = 0.0):
    """HumEnv robot first (its bodies, sensors and actuators must keep their indices), then the house.

    SMPL body frames are +x left, +y up, +z forward; the `eye` camera looks along the head's +z, tilted down by
    eye_pitch_deg (people look at the floor a few metres ahead when they walk).
    Ceiling geoms are put in group CEILING_GROUP.
    """
    import humenv
    import mujoco
    world = mujoco.MjSpec.from_file(os.path.join(os.path.dirname(humenv.__file__), "assets", "robot.xml"))
    world.compiler.boundmass, world.compiler.boundinertia = 1e-3, 1e-6
    world.geom("floor").pos = [0, 0, -0.003]
    world.geom("floor").group = 3  # still collides, but cameras do not see an endless floor through the windows
    a = math.radians(eye_pitch_deg)
    world.body("Head").add_camera(name="eye", pos=[0, 0.05, 0.10], xyaxes=[-1, 0, 0, 0, math.cos(a), math.sin(a)], fovy=fovy)
    world.attach(_house_spec(root, name, ceiling), prefix="h_", frame=world.worldbody.add_frame(pos=[0, 0, 0]))
    world.visual.global_.offwidth, world.visual.global_.offheight = 1280, 960
    world.visual.headlight.ambient = [0.45] * 3
    world.visual.headlight.diffuse = [0.5] * 3
    return world.compile()


def _inside_rooms(rooms: list[Room], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test for every grid cell centre, vectorised."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    out = np.zeros(X.shape, bool)
    for r in rooms:
        inside = np.zeros(X.shape, bool)
        p = r.polygon
        for (x1, y1), (x2, y2) in zip(p, np.roll(p, -1, axis=0)):
            crosses = (y1 > Y) != (y2 > Y)
            xint = (x2 - x1) * (Y - y1) / (y2 - y1 + 1e-12) + x1
            inside ^= crosses & (X < xint)
        out |= inside
    return out


def _erode(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Cells whose whole disk of the given radius is inside the mask."""
    out = mask.copy()
    nx, ny = mask.shape
    pad = np.zeros((nx + 2 * radius_cells, ny + 2 * radius_cells), bool)
    pad[radius_cells:radius_cells + nx, radius_cells:radius_cells + ny] = mask
    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            if di * di + dj * dj <= radius_cells * radius_cells:
                out &= pad[radius_cells + di:radius_cells + di + nx, radius_cells + dj:radius_cells + dj + ny]
    return out


def occupancy(model, rooms: list[Room], res: float = 0.1) -> tuple[Grid, np.ndarray]:
    """Walkable cells and wall cells.

    A cell is clear if it is inside a room, not a wall, and a ray cast straight down from the top of
    the body's height band reaches the floor. Walkable cells are clear cells with clear space for the
    body's radius all round. Wall cells are where a wall geom crosses eye height; they block sight,
    furniture does not.
    """
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pts = np.concatenate([r.polygon for r in rooms])
    lo, hi = pts.min(0) - 0.5, pts.max(0) + 0.5
    nx, ny = (np.ceil((hi - lo) / res)).astype(int)
    xs = lo[0] + (np.arange(nx) + 0.5) * res
    ys = lo[1] + (np.arange(ny) + 0.5) * res
    inside = _inside_rooms(rooms, xs, ys)
    walls = np.zeros((nx, ny), bool)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if "wall_" not in body:
            continue
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        w = (corners * h + c) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        if not (w[:, 2].min() < EYE_Z < w[:, 2].max()):
            continue
        a0, b0 = np.floor((w[:, :2].min(0) - lo) / res).astype(int)
        a1, b1 = np.ceil((w[:, :2].max(0) - lo) / res).astype(int)
        walls[max(a0, 0):max(a1, 0), max(b0, 0):max(b1, 0)] = True
    clear = inside & ~walls
    gid = np.zeros(1, np.int32)
    down = np.array([0.0, 0.0, -1.0])
    top = BODY_Z[1]
    for i, j in zip(*np.nonzero(clear)):
        dist = mujoco.mj_ray(model, data, np.array([xs[i], ys[j], top]), down, None, 1, -1, gid)
        if gid[0] >= 0 and top - dist > BODY_Z[0]:
            clear[i, j] = False
    free = _erode(clear, int(round(BODY_RADIUS / res)))
    return Grid(largest_component(free), lo, res), walls


def layout_from_house(root: str, name: str, res: float = 0.1) -> Layout:
    """Rooms and objects from the house JSON (THOR x, z map to MuJoCo x, y), grid from collision geometry."""
    _, _, js = house_files(root, name)
    house = json.load(open(js))
    rooms = [Room(r["id"], r["roomType"], np.array([[q["x"], q["z"]] for q in r["floorPolygon"]], float))
             for r in house["rooms"]]
    model = build_house(root, name, ceiling=False)
    grid, walls = occupancy(model, rooms, res)
    lay = Layout(name, grid, rooms, walls=walls)
    for o in house["objects"]:
        xy = np.array([o["position"]["x"], o["position"]["z"]], float)
        r = lay.room_at(xy)
        lay.objects.append(Obj(o["id"].split("|")[0], xy, r.id if r else None))
    return lay
