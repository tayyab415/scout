import math

import numpy as np

from scout.perception.selfmap import SelfMap, depth_to_points


def _flat_floor_depth(h=64, fovy=75, cam_z=1.5, pitch_deg=-30):
    """Analytic depth of a flat floor seen by a camera looking along +x, pitched down."""
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    vs, us = np.mgrid[0:h, 0:h]
    x = (us + 0.5 - h / 2) / f
    y = -(vs + 0.5 - h / 2) / f
    p = math.radians(pitch_deg)
    # camera frame: -z forward, +y up. World: forward = (cos p, 0, sin p), up = (-sin p, 0, cos p), right = (0, -1, 0)
    fwd, up, right = np.array([math.cos(p), 0, math.sin(p)]), np.array([-math.sin(p), 0, math.cos(p)]), np.array([0, -1.0, 0])
    dirs = x[..., None] * right + y[..., None] * up + fwd  # direction per unit view depth
    t = np.where(dirs[..., 2] < -1e-6, -cam_z / dirs[..., 2], 0.0)
    R = np.stack([right, up, -fwd], 1)  # columns: camera x, y, z axes in world
    return t, R


def test_floor_points_land_on_floor():
    depth, R = _flat_floor_depth()
    pts = depth_to_points(depth, [0, 0, 1.5], R, 75, stride=1)
    assert len(pts) > 100
    assert np.abs(pts[:, 2]).max() < 1e-6
    assert pts[:, 0].min() > 0  # all in front of the camera


def test_map_marks_floor_walkable_and_finds_path():
    depth, R = _flat_floor_depth()
    m = SelfMap(origin=[0, 0], size_m=20)
    m.integrate(depth_to_points(depth, [0, 0, 1.5], R, 75, stride=1))
    m.stand([0, 0])
    wk = m.walkable()
    assert wk[m.cell([2.0, 0])]
    assert not m.seen[m.cell([-3.0, 0])]
    p = m.path([0, 0], [2.0, 0])
    assert len(p) > 5 and np.linalg.norm(p[-1] - [2.0, 0]) < 0.4


def test_wall_blocks_and_frontiers_exist():
    m = SelfMap(origin=[0, 0], size_m=20)
    xs, ys = np.meshgrid(np.arange(0.5, 4, 0.05), np.arange(-2, 2, 0.05))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1)
    wall = np.stack([np.full(80, 2.0), np.linspace(-2, 2, 80), np.full(80, 1.0)], 1)
    m.integrate(np.concatenate([floor, wall, wall]))
    m.stand([0, 0])
    assert not m.walkable()[m.cell([2.0, 0])]
    assert len(m.frontiers()) >= 1


def test_ray_carving_connects_feet_to_far_floor():
    depth, R = _flat_floor_depth(pitch_deg=0)  # looking straight ahead: floor only visible far away
    m = SelfMap(origin=[0, 0], size_m=20)
    m.integrate(depth_to_points(depth, [0, 0, 1.5], R, 75, stride=1), cam_xy=[0, 0])
    assert m.floor[m.cell([0.5, 0])] > 0
