import numpy as np

from scout.world.grid import Grid, descend, geodesic, largest_component, path_length
from scout.world.layout import Layout, Obj, Room


def corridor_house():
    """Two 3x3 m rooms joined by a 1 m doorway, 0.1 m cells, wall between them at x = 3 m."""
    free = np.zeros((61, 30), bool)
    free[:29, :] = True
    free[32:, :] = True
    free[29:32, 10:20] = True  # doorway
    walls = np.zeros_like(free)
    walls[29:32, :] = True
    walls[29:32, 10:20] = False
    grid = Grid(free, np.array([0.0, 0.0]), 0.1)
    rooms = [Room("room|1", "Bedroom", np.array([[0, 0], [2.95, 0], [2.95, 3], [0, 3]], float)),
             Room("room|2", "Kitchen", np.array([[3.15, 0], [6.1, 0], [6.1, 3], [3.15, 3]], float))]
    objs = [Obj("Bed", np.array([1.0, 1.5]), "room|1"), Obj("Fridge", np.array([5.5, 2.5]), "room|2")]
    return Layout("toy", grid, rooms, objs, walls=walls)


def test_geodesic_goes_through_doorway():
    lay = corridor_house()
    g = lay.grid
    d = geodesic(g.free, [g.cell((5.5, 0.5))], g.res)
    start = g.cell((0.5, 0.5))
    straight = np.hypot(5.0, 0.0)
    assert np.isfinite(d[start])
    assert d[start] > straight  # has to detour up to the doorway
    path = descend(d, start)
    assert path[-1] == g.cell((5.5, 0.5))
    assert abs(path_length(path, g.res) - d[start]) < 0.05
    assert all(g.free[c] for c in path)


def test_unreachable_cell_is_infinite():
    free = np.ones((10, 10), bool)
    free[5, :] = False
    d = geodesic(free, [(0, 0)], 0.1)
    assert np.isinf(d[9, 9])
    assert descend(d, (9, 9)) == []


def test_largest_component_keeps_biggest_region():
    free = np.zeros((10, 10), bool)
    free[:3, :3] = True
    free[5:, 5:] = True
    keep = largest_component(free)
    assert keep[9, 9] and not keep[0, 0]
    assert keep.sum() == 25


def test_layout_round_trip(tmp_path):
    lay = corridor_house()
    path = tmp_path / "toy.npz"
    lay.save(path)
    back = Layout.load(path)
    assert back.name == "toy"
    assert np.array_equal(back.grid.free, lay.grid.free)
    assert np.array_equal(back.walls, lay.walls)
    assert [r.type for r in back.rooms] == ["Bedroom", "Kitchen"]
    assert np.allclose(back.rooms[1].polygon, lay.rooms[1].polygon)
    assert [(o.type, o.room) for o in back.objects] == [("Bed", "room|1"), ("Fridge", "room|2")]


def test_room_lookup():
    lay = corridor_house()
    assert lay.room_at((1, 1)).type == "Bedroom"
    assert lay.room_at((5, 1)).type == "Kitchen"
    assert lay.room_at((10, 10)) is None


def test_erode_keeps_body_radius_clear():
    from scout.world.house import _erode
    m = np.ones((20, 20), bool)
    m[10, 10] = False  # one obstacle cell
    e = _erode(m, 2)
    assert not e[10, 10] and not e[10, 12] and e[10, 13]
    assert not e[0, 0]  # the grid edge counts as blocked


def test_inside_rooms_vectorised_matches_polygon():
    from scout.world.house import _inside_rooms
    lay = corridor_house()
    xs = np.array([1.0, 3.05, 5.0, 7.0])
    ys = np.array([1.5])
    inside = _inside_rooms(lay.rooms, xs, ys).ravel()
    assert list(inside) == [lay.room_at((x, 1.5)) is not None for x in xs]
