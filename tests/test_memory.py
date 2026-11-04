import math

import numpy as np

from scout.memory.explored import Memory
from test_world import corridor_house


def test_walls_block_sight():
    lay = corridor_house()
    m = Memory(lay)
    m.look((1.0, 0.5), yaw=0.0)  # facing the wall, doorway is off to the left
    g = lay.grid
    assert m.seen[g.cell((2.0, 0.5))]
    assert not m.seen[g.cell((4.5, 0.5))]  # behind the wall


def test_sees_through_doorway_and_records_object():
    lay = corridor_house()
    m = Memory(lay)
    new = m.look((2.0, 1.5), yaw=0.0)  # looking straight through the doorway
    assert m.seen[lay.grid.cell((4.0, 1.5))]
    assert m.seen_of("Fridge") == [] or new  # fridge is at (5.5, 2.5), may be at the cone edge
    m.look((3.5, 1.5), yaw=math.atan2(1.0, 2.0))
    s = m.seen_of("Fridge")
    assert len(s) == 1 and np.allclose(s[0].xy, (5.5, 2.5)) and s[0].room == "room|2"


def test_object_behind_viewer_not_seen():
    lay = corridor_house()
    m = Memory(lay)
    m.look((2.0, 1.5), yaw=0.0)
    assert m.seen_of("Bed") == []  # bed at (1, 1.5) is behind
    m.look((2.0, 1.5), yaw=math.pi)
    assert len(m.seen_of("Bed")) == 1


def test_frontiers_shrink_as_house_is_explored():
    lay = corridor_house()
    m = Memory(lay)
    m.look_around((1.5, 1.5))
    first = m.frontiers()
    assert first, "doorway region should leave unexplored edges"
    m.look_around((4.5, 1.5))
    m.look_around((5.5, 2.5))
    m.look_around((5.5, 0.5))
    assert m.coverage() > 0.9
    assert len(m.frontiers()) <= len(first)


def test_memory_never_marks_unseen_rooms():
    lay = corridor_house()
    m = Memory(lay)
    m.look((2.5, 1.5), yaw=math.pi)  # facing away from the doorway
    seen_rooms = {t for _, t, _ in m.rooms_seen()}
    assert "Bedroom" in seen_rooms and "Kitchen" not in seen_rooms


def test_narrow_doorway_stays_an_opening():
    """Regression: after body clearance a doorway is ~4 cells wide and must still count as an opening."""
    lay = corridor_house()
    lay.grid.free[29:32, :] = False
    lay.grid.free[29:32, 13:17] = True  # 0.4 m of walkable width, like an eroded doorway
    m = Memory(lay)
    m.seen[:30, :] = True               # whole bedroom and the first doorway column seen
    m.seen[30:32, :] = True
    m.seen[30:32, 13:17] = False        # rest of the doorway and the kitchen unseen
    near_door = [f for f in m.frontiers() if abs(f.xy[0] - 2.95) < 0.2]
    assert near_door and near_door[0].size == 4, [(f.xy, f.size) for f in m.frontiers()]
