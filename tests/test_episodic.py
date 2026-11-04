import numpy as np

from scout.memory.episodic import Episodic


def test_same_label_nearby_is_one_object_far_is_two():
    m = Episodic()
    _, a, new_a = m.add("cup", 0.8, [1, 1, 0.9], 1.0, 0, [0, 0])
    _, b, new_b = m.add("cup", 0.7, [1.3, 1, 0.9], 2.0, 1, [0, 0])
    _, c, new_c = m.add("cup", 0.9, [5, 5, 0.9], 3.0, 2, [0, 0])
    assert new_a and not new_b and new_c
    assert a is b and c is not a
    assert "cup" in m.table()


def test_expected_in_view_respects_occlusion():
    m = Episodic()
    m.add("cup", 0.8, [3.0, 0, 1.5], 0.0, 0, [0, 0])
    # camera at origin looking along +x: camera -z axis = world +x
    # columns = camera x (right), y (up), z (backwards) in world
    R = np.stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], 1).astype(float)
    depth = np.full((64, 64), 10.0)
    assert len(m.expected_in_view([0, 0, 1.5], R, 75, depth)) == 1
    depth[:] = 1.0  # a wall at 1 m hides it
    assert len(m.expected_in_view([0, 0, 1.5], R, 75, depth)) == 0


def test_confident_needs_two_viewpoints_and_consistency():
    m = Episodic()
    for k in range(3):
        _, tr, _ = m.add("sofa", 0.9, [5, 5, 0.5], k, k, [0, 0])
    assert not tr.confident(m)  # same viewpoint three times
    m.add("sofa", 0.9, [5.1, 5, 0.5], 4, 4, [2, 0])
    assert tr.confident(m)
