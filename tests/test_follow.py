import math

import numpy as np

from scout.body.follow import DIRECTIONS, STAND, densify, lookahead, pick_direction, steer, wrap_deg


def test_pick_direction():
    assert pick_direction(10) == 0
    assert pick_direction(70) == 90
    assert pick_direction(-170) == 180
    assert pick_direction(-60) == -45


def test_lookahead_moves_forward_only():
    route = densify([[0, 0], [5, 0], [5, 5]])
    t, k = lookahead(route, np.array([2.0, 0.2]), 0)
    assert abs(t[0] - 3.0) < 0.15 and k > 0
    t2, k2 = lookahead(route, np.array([0.0, 0.0]), k)  # body slipped back: progress stays
    assert k2 >= k


def test_steer_turns_toward_the_corner():
    route = densify([[0, 0], [5, 0], [5, 5]])
    # facing north at the corner: the route goes north, so walk straight ahead
    skill, k, done = steer([5.0, 0.2], math.pi / 2, route, 45)
    assert skill == DIRECTIONS[0] and not done
    # facing east at the corner: the route now goes north, i.e. to the left
    skill, _, _ = steer([5.0, 0.2], 0.0, route, 45)
    assert skill == DIRECTIONS[90]


def test_steer_stops_at_the_end():
    route = densify([[0, 0], [2, 0]])
    skill, _, done = steer([1.9, 0.0], 0.0, route, len(route) - 1)
    assert done and skill == STAND


def test_simulated_follow_reaches_goal():
    """A unicycle that moves 1 m/s in the chosen egocentric direction follows an L-shaped route."""
    route = densify([[0, 0], [4, 0], [4, 4]])
    xy, yaw, k = np.array([0.0, 0.0]), 0.3, 0
    for _ in range(400):
        skill, k, done = steer(xy, yaw, route, k)
        if done:
            break
        ang = [a for a, s in DIRECTIONS.items() if s == skill][0]
        xy = xy + 0.033 * np.array([math.cos(yaw + math.radians(ang)), math.sin(yaw + math.radians(ang))])
        yaw += 0.002  # slow heading drift, like the real body
    assert done and np.linalg.norm(xy - route[-1]) < 0.6


def test_wrap():
    assert wrap_deg(190) == -170 and wrap_deg(-190) == 170
