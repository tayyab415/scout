"""Regression tests for the problems the independent review found (research/critic-review-20261103/REVIEW.md)."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from scout.perception.selfmap import BODY_RADIUS, SelfMap

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _clearance(m: SelfMap, path: np.ndarray, obstacle_xy: np.ndarray) -> float:
    """Distance from each path cell centre to the nearest edge of an obstacle cell."""
    half = m.res / 2
    d = []
    for p in path:
        dx = np.maximum(np.abs(obstacle_xy[:, 0] - p[0]) - half, 0)
        dy = np.maximum(np.abs(obstacle_xy[:, 1] - p[1]) - half, 0)
        d.append(np.sqrt(dx ** 2 + dy ** 2).min())
    return float(min(d))


def test_path_keeps_body_clearance_even_in_the_unseen_near_field():
    m = SelfMap(origin=[0, 0], size_m=12)
    # a short wall 0.45 m to the side of the start, nothing else seen: the near field (unseen floor) is filled in
    wall = np.stack([np.full(40, 0.0), np.linspace(0.45, 0.55, 40), np.full(40, 1.0)], 1)
    m.integrate(np.concatenate([wall, wall]))
    m.stand([0, 0])
    p = m.path([0, 0], [0.0, 0.8 + 0.3], goal_radius=0.1)
    obstacles = np.array([m.pos(c) for c in np.argwhere(m.blocked)])
    if len(p):
        assert _clearance(m, p[1:], obstacles) >= BODY_RADIUS - 1e-6


def test_walkable_cells_keep_body_clearance():
    m = SelfMap(origin=[0, 0], size_m=12)
    xs, ys = np.meshgrid(np.arange(-2, 2, 0.05), np.arange(-2, 2, 0.05))
    m.integrate(np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1))
    post = np.stack([np.zeros(30), np.zeros(30), np.linspace(0.3, 1.2, 30)], 1) + [1.0, 0.0, 0]
    m.integrate(np.concatenate([post, post]))
    wk = np.array([m.pos(c) for c in np.argwhere(m.walkable())])
    obstacles = np.array([m.pos(c) for c in np.argwhere(m.blocked)])
    assert _clearance(m, wk, obstacles) >= BODY_RADIUS - 1e-6


def _fake_run(d):
    """A two-look run: an object seen at [1, 1] at t=0 and re-estimated at [3, 1] at t=10; only frame 0 has a box."""
    from PIL import Image
    os.makedirs(f"{d}/frames", exist_ok=True)
    for i in range(3):
        Image.fromarray(np.full((512, 512, 3), 40 * i, np.uint8)).save(f"{d}/frames/{i:04d}.jpg")
    looks = [dict(t=0.0, frame=0, xy=[0, 0], yaw=0.0, dets=[dict(label="sofa", score=0.9, sighting=0, track=0, box=[10, 10, 100, 100])]),
             dict(t=5.0, frame=1, xy=[0, 0], yaw=0.0, dets=None),
             dict(t=10.0, frame=2, xy=[0, 0], yaw=0.0, dets=[dict(label="sofa", score=0.9, sighting=1, track=0, box=[10, 10, 100, 100])])]
    json.dump(dict(looks=looks, events=[], qa=[], observer_times=[0.0, 5.0, 10.0], observer_times_prefix=[]), open(f"{d}/log.json", "w"))
    json.dump(dict(sightings=[dict(id=0, label="sofa", score=0.9, xyz=[1, 1, 0.5], t=0.0, frame=0, body_xy=[0, 0]),
                              dict(id=1, label="sofa", score=0.9, xyz=[3, 1, 0.5], t=10.0, frame=2, body_xy=[0, 0])],
                   tracks=[dict(id=0, label="sofa", sightings=[0, 1], misses=[], status="ok", confident=False)]),
              open(f"{d}/memory.json", "w"))


def test_video_state_uses_only_information_available_at_that_time():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import env_video
    with tempfile.TemporaryDirectory() as d:
        _fake_run(d)
        st = env_video.State(d)
        assert st.pins(0.0) == [("sofa", [1, 1])]
        assert st.pins(10.0) == [("sofa", [3, 1])]
        assert st.boxes_for_frame(1) == []          # no detection on frame 1: draw nothing
        assert len(st.boxes_for_frame(0)) == 1
        assert st.head_frame(5.2) == 1               # the head image shown is the latest one at or before t


def test_agent_clock_has_no_extra_steps():
    """Every Agent.tick advances the env by exactly one step, also while recovering from a fall."""
    src = open(os.path.join(ROOT, "src/scout/env/agent.py")).read()
    tick = src[src.index("    def tick(self, env):"):src.index("    def quiz(")]
    assert tick.count("env.step(") == 3          # recovering, just fallen, normal
    assert "for " not in tick.split("# steer")[0].split("if self.recover is not None")[1].split("if body.head_z < 0.8")[0]


def _fake_episode(d, arm, verdict, quiz=None, visible=None):
    a = f"{d}/{arm}"
    os.makedirs(a, exist_ok=True)
    json.dump(dict(events=[], looks=[dict(t=10.0, frame=0, xy=[0, 0], yaw=0.0, dets=[])], qa=[dict(kind="question", text="q"),
                   dict(kind="answer", result=verdict, say="x")], quiz=quiz or [], falls=0, got_up=0, ended=None, t_end=100.0),
              open(f"{a}/log.json", "w"))
    json.dump(dict(sightings=[dict(id=0, label="sofa", score=0.9, xyz=[1, 1, 0.5], t=10.0, frame=0, body_xy=[0, 0])],
                   tracks=[dict(id=0, label="sofa", sightings=[0], misses=[], status="ok", confident=True)]), open(f"{a}/memory.json", "w"))
    json.dump(dict(objects_start=[dict(id=0, category="sofa", xy=[1.2, 1.0]), dict(id=1, category="bed", xy=[5, 5])], moves=[],
                   visibility=[dict(t=10.0, frame=0, visible=visible if visible is not None else {"0": 0.1})],
                   instances={"0": dict(category="sofa", key="a"), "1": dict(category="bed", key="b")}), open(f"{a}/evaluator.json", "w"))


def test_verify_scoring_needs_the_right_answer_per_arm_and_counts_abstentions():
    from scout.env.scoring import score_verify
    assert score_verify("unchanged", dict(qa=[dict(kind="answer", result="still there")]))["outcome"] == "correct"
    assert score_verify("moved", dict(qa=[dict(kind="answer", result="still there")]))["outcome"] == "wrong"
    assert score_verify("moved", dict(qa=[dict(kind="answer", result="not found")]))["outcome"] == "correct"
    assert score_verify("unchanged", dict(qa=[dict(kind="answer", result="not found")]))["outcome"] == "wrong"
    assert score_verify("moved", dict(qa=[dict(kind="answer", result="unclear")]))["outcome"] == "abstain"
    # a brain that always says "not found" gets the moved arm right and the unchanged arm wrong


def test_quiz_is_graded_against_what_the_camera_showed_not_what_the_house_contains():
    from scout.env.scoring import score_citations, score_quiz
    with tempfile.TemporaryDirectory() as d:
        quiz = [dict(category="sofa", observed=True, xy=[1, 1], cited=[0]),
                dict(category="bed", observed=False, xy=None, cited=[]),     # in the house, never visible: "no" is right
                dict(category="toilet", observed=True, xy=[9, 9], cited=[7])]  # never existed: wrong, bad citation
        _fake_episode(d, "unchanged", "still there", quiz)
        log, mem, ev = (json.load(open(f"{d}/unchanged/{f}")) for f in ("log.json", "memory.json", "evaluator.json"))
        q = score_quiz(log, mem, ev)
        assert [r["correct"] for r in q["rows"]] == [True, True, False]
        c = score_citations(log, mem, ev)
        assert (c["cited"], c["valid"], c["invalid_id"]) == (2, 1, 1)
        assert (c["affirmative"], c["affirmative_supported"]) == (2, 1)


def test_body_inside_a_clearance_margin_can_still_plan_out():
    """Regression (pilot cmp1): after a close inspection the body stood inside an obstacle margin and every path failed."""
    m = SelfMap(origin=[0, 0], size_m=12)
    xs, ys = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-3, 3, 0.05))
    m.integrate(np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1))
    post = np.stack([np.full(30, 0.2), np.zeros(30), np.linspace(0.3, 1.2, 30)], 1)  # an obstacle 0.2 m from the body
    m.integrate(np.concatenate([post, post]))
    m.stand([0, 0])
    assert m.inflated()[m.cell([0, 0])]
    p = m.path([0, 0], [-2.0, 0.0], goal_radius=0.2)
    assert len(p) > 5
    obstacles = np.array([m.pos(c) for c in np.argwhere(m.blocked)])
    far = [q for q in p if np.linalg.norm(q) > 0.7]  # beyond the escape zone the clearance holds
    assert _clearance(m, np.array(far), obstacles) >= BODY_RADIUS - 1e-6


def test_viewpoint_is_reachable_close_and_has_line_of_sight():
    from scout.env.agent import Agent
    a = Agent(brain=None, mind=False)
    m = SelfMap(origin=[0, 0], size_m=16)
    xs, ys = np.meshgrid(np.arange(-5, 5, 0.05), np.arange(-5, 5, 0.05))
    m.integrate(np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1))
    wall = np.stack([np.full(120, 2.0), np.linspace(-5, 1.0, 120), np.full(120, 1.0)], 1)  # wall x=2 with a gap above y=1
    m.integrate(np.concatenate([wall, wall]))
    m.stand([0, 0])
    a.smap = m
    obj = np.array([4.0, -2.0])  # behind the wall from the body's side
    p = a._viewpoint_path(np.array([0.0, 0.0]), obj)
    assert len(p) >= 2
    end = p[-1]
    assert 1.2 <= np.linalg.norm(end - obj) <= 3.5
    seg = end + np.linspace(0, 1, 30)[:, None] * (obj - end)
    ij = np.floor((seg[:-4] - m.origin) / m.res).astype(int)
    assert not m.blocked[ij[:, 0], ij[:, 1]].any()


def test_recorded_sample_episode_scores_offline():
    """A real recorded episode (examples/sample_episode) scores without a GPU, both arms correct."""
    import glob
    from scout.env.scoring import score_episode
    from scout.world.layout import Layout
    d = os.path.join(ROOT, "examples/sample_episode")
    r = score_episode(d, Layout.load(glob.glob(d + "/layout_*.npz")[0]))
    assert r["unchanged"]["verify"]["outcome"] == "correct" and r["moved"]["verify"]["outcome"] == "correct"
    assert 0 <= r["unchanged"]["quiz"]["accuracy"] <= 1 and r["unchanged"]["cite"]["cited"] > 0


def test_cells_the_body_stood_on_stay_passable():
    """Regression (dev3 val_227): obstacle hits under the body's own trail trapped it; its trail must stay plannable."""
    m = SelfMap(origin=[0, 0], size_m=12)
    xs, ys = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-3, 3, 0.05))
    m.integrate(np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], 1))
    for x in np.arange(-2.0, 0.01, 0.1):
        m.stand([x, 0.0])                                         # it walked here along y=0
    over = np.stack([np.linspace(-2, 0, 60), np.full(60, 0.0), np.full(60, 0.8)], 1)  # an overhang over its trail
    m.integrate(np.concatenate([over, over]))
    assert not m.blocked[m.cell([-1.0, 0.0])]
    assert len(m.path([0.0, 0.0], [-2.0, 0.0], goal_radius=0.1)) > 5
