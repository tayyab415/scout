"""Independent CPU contract probes against the frozen Scout source.

Run through ../critic-review-20261103/record_command.py so every run has video.
Fake physics exercises Python control flow; it does not establish MuJoCo fidelity.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).parent / ".."
sys.path.insert(0, str(ROOT / "src"))
from scout.env.agent import Agent
from scout.env.core import ScoutEnv
from scout.env.scoring import score_citations, score_episode, score_explore, score_quiz
from scout.brains import RuleBrain


def evidence():
    mem = dict(sightings=[dict(id=0, label="sofa", t=10., frame=0, xyz=[1, 1, .5], body_xy=[0, 0], score=.9),
                         dict(id=1, label="bed", t=10., frame=0, xyz=[5, 5, .5], body_xy=[0, 0], score=.9)],
               tracks=[dict(id=0, label="sofa", sightings=[0], misses=[], status="ok", confident=False),
                       dict(id=1, label="bed", sightings=[1], misses=[], status="ok", confident=False)])
    ev = dict(objects_start=[dict(id=0, category="sofa", xy=[1, 1]), dict(id=1, category="bed", xy=[5, 5])],
              moves=[], visibility=[dict(t=10., frame=0, visible={"0": .1, "1": .1})],
              instances={"0": dict(category="sofa", key="s"), "1": dict(category="bed", key="b")})
    log = dict(t_end=100., events=[], looks=[dict(t=10., frame=0, xy=[0, 0])], qa=[], quiz=[],
               falls=0, got_up=0, ended=None)
    return log, mem, ev


@pytest.mark.parametrize("xy", [[100., 100.], None], ids=["141m-away", "missing-location"])
def test_observed_answer_requires_correct_location(xy):
    log, mem, ev = evidence()
    log["quiz"] = [dict(category="sofa", observed=True, xy=xy, cited=[0])]
    result = score_quiz(log, mem, ev)
    print("quiz:", result)
    assert result["accuracy"] == 0., "Wrong or missing location received full credit"
    assert result["said_seen_correct"] == 0


def test_correct_location_control():
    log, mem, ev = evidence()
    log["quiz"] = [dict(category="sofa", observed=True, xy=[1, 1], cited=[0])]
    assert score_quiz(log, mem, ev)["accuracy"] == 1.


def test_citation_must_support_the_question_category():
    log, mem, ev = evidence()
    log["quiz"] = [dict(category="sofa", observed=True, xy=[5, 5], cited=[1])]
    result = score_citations(log, mem, ev)
    print("sofa answer cites a bed:", result)
    assert result["valid"] == 0


def test_citation_must_precede_its_question():
    log, mem, ev = evidence()
    log["quiz"] = [dict(t=5., category="sofa", observed=True, xy=[1, 1], cited=[0])]
    result = score_citations(log, mem, ev)
    print("question t=5, citation t=10, episode ends t=100:", result)
    assert result["future"] == 1 and result["valid"] == 0


def test_failed_brain_call_is_not_a_correct_negative():
    class FailedBrain:
        def answer(self, table, question):
            return {}  # exact LLMBrain._call fallback after provider/parse failure
    agent = Agent(FailedBrain(), mind=False)
    log, mem, ev = evidence()
    ev["visibility"] = []
    log["quiz"] = agent.quiz(["sofa"])
    result = score_quiz(log, mem, ev)
    print("failed call normalized to", log["quiz"], "score", result["accuracy"])
    assert result["accuracy"] == 0., "A failed call must remain an error/abstention in the denominator"


def write_episode(path, track=0, moved=True):
    log, mem, ev = evidence()
    log["qa"] = [dict(t=90., kind="question", text="Is the sofa still where you last saw it?", object_id=track),
                 dict(t=95., kind="answer", track=track, result="not found", say="I could not find it.",
                      before_frame=0, now_frames=[0])]
    move = dict(t=80., instance=0, from_xy=[1, 1], to_xy=[10, 10]) if moved else None
    ev["moves"] = [move] if move else []
    man = dict(episode="probe", house="synthetic", brain="probe", explore_s=80.,
               target=dict(instance=0, category="sofa", track_at_freeze=0, from_xy=[1, 1],
                           dest_xy=[10, 10] if moved else None), arms={"moved": dict(moved=move)})
    (path / "moved").mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps(man))
    for name, obj in [("log", log), ("memory", mem), ("evaluator", ev)]:
        (path / "moved" / (name + ".json")).write_text(json.dumps(obj))


def test_verify_does_not_reward_checking_the_wrong_object(tmp_path):
    write_episode(tmp_path, track=1)
    result = score_episode(str(tmp_path))["moved"]["verify"]
    print("asked sofa, answered about bed:", result)
    assert result["outcome"] != "correct"


def test_moved_arm_without_intervention_is_not_a_correct_moved_trial(tmp_path):
    write_episode(tmp_path, moved=False)
    result = score_episode(str(tmp_path))["moved"]["verify"]
    print("moved arm with no move:", result)
    assert result["outcome"] != "correct"


def test_verify_correct_target_control(tmp_path):
    write_episode(tmp_path)
    assert score_episode(str(tmp_path))["moved"]["verify"]["outcome"] == "correct"


class Vector(np.ndarray):
    def norm(self):
        return np.linalg.norm(self)

    def clone(self):
        return self.copy()


class ClockEnv:
    """CPU substitute for physics. Recovery-only views contain a sofa."""
    def __init__(self, fallen):
        self.k = 0
        self.fallen = fallen
        self.Z = {k: np.ones(2).view(Vector) for k in
                  ("move-ego-0-0", "move-ego-0-1", "move-ego-90-1", "move-ego--90-1", "yaw-L-2-free", "yaw-R-2-free")}
        self.body = NS(xy=np.zeros(2), yaw=0., head_z=.6 if fallen else 1.7)

    @property
    def t(self):
        return self.k / 30.

    def step(self, z):
        self.k += 1
        if self.k >= 31:
            self.body.head_z = 1.7
        return True

    def observe(self, detect=False):
        return dict(t=self.t, dets=None)

    def visible_instances(self):
        return {0: .1} if self.fallen and self.k < 31 else {}

    def observer_view(self):
        return np.zeros((2, 2, 3), np.uint8)


def clock_agent():
    agent = Agent(RuleBrain(), mind=False)
    # Keep actual tick/recovery logic; replace image processing and navigation.
    agent._perceive = lambda env, obs, detect: agent.looks.append(dict(t=round(obs["t"], 3), frame=len(agent.looks)))
    agent._explore_step = lambda *args: None
    return agent


@pytest.mark.parametrize("fallen", [False, True], ids=["normal", "fall-and-recovery"])
def test_tick_advances_once_behaviorally(fallen):
    env, agent = ClockEnv(fallen), clock_agent()
    for _ in range(90):
        before = env.k
        assert agent.tick(env)
        assert env.k == before + 1
    assert agent.falls == int(fallen)
    assert agent.got_up == int(fallen)
    print("90 ticks,", env.k, "physics steps, falls/recoveries", agent.falls, agent.got_up)


def run_runner_loop(fallen=True):
    """Execute the actual nested run_until AST without importing Modal or credentials."""
    tree = ast.parse((ROOT / "src/scout/env/episode.py").read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_until")
    env, agent = ClockEnv(fallen), clock_agent()
    namespace = dict(env=env, agent=agent, vis=[], maps=[], LOOK_EVERY=15, np=np)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "src/scout/env/episode.py"), "exec"), namespace)
    namespace["run_until"](3., NS(append_data=lambda frame: None), [])
    return namespace


def test_evaluator_visibility_has_one_matching_captured_head_frame():
    ns = run_runner_loop()
    actual = [(v["frame"], v["t"]) for v in ns["vis"]]
    expected = [(v["frame"], v["t"]) for v in ns["agent"].looks]
    print("evaluator frame/time:", actual, "captured frame/time:", expected)
    assert actual == expected


def test_recovery_only_view_does_not_make_quiz_answer_observable():
    ns = run_runner_loop()
    log, mem, ev = evidence()
    ev["visibility"] = ns["vis"]
    log["looks"] = ns["agent"].looks
    log["quiz"] = [dict(category="sofa", observed=False, xy=None, cited=[])]
    result = score_quiz(log, mem, ev)
    print("sofa visible only before captured frames:", result)
    assert result["accuracy"] == 1.


def reset_env():
    env = ScoutEnv.__new__(ScoutEnv)
    env.seed = 0
    env.k = 0
    env.moves = []
    env.M = NS(body_pos=np.array([[0., 0., 0.], [1., 1., .5]]), body_parentid=np.array([0, 0]))
    env.D = NS(xpos=env.M.body_pos, qpos=np.zeros(7), qvel=np.zeros(6), ctrl=np.zeros(2), act=np.zeros(0), time=0.)
    env.mj = NS(mj_forward=lambda model, data: None)
    env.instances = {0: dict(id=0, key="sofa", category="sofa", bodies=[1])}
    env.body = NS(place=lambda xy, yaw: None, step=lambda z: None)
    env.layout = NS(grid=NS(free=np.ones((1, 1), bool), pos=lambda ij: np.zeros(2)))
    env.observe = lambda: dict(t=env.t)
    return env


def test_reset_restores_house_and_clears_interventions():
    env = reset_env()
    env.reset()
    baseline = copy.deepcopy(env.truth_objects())
    env.move_object(0, [10, 10])
    env.reset()
    print("baseline:", baseline, "after reset:", env.truth_objects(), "moves:", env.moves)
    assert env.truth_objects() == baseline and env.moves == []


def test_snapshot_restores_object_and_moves_control():
    env = reset_env()
    snapshot = env.snapshot()
    baseline = copy.deepcopy(env.truth_objects())
    env.move_object(0, [10, 10])
    env.restore(snapshot)
    assert env.truth_objects() == baseline and env.moves == []


def video_state(path, same_view=False):
    spec = importlib.util.spec_from_file_location("critic_env_video", ROOT / "scripts/env_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log, mem, ev = evidence()
    if same_view:
        mem["sightings"] = [dict(mem["sightings"][0], id=i, t=float(i), frame=i) for i in range(3)]
        mem["tracks"] = [dict(mem["tracks"][0], sightings=[0, 1, 2])]
    (path / "log.json").write_text(json.dumps(log))
    (path / "memory.json").write_text(json.dumps(mem))
    return mod.State(str(path))


def test_confident_labels_require_distinct_viewpoints(tmp_path):
    state = video_state(tmp_path, same_view=True)
    print("three stationary sightings labelled confident:", state.pins(20., confident_only=True))
    assert state.pins(20., confident_only=True) == []


def test_no_head_frame_is_shown_before_first_capture(tmp_path):
    state = video_state(tmp_path)
    frame = state.head_frame(0.)
    print("first capture t=10, selected at t=0:", frame)
    assert frame is None


def test_floor_coverage_denominator_includes_unmapped_house_floor():
    log, mem, ev = evidence()
    layout = NS(grid=NS(free=np.ones((2, 1), bool), pos=lambda c: np.array([.5, .5]) if c[0] == 0 else np.array([10., 10.])),
                rooms=[NS(id=0)], room_at=lambda p: NS(id=0))
    maps = dict(shape=np.array([1, 1]), origin=np.array([0., 0.]), res=1., t=np.array([0.]),
                seen_0=np.packbits(np.ones((1, 1), bool)))
    result = score_explore(log, ev, layout, maps, 100.)
    print("one of two true floor cells mapped:", result)
    assert result["floor_seen"] == .5
