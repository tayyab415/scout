"""The student brain: a tiny trust network over the memory table. CPU only, no model calls."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scout.brains import make_brain
from scout.env.core import PROMPTS
from scout.env.scoring import score_citations, score_quiz
from scout.student.trust import CATEGORIES, FEATURE_NAMES, N_FEATURES, WEIGHTS, StudentBrain, TrustMLP, features, parse_table

ROOT = Path(__file__).parent / ".."
SAMPLE = ROOT / "examples/sample_episode/unchanged"


def test_table_parses_back_to_every_track_with_misses_and_verdicts():
    mem = json.loads((SAMPLE / "memory.json").read_text())
    rows = parse_table(mem["table"])
    assert [r["id"] for r in rows] == [tr["id"] for tr in mem["tracks"]]
    for r, tr in zip(rows, mem["tracks"]):
        assert r["label"] == tr["label"] and r["n"] == len(tr["sightings"]) and r["confident"] == tr["confident"]
        assert r["misses"] == len(tr["misses"])
        assert r["retracted"] == (tr["status"] == "retracted") and r["gone"] == (tr["status"] == "gone")
    X = features(rows)
    assert X.shape == (len(rows), N_FEATURES) and np.isfinite(X).all()
    assert len(FEATURE_NAMES) == N_FEATURES == 11 + len(CATEGORIES)


def test_features_use_nothing_but_the_table():
    """Two tables that print the same give the same features, whatever else the memory held."""
    table = ("object_id | object | first_seen_s | last_seen_s | times_seen | best_confidence | x_m | y_m | ids | confident\n"
             "0 | sofa | 10.0 | 200.0 | 7 | 0.91 | 1.0 | 2.0 | [3, 9, 40] | confident (3+ consistent sightings from 2+ viewpoints)\n"
             "1 | laptop | 50.0 | 50.0 | 1 | 0.55 | 4.0 | 2.0 | [12] | tentative\n"
             "1 | laptop NOT seen at its last known place at t=80.0s\n"
             "1 | laptop: a second look found nothing there, probably a misrecognition")
    rows = parse_table(table)
    assert rows[1]["misses"] == 1 and rows[1]["retracted"] and not rows[0]["retracted"]
    assert np.array_equal(features(rows), features(parse_table(table)))


def test_training_fits_a_separable_toy_and_saves_loadable_weights(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, N_FEATURES))
    y = (X[:, 1] + 0.5 * X[:, 0] > 0).astype(float)  # best_confidence and times_seen decide
    net = TrustMLP(hidden=8, seed=0)
    net.fit(X[:300], y[:300], X[300:], y[300:], epochs=60)
    acc = ((net.predict(X[300:]) >= 0.5) == (y[300:] > 0.5)).mean()
    assert acc > 0.9
    net.save(str(tmp_path / "w.npz"), note="toy")
    again = TrustMLP.load(str(tmp_path / "w.npz"))
    assert np.allclose(again.predict(X[:5]), net.predict(X[:5]))
    assert net.n_params() == N_FEATURES * 8 + 8 + 8 + 1 == 233


def test_shipped_weights_load_and_name_their_training_houses():
    assert os.path.exists(WEIGHTS)
    z = np.load(WEIGHTS, allow_pickle=False)
    plan = json.loads((ROOT / "configs/frozen_final3.json").read_text())
    trained_on = set(z["houses_train"].tolist()) | set(z["houses_val"].tolist())
    assert trained_on, "weights carry the houses they were trained on"
    assert not (trained_on & set(plan["houses"])), "no final-evaluation house entered training"
    assert TrustMLP.load().n_params() == 233


def test_student_answers_the_quiz_in_the_recorded_format_and_scores():
    brain = make_brain("student")
    assert isinstance(brain, StudentBrain)
    log = json.loads((SAMPLE / "log.json").read_text())
    mem = json.loads((SAMPLE / "memory.json").read_text())
    ev = json.loads((SAMPLE / "evaluator.json").read_text())
    quiz = []
    for cat in PROMPTS:
        q = f"Did you see a {cat}? If so, where was it?"
        a = brain.answer(mem["table"], q)
        assert isinstance(a["observed"], bool) and a["answer"].strip()
        if a["observed"]:
            assert mem["tracks"][a["object_id"]]["label"] == cat, "a yes names an object of the asked category"
        oid = a["object_id"]
        xy = mem["sightings"][mem["tracks"][oid]["sightings"][-1]]["xyz"][:2] if isinstance(oid, int) else None
        quiz.append(dict(t=log["quiz"][0]["t"], category=cat, question=q, answer=a["answer"], observed=a["observed"],
                         valid=True, error=None, object_id=oid, xy=xy, cited=a["cited_sightings"], malformed_citations=[]))
    s = score_quiz(dict(quiz=quiz, looks=log["looks"], t_end=log["t_end"]), mem, ev)
    c = score_citations(dict(quiz=quiz, looks=log["looks"], t_end=log["t_end"]), mem, ev)
    assert s["n"] == 16 and s["errors"] == 0 and 0.0 <= s["accuracy"] <= 1.0
    assert c["invalid_id"] == 0 and c["future"] == 0 and c["wrong_category"] == 0


def test_armchair_question_does_not_match_a_chair():
    table = ("object_id | object | first_seen_s | last_seen_s | times_seen | best_confidence | x_m | y_m | ids | confident\n"
             "0 | chair | 10.0 | 200.0 | 7 | 0.91 | 1.0 | 2.0 | [3, 9, 40] | confident")
    a = make_brain("student").answer(table, "Did you see an armchair? If so, where was it?")
    assert a["observed"] is False and a["object_id"] is None


def test_decide_explores_while_openings_remain_then_checks_the_least_certain_object():
    brain = make_brain("student")
    mem = json.loads((SAMPLE / "memory.json").read_text())
    assert brain.decide(mem["table"], "time 100 s; unexplored openings reachable: 3; ...") == {"do": "explore"}
    d = brain.decide(mem["table"], "time 300 s; unexplored openings reachable: 0; ...")
    assert d["do"] == "check"
    rows = brain.trust(mem["table"])
    live = [r for r in rows if not r["retracted"] and not r["gone"]]
    assert d["object_id"] == min(live, key=lambda r: abs(r["p"] - 0.5))["id"]


def test_decide_never_picks_an_object_the_agent_would_refuse():
    brain = make_brain("student")
    mem = json.loads((SAMPLE / "memory.json").read_text())
    first = brain.decide(mem["table"], "time 300 s; unexplored openings reachable: 0; ...")["object_id"]
    state = (f"time 320 s; position x=1.0 y=1.0; unexplored openings reachable: 0; last object checked: {first}; "
             f"objects already checked twice (do not choose): [{first}]; confident objects not seen for the longest: ")
    second = brain.decide(mem["table"], state)
    assert second["do"] == "check" and second["object_id"] != first
